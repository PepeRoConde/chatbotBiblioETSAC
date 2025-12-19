import os
import sys
import re
import argparse
import builtins
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.theme import Theme
from rich.progress import Progress, SpinnerColumn, TextColumn
from dotenv import load_dotenv


from .LLMManager import LLMManager
from .preprocessing.DocumentProcessor import DocumentProcessor
from .rag.RAGSystem import RAGSystem
from .utils.portada import titulo_ascii

def main():
    """Main function to run the Mistral RAG system."""
    parser = argparse.ArgumentParser(description='Mistral API RAG System for PDF and HTML documents')

    # Document Processing
    parser.add_argument('--state_dir', type=str, default='crawl', help='Folder containing PDF and HTML files')
    parser.add_argument('--docs_folder', type=str, default='text', help='Folder containing PDF and HTML files')
    parser.add_argument('--map_json', type=str, default='map.json', help='JSON file mapping filename to URL')
    parser.add_argument('--chunk_size', type=int, default=2000, help='Size of text chunks')
    parser.add_argument('--chunk_overlap', type=int, default=250, help='Overlap between chunks')
    parser.add_argument('--prefix_mode', type=str, default='source', help='Chunk prefix mode: none, source, llm')
    parser.add_argument('--check', type=bool, default=False, help='Either check changes in documents or not')
    
    # Vector Store
    parser.add_argument('--vector_store', type=str, default='local_vectorstore', help='Path to save/load vector store')
    parser.add_argument('--embedding_model', type=str, default='sentence-transformers/all-MiniLM-L6-v2', help='HuggingFace embedding model')
    parser.add_argument('--rebuild', action='store_true', help='Rebuild vector store even if it exists')
    
    # Retrieval
    parser.add_argument('--k', type=int, default=8, help='Number of documents to retrieve')
    parser.add_argument('--threshold', type=float, default=0.7, help='Similarity threshold for filtering')
    parser.add_argument('--search_type', type=str, default='mmr', help='Search type: similarity or mmr')
    
    # BM25
    parser.add_argument('--use_bm25', action='store_true', default=True, help='Use BM25 enhancement')
    parser.add_argument('--bm25_mode', type=str, default='hybrid', help='BM25 mode: rerank, hybrid, bm25, or filter')
    parser.add_argument('--bm25_weight', type=float, default=0.5, help='BM25 weight in hybrid mode (0.0-1.0)')
    parser.add_argument('--bm25_threshold', type=float, default=0.1, help='Minimum BM25 score for filter mode')
    
    # LLM Provider
    parser.add_argument('--provider', type=str, default='anthropic', help='LLM provider: mistral or claude')
    parser.add_argument('--model', type=str, default='claude-sonnet-4-5', help='Model name for final answers')
    parser.add_argument('--query_model', type=str, default="claude-3-haiku-20240307", help='Model name for query optimization (defaults to same as --model)')
    parser.add_argument('--use_query_optimization',default = True, action='store_true', help='Enable query optimization (two-stage LLM)')
    parser.add_argument('--api_key', type=str, default=None, help='API key (uses env var if not set)')
    parser.add_argument('--temperature', type=float, default=0.1, help='Temperature for generation (0.0-1.0)')
    parser.add_argument('--max_tokens', type=int, default=512, help='Maximum tokens to generate')
    
    # Conversation
    parser.add_argument('--language', type=str, default='galician', choices=['english', 'spanish', 'galician'], help='Language for prompts')
    parser.add_argument('--max_history_length', type=int, default=10, help='Maximum conversation turns to keep')
    
    # Cache & System
    parser.add_argument('--cache_dir', type=str, default='.doc_cache', help='Directory for cache storage')
    parser.add_argument('--clear-cache', action='store_true', help='Clear all caches before starting')
    parser.add_argument('--verbose', action='store_true', help='Show detailed information')
    parser.add_argument('--crop_chunks', action='store_true', help='Crop displayed chunks to 200 characters (default: show full chunks in verbose mode)')
    
    args = parser.parse_args() 

    # Set API key environment variable if provided
    if args.api_key:
        if args.provider == 'mistral':
            os.environ["MISTRAL_API_KEY"] = args.api_key
        elif args.provider == 'claude':
            os.environ["ANTHROPIC_API_KEY"] = args.api_key
    
    load_dotenv()
    
    # Create a custom theme for rich
    custom_theme = Theme({
        "info": "cyan",
        "warning": "yellow",
        "error": "bold red",
        "success": "bold green",
    })

    is_git_bash = 'MSYSTEM' in os.environ or 'MINGW' in os.environ.get('MSYSTEM', '')

    console = Console(
        theme=custom_theme,
        color_system="truecolor",
        legacy_windows=False if is_git_bash else None,
        force_terminal=True
    )
    
    # Share the console and verbose setting globally
    setattr(builtins, 'rich_console', console)
    setattr(builtins, 'verbose_mode', args.verbose)
    
    # Clear cache if requested
    if args.clear_cache:
        console.print("[yellow]Limpiando caché...[/yellow]")
        import shutil
        cache_dir = Path(".doc_cache")
        if cache_dir.exists():
            shutil.rmtree(cache_dir)
            console.print("[success]Caché eliminada[/success]")
    
    # Initialize the main LLM (for final answers)
    llm = LLMManager(
        provider=args.provider,
        model_name=args.model, 
        api_key=args.api_key
    ).llm
    
    # Initialize query optimization LLM if enabled
    llm_query = None
    if args.use_query_optimization:
        # If query_model is specified, use it; otherwise default to Haiku for Claude
        if args.query_model:
            query_model_name = args.query_model
        else:
            # Smart defaults based on provider
            if args.provider == 'claude':
                query_model_name = 'claude-haiku-4'  # Fast and cheap for query generation
            else:
                query_model_name = args.model  # Use same model for other providers
        
        llm_query = LLMManager(
            provider=args.provider,
            model_name=query_model_name,
            api_key=args.api_key
        ).llm
        
        if args.verbose:
            console.print(f"[info]Query optimization enabled:[/info]")
            console.print(f"  - Query model: [yellow]{query_model_name}[/yellow]")
            console.print(f"  - Answer model: [yellow]{args.model}[/yellow]")

    # Initialize the document processor
    with Progress(
        SpinnerColumn(),
        TextColumn("[success]Inicializando analizador de textos..."),
        console=console,
        transient=True
    ) as progress:
        progress.add_task("init", total=None)
        processor = DocumentProcessor(
            docs_folder=args.state_dir + '/' + args.docs_folder,
            embedding_model_name=args.embedding_model,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            verbose=args.verbose,
            cache_dir=args.cache_dir,
            prefix_mode=args.prefix_mode,
            llm=llm,
            map_json=args.state_dir + '/' + args.map_json,
            crawler_metadata_path=args.state_dir + '/' + 'metadata.json',
            text_folder=args.state_dir + '/' + 'text'
        )
    
    # Check if vector store exists and if we need to rebuild
    if not os.path.exists(args.vector_store) or args.rebuild:
        # Full rebuild
        with Progress(
            SpinnerColumn(),
            TextColumn("[success]Procesando textos e construíndo a base vectorial..."),
            console=console,
            transient=True
        ) as progress:
            task = progress.add_task("Procesando...", total=None)
            processor.process(force_reload=True, incremental=False)
            processor.save_vectorstore(args.vector_store)
    else:
        # Load existing vectorstore
        with Progress(
            SpinnerColumn(),
            TextColumn("[success]Cargando vectorstore..."),
            console=console,
            transient=True
        ) as progress:
            task = progress.add_task("cargando", total=None)
            processor.load_vectorstore(args.vector_store)
        
        if args.check:
            # Check for changes and rebuild if necessary (with cached embeddings)
            with Progress(
                SpinnerColumn(),
                TextColumn("[success]Comprobando actualizacións..."),
                console=console,
                transient=True
            ) as progress:
                task = progress.add_task("comprobando", total=None)
                processor.process(force_reload=False, incremental=True)
        processor.save_vectorstore(args.vector_store)
    
    # Show cache stats if verbose
    if args.verbose:
        stats = processor.get_cache_stats()
        cache_info = f"[bold blue]Estadísticas del sistema:[/bold blue]"
        cache_info += f"\nDocumentos totales: [yellow]{stats['total_documents']}[/yellow]"
        cache_info += f"\nCon embeddings: [green]{stats['embedded_documents']}[/green]"
        cache_info += f"\nPendientes: [yellow]{stats['needs_embeddings']}[/yellow]"
        cache_info += f"\nEmbeddings en caché: [yellow]{stats['embedding_cache']['cached_documents']}[/yellow]"
        cache_info += f"\nTamaño caché: [yellow]{stats['embedding_cache']['cache_size_mb']:.2f} MB[/yellow]"
        console.print(Panel.fit(cache_info, title="Información del sistema", border_style="cyan"))
    
    # Initialize the RAG system
    with Progress(
        SpinnerColumn(),
        TextColumn(f"[success]Inicializando sistema con {args.provider.upper()}..."),
        console=console,
        transient=True
    ) as progress:
        task = progress.add_task("init", total=None)
        
        # Common parameters for RAG system
        rag_params = {
            'vectorstore': processor.vectorstore,
            'k': args.k,
            'state_dir': args.state_dir,
            'threshold': args.threshold,
            'search_type': args.search_type,
            'language': args.language,
            'llm': llm,
            'llm_query': llm_query,
            'use_query_optimization': args.use_query_optimization,
            'provider': args.provider,
            'temperature': args.temperature,
            'max_tokens': args.max_tokens,
            'max_history_length': args.max_history_length,
            'use_bm25': args.use_bm25,
            'bm25_mode': args.bm25_mode,
            'bm25_weight': args.bm25_weight,
            'bm25_threshold': args.bm25_threshold
        }
        
        # Add BM25 components if needed
        if args.bm25_mode == "hybrid" or args.bm25_mode == "bm25":
            rag_params.update({
                'bm25_index': processor.bm25_index,
                'bm25_documents': processor.bm25_documents
            })
        
        rag = RAGSystem(**rag_params)
    
    # Welcome message with system info
    system_info = "[bold blue]Especificación actual do sistema[/bold blue]"
    if args.verbose:
        system_info += f"\nProvider: [yellow]{args.provider.upper()}[/yellow]"
        system_info += f"\nModelo: [yellow]{args.model}[/yellow]"
        if args.use_query_optimization:
            query_model_display = args.query_model if args.query_model else ('claude-haiku-4' if args.provider == 'claude' else args.model)
            system_info += f"\nModelo queries: [yellow]{query_model_display}[/yellow]"
            system_info += f"\nQuery optimization: [green]✓ ACTIVADO[/green]"
        system_info += f"\nLingua: [yellow]{args.language}[/yellow]"
        system_info += f"\nBase vectorial: [yellow]{args.vector_store}[/yellow]"
        system_info += f"\nTamaño dos textos: [yellow]{args.chunk_size}[/yellow]"
        system_info += f"\nTextos recuperados: [yellow]{args.k}[/yellow]"
        system_info += f"\nUmbral de recuperacion: [yellow]{args.threshold}[/yellow]"
        system_info += f"\nMétodo de recuperacion: [yellow]{args.search_type}[/yellow]"
        system_info += f"\nTemperatura: [yellow]{args.temperature}[/yellow]"
        system_info += f"\nMax tokens: [yellow]{args.max_tokens}[/yellow]"
    
    system_info += "\nEscriba [bold red]'sair'[/bold red] para saír."
    
    console.print(Panel.fit(
        system_info,
        title="O sistema está listo",
        border_style="green"
    ))
    
    console.print(titulo_ascii, style="rgb(196,45,137)")
    
    console.print("\n[dim]Comandos especiales:[/dim]")
    console.print("[dim]  - 'sair' → Saír do programa[/dim]")
    console.print("[dim]  - 'limpar' → Limpar historial de conversación[/dim]")
    console.print("[dim]  - 'historial' → Ver historial de conversación[/dim]\n")

    while True:
        question = console.input("\n[bold cyan]Insire a súa pregunta: [/bold cyan]")
        
        # Comandos especiales
        if question.lower() == 'sair':
            console.print("[yellow]Adeus! 👋[/yellow]")
            break
        
        if question.lower() == 'limpar':
            rag.clear_history()
            console.print("[green]✓ Historial de conversación limpo[/green]")
            continue
        
        if question.lower() == 'historial':
            history = rag.get_history()
            if not history:
                console.print("[yellow]Non hai historial de conversación[/yellow]")
            else:
                console.print(Panel.fit(
                    "\n".join([
                        f"[cyan]P{i+1}:[/cyan] {h['question']}\n[green]R{i+1}:[/green] {h['answer'][:100]}..."
                        for i, h in enumerate(history)
                    ]),
                    title=f"Historial ({len(history)} interaccións)",
                    border_style="blue"
                ))
            continue
        
        if not question.strip():
            console.print("[yellow]Por favor, insire unha pregunta[/yellow]")
            continue
            
        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[success]Procesando consulta..."),
                console=console,
                transient=True
            ) as progress:
                task = progress.add_task("procesando", total=None)
                # Obtener costes si verbose está activo
                if args.verbose:
                    answer, sources, cost_info = rag.query(question, use_history=True, return_costs=True)
                else:
                    answer, sources = rag.query(question, use_history=True, return_costs=False)
            
            # Construir título del panel con costes si verbose
            if args.verbose and cost_info and cost_info.get("total_cost", 0) > 0:
                by_stage = cost_info.get("by_stage", {})

                query_cost = by_stage.get("query_decision", 0.0)
                answer_cost = by_stage.get("answer", 0.0)
                
                title = (
                    f"Resposta | "
                    f"💰 ${cost_info['total_cost']:.6f} "
                    f"(🤖 ${query_cost:.6f} + "
                    f"🧠 ${answer_cost:.6f})"
                )
            else:
                title = "Resposta"
            
            console.print(Panel(
                Markdown(answer), 
                title=title, 
                border_style="green"
            ))
            
            if args.verbose:
                history_len = len(rag.get_history())
                console.print(f"[dim]Conversacións no historial: {history_len}[/dim]")
                console.print("\n[bold]Textos dos que extráese a información:[/bold]")
                
                for i, doc in enumerate(sources[:args.k]):
                    text = doc.page_content
                    first_line, _, rest = text.partition("\n")
                    
                    parts = first_line.split('|')
                    
                    # Get relevance scores and method from metadata
                    relevance_score = doc.metadata.get('relevance_score', None)
                    vector_score = doc.metadata.get('vector_score', None)
                    bm25_score_meta = doc.metadata.get('bm25_score', None)
                    retrieval_method = doc.metadata.get('retrieval_method', 'unknown')
                    
                    # Format score information
                    score_info = []
                    if relevance_score is not None:
                        score_info.append(f"[bold cyan]Score: {relevance_score:.4f}[/bold cyan]")
                    
                    if retrieval_method == 'hybrid':
                        method_emoji = "🔄"
                        method_text = "[bold magenta]Hybrid[/bold magenta] (Vector + BM25)"
                        if vector_score is not None:
                            score_info.append(f"[cyan]Vector: {vector_score:.4f}[/cyan]")
                        if bm25_score_meta is not None:
                            score_info.append(f"[yellow]BM25: {bm25_score_meta:.4f}[/yellow]")
                    elif retrieval_method == 'vector':
                        method_emoji = "🎯"
                        method_text = "[bold cyan]Vector[/bold cyan]"
                        if vector_score is not None:
                            score_info.append(f"[cyan]Vector: {vector_score:.4f}[/cyan]")
                    elif retrieval_method == 'bm25':
                        method_emoji = "📊"
                        method_text = "[bold yellow]BM25[/bold yellow]"
                        if bm25_score_meta is not None:
                            score_info.append(f"[yellow]BM25: {bm25_score_meta:.4f}[/yellow]")
                    else:
                        method_emoji = "❓"
                        method_text = "[dim]Unknown[/dim]"
                    
                    score_line = " | ".join(score_info) if score_info else "[dim]No score info[/dim]"
                    
                    if len(parts) >= 5:
                        filename, file_type, url, last_modified, tipo_data = parts[:5]
                        
                        title = f"{method_emoji} Texto {i+1} - {method_text}"
                        metadata = f"{score_line}\n\n[bold]{filename} ({file_type})[/bold]\n[dim]🔗 {url}[/dim]"
                        
                        if last_modified and last_modified.lower() != "no hai data":
                            date_emoji = "📅" if "modificación" in tipo_data.lower() else "🕐" if "crawl" in tipo_data.lower() else "❓"
                            metadata += f"\n[dim]{date_emoji} {last_modified}"
                            if tipo_data and tipo_data.lower() != "no hai data":
                                metadata += f" ({tipo_data})[/dim]"
                            else:
                                metadata += "[/dim]"
                        
                        # Show full chunks by default in verbose mode, crop only if --crop_chunks is specified
                        chunk_text = rest.strip()
                        if args.crop_chunks:
                            chunk_text = chunk_text[:200] + "..."
                        content = f"{metadata}\n\n{chunk_text}"
                    else:
                        title = f"{method_emoji} Texto {i+1} - {method_text}"
                        # Show full chunks by default in verbose mode, crop only if --crop_chunks is specified
                        chunk_text = text.strip()
                        if args.crop_chunks:
                            chunk_text = chunk_text[:200] + "..."
                        content = f"{score_line}\n\n{chunk_text}"
                    
                    console.print(Panel(
                        content,
                        title=title,
                        border_style="blue" if retrieval_method == "vector" else "yellow" if retrieval_method == "bm25" else "magenta",
                        padding=(1, 2)
                    ))

        except Exception as e:
            console.print(f"[error]Error procesando consulta:[/error] {e}")
            if args.verbose:
                import traceback
                console.print(traceback.format_exc())
if __name__ == "__main__":
    if sys.platform == 'win32':
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')

    main()
