import os
import sys
import re
import argparse
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.theme import Theme
from rich.progress import Progress, SpinnerColumn, TextColumn
from dotenv import load_dotenv


from LLMManager import LLMManager
from DocumentProcessor import DocumentProcessor
from RAGSystem import RAGSystem
from portada import titulo_ascii

def main():
    """Main function to run the Mistral RAG system."""
    parser = argparse.ArgumentParser(description='Mistral API RAG System for PDF and HTML documents')

    # Document Processing
    parser.add_argument('--docs_folder', type=str, default='crawl/text', help='Folder containing PDF and HTML files')
    parser.add_argument('--map_json', type=str, default='crawl/map.json', help='JSON file mapping filename to URL')
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
    
    # TF-IDF
    parser.add_argument('--use_tfidf', action='store_true', default=True, help='Use TF-IDF enhancement')
    parser.add_argument('--tfidf_mode', type=str, default='hybrid', help='TF-IDF mode: rerank, hybrid, tfidf, or filter')
    parser.add_argument('--tfidf_weight', type=float, default=0.3, help='TF-IDF weight in hybrid mode (0.0-1.0)')
    parser.add_argument('--tfidf_threshold', type=float, default=0.1, help='Minimum TF-IDF score for filter mode')
    
    # LLM Provider
    parser.add_argument('--provider', type=str, default='claude', help='LLM provider: mistral or claude')
    parser.add_argument('--model', type=str, default='claude-3-5-haiku-20241022', help='Model name')
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
    import builtins
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
    
    
    # Initialize the LLM
    llm = LLMManager(
        provider=args.provider,
        model_name=args.model, 
        api_key=args.api_key
    ).llm

    # Initialize the document processor
    with Progress(
        SpinnerColumn(),
        TextColumn("[success]Inicializando analizador de textos..."),
        console=console,
        transient=True
    ) as progress:
        progress.add_task("init", total=None)
        print(args.docs_folder)
        processor = DocumentProcessor(
            docs_folder=args.docs_folder,
            embedding_model_name=args.embedding_model,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            verbose=args.verbose,
            cache_dir=args.cache_dir,
            prefix_mode=args.prefix_mode,
            llm=llm,
            map_json=args.map_json,
            crawler_metadata_path='crawl/metadata.json',
            text_folder='crawl/text'
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
                # Save vectorstore (will save only if there were changes)
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
        if args.tfidf_mode == "hybrid" or args.tfidf_mode == "tfidf":
            rag = RAGSystem(
                vectorstore=processor.vectorstore,
                k=args.k,
                threshold=args.threshold,
                search_type=args.search_type,
                language=args.language,
                llm=llm,
                provider=args.provider,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                max_history_length=args.max_history_length,
                use_tfidf=args.use_tfidf,
                tfidf_mode=args.tfidf_mode,
                tfidf_weight=args.tfidf_weight,
                tfidf_threshold=args.tfidf_threshold,
                tfidf_vectorizer=processor.tfidf_vectorizer,
                tfidf_matrix=processor.tfidf_matrix,
                tfidf_documents=processor.tfidf_documents
            )
        else:
            rag = RAGSystem(
                vectorstore=processor.vectorstore,
                k=args.k,
                threshold=args.threshold,
                search_type=args.search_type,
                language=args.language,
                llm=llm,
                provider=args.provider,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                max_history_length=args.max_history_length,
                use_tfidf=args.use_tfidf,
                tfidf_mode=args.tfidf_mode,
                tfidf_weight=args.tfidf_weight,
                tfidf_threshold=args.tfidf_threshold
            )
    
    # Welcome message with system info
    system_info = "[bold blue]Especificación actual do sistema[/bold blue]"
    if args.verbose:
        system_info += f"\nProvider: [yellow]{args.provider.upper()}[/yellow]"
        system_info += f"\nModelo: [yellow]{args.model}[/yellow]"
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
    
    question = ''
    answer = ''



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
                answer, sources = rag.query(question, use_history=True)  # MODIFICADO: use_history=True
            
            console.print(Panel(
                Markdown(answer), 
                title="Resposta", 
                border_style="green"
            ))
            
            if args.verbose:
                history_len = len(rag.get_history())
                console.print(f"[dim]Conversacións no historial: {history_len}[/dim]")

                console.print("\n[bold]Textos dos que extráese a información:[/bold]")

                # Parse it
                for i, doc in enumerate(sources[:args.k]):
                    text = doc.page_content
                    first_line, _, rest = text.partition("\n")
                    
                    parts = first_line.split('|')
                    
                    if len(parts) >= 5:
                        filename, file_type, url, last_modified, tipo_data = parts[:5]
                        
                        title = f"Texto {i+1}"
                        metadata = f"[bold]{filename} ({file_type})[/bold]\n[dim]🔗 {url}[/dim]"
                        
                        if last_modified and last_modified.lower() != "no hai data":
                            date_emoji = "📅" if "modificación" in tipo_data.lower() else "🕐" if "crawl" in tipo_data.lower() else "❓"
                            metadata += f"\n[dim]{date_emoji} {last_modified}"
                            if tipo_data and tipo_data.lower() != "no hai data":
                                metadata += f" ({tipo_data})[/dim]"
                            else:
                                metadata += "[/dim]"
                        
                        content = f"{metadata}\n\n{rest.strip()[:200]}..."
                    else:
                        title = f"Texto {i+1}"
                        content = text.strip()[:200] + "..."
                    
                    console.print(Panel(
                        content,
                        title=title,
                        border_style="blue",
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
