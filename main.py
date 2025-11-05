import os
import argparse
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.theme import Theme
from rich.progress import Progress, SpinnerColumn, TextColumn

# Try to load .env file if python-dotenv is available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from DocumentProcessor import DocumentProcessor
from MistralRAGSystem import MistralRAGSystem

def main():
    """Main function to run the Mistral RAG system."""
    parser = argparse.ArgumentParser(description='Mistral API RAG System for PDF and HTML documents')
    parser.add_argument('--docs_folder', type=str, help='Folder containing PDF and HTML files', default='/documentacion')
    parser.add_argument('--vector_store', type=str, default='local_vectorstore', help='Path to save/load vector store')
    parser.add_argument('--rebuild', action='store_true', help='Rebuild vector store even if it exists')
    parser.add_argument('--clear-cache', action='store_true', help='Clear all caches before starting')
    parser.add_argument('--language', type=str, default='galician', choices=['english', 'spanish', 'galician'], 
                       help='Language for the prompt template')
    parser.add_argument('--chunk_size', type=int, default=300, help='Size of text chunks')
    parser.add_argument('--chunk_overlap', type=int, default=15, help='Overlap between chunks')
    parser.add_argument('--k', type=int, default=4, help='Number of documents to retrieve')
    parser.add_argument('--threshold', type=float, default=0.7, help='How hard filter documents')
    parser.add_argument('--search_type', type=str, default="mmr", 
                        help='Way of performing search (default: mmr, possible: similarity)')
    parser.add_argument('--provider', type=str, default="claude", 
                       help='Servidor del LM (mistral, claude)')
    parser.add_argument('--model', type=str, default="claude-3-5-haiku-20241022", 
                       help='Claude or Mistral model name (default claude-3-5-sonnet-20241022)')
    parser.add_argument('--api_key', type=str, default=None,
                       help='API key (if not set, will use MISTRAL_API_KEY or ANTHROPIC_API_KEY environment variable)')
    parser.add_argument('--embedding_model', type=str, default="sentence-transformers/all-MiniLM-L6-v2",
                       help='HuggingFace model to use for embeddings')
    parser.add_argument('--verbose', action='store_true', help='Show detailed information including sources')
    parser.add_argument('--temperature', type=float, default=0.1,
                       help='Temperature for text generation (0.0-1.0)')
    parser.add_argument('--max_tokens', type=int, default=512,
                       help='Maximum tokens to generate')
    args = parser.parse_args()
    
    # Set API key environment variable if provided
    if args.api_key:
        if args.provider == 'mistral':
            os.environ["MISTRAL_API_KEY"] = args.api_key
        elif args.provider == 'claude':
            os.environ["ANTHROPIC_API_KEY"] = args.api_key
    
    # Create a custom theme for rich
    custom_theme = Theme({
        "info": "cyan",
        "warning": "yellow",
        "error": "bold red",
        "success": "bold green",
    })
    
    # Initialize rich console with our theme
    console = Console(theme=custom_theme, color_system="truecolor")
    
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
            cache_dir=".doc_cache"
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
        cache_info = f"[bold blue]Estadísticas de caché:[/bold blue]"
        cache_info += f"\nArquivos rastreados: [yellow]{stats['tracked_files']}[/yellow]"
        cache_info += f"\nPDFs: [yellow]{stats['files_by_type']['pdf']}[/yellow]"
        cache_info += f"\nHTMLs: [yellow]{stats['files_by_type']['html']}[/yellow]"
        cache_info += f"\nEmbeddings en caché: [yellow]{stats['embedding_cache']['cached_documents']}[/yellow]"
        cache_info += f"\nTamaño caché: [yellow]{stats['embedding_cache']['cache_size_mb']:.2f} MB[/yellow]"
        console.print(Panel.fit(cache_info, title="Información de caché", border_style="cyan"))
    
    # Initialize the RAG system
    with Progress(
        SpinnerColumn(),
        TextColumn(f"[success]Inicializando sistema con {args.provider.upper()}..."),
        console=console,
        transient=True
    ) as progress:
        task = progress.add_task("init", total=None)
        rag = MistralRAGSystem(
            vectorstore=processor.vectorstore,
            k=args.k,
            threshold=args.threshold,
            search_type=args.search_type,
            language=args.language,
            provider=args.provider,
            model_name=args.model,
            api_key=args.api_key,
            temperature=args.temperature,
            max_tokens=args.max_tokens
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
    
    titulo_ascii = '''
             ______ _______ _____         _____   _______ ____  
            |  ____|__   __/ ____|  /\\   / ____| |__   __/ __ \\ 
            | |__     | | | (___   /  \\ | |   ______| | | |  | |
            |  __|    | |  \\___ \\ / /\\ \\| |  |______| | | |  | |
            | |____   | |  ____) / ____ \\ |____     | | | |__| |
            |______|  |_| |_____/_/    \\_\\_____|    |_|  \\____/ 


                                      :+***+:                                  
                         *********************************                     
                   *********************************************               
               *****************************************************           
                   *********************************************               
         **+           *************************************           =**     
       ***********         -***************************-         ***********   
      *******************       *******************       *******************  
       ************************     ***********     ************************   
                                -*****  ***  *****-                            
                              -******   ***   ******-                          
     *************************+    .***********.    +************************* 
      ******************       *********************       ******************  
       **********          *****************************          **********   
         **            *************************************            **     
                   *********************************************               
               *****************************************************           
                   *********************************************               
                         *********************************                     
                                                                               
                                                                               
            ¡Bo día/Boa tarde! 👋

        Son o teu asistente virtual especializado na normativa e servizos da Universidade. Estou aquí para axudarche con calquera dúbida que teñas sobre:
        
        📚 Biblioteca (normas, préstamos, dereitos, guías...)
        🎓 Matrícula (grao, máster, doutoramento)
        💰 Bolsas e axudas (becas, Santander, dificultades económicas, comedor...)
        🌍 Mobilidade (programas de intercambio, axudas MilleniumBus...)
        🏛 Normativa xeral (dereitos, regulamentos internos, preguntas frecuentes)
        
        Pregúntame o que necesites e intentarei atopar a información máis actualizada nos documentos oficiais da Universidade.
'''

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
                # Mostrar número de interacciones en el historial
                history_len = len(rag.get_history())
                console.print(f"[dim]Conversacións no historial: {history_len}[/dim]")
                
                console.print("\n[bold]Textos dos que extráese a información:[/bold]")
                for i, doc in enumerate(sources[:args.k]):
                    console.print(Panel(
                        doc.page_content[:200] + "...", 
                        title=f"Texto {i+1}", 
                        border_style="blue"
                    ))
                
        except Exception as e:
            console.print(f"[error]Error procesando consulta:[/error] {e}")
            if args.verbose:
                import traceback
                console.print(traceback.format_exc())

if __name__ == "__main__":
    main()