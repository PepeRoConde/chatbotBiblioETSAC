import os
import argparse
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.theme import Theme
from rich.progress import Progress, SpinnerColumn, TextColumn

from DocumentProcessor import DocumentProcessor
from MistralRAGSystem import MistralRAGSystem

def main():
    """Main function to run the Mistral RAG system."""
    parser = argparse.ArgumentParser(description='Mistral API RAG System for PDF and HTML documents')
    parser.add_argument('--docs_folder', type=str, help='Folder containing PDF and HTML files', default='/Users/pepe/OneDrive - Universidade da Coruña/documentacion_y_normativa')
    parser.add_argument('--vector_store', type=str, default='local_vectorstore', help='Path to save/load vector store')
    parser.add_argument('--rebuild', action='store_true', help='Rebuild vector store even if it exists')
    parser.add_argument('--language', type=str, default='galician', choices=['english', 'spanish', 'galician'], 
                       help='Language for the prompt template')
    parser.add_argument('--chunk_size', type=int, default=300, help='Size of text chunks')
    parser.add_argument('--chunk_overlap', type=int, default=15, help='Overlap between chunks')
    parser.add_argument('--k', type=int, default=4, help='Number of documents to retrieve')
    parser.add_argument('--model', type=str, default="mistral-medium", 
                       help='Mistral model name (tiny, small, medium, large)')
    parser.add_argument('--api_key', type=str, default=None,
                       help='Mistral API key (if not set, will use MISTRAL_API_KEY environment variable)')
    parser.add_argument('--embedding_model', type=str, default="sentence-transformers/all-MiniLM-L6-v2",
                       help='HuggingFace model to use for embeddings')
    parser.add_argument('--verbose', action='store_true', help='Show detailed information including sources')
    args = parser.parse_args()
    
    # Set API key environment variable if provided
    if args.api_key:
        os.environ["MISTRAL_API_KEY"] = args.api_key
    
    # Check if API key is available
    if not args.api_key and not os.environ.get("MISTRAL_API_KEY"):
        raise ValueError("Mistral API key must be provided either via --api_key argument or MISTRAL_API_KEY environment variable")
    
    # Create a custom theme for rich
    custom_theme = Theme({
        "info": "cyan",
        "warning": "yellow",
        "error": "bold red",
        "success": "bold green",
    })
    
    # Initialize rich console with our theme
    console = Console(theme=custom_theme,color_system="truecolor")
    
    # Share the console and verbose setting globally
    # This is used by other classes to respect the verbose setting
    import builtins
    setattr(builtins, 'rich_console', console)
    setattr(builtins, 'verbose_mode', args.verbose)
    
    # Initialize the document processor with fancy progress display
    with Progress(
        SpinnerColumn(),
        TextColumn("[success]Inicializando analizador de textos..."),
        console=console,
        transient=True
    ) as progress:
        progress.add_task("init", total=None)
        processor = DocumentProcessor(
            docs_folder=args.docs_folder,
            embedding_model_name=args.embedding_model,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            verbose=args.verbose
        )
    
    # Check if vector store exists and if we need to rebuild
    if not os.path.exists(args.vector_store) or args.rebuild:
        with Progress(
            SpinnerColumn(),
            TextColumn("[success]Procesando textos e construíndo a base vectorial (esto faise unha única vez)..."),
            console=console,
            transient=True
        ) as progress:
            task = progress.add_task("Procesando...", total=None)
            processor.process()
            processor.save_vectorstore(args.vector_store)
    else:
        with Progress(
            SpinnerColumn(),
            TextColumn("[success]Cargando base vectorial existente..."),
            console=console,
            transient=True
        ) as progress:
            task = progress.add_task("cargando", total=None)
            processor.load_vectorstore(args.vector_store)
    
    # Initialize the RAG system with Mistral
    with Progress(
        SpinnerColumn(),
        TextColumn("[success]Inicializando sistema con Mistral..."),
        console=console,
        transient=True
    ) as progress:
        task = progress.add_task("init", total=None)
        rag = MistralRAGSystem(
            vectorstore=processor.vectorstore,
            k=args.k,
            language=args.language,
            model_name=args.model,
            api_key=args.api_key
        )
    
    # Welcome message with system info
    system_info = "[bold blue]Especificación actual do sistema[/bold blue]"
    if args.verbose:
        system_info += f"\nMistral Modelo da lingua: [yellow]{args.model}[/yellow]"
        system_info += f"\nLingua: [yellow]{args.language}[/yellow]"
        system_info += f"\nBase vectorial: [yellow]{args.vector_store}[/yellow]"
        system_info += f"\nTamaño dos textos (en palabras): [yellow]{args.chunk_size}[/yellow]"
        system_info += f"\nTextos recuperados por consulta: [yellow]{args.k}[/yellow]"
    
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
'''

    console.print(titulo_ascii, style="rgb(196,45,137)")
    # Interactive query loop
    while True:
        question = console.input("\n[bold cyan]Insire a súa pregunta: [/bold cyan]")
        if question.lower() == 'sair':
            break
            
        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[success]Procesando consulta..."),
                console=console,
                transient=True
            ) as progress:
                task = progress.add_task("procesando", total=None)
                answer, sources = rag.query(question)
            
            # Display formatted answer
            console.print(Panel(
                Markdown(answer), 
                title="Resposta", 
                border_style="green"
            ))
            
            # Only show sources in verbose mode
            if args.verbose:
                console.print("\n[bold]Textos dos que extráese a información:[/bold]")
            for i, doc in enumerate(sources[:args.k]):  # Show top 3 sources
                    console.print(Panel(
                        doc.page_content + "...", 
                        title=f"Texto {i+1}", 
                        border_style="blue"
                    ))
                
        except Exception as e:
            console.print(f"[error]Error procesando consulta:[/error] {e}")


if __name__ == "__main__":
    main()
