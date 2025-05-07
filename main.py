import os
import argparse
import re
from pathlib import Path
import numpy as np
from typing import List, Dict, Tuple, Optional, Union, Any
from bs4 import BeautifulSoup
from langchain_community.document_loaders import PyPDFLoader, BSHTMLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich import print as rprint
from rich.theme import Theme
from rich.progress import Progress, SpinnerColumn, TextColumn

from DocumentProcessor import DocumentProcessor
# Import our updated RAGSystem that uses MistralLLM
from RAGSystem import RAGSystem 

def main():
    """Main function to run the RAG system."""
    parser = argparse.ArgumentParser(description='RAG System using Mistral API')
    parser.add_argument('--docs_folder', type=str, help='Folder containing PDF and HTML files', default='/Users/pepe/OneDrive - Universidade da Coruña/documentacion_y_normativa')
    parser.add_argument('--vector_store', type=str, default='local_vectorstore', help='Path to save/load vector store')
    parser.add_argument('--rebuild', action='store_true', help='Rebuild vector store even if it exists')
    parser.add_argument('--language', type=str, default='english', choices=['english', 'spanish', 'galician'], 
                       help='Language for the prompt template')
    parser.add_argument('--chunk_size', type=int, default=300, help='Size of text chunks')
    parser.add_argument('--chunk_overlap', type=int, default=15, help='Overlap between chunks')
    parser.add_argument('--k', type=int, default=4, help='Number of documents to retrieve')
    parser.add_argument('--model', type=str, default="mistral-large-latest", 
                       help='Mistral model to use')
    parser.add_argument('--api_key', type=str, default=None,
                       help='Mistral API key (defaults to MISTRAL_API_KEY env var)')
    parser.add_argument('--embedding_model', type=str, default="sentence-transformers/all-MiniLM-L6-v2",
                       help='HuggingFace model to use for embeddings')
    parser.add_argument('--temperature', type=float, default=0.1,
                       help='Temperature for text generation')
    parser.add_argument('--max_tokens', type=int, default=512,
                       help='Maximum tokens to generate')
    parser.add_argument('--verbose', action='store_true', help='Show detailed information including sources')
    args = parser.parse_args()
    
    # Make sure API key is available
    api_key = args.api_key or os.environ.get("MISTRAL_API_KEY")
    if not api_key:
        print("Error: Mistral API key not provided. Use --api_key or set MISTRAL_API_KEY environment variable.")
        return
    
    # Create a custom theme for rich
    custom_theme = Theme({
        "info": "cyan",
        "warning": "yellow",
        "error": "bold red",
        "success": "bold green",
    })
    
    # Initialize rich console with our theme
    console = Console(theme=custom_theme)
    
    # Share the console and verbose setting globally
    # This is used by other classes to respect the verbose setting
    import builtins
    setattr(builtins, 'rich_console', console)
    setattr(builtins, 'verbose_mode', args.verbose)
    
    # Initialize the document processor with fancy progress display
    with Progress(
        SpinnerColumn(),
        TextColumn("[success]Initializing document processor..."),
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
            TextColumn("[success]Processing documents and building vector store..."),
            console=console,
            transient=True
        ) as progress:
            task = progress.add_task("processing", total=None)
            processor.process()
            processor.save_vectorstore(args.vector_store)
    else:
        with Progress(
            SpinnerColumn(),
            TextColumn("[success]Loading existing vector store..."),
            console=console,
            transient=True
        ) as progress:
            task = progress.add_task("loading", total=None)
            processor.load_vectorstore(args.vector_store)
    
    # Initialize the RAG system with Mistral
    with Progress(
        SpinnerColumn(),
        TextColumn("[success]Initializing RAG system with Mistral API..."),
        console=console,
        transient=True
    ) as progress:
        task = progress.add_task("init", total=None)
        rag = RAGSystem(
            vectorstore=processor.vectorstore,
            api_key=api_key,
            k=args.k,
            language=args.language,
            model_name=args.model,
            temperature=args.temperature,
            max_tokens=args.max_tokens
        )
    
    # Always use fancy welcome message, just vary the detail
    system_info = "[bold blue]Mistral-powered RAG System Ready![/bold blue]"
    if args.verbose:
        system_info += f"\nLanguage: [yellow]{args.language}[/yellow]"
        system_info += f"\nModel: [yellow]{args.model}[/yellow]"
        system_info += f"\nVector Store: [yellow]{args.vector_store}[/yellow]"
        system_info += f"\nChunk Size: [yellow]{args.chunk_size}[/yellow]"
        system_info += f"\nRetrieved Documents: [yellow]{args.k}[/yellow]"
    
    system_info += "\nType [bold red]'exit'[/bold red] to quit."
    
    console.print(Panel.fit(
        system_info,
        title="Mistral RAG System Initialized",
        border_style="green"
    ))
    
    # Interactive query loop
    while True:
        question = console.input("\n[bold cyan]Enter your question: [/bold cyan]")
        if question.lower() == 'exit':
            break
            
        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[success]Processing query with Mistral..."),
                console=console,
                transient=True
            ) as progress:
                task = progress.add_task("processing", total=None)
                answer, sources = rag.query(question)
            
            # Always use fancy formatting for the answer
            console.print(Panel(
                Markdown(answer), 
                title="Answer", 
                border_style="green"
            ))
            
            # Only show sources in verbose mode
            if args.verbose:
                console.print("\n[bold]Sources:[/bold]")
                for i, doc in enumerate(sources[:3]):  # Show top 3 sources
                    console.print(Panel(
                        doc.page_content[:200] + "...", 
                        title=f"Source {i+1}", 
                        border_style="blue"
                    ))
                
        except Exception as e:
            console.print(f"[error]Error processing query:[/error] {e}")


if __name__ == "__main__":
    main()
