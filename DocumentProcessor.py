from typing import List, Dict, Tuple, Optional, Union, Any
from pathlib import Path
import os
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, BSHTMLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from LocalEmbeddings import LocalEmbeddings

class DocumentProcessor:
    """Class for processing multiple document types (PDF, HTML)."""
    
    def __init__(
        self, 
        docs_folder: str,
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        chunk_size: int = 300,
        chunk_overlap: int = 15,
        verbose: bool = False
    ):
        """Initialize document processor.
        
        Args:
            docs_folder: Folder containing documents
            embedding_model_name: Name of embedding model
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
            verbose: Whether to show detailed information
        """
        self.docs_folder = Path(docs_folder)
        self.embeddings = LocalEmbeddings(model_name=embedding_model_name)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        self.documents = []
        self.vectorstore = None
        self.verbose = verbose
        
        # Use the global rich console if available
        try:
            self.console = __builtins__.rich_console
        except (AttributeError, NameError):
            # Fallback to a new console if not running from main.py
            from rich.console import Console
            self.console = Console()
        
    def log(self, message: str, level: str = "info") -> None:
        """Log a message with appropriate styling based on level.
        
        Args:
            message: Message to log
            level: Log level (info, success, warning, error)
        """
        self.console.print(message)
        
    def load_documents(self) -> None:
        """Load all PDFs and HTML files from the specified folder."""
        pdf_files = list(self.docs_folder.glob("*.pdf"))
        html_files = list(self.docs_folder.glob("*.html")) + list(self.docs_folder.glob("*.htm"))
        
        self.log(f"Found {len(pdf_files)} PDF files and {len(html_files)} HTML files")
        
        # Process PDF files
        for pdf_path in pdf_files:
            try:
                if self.verbose:
                    self.log(f"Processing PDF: {pdf_path}")
                loader = PyPDFLoader(str(pdf_path))
                self.documents.extend(loader.load())
            except Exception as e:
                self.log(f"Error processing PDF {pdf_path}: {e}", "error")
        
        # Process HTML files
        for html_path in html_files:
            try:
                if self.verbose:
                    self.log(f"Processing HTML: {html_path}")
                loader = BSHTMLLoader(str(html_path))
                self.documents.extend(loader.load())
            except Exception as e:
                self.log(f"Error processing HTML {html_path}: {e}", "error")
            
        self.log(f"Loaded {len(self.documents)} documents in total", "success")
        
    def split_documents(self) -> List:
        """Split documents into chunks.
        
        Returns:
            List of document chunks
        """
        if self.verbose:
            self.log("Splitting documents into chunks...")
        chunks = self.text_splitter.split_documents(self.documents)
        if self.verbose:
            self.log(f"Created {len(chunks)} document chunks", "success")
        return chunks
    
    def create_vectorstore(self, chunks: List) -> None:
        """Create a vector store from document chunks.
        
        Args:
            chunks: List of document chunks
        """
        if self.verbose:
            self.log("Creating vector store...")
        self.vectorstore = FAISS.from_documents(chunks, self.embeddings)
        if self.verbose:
            self.log("Vector store creation complete!", "success")
        
    def process(self) -> None:
        """Process all documents and create the vector store."""
        self.load_documents()
        chunks = self.split_documents()
        self.create_vectorstore(chunks)
        
    def save_vectorstore(self, path: str) -> None:
        """Save the vector store to disk.
        
        Args:
            path: Path to save the vector store
        """
        if self.vectorstore:
            self.vectorstore.save_local(path)
            self.log(f"Vector store saved to {path}", "success")
        else:
            self.log("No vector store to save. Run process() first.", "warning")
            
    def load_vectorstore(self, path: str) -> None:
        """Load a vector store from disk.
        
        Args:
            path: Path to load the vector store from
        """
        if os.path.exists(path):
            self.vectorstore = FAISS.load_local(
                path, 
                self.embeddings, 
                allow_dangerous_deserialization=True
            )
            self.log(f"Vector store loaded from {path}", "success")
        else:
            self.log(f"No vector store found at {path}", "warning")