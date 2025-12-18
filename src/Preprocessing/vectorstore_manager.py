"""Vectorstore management functionality."""
from typing import List
from langchain_community.vectorstores import FAISS
from src.LocalEmbeddings import LocalEmbeddings


class VectorstoreManager:
    """Manages FAISS vectorstore operations."""
    
    def __init__(self, embeddings: LocalEmbeddings, batch_size: int = 128, verbose: bool = False):
        """Initialize vectorstore manager.
        
        Args:
            embeddings: LocalEmbeddings instance
            batch_size: Batch size for embedding generation
            verbose: Whether to show detailed information
        """
        self.embeddings = embeddings
        self.batch_size = batch_size
        self.verbose = verbose
        self.vectorstore = None
        
        # Use centralized Rich console utility
        from src.utils.rich_utils import get_console
        self.console = get_console()
    
    def create_vectorstore(self, chunks: List) -> None:
        """Create a vector store from document chunks with batched processing.
        
        Args:
            chunks: List of document chunks to embed
        """
        if not chunks:
            self.log("[yellow]Non hai chunks para crear vectorstore[/yellow]")
            return
        
        if self.verbose:
            self.log(f"[cyan]Creando vectorstore con {len(chunks)} chunks (batch_size={self.batch_size})...[/cyan]")
        
        # Process in batches to manage memory and show progress
        for i in range(0, len(chunks), self.batch_size):
            batch = chunks[i:i + self.batch_size]
            
            if i == 0:
                # Create initial vectorstore
                self.vectorstore = FAISS.from_documents(batch, self.embeddings)
            else:
                # Merge additional batches
                batch_vectorstore = FAISS.from_documents(batch, self.embeddings)
                self.vectorstore.merge_from(batch_vectorstore)
            
            if self.verbose and i > 0:
                progress = min(i + self.batch_size, len(chunks))
                self.log(f"[dim]Progreso vectorstore: {progress}/{len(chunks)} chunks[/dim]")
        
        if self.verbose:
            self.log("[green]✓ Vectorstore creado![/green]", "success")
    
    def save_vectorstore(self, path: str) -> None:
        """Save the vector store to disk.
        
        Args:
            path: Path to save the vectorstore
        """
        if self.vectorstore:
            self.vectorstore.save_local(path)
            if self.verbose:
                self.log(f"Vectorstore gardado en {path}", "success")
        else:
            self.log("Non hai vectorstore para gardar. Execute create_vectorstore() primeiro.", "warning")
    
    def load_vectorstore(self, path: str) -> None:
        """Load a vector store from disk.
        
        Args:
            path: Path to load the vectorstore from
        """
        import os
        if os.path.exists(path):
            self.vectorstore = FAISS.load_local(
                path, 
                self.embeddings, 
                allow_dangerous_deserialization=True
            )
            if self.verbose:
                self.log(f"Vectorstore cargado desde {path}", "success")
        else:
            if self.verbose:
                self.log(f"Non se atopou vectorstore en {path}", "warning")
    
    def log(self, message: str, level: str = "info") -> None:
        """Log a message with appropriate styling based on level."""
        self.console.print(message)

