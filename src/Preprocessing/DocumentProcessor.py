"""Main document processor that orchestrates document loading, chunking, and indexing."""
from typing import List, Dict, Optional, Any
from pathlib import Path
from datetime import datetime

from src.LocalEmbeddings import LocalEmbeddings
from .metadata_manager import MetadataManager
from .document_loader import DocumentLoader
from .chunk_splitter import ChunkSplitter
from .vectorstore_manager import VectorstoreManager
from .bm25_manager import BM25Manager


class DocumentProcessor:
    """Orchestrates document processing pipeline with parallel processing support."""
    
    def __init__(
        self, 
        docs_folder: str,
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        chunk_size: int = 300,
        chunk_overlap: int = 15,
        verbose: bool = False,
        cache_dir: str = ".doc_cache",
        prefix_mode: str = "source",
        llm: Optional[Any] = None,
        map_json: str = 'crawl/map.json',
        max_workers: int = 7,
        batch_size: int = 128,
        crawler_metadata_path: str = "crawl/metadata.json",
        text_folder: str = "crawl/text"
    ):
        """Initialize document processor with parallel processing support.
        
        Args:
            docs_folder: Folder containing documents (legacy, for compatibility)
            embedding_model_name: Name of embedding model
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
            verbose: Whether to show detailed information
            cache_dir: Directory for caching file metadata
            prefix_mode: How to prefix chunks ("none", "source", "llm")
            llm: LLM instance for prefix generation (required if prefix_mode="llm")
            map_json: Path to JSON file mapping filenames to URLs
            max_workers: Maximum number of parallel workers (default: 7)
            batch_size: Batch size for embedding generation (default: 128)
            crawler_metadata_path: Path to crawler's metadata.json
            text_folder: Folder containing plain text files from crawler
        """
        self.docs_folder = Path(docs_folder)
        self.text_folder = Path(text_folder)
        self.crawler_metadata_path = Path(crawler_metadata_path)
        self.cache_dir = Path(cache_dir)
        self.verbose = verbose
        self.max_workers = max_workers
        self.batch_size = batch_size
        self.map_json = map_json
        
        # Initialize embeddings
        self.embeddings = LocalEmbeddings(
            model_name=embedding_model_name,
            cache_dir=str(self.cache_dir / "embeddings")
        )
        
        # Initialize component managers
        self.metadata_manager = MetadataManager(
            crawler_metadata_path=self.crawler_metadata_path,
            cache_dir=self.cache_dir,
            verbose=self.verbose
        )
        
        self.document_loader = DocumentLoader(
            crawler_metadata_path=self.crawler_metadata_path,
            text_folder=self.text_folder,
            max_workers=self.max_workers,
            verbose=self.verbose
        )
        
        self.chunk_splitter = ChunkSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            prefix_mode=prefix_mode,
            llm=llm,
            max_workers=self.max_workers,
            verbose=self.verbose
        )
        
        self.vectorstore_manager = VectorstoreManager(
            embeddings=self.embeddings,
            batch_size=self.batch_size,
            verbose=self.verbose
        )
        
        self.bm25_manager = BM25Manager(verbose=self.verbose)
        
        # Document storage
        self.documents = []
        
        # Expose vectorstore and BM25 for external access
        self.vectorstore = None
        self.bm25_index = None
        self.bm25_documents = []
        
        if self.verbose:
            mode_desc = {
                "none": "sen prefixo",
                "source": "co nome do documento",
                "llm": "xerado por LLM"
            }[prefix_mode]
            self.log(f"[cyan]Inicializado con {max_workers} workers paralelos[/cyan]")
            self.log(f"Dividindo documentos en fragmentos ({mode_desc})...")
    
    def load_documents(self, force_reload: bool = False) -> Dict[str, List]:
        """Load plain text files that need embeddings based on crawler metadata.
        
        Args:
            force_reload: If True, process all files (ignores needs_embeddings flag)
            
        Returns:
            Dictionary with 'processed' and 'skipped' document lists
        """
        # Get documents needing processing
        if force_reload:
            # Process all documents in text folder
            docs_to_process = []
            for url, meta in self.metadata_manager.crawler_metadata.items():
                if isinstance(meta, dict) and 'text_path' in meta:
                    docs_to_process.append((url, meta))
            self.log(f"[yellow]Modo force_reload: procesando {len(docs_to_process)} documentos[/yellow]")
        else:
            docs_to_process = self.metadata_manager.get_documents_needing_embeddings()
            self.log(f"[cyan]Documentos que necesitan embeddings: {len(docs_to_process)}[/cyan]")
        
        # Load documents using DocumentLoader
        result = self.document_loader.load_documents(docs_to_process, self.documents)
        
        return result
    
    def split_documents(self) -> List:
        """Split documents into chunks with optional prefixing.
        
        Returns:
            List of split Document chunks
        """
        return self.chunk_splitter.split_documents(self.documents)
    
    def create_vectorstore(self, chunks: List) -> None:
        """Create a vector store from document chunks.
        
        Args:
            chunks: List of document chunks to embed
        """
        self.vectorstore_manager.create_vectorstore(chunks)
        self.vectorstore = self.vectorstore_manager.vectorstore
    
    def process(self, force_reload: bool = False, incremental: bool = True) -> bool:
        """Process documents and rebuild vector store when embeddings change.
        
        Args:
            force_reload: Force reprocessing of all files
            incremental: Use incremental updates (only process docs with needs_embeddings=True)
            
        Returns:
            True if vectorstore was modified, False otherwise
        """
        vectorstore_modified = False
        
        # Load documents that need embeddings
        result = self.load_documents(force_reload=force_reload)
        processed_urls = result.get('processed', [])
        
        if not processed_urls:
            self.log("[green]✓ No hay cambios - vectorstore actualizado[/green]")
            return False
        
        self.log(f"[cyan]Procesando {len(processed_urls)} documentos con cambios...[/cyan]")
        
        # Generate chunks and embeddings
        if self.documents:
            chunks = self.split_documents()
            
            if chunks:
                # Generate embeddings (with cache)
                self.log("[cyan]Generando embeddings (usando caché para documentos sin cambios)...[/cyan]")
                
                # ALWAYS rebuild FAISS from scratch with all chunks
                self.log("[yellow]Reconstruyendo FAISS desde cero...[/yellow]")
                
                # Load ALL documents (not just changed ones) for complete FAISS rebuild
                _ = self.load_documents(force_reload=True)
                all_chunks = self.split_documents()
                
                if all_chunks:
                    self.create_vectorstore(all_chunks)
                    self.log("[green]✓ Vectorstore reconstruido exitosamente![/green]")
                    vectorstore_modified = True
                    
                    # Update crawler metadata: mark processed docs as embedded
                    self.metadata_manager.mark_as_embedded(processed_urls)
                    self.log(f"[green]✓ Marcados {len(processed_urls)} documentos como embedded[/green]")
                else:
                    self.log("[yellow]⚠ Non se crearon chunks[/yellow]")
            else:
                self.log("[yellow]⚠ Non se crearon chunks[/yellow]")
        else:
            self.log("[yellow]⚠ Non hai documentos para procesar[/yellow]")
        
        # Clear documents from memory
        self.documents = []
        
        return vectorstore_modified

    def save_vectorstore(self, path: str) -> None:
        """Save the vector store and BM25 index to disk.
        
        Args:
            path: Path to save the vectorstore
        """
        if self.vectorstore:
            self.vectorstore_manager.save_vectorstore(path)
            self.bm25_manager.build_index_from_vectorstore(path)
            self.bm25_manager.save_bm25(path)
            # Update exposed attributes
            self.bm25_index = self.bm25_manager.bm25_index
            self.bm25_documents = self.bm25_manager.bm25_documents
            if self.verbose:
                self.log(f"Vectorstore gardado en {path}", "success")
        else:
            self.log("Non hai vectorstore para gardar. Execute process() primeiro.", "warning")

    def load_vectorstore(self, path: str) -> None:
        """Load a vector store and BM25 index from disk.
        
        Args:
            path: Path to load the vectorstore from
        """
        self.vectorstore_manager.load_vectorstore(path)
        self.vectorstore = self.vectorstore_manager.vectorstore
        self.bm25_manager.load_bm25(path)
        # Update exposed attributes
        self.bm25_index = self.bm25_manager.bm25_index
        self.bm25_documents = self.bm25_manager.bm25_documents

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get statistics about cached data and parallel processing."""
        embedding_stats = self.embeddings.get_cache_stats()
        metadata_stats = self.metadata_manager.get_cache_stats()
        
        return {
            **metadata_stats,
            'embedding_cache': embedding_stats,
            'text_folder': str(self.text_folder),
            'max_workers': self.max_workers,
            'batch_size': self.batch_size
        }

    def clear_cache(self) -> None:
        """Clear all caches."""
        self.metadata_manager.file_metadata = {}
        self.metadata_manager.save_file_metadata()
        self.embeddings.clear_cache()
        self.log("[green]Todos os cachés limpos[/green]", "success")
    
    def log(self, message: str, level: str = "info") -> None:
        """Log a message with appropriate styling based on level."""
        from src.utils.rich_utils import get_console
        console = get_console()
        console.print(message)
