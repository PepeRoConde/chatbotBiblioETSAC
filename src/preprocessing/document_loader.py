"""Document loading functionality."""
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

from langchain_core.documents import Document


class DocumentLoader:
    """Handles loading documents from text files."""
    
    def __init__(self, crawler_metadata_path: Path, text_folder: Path, max_workers: int = 7, verbose: bool = False):
        """Initialize document loader.
        
        Args:
            crawler_metadata_path: Path to crawler's metadata.json
            text_folder: Folder containing plain text files from crawler
            max_workers: Maximum number of parallel workers
            verbose: Whether to show detailed information
        """
        self.crawler_metadata_path = crawler_metadata_path
        self.text_folder = text_folder
        self.max_workers = max_workers
        self.verbose = verbose
        self.doc_lock = Lock()
        
        # Use centralized Rich console utility
        from src.utils.rich_utils import get_console
        self.console = get_console()
    
    def _load_single_text_file(self, url: str, meta: Dict) -> Tuple[Optional[List], str, Optional[str]]:
        """Load a single plain text file and return documents, url, and error.
        
        This method is designed to be called by parallel workers.
        
        Args:
            url: Source URL for the document
            meta: Metadata dict from crawler
            
        Returns:
            (documents, url, error_message)
        """
        try:
            text_path_rel = meta.get('text_path')
            if not text_path_rel:
                return None, url, "No text_path in metadata"
            
            # Convert relative to absolute path
            text_path_rel = text_path_rel.replace("\\", os.sep)

            # Construir path de forma portable
            text_path = Path(self.crawler_metadata_path.parent) / text_path_rel
            
            if not text_path.exists():
                return None, url, f"Text file not found: {text_path}"

 
            if not text_path.exists():
                return None, url, f"Text file not found: {text_path}"
            
            # Load plain text
            with open(text_path, 'r', encoding='utf-8') as f:
                text_content = f.read()
            
            # Create Document object
            doc = Document(
                page_content=text_content,
                metadata={
                    'source_url': url,
                    'text_path': str(text_path),
                    'text_hash': meta.get('text_hash', ''),
                    'original_format': meta.get('original_format', 'unknown'),
                    'last_crawl': meta.get('last_crawl', '')
                }
            )
            
            return [doc], url, None
            
        except Exception as e:
            return None, url, str(e)
    
    def load_documents(self, docs_to_process: List[Tuple[str, Dict]], documents: List[Document]) -> Dict[str, List]:
        """Load plain text files in parallel.
        
        Args:
            docs_to_process: List of (url, metadata) tuples to process
            documents: List to append loaded documents to (thread-safe)
            
        Returns:
            Dictionary with 'processed' and 'skipped' document lists
        """
        if not docs_to_process:
            self.log("[green]✓ Todos los documentos tienen embeddings actualizados[/green]")
            return {'processed': [], 'skipped': []}
        
        processed_docs = []
        skipped_docs = []
        
        # Process files in parallel
        self.log(f"[cyan]Cargando {len(docs_to_process)} textos planos en paralelo con {self.max_workers} workers...[/cyan]")
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_url = {
                executor.submit(self._load_single_text_file, url, meta): url
                for url, meta in docs_to_process
            }
            
            # Process completed tasks as they finish
            for future in as_completed(future_to_url):
                url = future_to_url[future]
                docs, _, error = future.result()
                
                if error:
                    self.log(f"[red]Error cargando {url}: {error}[/red]", "error")
                    skipped_docs.append(url)
                    continue
                
                # Thread-safe update of shared state
                with self.doc_lock:
                    documents.extend(docs)
                
                processed_docs.append(url)
                
                if self.verbose:
                    text_path = docs[0].metadata.get('text_path', 'unknown')
                    self.log(f"[cyan]✓ Cargado: {Path(text_path).name}[/cyan]")
        
        # Summary
        if processed_docs:
            self.log(
                f"[green]Cargados {len(processed_docs)} textos planos, "
                f"{len(skipped_docs)} con errores[/green]",
                "success"
            )
        
        return {
            'processed': processed_docs,
            'skipped': skipped_docs
        }
    
    def log(self, message: str, level: str = "info") -> None:
        """Log a message with appropriate styling based on level."""
        self.console.print(message)

