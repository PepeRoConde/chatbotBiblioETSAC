"""Metadata management for document processing."""
from typing import Dict, Any
from pathlib import Path
import json
import pickle
import hashlib
from datetime import datetime


class MetadataManager:
    """Manages file metadata and caching for document processing."""
    
    def __init__(self, crawler_metadata_path: Path, cache_dir: Path, verbose: bool = False):
        """Initialize metadata manager.
        
        Args:
            crawler_metadata_path: Path to crawler's metadata.json
            cache_dir: Directory for caching file metadata
            verbose: Whether to show detailed information
        """
        self.crawler_metadata_path = crawler_metadata_path
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(exist_ok=True)
        self.verbose = verbose
        
        self.crawler_metadata = self._load_crawler_metadata()
        self.metadata_file = self.cache_dir / "processor_cache.json"
        self.file_metadata = self._load_file_metadata()
        
        # Use centralized Rich console utility
        from src.utils.rich_utils import get_console
        self.console = get_console()
    
    def _load_crawler_metadata(self) -> Dict[str, Any]:
        """Load metadata from crawler (shared metadata.json)."""
        if self.crawler_metadata_path.exists():
            try:
                with open(self.crawler_metadata_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                self.log(f"Could not load crawler metadata: {e}", "warning")
                return {}
        return {}
    
    def save_crawler_metadata(self) -> None:
        """Save updated crawler metadata."""
        try:
            with open(self.crawler_metadata_path, 'w', encoding='utf-8') as f:
                json.dump(self.crawler_metadata, f, ensure_ascii=False, indent=2)
        except Exception as e:
            self.log(f"Could not save crawler metadata: {e}", "error")
    
    def _load_file_metadata(self) -> Dict[str, Dict]:
        """Load processor's internal cache."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                self.log(f"Could not load processor cache: {e}", "warning")
                return {}
        return {}
    
    def save_file_metadata(self) -> None:
        """Save file metadata to disk."""
        try:
            with open(self.metadata_file, 'wb') as f:
                pickle.dump(self.file_metadata, f)
        except Exception as e:
            self.log(f"Could not save metadata: {e}", "error")
    
    def _get_file_hash(self, file_path: Path) -> str:
        """Calculate hash of a file's content."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    
    def _get_file_info(self, file_path: Path) -> Dict:
        """Get file information (hash, mod time, size)."""
        stat = file_path.stat()
        return {
            'hash': self._get_file_hash(file_path),
            'mtime': stat.st_mtime,
            'size': stat.st_size,
            'last_processed': datetime.now().isoformat()
        }
    
    def file_has_changed(self, file_path: Path) -> bool:
        """Check if a file has changed since last processing."""
        file_key = str(file_path)
        
        if file_key not in self.file_metadata:
            return True
        
        current_mtime = file_path.stat().st_mtime
        stored_mtime = self.file_metadata[file_key].get('mtime', 0)
        
        if current_mtime > stored_mtime:
            current_hash = self._get_file_hash(file_path)
            stored_hash = self.file_metadata[file_key].get('hash', '')
            return current_hash != stored_hash
        
        return False
    
    def get_documents_needing_embeddings(self) -> list:
        """Get documents that need embeddings based on crawler metadata.
        
        Returns:
            List of (url, metadata) tuples for documents needing processing
        """
        # Reload crawler metadata to get latest
        self.crawler_metadata = self._load_crawler_metadata()
        
        docs_needing_processing = []
        for url, meta in self.crawler_metadata.items():
            if isinstance(meta, dict) and meta.get('needs_embeddings', False):
                text_path = meta.get('text_path')
                if text_path:
                    # Convert relative path to absolute
                    full_path = self.crawler_metadata_path.parent / text_path
                    if full_path.exists():
                        docs_needing_processing.append((url, meta))
                    else:
                        self.log(f"[yellow]Text file not found: {text_path}[/yellow]", "warning")
        
        return docs_needing_processing
    
    def mark_as_embedded(self, urls: list) -> None:
        """Mark documents as embedded in crawler metadata."""
        for url in urls:
            if url in self.crawler_metadata and isinstance(self.crawler_metadata[url], dict):
                self.crawler_metadata[url]['needs_embeddings'] = False
                self.crawler_metadata[url]['last_embedded'] = datetime.now().isoformat()
        self.save_crawler_metadata()
    
    def log(self, message: str, level: str = "info") -> None:
        """Log a message with appropriate styling based on level."""
        self.console.print(message)
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get statistics about cached data."""
        # Count documents by status
        total_docs = len(self.crawler_metadata)
        needs_embeddings = sum(1 for meta in self.crawler_metadata.values() 
                                if isinstance(meta, dict) and meta.get('needs_embeddings', False))
        embedded_docs = total_docs - needs_embeddings
        
        return {
            'total_documents': total_docs,
            'embedded_documents': embedded_docs,
            'needs_embeddings': needs_embeddings,
            'cache_dir': str(self.cache_dir),
        }

