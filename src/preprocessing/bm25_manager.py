"""BM25 index management functionality."""
import os
import pickle
from typing import List
from pathlib import Path
from langchain_core.documents import Document
from .BM25 import BM25


class BM25Manager:
    """Manages BM25 index operations."""
    
    def __init__(self, verbose: bool = False):
        """Initialize BM25 manager.
        
        Args:
            verbose: Whether to show detailed information
        """
        self.bm25_index = None
        self.bm25_documents = []
        self.verbose = verbose
        
        # Use centralized Rich console utility
        from src.utils.rich_utils import get_console
        self.console = get_console()
    
    def build_index_from_vectorstore(self, vectorstore_path: str) -> None:
        """Build BM25 index from FAISS index.pkl file.
        
        Args:
            vectorstore_path: Path to vectorstore directory
        """
        index_path = f"{vectorstore_path}/index.pkl"
        if not os.path.exists(index_path):
            if self.verbose:
                self.log(f"FAISS index not found at {index_path}, skipping BM25 build")
            return

        if self.verbose:
            self.log("Building BM25 index from FAISS index.pkl...")

        try:
            with open(index_path, 'rb') as f:
                data = pickle.load(f)

            docstore = data[0]
            index_to_id = data[1]

            # Extract texts and documents in FAISS order
            chunk_texts = []
            self.bm25_documents = []

            for idx in sorted(index_to_id.keys()):
                doc_id = index_to_id[idx]
                doc = docstore._dict[doc_id]
                chunk_texts.append(doc.page_content)
                self.bm25_documents.append(doc)

            # Build BM25 index
            self.bm25_index = BM25(b=0.75, k1=1.6)
            self.bm25_index.fit(chunk_texts)

            if self.verbose:
                self.log(f"BM25 index built on {len(chunk_texts)} chunks from FAISS")

        except Exception as e:
            self.log(f"Error building BM25 index: {e}", "error")
            self.bm25_index = None
            self.bm25_documents = []
    
    def save_bm25(self, path: str) -> None:
        """Save BM25 index to disk.
        
        Args:
            path: Path to save the BM25 index
        """
        if self.bm25_index is not None:
            data = {
                'bm25_index': self.bm25_index,
                'documents': self.bm25_documents
            }
            with open(f"{path}/bm25_index.pkl", 'wb') as f:
                pickle.dump(data, f)
            if self.verbose:
                self.log(f"BM25 index saved to {path}/bm25_index.pkl")

    def load_bm25(self, path: str) -> None:
        """Load BM25 index from disk.
        
        Args:
            path: Path to load the BM25 index from
        """
        bm25_path = f"{path}/bm25_index.pkl"
        if os.path.exists(bm25_path):
            with open(bm25_path, 'rb') as f:
                data = pickle.load(f)
            self.bm25_index = data['bm25_index']
            self.bm25_documents = data['documents']
            if self.verbose:
                self.log(f"BM25 index loaded from {bm25_path}")
        else:
            if self.verbose:
                self.log(f"BM25 index not found at {bm25_path}")
    
    def log(self, message: str, level: str = "info") -> None:
        """Log a message with appropriate styling based on level."""
        self.console.print(message)

