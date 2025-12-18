from typing import List, Dict, Tuple, Optional, Union, Any
import csv
from langchain_core.documents import Document

class BM25Reranker:
    """BM25 based reranker for document relevance scoring."""
    
    def __init__(self, b=0.75, k1=1.6, stopwords_file: str = None):
        """Initialize BM25 reranker.
        
        Args:
            b: BM25 parameter for length normalization (default: 0.75)
            k1: BM25 parameter for term frequency saturation (default: 1.6)
            stopwords_file: Optional path to CSV file with stopwords
        """
        from src.preprocessing.BM25 import BM25
        
        self.bm25 = BM25(b=b, k1=k1)
        self.documents = []
        self.fitted = False
        
        # Load stopwords if provided
        self.stopwords = []
        if stopwords_file:
            self.stopwords = self.load_stopwords_from_csv(stopwords_file)

    @staticmethod
    def load_stopwords_from_csv(csv_file: str) -> list:
        """Load words from CSV file to use as stopwords.
        
        Args:
            csv_file: Path to CSV file (word,count format)
            
        Returns:
            List of words (first column of CSV)
        """
        stopwords = []
        try:
            with open(csv_file, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                for row in reader:
                    if row:  # Check if row is not empty
                        stopwords.append(row[0])
            print(f"Loaded {len(stopwords)} stopwords from {csv_file}")
        except FileNotFoundError:
            print(f"Warning: {csv_file} not found, no custom stopwords loaded")
        return stopwords

    def fit(self, documents: List[Document]) -> None:
        """Fit the BM25 index on a corpus of documents.
        
        Args:
            documents: List of LangChain Document objects
        """
        self.documents = documents
        texts = [doc.page_content for doc in documents]
        self.bm25.fit(texts)
        self.fitted = True
    
    def rerank(self, query: str, documents: List[Document], top_k: int = None) -> List[Tuple[Document, float]]:
        """Rerank documents based on BM25 similarity to query.
        
        Args:
            query: Query string
            documents: Documents to rerank
            top_k: Number of top documents to return (None = return all)
            
        Returns:
            List of (document, score) tuples sorted by relevance
        """
        if not documents:
            return []
        
        if not self.fitted:
            # Fit on the fly if not already fitted
            texts = [doc.page_content for doc in documents]
            self.bm25.fit(texts)
            self.fitted = True
        
        # Get BM25 scores
        doc_texts = [doc.page_content for doc in documents]
        scores = self.bm25.transform(query, doc_texts)
        
        # Create (document, score) pairs and sort
        doc_scores = list(zip(documents, scores))
        doc_scores.sort(key=lambda x: x[1], reverse=True)
        
        if top_k is not None:
            doc_scores = doc_scores[:top_k]
        
        return doc_scores
    
    def get_top_keywords(self, document: Document, top_n: int = 10) -> List[Tuple[str, float]]:
        """Extract top keywords from a document (placeholder - BM25 doesn't have direct keyword extraction).
        
        Args:
            document: Document to analyze
            top_n: Number of top keywords to return
            
        Returns:
            List of (keyword, score) tuples (empty for now)
        """
        # BM25 doesn't have direct keyword extraction like TF-IDF
        # This is a placeholder that returns empty list
        return []

