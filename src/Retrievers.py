"""
Retrievers for RAG system.
Contains TF-IDF reranker, BM25, and hybrid retrieval strategies.
"""

from typing import List, Dict, Tuple, Optional, Union, Any
import csv
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun

try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False
    print("Warning: rank_bm25 not installed. BM25Retriever will not be available.")


class TFIDFReranker:
    """TF-IDF based reranker for document relevance scoring."""
    
    def __init__(self, max_features: int = 5000, ngram_range: Tuple[int, int] = (1, 2), 
                 stopwords_file: str = 'top_words.csv'):
        """Initialize TF-IDF reranker.
        
        Args:
            max_features: Maximum number of features for TF-IDF
            ngram_range: Range of n-grams to consider (unigrams and bigrams by default)
            stopwords_file: Path to CSV file with stopwords
        """
        # Load stopwords from CSV
        stopwords = self.load_stopwords_from_csv(stopwords_file)
        
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            stop_words=stopwords if stopwords else None,
            lowercase=True,
            token_pattern=r'\b\w+\b'
        )
        self.doc_vectors = None
        self.documents = []
        self.fitted = False 

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
        """Fit the TF-IDF vectorizer on a corpus of documents.
        
        Args:
            documents: List of LangChain Document objects
        """
        self.documents = documents
        texts = [doc.page_content for doc in documents]
        self.doc_vectors = self.vectorizer.fit_transform(texts)
        self.fitted = True
    
    def rerank(self, query: str, documents: List[Document], top_k: int = None) -> List[Tuple[Document, float]]:
        """Rerank documents based on TF-IDF similarity to query.
        
        Args:
            query: Query string
            documents: Documents to rerank
            top_k: Number of top documents to return (None = return all)
            
        Returns:
            List of (document, score) tuples sorted by relevance
        """
        if not documents:
            return []
        
        # Transform query and documents
        query_vector = self.vectorizer.transform([query])
        doc_texts = [doc.page_content for doc in documents]
        doc_vectors = self.vectorizer.transform(doc_texts)
        
        # Calculate cosine similarities
        similarities = cosine_similarity(query_vector, doc_vectors)[0]
        
        # Create (document, score) pairs and sort
        doc_scores = list(zip(documents, similarities))
        doc_scores.sort(key=lambda x: x[1], reverse=True)
        
        if top_k is not None:
            doc_scores = doc_scores[:top_k]
        
        return doc_scores
    
    def get_top_keywords(self, document: Document, top_n: int = 10) -> List[Tuple[str, float]]:
        """Extract top TF-IDF keywords from a document.
        
        Args:
            document: Document to analyze
            top_n: Number of top keywords to return
            
        Returns:
            List of (keyword, score) tuples
        """
        if not self.fitted:
            raise ValueError("Vectorizer must be fitted before extracting keywords")
        
        doc_vector = self.vectorizer.transform([document.page_content])
        feature_names = self.vectorizer.get_feature_names_out()
        
        # Get TF-IDF scores for this document
        scores = doc_vector.toarray()[0]
        
        # Get top N features
        top_indices = scores.argsort()[-top_n:][::-1]
        top_keywords = [(feature_names[i], scores[i]) for i in top_indices if scores[i] > 0]
        
        return top_keywords


class TFIDFRetriever(BaseRetriever):
    """Pure TF-IDF retriever."""
    
    vectorizer: Any
    matrix: Any
    documents: List
    k: int = 4

    def __init__(self, vectorizer, matrix, documents, k=4):
        super().__init__(vectorizer=vectorizer, matrix=matrix, documents=documents, k=k)
        self.vectorizer = vectorizer
        self.matrix = matrix
        self.documents = documents
        self.k = k

    def _get_relevant_documents(self, query: str, *, run_manager) -> List[Document]:
        query_vec = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.matrix)[0]
        top_indices = similarities.argsort()[-self.k:][::-1]
        return [self.documents[i] for i in top_indices]

    def get_relevant_documents_with_scores(self, query: str, k: int) -> List[Tuple[Document, float]]:
        query_vec = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.matrix)[0]
        top_indices = similarities.argsort()[-k:][::-1]
        return [(self.documents[i], similarities[i]) for i in top_indices]


class HybridRetriever(BaseRetriever):
    """Hybrid retriever combining vector similarity and TF-IDF reranking."""
    
    vector_retriever: Any
    tfidf_reranker: Any  # TFIDFReranker
    vector_weight: float = 0.7
    tfidf_weight: float = 0.3
    verbose: bool = False
    
    def __init__(
        self,
        vector_retriever: Any,
        tfidf_reranker: Any,
        vector_weight: float = 0.7,
        tfidf_weight: float = 0.3,
        verbose: bool = False
    ):
        """Initialize hybrid retriever.
        
        Args:
            vector_retriever: LangChain vector store retriever
            tfidf_reranker: TF-IDF reranker instance
            vector_weight: Weight for vector similarity scores (0-1)
            tfidf_weight: Weight for TF-IDF scores (0-1)
            verbose: Whether to log detailed information
        """
        # Normalize weights
        total = vector_weight + tfidf_weight
        normalized_vector_weight = vector_weight / total
        normalized_tfidf_weight = tfidf_weight / total
        
        # Call parent __init__ with all fields
        super().__init__(
            vector_retriever=vector_retriever,
            tfidf_reranker=tfidf_reranker,
            vector_weight=normalized_vector_weight,
            tfidf_weight=normalized_tfidf_weight,
            verbose=verbose
        )
    
    def _get_relevant_documents(
        self, 
        query: str,
        *, 
        run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """Retrieve documents using hybrid approach.
        
        Args:
            query: Query string
            run_manager: Callback manager
            
        Returns:
            List of ranked documents
        """
        # Step 1: Get initial candidates from vector store using invoke
        vector_docs = self.vector_retriever.invoke(query)
        
        if not vector_docs:
            return []
        
        # Step 2: Rerank with TF-IDF
        tfidf_scores = self.tfidf_reranker.rerank(query, vector_docs)
        
        # Step 3: Combine scores (assuming vector retriever returns docs in order of relevance)
        # Assign decreasing vector scores based on position
        vector_scores = np.linspace(1.0, 0.1, len(vector_docs))
        
        # Create a mapping of documents to combined scores
        doc_to_combined_score = {}
        for i, (doc, tfidf_score) in enumerate(tfidf_scores):
            vector_score = vector_scores[i] if i < len(vector_scores) else 0.1
            combined_score = (self.vector_weight * vector_score) + (self.tfidf_weight * tfidf_score)
            doc_to_combined_score[id(doc)] = (doc, combined_score)
        
        # Sort by combined score
        ranked_docs = sorted(doc_to_combined_score.values(), key=lambda x: x[1], reverse=True)
        
        if self.verbose:
            print(f"Hybrid retrieval: {len(ranked_docs)} documents ranked")
        
        return [doc for doc, score in ranked_docs]


class TrueHybridRetriever(BaseRetriever):
    """True hybrid retriever that combines FAISS and TF-IDF candidates before scoring."""
    
    vectorstore: Any
    tfidf_retriever: Any
    k: int = 4
    verbose: bool = False

    def __init__(self, vectorstore, tfidf_retriever, k=4, verbose=False):
        super().__init__(vectorstore=vectorstore, tfidf_retriever=tfidf_retriever, k=k, verbose=verbose)
        self.vectorstore = vectorstore
        self.tfidf_retriever = tfidf_retriever
        self.k = k
        self.verbose = verbose

    def _get_relevant_documents(self, query: str, *, run_manager) -> List[Document]:
        # Get from vector store with scores
        vector_docs_scores = self.vectorstore.similarity_search_with_score(query, k=self.k*2)
        # Get from TF-IDF with scores
        tfidf_docs_scores = self.tfidf_retriever.get_relevant_documents_with_scores(query, self.k*2)
        # Combine and deduplicate
        doc_to_score = {}
        for doc, score in vector_docs_scores + tfidf_docs_scores:
            key = id(doc)
            if key not in doc_to_score:
                doc_to_score[key] = (doc, score)
            else:
                # If already present, take the max score
                existing_score = doc_to_score[key][1]
                doc_to_score[key] = (doc, max(existing_score, score))
        # Sort by score descending
        ranked = sorted(doc_to_score.values(), key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in ranked[:self.k]]


class BM25Retriever(BaseRetriever):
    """BM25-based retriever for keyword matching."""
    
    documents: List[Document]
    bm25: Any = None
    k: int = 4
    tokenized_corpus: List[List[str]] = []
    
    def __init__(self, documents: List[Document], k: int = 4):
        """Initialize BM25 retriever.
        
        Args:
            documents: List of documents to index
            k: Number of documents to retrieve
        """
        if not BM25_AVAILABLE:
            raise ImportError("rank_bm25 not installed. Install with: pip install rank-bm25")
        
        # Tokenize corpus
        tokenized_corpus = [doc.page_content.lower().split() for doc in documents]
        
        # Initialize BM25
        bm25 = BM25Okapi(tokenized_corpus)
        
        super().__init__(
            documents=documents,
            bm25=bm25,
            k=k,
            tokenized_corpus=tokenized_corpus
        )
    
    def _get_relevant_documents(
        self, 
        query: str, 
        *, 
        run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """Retrieve documents using BM25.
        
        Args:
            query: Query string
            run_manager: Callback manager
            
        Returns:
            Top-k documents ranked by BM25 score
        """
        # Tokenize query
        tokenized_query = query.lower().split()
        
        # Get BM25 scores
        scores = self.bm25.get_scores(tokenized_query)
        
        # Get top-k indices
        top_k_indices = np.argsort(scores)[-self.k:][::-1]
        
        return [self.documents[i] for i in top_k_indices]
    
    def get_relevant_documents_with_scores(
        self, 
        query: str, 
        k: int
    ) -> List[Tuple[Document, float]]:
        """Get documents with their BM25 scores.
        
        Args:
            query: Query string
            k: Number of documents to retrieve
            
        Returns:
            List of (document, score) tuples
        """
        # Tokenize query
        tokenized_query = query.lower().split()
        
        # Get BM25 scores
        scores = self.bm25.get_scores(tokenized_query)
        
        # Get top-k indices
        top_k_indices = np.argsort(scores)[-k:][::-1]
        
        return [(self.documents[i], scores[i]) for i in top_k_indices]

