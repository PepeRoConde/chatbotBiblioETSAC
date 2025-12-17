from typing import List, Dict, Tuple, Optional, Union, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from langchain_core.documents import Document

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
