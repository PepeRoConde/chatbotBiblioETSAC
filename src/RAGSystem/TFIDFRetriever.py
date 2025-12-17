from typing import List, Dict, Tuple, Optional, Union, Any
from sklearn.metrics.pairwise import cosine_similarity
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document

class TFIDFRetriever(BaseRetriever):
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
