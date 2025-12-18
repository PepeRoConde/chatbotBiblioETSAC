from typing import List, Dict, Tuple, Optional, Union, Any
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document

class BM25Retriever(BaseRetriever):
    bm25_index: Any
    documents: List
    k: int = 4

    def __init__(self, bm25_index, documents, k=4):
        super().__init__(bm25_index=bm25_index, documents=documents, k=k)
        self.bm25_index = bm25_index
        self.documents = documents
        self.k = k

    def _get_relevant_documents(self, query: str, *, run_manager) -> List[Document]:
        # Get BM25 scores for query
        scores = self.bm25_index.transform(query, [doc.page_content for doc in self.documents])
        # Get top k indices
        top_indices = scores.argsort()[-self.k:][::-1]
        return [self.documents[i] for i in top_indices]

    def get_relevant_documents_with_scores(self, query: str, k: int) -> List[Tuple[Document, float]]:
        # Get BM25 scores for query
        scores = self.bm25_index.transform(query, [doc.page_content for doc in self.documents])
        # Get top k indices
        top_indices = scores.argsort()[-k:][::-1]
        return [(self.documents[i], float(scores[i])) for i in top_indices]

