from typing import List, Dict, Tuple, Optional, Union, Any
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document

class HybridRetriever(BaseRetriever):
    vectorstore: Any
    bm25_retriever: Any
    k: int = 4
    verbose: bool = False
    bm25_score_factor: int = 100

    def __init__(self, vectorstore, bm25_retriever, k=4, verbose=False, bm25_score_factor=100):
        super().__init__(vectorstore=vectorstore, bm25_retriever=bm25_retriever, k=k, verbose=verbose)
        self.vectorstore = vectorstore
        self.bm25_retriever = bm25_retriever
        self.k = k
        self.verbose = verbose
        self.bm25_score_factor = bm25_score_factor

    def _get_relevant_documents(self, query: str, *, run_manager) -> List[Document]:
        # Get from vector store with scores
        vector_docs_scores = self.vectorstore.similarity_search_with_score(query, k=self.k*2)
        # Get from BM25 with scores
        bm25_docs_scores = self.bm25_retriever.get_relevant_documents_with_scores(query, self.k*2)
        
        # Track which retriever found each document
        doc_to_info = {}
        
        # Add vector documents
        for doc, score in vector_docs_scores:
            key = id(doc)
            if key not in doc_to_info:
                doc_to_info[key] = {
                    'doc': doc,
                    'score': score,
                    'vector_score': score,
                    'bm25_score': None,
                    'retrieval_method': 'vector'
                }
        
        # Add BM25 documents
        for doc, score in bm25_docs_scores:
            key = id(doc)
            if key not in doc_to_info:
                doc_to_info[key] = {
                    'doc': doc,
                    'score': score * self.bm25_score_factor,
                    'vector_score': None,
                    'bm25_score': score,
                    'retrieval_method': 'bm25'
                }
            else:
                # Document found by both methods
                existing_score = doc_to_info[key]['score']
                doc_to_info[key]['bm25_score'] = score
                doc_to_info[key]['score'] = max(existing_score, score * self.bm25_score_factor)
                doc_to_info[key]['retrieval_method'] = 'hybrid'
        
        # Sort by score descending
        ranked = sorted(doc_to_info.values(), key=lambda x: x['score'], reverse=True)
        
        # Add metadata to documents
        result_docs = []
        for info in ranked[:self.k]:
            doc = info['doc']
            doc.metadata['relevance_score'] = float(info['score'])
            doc.metadata['vector_score'] = float(info['vector_score']) if info['vector_score'] is not None else None
            doc.metadata['bm25_score'] = float(info['bm25_score']) if info['bm25_score'] is not None else None
            doc.metadata['retrieval_method'] = info['retrieval_method']
            result_docs.append(doc)
        
        return result_docs
