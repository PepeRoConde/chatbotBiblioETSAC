from typing import List, Dict, Tuple, Optional, Union, Any
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document

class HybridRetriever(BaseRetriever):
    vectorstore: Any
    tfidf_retriever: Any
    k: int = 4
    verbose: bool = False
    tfidf_score_factor: int = 100

    def __init__(self, vectorstore, tfidf_retriever, k=4, verbose=False, tfidf_score_factor= 100):
        super().__init__(vectorstore=vectorstore, tfidf_retriever=tfidf_retriever, k=k, verbose=verbose)
        self.vectorstore = vectorstore
        self.tfidf_retriever = tfidf_retriever
        self.k = k
        self.verbose = verbose
        self.tfidf_score_factor = tfidf_score_factor

    def _get_relevant_documents(self, query: str, *, run_manager) -> List[Document]:
        # Get from vector store with scores
        vector_docs_scores = self.vectorstore.similarity_search_with_score(query, k=self.k*2)
        # Get from TF-IDF with scores
        tfidf_docs_scores = self.tfidf_retriever.get_relevant_documents_with_scores(query, self.k*2)
        
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
                    'tfidf_score': None,
                    'retrieval_method': 'vector'
                }
        
        # Add TF-IDF documents
        for doc, score in tfidf_docs_scores:
            key = id(doc)
            if key not in doc_to_info:
                doc_to_info[key] = {
                    'doc': doc,
                    'score': score * self.tfidf_score_factor,
                    'vector_score': None,
                    'tfidf_score': score,
                    'retrieval_method': 'tfidf'
                }
            else:
                # Document found by both methods
                existing_score = doc_to_info[key]['score']
                doc_to_info[key]['tfidf_score'] = score
                doc_to_info[key]['score'] = max(existing_score, score)
                doc_to_info[key]['retrieval_method'] = 'hybrid'
        
        # Sort by score descending
        ranked = sorted(doc_to_info.values(), key=lambda x: x['score'], reverse=True)
        
        # Add metadata to documents
        result_docs = []
        for info in ranked[:self.k]:
            doc = info['doc']
            doc.metadata['relevance_score'] = float(info['score'])
            doc.metadata['vector_score'] = float(info['vector_score']) if info['vector_score'] is not None else None
            doc.metadata['tfidf_score'] = float(info['tfidf_score']) if info['tfidf_score'] is not None else None
            doc.metadata['retrieval_method'] = info['retrieval_method']
            result_docs.append(doc)
        
        return result_docs
