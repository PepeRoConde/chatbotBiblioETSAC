from typing import List, Dict, Tuple, Optional, Union, Any
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document

class HybridRetriever(BaseRetriever):
    vectorstore: Any
    bm25_retriever: Any
    k: int = 4
    verbose: bool = False
    bm25_weight: float = 0.5
    vector_weight: float = 0.5

    def __init__(self, vectorstore, bm25_retriever, k=4, verbose=False, bm25_weight=0.5):
        super().__init__(vectorstore=vectorstore, bm25_retriever=bm25_retriever, k=k, verbose=verbose)
        self.vectorstore = vectorstore
        self.bm25_retriever = bm25_retriever
        self.k = k
        self.verbose = verbose
        self.bm25_weight = bm25_weight
        self.vector_weight = 1.0 - bm25_weight

    def _get_relevant_documents(self, query: str, *, run_manager) -> List[Document]:
        # Get from vector store with scores
        vector_docs_scores = self.vectorstore.similarity_search_with_score(query, k=self.k*2)
        # Get from BM25 with scores
        bm25_docs_scores = self.bm25_retriever.get_relevant_documents_with_scores(query, self.k*2)
        
        # Normalize scores to [0, 1] range for proper weighted combination
        if vector_docs_scores:
            max_vector_score = max(score for _, score in vector_docs_scores) if vector_docs_scores else 1.0
            min_vector_score = min(score for _, score in vector_docs_scores) if vector_docs_scores else 0.0
            vector_range = max_vector_score - min_vector_score if max_vector_score != min_vector_score else 1.0
        else:
            vector_range = 1.0
        
        if bm25_docs_scores:
            max_bm25_score = max(score for _, score in bm25_docs_scores) if bm25_docs_scores else 1.0
            min_bm25_score = min(score for _, score in bm25_docs_scores) if bm25_docs_scores else 0.0
            bm25_range = max_bm25_score - min_bm25_score if max_bm25_score != min_bm25_score else 1.0
        else:
            bm25_range = 1.0
        
        # Track which retriever found each document
        doc_to_info = {}
        
        # Add vector documents
        for doc, score in vector_docs_scores:
            key = id(doc)
            # Normalize vector score
            normalized_vector = (score - min_vector_score) / vector_range if vector_range > 0 else 0.0
            if key not in doc_to_info:
                doc_to_info[key] = {
                    'doc': doc,
                    'vector_score': score,
                    'bm25_score': None,
                    'normalized_vector': normalized_vector,
                    'normalized_bm25': 0.0,
                    'retrieval_method': 'vector'
                }
            else:
                doc_to_info[key]['normalized_vector'] = normalized_vector
                doc_to_info[key]['vector_score'] = score
        
        # Add BM25 documents
        for doc, score in bm25_docs_scores:
            key = id(doc)
            # Normalize BM25 score
            normalized_bm25 = (score - min_bm25_score) / bm25_range if bm25_range > 0 else 0.0
            if key not in doc_to_info:
                doc_to_info[key] = {
                    'doc': doc,
                    'vector_score': None,
                    'bm25_score': score,
                    'normalized_vector': 0.0,
                    'normalized_bm25': normalized_bm25,
                    'retrieval_method': 'bm25'
                }
            else:
                # Document found by both methods
                doc_to_info[key]['bm25_score'] = score
                doc_to_info[key]['normalized_bm25'] = normalized_bm25
                doc_to_info[key]['retrieval_method'] = 'hybrid'
        
        # Calculate combined scores using weighted combination
        for key, info in doc_to_info.items():
            combined_score = (self.vector_weight * info['normalized_vector'] + 
                            self.bm25_weight * info['normalized_bm25'])
            info['score'] = combined_score
        
        # Sort by combined score descending
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
