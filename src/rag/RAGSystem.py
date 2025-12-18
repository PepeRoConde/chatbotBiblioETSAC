from typing import List, Dict, Tuple, Optional, Union, Any
import re
import os
import json
import numpy as np
from datetime import datetime

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

from .cost.cost_tracker import CostTracker
from .cost.claude_cost_callback import ClaudeCostCallback

from .system_prompt import retrieval_prompt, no_retrieval_prompt
from .HybridRetriever import HybridRetriever
from .BM25Retriever import BM25Retriever
from .BM25Reranker import BM25Reranker
from .query_optimizer import QueryOptimizer
from .conversation_history import ConversationHistory

class RAGSystem:
    """RAG system with BM25 integration and query optimization."""
    
    def __init__(
        self, 
        vectorstore: Any,
        llm: Any,
        llm_query: Optional[Any] = "claude-3-haiku-20240307",  
        k: int = 4,
        state_dir: str = 'crawl',
        threshold: float = 0.7,
        search_type: str = "mmr",
        language: str = "english",
        provider: str = 'claude',
        temperature: float = 0.1,
        max_tokens: int = 512,
        max_history_length: int = 10,
        use_bm25: bool = True,
        bm25_mode: str = "rerank",
        bm25_weight: float = 0.3,
        bm25_threshold: float = 0.1,
        bm25_index: Optional[Any] = None,
        bm25_documents: Optional[List] = None,
        use_query_optimization: bool = True,  
    ):
        """Initialize the RAG system with query optimization support.
        
        Args:
            vectorstore: Vector store for retrieval
            llm: Language model for generating final answers
            llm_query: Optional separate model for query generation (defaults to llm if None)
            k: Number of documents to retrieve
            threshold: Filter unrelevant documents
            search_type: Way of performing the retrieval
            language: Language for prompt template
            provider: LLM provider ('mistral' or 'claude')
            max_history_length: Maximum number of conversation turns to keep
            use_bm25: Whether to use BM25 enhancement
            bm25_mode: How to use BM25 ("rerank", "hybrid", or "filter")
            bm25_weight: Weight for BM25 in hybrid mode (0-1)
            bm25_threshold: Minimum BM25 score for filtering
            bm25_index: BM25 index object
            bm25_documents: List of documents for BM25
            use_query_optimization: Enable two-stage query optimization (Haiku + Sonnet)
        """
        self.vectorstore = vectorstore
        self.language = language
        self.k = k
        self.state_dir = str(state_dir)  # Ensure it's a string
        self.threshold = threshold
        self.search_type = search_type
        self.max_history_length = max_history_length
        self.use_bm25 = use_bm25
        self.bm25_mode = bm25_mode
        self.bm25_threshold = bm25_threshold
        self.bm25_index = bm25_index
        self.bm25_documents = bm25_documents
        self.bm25_weight = bm25_weight
        self.use_query_optimization = use_query_optimization
        
        # Console setup (must be before component initialization)
        from src.utils.rich_utils import get_console, get_verbose
        self.console = get_console()
        self.verbose = get_verbose()
        
        # Cost tracking
        self.total_cost = 0.0
        self.query_costs = []
        
        # Claude pricing (per 1M tokens) - Actualizado a precios de Dic 2024
        # Claude pricing (per 1M tokens) - Actualizado Diciembre 2024
        self.pricing = {
            'claude-3-haiku-20240307': {'input': 0.25, 'output': 1.25},
            'claude-haiku-4-5': {'input': 1.00, 'output': 5.00},
            'claude-3-5-sonnet-20241022': {'input': 3.00, 'output': 15.00},
            'claude-3-5-sonnet-20240620': {'input': 3.00, 'output': 15.00},
            'claude-sonnet-4-20250514': {'input': 3.00, 'output': 15.00},
            'claude-sonnet-4-5': {'input': 3.00, 'output': 15.00},
            'claude-3-opus-20240229': {'input': 15.00, 'output': 75.00},
        }

        self.cost_tracker = CostTracker(self.pricing)
        
        # Initialize component managers
        self.conversation_history = ConversationHistory(
            max_history_length=max_history_length,
            language=language,
            verbose=self.verbose
        )
        
        # LLM setup
        self.llm = llm  # Main LLM for answers (Sonnet)
        self.llm_query = llm_query if llm_query else llm  # Query optimization LLM (Haiku or same as main)
        
        # Query optimizer
        self.query_optimizer = QueryOptimizer(
            llm_query=self.llm_query,
            cost_tracker=self.cost_tracker,
            language=language,
            verbose=self.verbose
        ) if self.use_query_optimization else None
        
        # BM25 setup
        self.bm25_reranker = None
        if self.use_bm25:
            self._setup_bm25()
        
        # Create the retriever (base or hybrid)
        if self.use_bm25 and self.bm25_mode == "hybrid":
            if bm25_index is None or bm25_documents is None:
                raise ValueError("BM25 components required for hybrid mode")
            bm25_retriever = BM25Retriever(bm25_index, bm25_documents, k=self.k)
            self.retriever = HybridRetriever(vectorstore=self.vectorstore, bm25_retriever=bm25_retriever, k=self.k, verbose=self.verbose, bm25_weight=self.bm25_weight)
        elif self.use_bm25 and self.bm25_mode == "bm25":
            if bm25_index is None or bm25_documents is None:
                raise ValueError("BM25 components required for bm25 mode")
            self.retriever = BM25Retriever(bm25_index, bm25_documents, k=self.k)
        elif self.use_bm25 and self.bm25_mode == "rerank":
            # For rerank mode, use base vector retriever and apply BM25 reranking later
            self.retriever = self.vectorstore.as_retriever(
                search_kwargs={"k": self.k * 2, "score_threshold": self.threshold},
                search_type=self.search_type
            )
        else:
            self.retriever = self.vectorstore.as_retriever(
                search_kwargs={"k": self.k, "score_threshold": self.threshold},
                search_type=self.search_type
            )
        
        # Create the prompt template
        self.prompt = self._create_prompt_template()
        
        # Create the RAG chain
        self._create_rag_chain()
        
        if self.verbose:
            bm25_status = f" with BM25 ({self.bm25_mode} mode)" if self.use_bm25 else ""
            query_opt_status = " + Query Optimization (Haiku→Sonnet)" if self.use_query_optimization else ""
            self.log(f"RAG system initialized{bm25_status}{query_opt_status} with {provider.upper()}", "success")
            self.log(f"Language: {language}", "info")
            if self.use_query_optimization:
                self.log(f"Query model: {type(self.llm_query).__name__}", "info")
                self.log(f"Answer model: {type(self.llm).__name__}", "info")
    
    def _get_model_name(self, llm) -> str:
        """Extract model name from LLM object.
        
        Args:
            llm: LangChain LLM object
            
        Returns:
            Model name string
        """
        if hasattr(llm, 'model'):
            return llm.model
        elif hasattr(llm, 'model_name'):
            return llm.model_name
        else:
            return 'unknown'
    

    
    def _setup_bm25(self) -> None:
        """Setup BM25 reranker by fitting on all documents in vectorstore."""
        if self.verbose:
            self.log("Initializing BM25 reranker...", "info")
        
        # Get all documents from vectorstore
        try:
            # Try to get docstore if available (FAISS has this)
            if hasattr(self.vectorstore, 'docstore'):
                all_docs = list(self.vectorstore.docstore._dict.values())
            else:
                # Fallback: retrieve many documents with a generic query
                temp_retriever = self.vectorstore.as_retriever(search_kwargs={"k": 1000})
                all_docs = temp_retriever.get_relevant_documents("*")
            
            if all_docs:
                stopwords_file = None
                top_words_path = os.path.join(self.state_dir, 'top_words.csv')
                if os.path.exists(top_words_path):
                    stopwords_file = top_words_path
                self.bm25_reranker = BM25Reranker(b=0.75, k1=1.6, stopwords_file=stopwords_file)
                self.bm25_reranker.fit(all_docs)
                
                if self.verbose:
                    self.log(f"BM25 fitted on {len(all_docs)} documents", "success")
            else:
                self.log("Warning: Could not load documents for BM25", "warning")
                self.use_bm25 = False
                
        except Exception as e:
            self.log(f"Error setting up BM25: {e}", "error")
            self.use_bm25 = False
    
    
    def log(self, message: str, level: str = "info") -> None:
        """Log a message with appropriate styling."""
        self.console.print(message)
    
    def _create_prompt_template(self) -> ChatPromptTemplate:
        """Create prompt template based on selected language."""
        
        template = retrieval_prompt.get(self.language.lower(), retrieval_prompt["english"])
        return ChatPromptTemplate.from_template(template)
    
    
    def _create_rag_chain(self) -> None:
        """Create the RAG chain using LangChain 1.0 API."""
        document_chain = create_stuff_documents_chain(
            llm=self.llm,  # Uses main LLM (Sonnet) for final answer
            prompt=self.prompt
        )
        
        self.rag_chain = create_retrieval_chain(
            retriever=self.retriever,
            combine_docs_chain=document_chain
        )
    
    def _apply_bm25_reranking(self, query: str, documents: List[Document]) -> List[Document]:
        """Apply BM25 reranking to retrieved documents.
        
        Args:
            query: Original query
            documents: Documents to rerank
            
        Returns:
            Reranked list of documents
        """
        if not self.use_bm25 or not self.bm25_reranker or not documents:
            return documents
        
        if self.bm25_mode == "rerank":
            # Pure reranking: sort by BM25 scores
            reranked = self.bm25_reranker.rerank(query, documents, top_k=len(documents))
            return [doc for doc, score in reranked]
        
        elif self.bm25_mode == "filter":
            # Filtering: remove documents below threshold
            scored_docs = self.bm25_reranker.rerank(query, documents, top_k=None)
            filtered = [(doc, score) for doc, score in scored_docs if score >= self.bm25_threshold]
            return [doc for doc, score in filtered]
        
        return documents
    
    def extract_answer_only(self, full_response: str) -> str:
        """Extract just the actual answer from LLM response."""
        answer_only = full_response
        answer_only = re.sub(r'^.*Answer:\s*', '', answer_only, flags=re.DOTALL)
        answer_only = re.sub(r'^.*Respuesta:\s*', '', answer_only, flags=re.DOTALL)
        answer_only = re.sub(r'^.*Resposta:\s*', '', answer_only, flags=re.DOTALL)
        answer_only = re.sub(r'<.*?>.*', '', answer_only, flags=re.DOTALL)
        return answer_only.strip()
    
    def query(
        self,
        question: str,
        use_history: bool = True,
        return_costs: bool = False
    ):
        """
        Query the RAG system with intelligent retrieval decision.
        """

        # Reset cost tracker for this query
        self.cost_tracker.reset()

        # -----------------------------
        # 1. Prepare conversation history
        # -----------------------------
        history_text = ""
        if use_history:
            history_text = self.conversation_history.format_history(max_turns=5)

        # -----------------------------
        # 2. Decide whether to retrieve (Haiku)
        # -----------------------------
        if self.query_optimizer:
            should_retrieve, search_query = self.query_optimizer.should_retrieve_documents(
                question,
                history_text
            )
        else:
            should_retrieve, search_query = True, question

        if self.verbose:
            self.log(f"📝 Pregunta: {question}", "info")
            if should_retrieve:
                self.log(f"🔍 Query optimizada: {search_query}", "success")
            else:
                self.log("💬 Sin búsqueda en documentos", "warning")

        # -----------------------------
        # 3. Generate answer (Sonnet)
        # -----------------------------
        used_retrieval = False

        if not should_retrieve:
            # ---- Direct answer (no retrieval) ----

            template = no_retrieval_prompt.get(
                self.language.lower(),
                no_retrieval_prompt["english"]
            )

            prompt = ChatPromptTemplate.from_template(template)
            chain = prompt | self.llm | StrOutputParser()

            callback = ClaudeCostCallback(
                stage="answer",
                model=self._get_model_name(self.llm),
                cost_tracker=self.cost_tracker,
            )

            answer = chain.invoke(
                {
                    "input": question,
                    "history": history_text,
                },
                config={"callbacks": [callback]},
            )

            source_docs = []

        else:
            # ---- RAG answer with retrieval ----
            callback = ClaudeCostCallback(
                stage="answer",
                model=self._get_model_name(self.llm),
                cost_tracker=self.cost_tracker,
            )

            result = self.rag_chain.invoke(
                {
                    "input": search_query,
                    "history": history_text,
                },
                config={"callbacks": [callback]},
            )

            answer = result.get("answer", "")
            source_docs = result.get("context", [])
            used_retrieval = True

            # Optional BM25 post-processing
            if self.use_bm25 and self.bm25_mode in ("rerank", "filter") and source_docs:
                original_count = len(source_docs)
                source_docs = self._apply_bm25_reranking(
                    search_query,
                    source_docs
                )

                if self.verbose and self.bm25_mode == "filter":
                    self.log(
                        f"Filtrado BM25: {original_count} -> {len(source_docs)} docs",
                        "info"
                    )

            if self.use_bm25 and self.bm25_reranker and source_docs:
                scored_docs = self.bm25_reranker.rerank(
                    search_query,
                    source_docs,
                    top_k=None
                )
                for doc, score in scored_docs:
                    doc.metadata["bm25_score"] = float(score)

            if not source_docs:
                no_info_messages = {
                    "galician": "Non atopei información relevante nos documentos.",
                    "spanish": "No encontré información relevante en los documentos.",
                    "english": "I couldn't find relevant information in the documents."
                }
                answer = no_info_messages.get(
                    self.language.lower(),
                    no_info_messages["english"]
                )

        # -----------------------------
        # 4. Finalize
        # -----------------------------
        clean_answer = self.extract_answer_only(answer)

        if use_history:
            self.conversation_history.add_interaction(question, clean_answer)

        cost_summary = self.cost_tracker.summary()

        if self.verbose:
            self.log(f"📄 Documentos usados: {len(source_docs)}", "info")
            self.log(
                f"💵 Coste total: ${cost_summary['total_cost']:.6f}",
                "success"
            )

        if return_costs:
            cost_info = {
                "total_cost": cost_summary["total_cost"],
                "by_stage": cost_summary["by_stage"],
                "calls": cost_summary["calls"],
                "used_retrieval": used_retrieval,
                "query_model": self._get_model_name(self.llm_query),
                "answer_model": self._get_model_name(self.llm),
            }
            return clean_answer, source_docs, cost_info

        return clean_answer, source_docs

    
    def get_document_keywords(self, document: Document, top_n: int = 10) -> List[Tuple[str, float]]:
        """Extract top keywords from a document (placeholder for BM25).
        
        Args:
            document: Document to analyze
            top_n: Number of keywords to return
            
        Returns:
            List of (keyword, score) tuples (empty for BM25)
        """
        if not self.use_bm25 or not self.bm25_reranker:
            return []
        
        return self.bm25_reranker.get_top_keywords(document, top_n)
    
    # ========== History Management Methods  ==========
    
    def get_cost_summary(self) -> Dict[str, Any]:
        """Get summary of all costs incurred.
        
        Returns:
            Dictionary with cost statistics
        """
        if not self.query_costs:
            return {
                'total_cost': 0.0,
                'num_queries': 0,
                'avg_cost_per_query': 0.0,
                'total_query_costs': 0.0,
                'total_answer_costs': 0.0
            }
        
        return {
            'total_cost': self.total_cost,
            'num_queries': len(self.query_costs),
            'avg_cost_per_query': self.total_cost / len(self.query_costs),
            'total_query_costs': sum(c['query_cost'] for c in self.query_costs),
            'total_answer_costs': sum(c['answer_cost'] for c in self.query_costs),
            'queries_with_retrieval': sum(1 for c in self.query_costs if c['used_retrieval']),
            'queries_without_retrieval': sum(1 for c in self.query_costs if not c['used_retrieval']),
        }
    
    def print_cost_summary(self) -> None:
        """Print formatted cost summary."""
        summary = self.get_cost_summary()
        
        if summary['num_queries'] == 0:
            self.log("No hay queries registradas aún", "info")
            return
        
        self.log("\n" + "="*50, "info")
        self.log("RESUMEN DE COSTES", "success")
        self.log("="*50, "info")
        self.log(f"💵 Coste total: ${summary['total_cost']:.6f}", "success")
        self.log(f"Número de queries: {summary['num_queries']}", "info")
        self.log(f"Coste medio por query: ${summary['avg_cost_per_query']:.6f}", "info")
        self.log(f"Queries con retrieval: {summary['queries_with_retrieval']}", "info")
        self.log(f"Queries sin retrieval: {summary['queries_without_retrieval']}", "info")
        self.log(f"Coste total Haiku (queries): ${summary['total_query_costs']:.6f}", "info")
        self.log(f"Coste total Sonnet (respuestas): ${summary['total_answer_costs']:.6f}", "info")
        self.log("="*50 + "\n", "info")
    
    def reset_costs(self) -> None:
        """Reset cost tracking."""
        self.total_cost = 0.0
        self.query_costs = []
        if self.verbose:
            self.log("Costes reseteados", "success")
    
    def clear_history(self) -> None:
        """Clear conversation history."""
        self.conversation_history.clear()
    
    def get_history(self) -> List[Dict[str, str]]:
        """Get conversation history."""
        return self.conversation_history.get_history()
    
    def get_history_summary(self) -> str:
        """Get conversation history summary."""
        return self.conversation_history.get_history_summary()
    
    def save_history(self, filepath: str = "conversation_history.json") -> None:
        """Save conversation history to JSON."""
        self.conversation_history.save_history(filepath)
    
    def load_history(self, filepath: str = "conversation_history.json") -> None:
        """Load conversation history from JSON."""
        self.conversation_history.load_history(filepath)
    
    def export_history_markdown(self, filepath: str = "conversation_history.md") -> None:
        """Export conversation history to Markdown."""
        self.conversation_history.export_history_markdown(filepath)
