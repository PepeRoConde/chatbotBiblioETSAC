from typing import List, Dict, Tuple, Optional, Union, Any
import re
import os
import json
import csv
import numpy as np
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity



from cost_tracker import CostTracker
from claude_cost_callback import ClaudeCostCallback

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_classic.chains import create_retrieval_chain
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

from .system_prompt import retrieval_prompt, no_retrieval_prompt, few_shot_classification_prompt
from .HybridRetriever import HybridRetriever
from .TFIDFRetriever import TFIDFRetriever
from .TFIDFReranker import TFIDFReranker

class RAGSystem:
    """RAG system with TF-IDF integration and query optimization."""
    
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
        use_tfidf: bool = True,
        tfidf_mode: str = "rerank",
        tfidf_weight: float = 0.3,
        tfidf_threshold: float = 0.1,
        tfidf_vectorizer: Optional[Any] = None,
        tfidf_matrix: Optional[Any] = None,
        tfidf_documents: Optional[List] = None,
        tfidf_score_factor = 100,
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
            use_tfidf: Whether to use TF-IDF enhancement
            tfidf_mode: How to use TF-IDF ("rerank", "hybrid", or "filter")
            tfidf_weight: Weight for TF-IDF in hybrid mode (0-1)
            tfidf_threshold: Minimum TF-IDF score for filtering
            use_query_optimization: Enable two-stage query optimization (Haiku + Sonnet)
        """
        self.vectorstore = vectorstore
        self.language = language
        self.k = k
        self.threshold = threshold
        self.search_type = search_type
        self.max_history_length = max_history_length
        self.use_tfidf = use_tfidf
        self.tfidf_mode = tfidf_mode
        self.tfidf_threshold = tfidf_threshold
        self.tfidf_vectorizer = tfidf_vectorizer
        self.tfidf_matrix = tfidf_matrix
        self.tfidf_documents = tfidf_documents
        self.tfidf_score_factor = tfidf_score_factor
        self.use_query_optimization = use_query_optimization
        
        # Conversation history
        self.conversation_history: List[Dict[str, str]] = []
        
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
        # Console setup
        try:
            self.console = __builtins__.rich_console
            self.verbose = __builtins__.verbose_mode
        except (AttributeError, NameError):
            from rich.console import Console
            self.console = Console()
            self.verbose = True
        
        # LLM setup
        self.llm = llm  # Main LLM for answers (Sonnet)
        self.llm_query = llm_query if llm_query else llm  # Query optimization LLM (Haiku or same as main)
        
        # TF-IDF setup
        self.tfidf_reranker = None
        if self.use_tfidf:
            self._setup_tfidf()
        
        # Create the retriever (base or hybrid)
        if self.use_tfidf and self.tfidf_mode == "hybrid":
            if tfidf_vectorizer is None or tfidf_matrix is None or tfidf_documents is None:
                raise ValueError("TF-IDF components required for hybrid mode")
            tfidf_retriever = TFIDFRetriever(tfidf_vectorizer, tfidf_matrix, tfidf_documents, k=self.k)
            self.retriever = HybridRetriever(vectorstore=self.vectorstore, tfidf_retriever=tfidf_retriever, k=self.k, verbose=self.verbose, tfidf_score_factor=self.tfidf_score_factor)
        elif self.use_tfidf and self.tfidf_mode == "tfidf":
            if tfidf_vectorizer is None or tfidf_matrix is None or tfidf_documents is None:
                raise ValueError("TF-IDF components required for tfidf mode")
            self.retriever = TFIDFRetriever(tfidf_vectorizer, tfidf_matrix, tfidf_documents, k=self.k)
        elif self.use_tfidf and self.tfidf_mode == "rerank":
            self.retriever = self._create_hybrid_retriever(tfidf_weight)
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
            tfidf_status = f" with TF-IDF ({self.tfidf_mode} mode)" if self.use_tfidf else ""
            query_opt_status = " + Query Optimization (Haiku→Sonnet)" if self.use_query_optimization else ""
            self.log(f"RAG system initialized{tfidf_status}{query_opt_status} with {provider.upper()}", "success")
            self.log(f"Language: {language}", "info")
            if self.use_query_optimization:
                self.log(f"Query model: {type(self.llm_query).__name__}", "info")
                self.log(f"Answer model: {type(self.llm).__name__}", "info")
    
    def _calculate_cost(self, model_name: str, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost for a model call.
        
        Args:
            model_name: Name of the model used
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            
        Returns:
            Cost in dollars
        """
        if model_name not in self.pricing:
            # Default pricing if model not found
            return 0.0
        
        prices = self.pricing[model_name]
        input_cost = (input_tokens / 1_000_000) * prices['input']
        output_cost = (output_tokens / 1_000_000) * prices['output']
        
        return input_cost + output_cost
    
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
    
    def _should_retrieve_documents(
        self,
        user_question: str,
        history_text: str
    ) -> tuple[bool, str]:
        """
        Determina si es necesario buscar en documentos y genera una query optimizada.

        Returns:
            (should_retrieve, optimized_query)
        """


        template = few_shot_classification_prompt.get(self.language.lower(), few_shot_classification_prompt["english"])
        query_prompt = ChatPromptTemplate.from_template(template)

        callback = ClaudeCostCallback(
            stage="query_decision",
            model=self._get_model_name(self.llm_query),
            cost_tracker=self.cost_tracker,
        )

        chain = query_prompt | self.llm_query | StrOutputParser()

        try:
            response = chain.invoke(
                {
                    "question": user_question,
                    "history": history_text or "No previous conversation.",
                },
                config={"callbacks": [callback]},
            )

            response = response.strip()

            # Decisión explícita de NO retrieval
            if response.upper().startswith("NO_RETRIEVAL"):
                return False, ""

            # Limpieza de prefijos comunes
            prefixes = (
                "query optimizada:",
                "optimized query:",
                "query:",
                "search query:",
                "búsqueda:",
                "busca:",
                "respuesta:",
            )

            for prefix in prefixes:
                if response.lower().startswith(prefix):
                    response = response[len(prefix):].strip()
                    break

            # Seguridad extra: limitar longitud
            response_words = response.split()
            optimized_query = " ".join(response_words[:15])

            return True, optimized_query

        except Exception as e:
            self.log(
                f"Error en análisis de necesidad de búsqueda: {e}",
                "error"
            )
            # Fail-safe: mejor buscar que no buscar
            return True, user_question

    
    def _setup_tfidf(self) -> None:
        """Setup TF-IDF reranker by fitting on all documents in vectorstore."""
        if self.verbose:
            self.log("Initializing TF-IDF reranker...", "info")
        
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
                self.tfidf_reranker = TFIDFReranker(max_features=5000, ngram_range=(1, 2), stopwords_file=self.state_dir + '/' + 'top_words.csv')
                self.tfidf_reranker.fit(all_docs)
                
                if self.verbose:
                    self.log(f"TF-IDF fitted on {len(all_docs)} documents", "success")
            else:
                self.log("Warning: Could not load documents for TF-IDF", "warning")
                self.use_tfidf = False
                
        except Exception as e:
            self.log(f"Error setting up TF-IDF: {e}", "error")
            self.use_tfidf = False
    
    def _create_hybrid_retriever(self, tfidf_weight: float) -> HybridRetriever:
        """Create a hybrid retriever combining vector and TF-IDF."""
        base_retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": self.k * 2, "score_threshold": self.threshold},  
            search_type=self.search_type
        )
        
        return HybridRetriever(
            vector_retriever=base_retriever,
            tfidf_reranker=self.tfidf_reranker,
            vector_weight=1.0 - tfidf_weight,
            tfidf_weight=tfidf_weight,
            verbose=self.verbose
        )
    
    def log(self, message: str, level: str = "info") -> None:
        """Log a message with appropriate styling."""
        self.console.print(message)
    
    def _create_prompt_template(self) -> ChatPromptTemplate:
        """Create prompt template based on selected language."""
        
        template = retrieval_prompt.get(self.language.lower(), retrieval_prompt["english"])
        return ChatPromptTemplate.from_template(template)
    
    def _format_history(self, max_turns: int = 5) -> str:
        """Format conversation history for inclusion in prompt."""
        if not self.conversation_history:
            return ""
        
        recent_history = self.conversation_history[-max_turns:]
        
        if self.language.lower() == "galician":
            user_label, assistant_label, header = "Usuario", "Asistente", "Historial da conversación:"
        elif self.language.lower() == "spanish":
            user_label, assistant_label, header = "Usuario", "Asistente", "Historial de la conversación:"
        else:
            user_label, assistant_label, header = "User", "Assistant", "Conversation history:"
        
        history_lines = [header]
        for interaction in recent_history:
            answer = interaction['answer']
            if len(answer) > 300:
                answer = answer[:300] + "..."
            history_lines.append(f"{user_label}: {interaction['question']}")
            history_lines.append(f"{assistant_label}: {answer}\n")
        
        return "\n".join(history_lines)
    
    def _add_to_history(self, question: str, answer: str) -> None:
        """Add interaction to conversation history."""
        self.conversation_history.append({
            'question': question,
            'answer': answer,
            'timestamp': datetime.now().isoformat()
        })
        
        if len(self.conversation_history) > self.max_history_length:
            self.conversation_history = self.conversation_history[-self.max_history_length:]
        
        if self.verbose:
            self.log(f"Historial actualizado: {len(self.conversation_history)} interaccións", "info")
    
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
    
    def _apply_tfidf_reranking(self, query: str, documents: List[Document]) -> List[Document]:
        """Apply TF-IDF reranking to retrieved documents.
        
        Args:
            query: Original query
            documents: Documents to rerank
            
        Returns:
            Reranked list of documents
        """
        if not self.use_tfidf or not self.tfidf_reranker or not documents:
            return documents
        
        if self.tfidf_mode == "rerank":
            # Pure reranking: sort by TF-IDF scores
            reranked = self.tfidf_reranker.rerank(query, documents, top_k=len(documents))
            return [doc for doc, score in reranked]
        
        elif self.tfidf_mode == "filter":
            # Filtering: remove documents below threshold
            scored_docs = self.tfidf_reranker.rerank(query, documents, top_k=None)
            filtered = [(doc, score) for doc, score in scored_docs if score >= self.tfidf_threshold]
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
            history_text = self._format_history(max_turns=5)

        # -----------------------------
        # 2. Decide whether to retrieve (Haiku)
        # -----------------------------
        should_retrieve, search_query = self._should_retrieve_documents(
            question,
            history_text
        )

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

            # Optional TF-IDF post-processing
            if self.use_tfidf and self.tfidf_mode in ("rerank", "filter") and source_docs:
                original_count = len(source_docs)
                source_docs = self._apply_tfidf_reranking(
                    search_query,
                    source_docs
                )

                if self.verbose and self.tfidf_mode == "filter":
                    self.log(
                        f"Filtrado TF-IDF: {original_count} -> {len(source_docs)} docs",
                        "info"
                    )

            if self.use_tfidf and self.tfidf_reranker and source_docs:
                scored_docs = self.tfidf_reranker.rerank(
                    search_query,
                    source_docs,
                    top_k=None
                )
                for doc, score in scored_docs:
                    doc.metadata["tfidf_score"] = float(score)

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
            self._add_to_history(question, clean_answer)

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
        """Extract top TF-IDF keywords from a document.
        
        Args:
            document: Document to analyze
            top_n: Number of keywords to return
            
        Returns:
            List of (keyword, score) tuples
        """
        if not self.use_tfidf or not self.tfidf_reranker:
            return []
        
        return self.tfidf_reranker.get_top_keywords(document, top_n)
    
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
        self.conversation_history = []
        if self.verbose:
            self.log("Historial de conversación limpo", "success")
    
    def get_history(self) -> List[Dict[str, str]]:
        """Get conversation history."""
        return self.conversation_history.copy()
    
    def get_history_summary(self) -> str:
        """Get conversation history summary."""
        if not self.conversation_history:
            return "Non hai historial de conversación" if self.language == "galician" else "No conversation history"
        
        summary_lines = []
        for i, interaction in enumerate(self.conversation_history, 1):
            summary_lines.append(f"\n--- Interacción {i} ---")
            summary_lines.append(f"Pregunta: {interaction['question']}")
            summary_lines.append(f"Resposta: {interaction['answer'][:150]}...")
            if 'timestamp' in interaction:
                summary_lines.append(f"Hora: {interaction['timestamp']}")
        
        return "\n".join(summary_lines)
    
    def save_history(self, filepath: str = "conversation_history.json") -> None:
        """Save conversation history to JSON."""
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(self.conversation_history, f, ensure_ascii=False, indent=2)
            if self.verbose:
                self.log(f"Historial gardado en {filepath}", "success")
        except Exception as e:
            self.log(f"Error gardando historial: {e}", "error")
    
    def load_history(self, filepath: str = "conversation_history.json") -> None:
        """Load conversation history from JSON."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                self.conversation_history = json.load(f)
            if self.verbose:
                self.log(f"Historial cargado desde {filepath} ({len(self.conversation_history)} interaccións)", "success")
        except FileNotFoundError:
            if self.verbose:
                self.log(f"Non se atopou o arquivo {filepath}", "warning")
            self.conversation_history = []
        except Exception as e:
            self.log(f"Error cargando historial: {e}", "error")
            self.conversation_history = []
    
    def export_history_markdown(self, filepath: str = "conversation_history.md") -> None:
        """Export conversation history to Markdown."""
        if not self.conversation_history:
            self.log("Non hai historial para exportar", "warning")
            return
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write("# Historial de Conversación\n\n")
                for i, interaction in enumerate(self.conversation_history, 1):
                    f.write(f"## Interacción {i}\n\n")
                    if 'timestamp' in interaction:
                        f.write(f"**Hora:** {interaction['timestamp']}\n\n")
                    f.write(f"**Pregunta:** {interaction['question']}\n\n")
                    f.write(f"**Resposta:**\n\n{interaction['answer']}\n\n")
                    f.write("---\n\n")
            if self.verbose:
                self.log(f"Historial exportado a {filepath}", "success")
        except Exception as e:
            self.log(f"Error exportando historial: {e}", "error")
