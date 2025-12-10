from typing import List, Dict, Tuple, Optional, Union, Any
import re
import os
import json
import numpy as np
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain


class TFIDFReranker:
    """TF-IDF based reranker for document relevance scoring."""
    
    def __init__(self, max_features: int = 5000, ngram_range: Tuple[int, int] = (1, 2)):
        """Initialize TF-IDF reranker.
        
        Args:
            max_features: Maximum number of features for TF-IDF
            ngram_range: Range of n-grams to consider (unigrams and bigrams by default)
        """
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            stop_words=None,  # Can set to 'english' or custom list
            lowercase=True,
            token_pattern=r'\b\w+\b'
        )
        self.doc_vectors = None
        self.documents = []
        self.fitted = False
    
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


class HybridRetriever:
    """Hybrid retriever combining vector similarity and TF-IDF."""
    
    def __init__(
        self,
        vector_retriever: Any,
        tfidf_reranker: TFIDFReranker,
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
        self.vector_retriever = vector_retriever
        self.tfidf_reranker = tfidf_reranker
        self.vector_weight = vector_weight
        self.tfidf_weight = tfidf_weight
        self.verbose = verbose
        
        # Normalize weights
        total = vector_weight + tfidf_weight
        self.vector_weight = vector_weight / total
        self.tfidf_weight = tfidf_weight / total
    
    def get_relevant_documents(self, query: str) -> List[Document]:
        """Retrieve documents using hybrid approach.
        
        Args:
            query: Query string
            
        Returns:
            List of ranked documents
        """
        # Step 1: Get initial candidates from vector store
        vector_docs = self.vector_retriever.get_relevant_documents(query)
        
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
    
    def invoke(self, query: str) -> List[Document]:
        """Invoke method for LangChain compatibility."""
        return self.get_relevant_documents(query)


class RAGSystem:
    """RAG system with TF-IDF integration for improved retrieval and reranking."""
    
    def __init__(
        self, 
        vectorstore: Any,
        k: int = 4,
        threshold: float = 0.7,
        search_type: str = "mmr",
        language: str = "english",
        llm: Optional[Any] = None,
        provider: str = 'claude',
        temperature: float = 0.1,
        max_tokens: int = 512,
        max_history_length: int = 10,
        use_tfidf: bool = True,
        tfidf_mode: str = "rerank",  # "rerank", "hybrid", or "filter"
        tfidf_weight: float = 0.3,
        tfidf_threshold: float = 0.1
    ):
        """Initialize the RAG system with TF-IDF support.
        
        Args:
            vectorstore: Vector store for retrieval
            k: Number of documents to retrieve
            threshold: Filter unrelevant documents
            search_type: Way of performing the retrieval
            language: Language for prompt template
            llm: Language model instance
            provider: LLM provider ('mistral' or 'claude')
            max_history_length: Maximum number of conversation turns to keep
            use_tfidf: Whether to use TF-IDF enhancement
            tfidf_mode: How to use TF-IDF ("rerank", "hybrid", or "filter")
            tfidf_weight: Weight for TF-IDF in hybrid mode (0-1)
            tfidf_threshold: Minimum TF-IDF score for filtering
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
        
        # Conversation history
        self.conversation_history: List[Dict[str, str]] = []
        
        # Console setup
        try:
            self.console = __builtins__.rich_console
            self.verbose = __builtins__.verbose_mode
        except (AttributeError, NameError):
            from rich.console import Console
            self.console = Console()
            self.verbose = True
        
        self.llm = llm
        
        # TF-IDF setup
        self.tfidf_reranker = None
        if self.use_tfidf:
            self._setup_tfidf()
        
        # Create the retriever (base or hybrid)
        if self.use_tfidf and self.tfidf_mode == "hybrid":
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
            self.log(f"RAG system initialized{tfidf_status} with {provider.upper()}", "success")
            self.log(f"Language: {language}", "info")
    
    def _setup_tfidf(self) -> None:
        """Setup TF-IDF reranker by fitting on all documents in vectorstore."""
        if self.verbose:
            self.log("Initializing TF-IDF reranker...", "info")
        
        # Get all documents from vectorstore
        # This is a workaround since FAISS doesn't expose all docs directly
        # We do a broad search to get a large sample
        try:
            # Try to get docstore if available (FAISS has this)
            if hasattr(self.vectorstore, 'docstore'):
                all_docs = list(self.vectorstore.docstore._dict.values())
            else:
                # Fallback: retrieve many documents with a generic query
                temp_retriever = self.vectorstore.as_retriever(search_kwargs={"k": 1000})
                all_docs = temp_retriever.get_relevant_documents("*")
            
            if all_docs:
                self.tfidf_reranker = TFIDFReranker(max_features=5000, ngram_range=(1, 2))
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
            search_kwargs={"k": self.k * 2, "score_threshold": self.threshold},  # Get more candidates
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
        templates = {
            "english": """Use the following context from documents and the conversation history (if any) to answer the question. 
Be concise and extract important information from the text. 
If the question refers to something mentioned earlier in the conversation, use that information.
If you don't know, politely say you don't know instead of making up an answer. 
The answer should be pleasant and clear.

Context from documents:
{context}

{history}

Question: {input}

Answer:""",
            "spanish": """Usa el siguiente contexto de los documentos y el historial de conversación (si existe) para responder a la pregunta.
Sé conciso, extrae información importante del texto.
Si la pregunta hace referencia a algo mencionado anteriormente en la conversación, usa esa información.
Si no sabes, di educadamente que no sabes, no intentes inventar la respuesta.
La respuesta debe ser agradable y clara.

Contexto de los documentos:
{context}

{history}

Pregunta: {input}

Respuesta:""",
            "galician": """Usa o seguinte contexto dos documentos e o historial de conversación (se existe) para responder á pregunta.
Responde en galego e NON en portugués. Sé conciso, extrae información importante do texto.
Se a pregunta fai referencia a algo mencionado anteriormente na conversación, usa esa información.
Se non sabes a resposta, di educadamente que non o sabes, non intentes inventar.
A resposta debe ser agradable e clara.

Contexto dos documentos:
{context}

{history}

Pregunta: {input}

Resposta:"""
        }
        
        template = templates.get(self.language.lower(), templates["english"])
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
            llm=self.llm,
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
    
    def query(self, question: str, use_history: bool = True) -> Tuple[str, List[Document]]:
        """Query the RAG system with optional TF-IDF enhancement.
        
        Args:
            question: Question to ask
            use_history: Whether to include conversation history
        
        Returns:
            Answer and source documents (with TF-IDF scores in metadata if applicable)
        """
        # Format conversation history
        history_text = ""
        if use_history:
            history_text = self._format_history(max_turns=5)
        
        # Invoke the RAG chain
        result = self.rag_chain.invoke({
            "input": question,
            "history": history_text
        })
        
        answer = result.get("answer", "No answer found")
        source_docs = result.get("context", [])
        
        # Apply TF-IDF reranking if enabled and mode is "rerank" or "filter"
        if self.use_tfidf and self.tfidf_mode in ["rerank", "filter"] and source_docs:
            original_count = len(source_docs)
            source_docs = self._apply_tfidf_reranking(question, source_docs)
            
            if self.verbose and self.tfidf_mode == "filter":
                self.log(f"TF-IDF filtering: {original_count} → {len(source_docs)} documents", "info")
        
        # Add TF-IDF scores to document metadata for debugging
        if self.use_tfidf and self.tfidf_reranker and source_docs:
            scored_docs = self.tfidf_reranker.rerank(question, source_docs, top_k=None)
            for doc, score in scored_docs:
                doc.metadata['tfidf_score'] = float(score)
        
        if not source_docs:
            no_info_messages = {
                "galician": "Non atopei información relevante nos documentos para responder a esa pregunta.",
                "spanish": "No encontré información relevante en los documentos para responder a esa pregunta.",
                "english": "I couldn't find relevant information in the documents to answer that question."
            }
            answer = no_info_messages.get(self.language.lower(), no_info_messages["english"])
        
        if self.verbose:
            self.log(f"Retrieved {len(source_docs)} documents for the query", "info")
        
        clean_answer = self.extract_answer_only(answer)
        
        if use_history:
            self._add_to_history(question, clean_answer)
        
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
