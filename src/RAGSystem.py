from typing import List, Dict, Tuple, Optional, Union, Any
import re
import os
import json
import numpy as np
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity



from cost_tracker import CostTracker
from claude_cost_callback import ClaudeCostCallback

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain


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

class RAGSystem:
    """RAG system with TF-IDF integration and query optimization."""
    
    def __init__(
        self, 
        vectorstore: Any,
        llm: Any,
        llm_query: Optional[Any] = "claude-3-haiku-20240307",  
        k: int = 4,
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

        templates = {
            "english": """You are a university assistant analyzing if a student's question requires searching in university documents (regulations, guides, academic procedures, course information, etc.).

    Student asked: {question}

    Recent conversation history:
    {history}

    Analyze if this question needs information from university documents or can be answered from:
    1. General knowledge or common courtesy responses
    2. Previous conversation context
    3. Greetings, thanks, clarifications about previous answers

    If NO retrieval needed (greetings, thanks, clarifications), respond: NO_RETRIEVAL
    If retrieval IS needed (academic info, regulations, procedures), respond with an optimized search query (max 15 words).

    Examples:
    - "Hello" -> NO_RETRIEVAL
    - "Thanks for the info" -> NO_RETRIEVAL
    - "Can you repeat that?" -> NO_RETRIEVAL
    - "What are the enrollment deadlines?" -> enrollment deadlines registration periods

    Your response:""",

            "spanish": """Eres un asistente universitario analizando si la pregunta de un estudiante requiere buscar en documentos de la universidad (normativas, guías, procedimientos académicos, información de cursos, etc.).

    El estudiante preguntó: {question}

    Historial reciente:
    {history}

    Si NO necesita búsqueda, responde: NO_RETRIEVAL  
    Si SÍ necesita búsqueda, responde con una query optimizada (máx 15 palabras).

    Tu respuesta:""",

            "galician": """Es un asistente universitario analizando se a pregunta dun estudante require buscar en documentos da universidade (normativas, guías, procedementos académicos, información de cursos, etc.).

    O estudante preguntou: {question}

    Historial recente:
    {history}

    Se NON necesita busca, responde: NO_RETRIEVAL  
    Se SI necesita busca, responde cunha query optimizada (máx 15 palabras).

    A túa resposta:"""
        }

        template = templates.get(self.language.lower(), templates["english"])
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
        templates = {
            "english": """Youre a RAGsystem of the University of A Coruña (UDC) use the following context from documents and the conversation history (if any) to answer the question. 
Be concise and extract important information from the text. 
If the question refers to something mentioned earlier in the conversation, use that information.
If you don't know, politely say you don't know instead of making up an answer. 
The answer should be pleasant and clear.

Context from documents:
{context}

{history}

Question: {input}

Answer:""",
            "spanish": """Eres un RAGsystem de la universidad da couruña (UDC) usa el siguiente contexto de los documentos y el historial de conversación (si existe) para responder a la pregunta.
Sé conciso, extrae información importante del texto.
Si la pregunta hace referencia a algo mencionado anteriormente en la conversación, usa esa información.
Si no sabes, di educadamente que no sabes, no intentes inventar la respuesta.
La respuesta debe ser agradable y clara.

Contexto de los documentos:
{context}

{history}

Pregunta: {input}

Respuesta:""",
            "galician": """Es un RAGsystem da universidade da couruña (UDC) usa o seguinte contexto dos documentos e o historial de conversación (se existe) para responder á pregunta.
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
            direct_templates = {
                "english": """Answer the following question directly. Use conversation history if relevant.

    {history}

    Question: {input}

    Answer:""",
                "spanish": """Responde directamente a la siguiente pregunta. Usa el historial si es relevante.

    {history}

    Pregunta: {input}

    Respuesta:""",
                "galician": """Responde directamente á seguinte pregunta en galego (NON en portugués). Usa o historial se é relevante.

    {history}

    Pregunta: {input}

    Resposta:"""
            }

            template = direct_templates.get(
                self.language.lower(),
                direct_templates["english"]
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
        self.log("📊 RESUMEN DE COSTES", "success")
        self.log("="*50, "info")
        self.log(f"💵 Coste total: ${summary['total_cost']:.6f}", "success")
        self.log(f"📝 Número de queries: {summary['num_queries']}", "info")
        self.log(f"📊 Coste medio por query: ${summary['avg_cost_per_query']:.6f}", "info")
        self.log(f"🔍 Queries con retrieval: {summary['queries_with_retrieval']}", "info")
        self.log(f"💬 Queries sin retrieval: {summary['queries_without_retrieval']}", "info")
        self.log(f"🤖 Coste total Haiku (queries): ${summary['total_query_costs']:.6f}", "info")
        self.log(f"🧠 Coste total Sonnet (respuestas): ${summary['total_answer_costs']:.6f}", "info")
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