from typing import List, Dict, Tuple, Optional, Union, Any
import re
import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain

from LLMManager import LLMManager

class MistralRAGSystem:
    """Retrieval Augmented Generation system using LangChain 1.0."""
    
    def __init__(
        self, 
        vectorstore: Any,
        k: int = 4,
        language: str = "english",
        model_name: str = "claude-3-5-sonnet-20241022",
        provider: str = 'claude',
        temperature: float = 0.1,
        max_tokens: int = 512,
        api_key: str = None
    ):
        """Initialize the RAG system.
        
        Args:
            vectorstore: Vector store for retrieval
            k: Number of documents to retrieve
            language: Language for prompt template
            model_name: Model name to use
            provider: LLM provider ('mistral' or 'claude')
            api_key: API key for the provider
        """
        self.vectorstore = vectorstore
        self.language = language
        self.k = k
        
        # Use the global rich console if available
        try:
            self.console = __builtins__.rich_console
            self.verbose = __builtins__.verbose_mode
        except (AttributeError, NameError):
            # Fallback to a new console if not running from main.py
            from rich.console import Console
            self.console = Console()
            self.verbose = True
        
        # Initialize the LLM
        self.llm_manager = LLMManager(
            provider=provider,
            model_name=model_name, 
            api_key=api_key
        )
        self.llm = self.llm_manager.llm
        
        # Create the retriever
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": k})
        
        # Create the prompt template based on language
        self.prompt = self._create_prompt_template()
        
        # Create the RAG chain using LangChain 1.0 API
        self._create_rag_chain()
        
        if self.verbose:
            self.log(f"RAG system initialized with {provider.upper()} using model: {model_name}", "success")
            self.log(f"Language: {language}", "info")
    
    def log(self, message: str, level: str = "info") -> None:
        """Log a message with appropriate styling based on level.
        
        Args:
            message: Message to log
            level: Log level (info, success, warning, error)
        """
        self.console.print(message)
    
    def _create_prompt_template(self) -> ChatPromptTemplate:
        """Create prompt template based on selected language.
        
        Returns:
            Chat prompt template
        """
        templates = {
            "english": """Use the following context to answer the question. Be concise and extract important information from the text. If you don't know, politely say you don't know instead of making up an answer. The answer should be pleasant and clear.

Context:
{context}

Question: {input}

Answer:""",
            "spanish": """Usa el siguiente contexto para responder a la pregunta. Sé conciso, extrae información importante del texto. Si no sabes, di educadamente que no sabes, no intentes inventar la respuesta. La respuesta debe ser agradable y clara.

Contexto:
{context}

Pregunta: {input}

Respuesta:""",
            "galician": """Usa o seguinte contexto para responder á pregunta. Se conciso, extrae información importante do texto. A resposta debe ser agradable e clara.

Contexto:
{context}

Pregunta: {input}

Resposta:"""
        }
        
        template = templates.get(self.language.lower(), templates["english"])
        if self.verbose:
            self.log(f"Using {self.language} prompt template", "info")
        
        return ChatPromptTemplate.from_template(template)
    
    def _create_rag_chain(self) -> None:
        """Create the RAG chain using LangChain 1.0 API."""
        # Create the document chain (combines documents with the prompt)
        document_chain = create_stuff_documents_chain(
            llm=self.llm,
            prompt=self.prompt
        )
        
        # Create the retrieval chain
        self.rag_chain = create_retrieval_chain(
            retriever=self.retriever,
            combine_docs_chain=document_chain
        )
        
    def extract_answer_only(self, full_response: str) -> str:
        """Extract just the actual answer part from a full LLM response.
        
        This removes any system instructions, context, or extraneous text.
        
        Args:
            full_response: The complete response from the RAG system
            
        Returns:
            The cleaned answer text only
        """
        answer_only = full_response
        
        # Remove any "Answer:" prefix if present
        answer_only = re.sub(r'^.*Answer:\s*', '', answer_only, flags=re.DOTALL)
        answer_only = re.sub(r'^.*Respuesta:\s*', '', answer_only, flags=re.DOTALL)
        answer_only = re.sub(r'^.*Resposta:\s*', '', answer_only, flags=re.DOTALL)
        
        # Clean up any trailing system instructions
        answer_only = re.sub(r'<.*?>.*', '', answer_only, flags=re.DOTALL)
        
        return answer_only.strip()
    
    def query(self, question: str) -> Tuple[str, List[Document]]:
        """Query the RAG system with a question.
        
        Args:
            question: Question to ask
            
        Returns:
            Answer and source documents
        """
        # Invoke the chain with the new API
        result = self.rag_chain.invoke({"input": question})
        
        # Extract answer and sources from the new result structure
        answer = result.get("answer", "No answer found")
        source_docs = result.get("context", [])
        
        if self.verbose:
            self.log(f"Retrieved {len(source_docs)} documents for the query", "info")
        
        # Clean the answer before returning
        clean_answer = self.extract_answer_only(answer)
        
        return clean_answer, source_docs