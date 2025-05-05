from typing import List, Dict, Tuple, Optional, Union, Any
import re
from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document
from langchain.chains import RetrievalQA

from LocalLLM import LocalLLM

class RAGSystem:
    """Retrieval Augmented Generation system."""
    
    def __init__(
        self, 
        vectorstore: Any,
        k: int = 4,
        language: str = "english"
    ):
        """Initialize the RAG system.
        
        Args:
            vectorstore: Vector store for retrieval
            k: Number of documents to retrieve
            language: Language for prompt template
        """
        self.vectorstore = vectorstore
        self.language = language
        
        # Use the global rich console if available
        try:
            self.console = __builtins__.rich_console
            self.verbose = __builtins__.verbose_mode
        except (AttributeError, NameError):
            # Fallback to a new console if not running from main.py
            from rich.console import Console
            self.console = Console()
            self.verbose = True
        
        # Initialize the local LLM
        self.local_llm = LocalLLM()
        self.llm = self.local_llm.llm
        
        # Create the retriever
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": k})
        
        # Create the prompt template based on language
        self.prompt = self._create_prompt_template()
        
        # Create the QA chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": self.prompt}
        )
        
        if self.verbose:
            self.log(f"RAG system initialized with language: {language}", "success")
    
    def log(self, message: str, level: str = "info") -> None:
        """Log a message with appropriate styling based on level.
        
        Args:
            message: Message to log
            level: Log level (info, success, warning, error)
        """
        self.console.print(message)
    
    def _create_prompt_template(self) -> PromptTemplate:
        """Create prompt template based on selected language.
        
        Returns:
            Prompt template
        """
        templates = {
            "english": """Use the following information to answer the question. Be concise and extract important information from the text. If you don't know, politely say you don't know instead of making up an answer. The answer should be pleasant and clear.

Context:
{context}

Question: {question}

Answer:""",
            "spanish": """Usa la siguiente información para responder a la pregunta. Sé conciso, extrae información importante del texto. Si no sabes, di educadamente que no sabes, no intentes inventar la respuesta. La respuesta debe ser agradable y clara.

Contexto:
{context}

Pregunta: {question}

Respuesta:""",
            "galician": """Usa a seguinte informacion para responder á pregunta. Se conciso, extrae informacion importante do texto. Si non sabes, di educadamente que non sabes, non intentes inventar a resposta. A resposta debe ser agradable e clara.

Contexto:
{context}

Pregunta: {question}

Resposta:"""
        }
        
        template = templates.get(self.language.lower(), templates["english"])
        if self.verbose:
            self.log(f"Using {self.language} prompt template", "info")
        return PromptTemplate.from_template(template)
        
    def extract_answer_only(self, full_response: str) -> str:
        """Extract just the actual answer part from a full LLM response.
        
        This removes any system instructions, context, or extraneous text.
        
        Args:
            full_response: The complete response from the RAG system
            
        Returns:
            The cleaned answer text only
        """
        # Try to find where the actual answer begins after all the prompting
        # This pattern might need adjustment based on how the LLM formats answers
        answer_only = full_response
        
        # Remove any "Answer:" prefix if present
        answer_only = re.sub(r'^.*Answer:\s*', '', answer_only, flags=re.DOTALL)
        
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
        # Use invoke instead of __call__ to avoid deprecation warning
        result = self.qa_chain.invoke({"query": question})
        
        # Extract answer and sources based on the returned structure
        if "result" in result:
            answer = result["result"]
        else:
            answer = result.get("answer", "No answer found")
            
        source_docs = result.get("source_documents", [])
        
        if self.verbose:
            self.log(f"Retrieved {len(source_docs)} documents for the query", "info")
        
        # Clean the answer before returning
        clean_answer = self.extract_answer_only(answer)
        
        return clean_answer, source_docs
