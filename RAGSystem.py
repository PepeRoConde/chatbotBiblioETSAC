from typing import List, Dict, Tuple, Optional, Union, Any
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
        return PromptTemplate.from_template(template)
        
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
        
        return answer, source_docs
