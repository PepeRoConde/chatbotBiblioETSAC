from typing import List, Dict, Tuple, Optional, Union, Any
import re
import os
import json
from datetime import datetime
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain

from LLMManager import LLMManager

class MistralRAGSystem:
    """Retrieval Augmented Generation system using LangChain 1.0 with conversation history."""
    
    def __init__(
        self, 
        vectorstore: Any,
        k: int = 4,
        threshold: float = 0.7,
        search_type: str = "mmr",
        language: str = "english",
        model_name: str = "claude-3-5-sonnet-20241022",
        provider: str = 'claude',
        temperature: float = 0.1,
        max_tokens: int = 512,
        api_key: str = None,
        max_history_length: int = 10
    ):
        """Initialize the RAG system.
        
        Args:
            vectorstore: Vector store for retrieval
            k: Number of documents to retrieve
            threshold: Filter unrelevant documents
            search_type: Way of performing the retrieval
            language: Language for prompt template
            model_name: Model name to use
            provider: LLM provider ('mistral' or 'claude')
            api_key: API key for the provider
            max_history_length: Maximum number of conversation turns to keep
        """
        self.vectorstore = vectorstore
        self.language = language
        self.k = k
        self.threshold = threshold
        self.search_type = search_type
        self.max_history_length = max_history_length
        
        # NUEVO: Historial de conversación
        self.conversation_history: List[Dict[str, str]] = []
        
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
        self.retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": self.k, "score_threshold": self.threshold},
            search_type=self.search_type
        )
        
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
        if self.verbose:
            self.log(f"Using {self.language} prompt template", "info")
        
        return ChatPromptTemplate.from_template(template)
    
    def _format_history(self, max_turns: int = 5) -> str:
        """Formatea el historial de conversación para incluirlo en el prompt.
        
        Args:
            max_turns: Número máximo de turnos de conversación a incluir
            
        Returns:
            Historial formateado como string
        """
        if not self.conversation_history:
            return ""
        
        # Tomar solo los últimos N turnos
        recent_history = self.conversation_history[-max_turns:]
        
        # Determinar etiquetas según el idioma
        if self.language.lower() == "galician":
            user_label = "Usuario"
            assistant_label = "Asistente"
            header = "Historial da conversación:"
        elif self.language.lower() == "spanish":
            user_label = "Usuario"
            assistant_label = "Asistente"
            header = "Historial de la conversación:"
        else:
            user_label = "User"
            assistant_label = "Assistant"
            header = "Conversation history:"
        
        # Formatear historial
        history_lines = [header]
        for interaction in recent_history:
            # Truncar respuestas muy largas para no saturar el contexto
            answer = interaction['answer']
            if len(answer) > 300:
                answer = answer[:300] + "..."
            
            history_lines.append(f"{user_label}: {interaction['question']}")
            history_lines.append(f"{assistant_label}: {answer}\n")
        
        return "\n".join(history_lines)
    
    def _add_to_history(self, question: str, answer: str) -> None:
        """Añade una interacción al historial.
        
        Args:
            question: Pregunta del usuario
            answer: Respuesta del asistente
        """
        self.conversation_history.append({
            'question': question,
            'answer': answer,
            'timestamp': datetime.now().isoformat()
        })
        
        # Mantener solo las últimas N interacciones
        if len(self.conversation_history) > self.max_history_length:
            self.conversation_history = self.conversation_history[-self.max_history_length:]
        
        if self.verbose:
            self.log(f"Historial actualizado: {len(self.conversation_history)} interaccións", "info")
    
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
    
    def query(self, question: str, use_history: bool = True) -> Tuple[str, List[Document]]:
        """Query the RAG system with a question.
        
        Args:
            question: Question to ask
            use_history: Whether to include conversation history in the context
        
        Returns:
            Answer and source documents
        """
        # Formatear historial si está habilitado
        history_text = ""
        if use_history:
            history_text = self._format_history(max_turns=5)
        
        # Invoke the chain with the new API, incluyendo historial
        result = self.rag_chain.invoke({
            "input": question,
            "history": history_text
        })
        
        # Extract answer and sources from the new result structure
        answer = result.get("answer", "No answer found")
        source_docs = result.get("context", [])
        
        # Si no hay documentos relevantes
        if not source_docs:
            no_info_messages = {
                "galician": "Non atopei información relevante nos documentos para responder a esa pregunta.",
                "spanish": "No encontré información relevante en los documentos para responder a esa pregunta.",
                "english": "I couldn't find relevant information in the documents to answer that question."
            }
            answer = no_info_messages.get(self.language.lower(), no_info_messages["english"])
        
        if self.verbose:
            self.log(f"Retrieved {len(source_docs)} documents for the query", "info")
        
        # Clean the answer before returning
        clean_answer = self.extract_answer_only(answer)
        
        # Añadir al historial
        if use_history:
            self._add_to_history(question, clean_answer)
        
        return clean_answer, source_docs
    
    def clear_history(self) -> None:
        """Limpia el historial de conversación."""
        self.conversation_history = []
        if self.verbose:
            self.log("Historial de conversación limpo", "success")
    
    def get_history(self) -> List[Dict[str, str]]:
        """Obtiene el historial de conversación.
        
        Returns:
            Lista de interacciones (pregunta-respuesta con timestamp)
        """
        return self.conversation_history.copy()
    
    def get_history_summary(self) -> str:
        """Obtiene un resumen del historial de conversación.
        
        Returns:
            String con resumen del historial
        """
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
        """Guarda el historial en un archivo JSON.
        
        Args:
            filepath: Ruta del archivo donde guardar el historial
        """
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(self.conversation_history, f, ensure_ascii=False, indent=2)
            
            if self.verbose:
                self.log(f"Historial gardado en {filepath}", "success")
        except Exception as e:
            self.log(f"Error gardando historial: {e}", "error")
    
    def load_history(self, filepath: str = "conversation_history.json") -> None:
        """Carga el historial desde un archivo JSON.
        
        Args:
            filepath: Ruta del archivo desde donde cargar el historial
        """
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
        """Exporta el historial a formato Markdown.
        
        Args:
            filepath: Ruta del archivo donde exportar
        """
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