"""Conversation history management."""
from typing import List, Dict
from datetime import datetime
import json


class ConversationHistory:
    """Manages conversation history for RAG system."""
    
    def __init__(self, max_history_length: int = 10, language: str = "english", verbose: bool = False):
        """Initialize conversation history manager.
        
        Args:
            max_history_length: Maximum number of conversation turns to keep
            language: Language for formatting
            verbose: Whether to show detailed information
        """
        self.max_history_length = max_history_length
        self.language = language
        self.verbose = verbose
        self.conversation_history: List[Dict[str, str]] = []
        
        # Use centralized Rich console utility
        from src.utils.rich_utils import get_console
        self.console = get_console()
    
    def format_history(self, max_turns: int = 5) -> str:
        """Format conversation history for inclusion in prompt.
        
        Args:
            max_turns: Maximum number of recent turns to include
            
        Returns:
            Formatted history string
        """
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
    
    def add_interaction(self, question: str, answer: str) -> None:
        """Add interaction to conversation history.
        
        Args:
            question: User question
            answer: Assistant answer
        """
        self.conversation_history.append({
            'question': question,
            'answer': answer,
            'timestamp': datetime.now().isoformat()
        })
        
        if len(self.conversation_history) > self.max_history_length:
            self.conversation_history = self.conversation_history[-self.max_history_length:]
        
        if self.verbose:
            self.log(f"Historial actualizado: {len(self.conversation_history)} interaccións", "info")
    
    def clear(self) -> None:
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
    
    def log(self, message: str, level: str = "info") -> None:
        """Log a message with appropriate styling based on level."""
        self.console.print(message)

