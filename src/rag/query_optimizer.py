"""Query optimization functionality."""
from typing import Tuple, Any
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from .cost.claude_cost_callback import ClaudeCostCallback
from .system_prompt import few_shot_classification_prompt


class QueryOptimizer:
    """Handles query optimization and retrieval decision."""
    
    def __init__(self, llm_query: Any, cost_tracker: Any, language: str = "english", verbose: bool = False):
        """Initialize query optimizer.
        
        Args:
            llm_query: LLM instance for query optimization
            cost_tracker: CostTracker instance
            language: Language for prompts
            verbose: Whether to show detailed information
        """
        self.llm_query = llm_query
        self.cost_tracker = cost_tracker
        self.language = language
        self.verbose = verbose
        
        # Use centralized Rich console utility
        from src.utils.rich_utils import get_console
        self.console = get_console()
    
    def _get_model_name(self, llm) -> str:
        """Extract model name from LLM object."""
        if hasattr(llm, 'model'):
            return llm.model
        elif hasattr(llm, 'model_name'):
            return llm.model_name
        else:
            return 'unknown'
    
    def should_retrieve_documents(self, user_question: str, history_text: str) -> Tuple[bool, str]:
        """Determine if retrieval is needed and generate optimized query.
        
        Args:
            user_question: Original user question
            history_text: Formatted conversation history
            
        Returns:
            (should_retrieve, optimized_query)
        """
        template = few_shot_classification_prompt.get(
            self.language.lower(), 
            few_shot_classification_prompt["english"]
        )
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

            # Explicit decision to NOT retrieve
            if response.upper().startswith("NO_RETRIEVAL"):
                return False, ""

            # Clean common prefixes
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

            # Safety: limit length
            response_words = response.split()
            optimized_query = " ".join(response_words[:15])

            return True, optimized_query

        except Exception as e:
            self.log(
                f"Error en análisis de necesidad de búsqueda: {e}",
                "error"
            )
            # Fail-safe: better to search than not search
            return True, user_question
    
    def log(self, message: str, level: str = "info") -> None:
        """Log a message with appropriate styling based on level."""
        self.console.print(message)

