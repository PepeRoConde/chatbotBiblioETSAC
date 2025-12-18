from typing import Any, List, Optional
import os
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI


class LLMManager:
    """Class for managing LLM providers (Anthropic and OpenAI)."""
    
    SUPPORTED_PROVIDERS = ['anthropic', 'openai']
    
    def __init__(
        self, 
        provider: str = 'anthropic', 
        model_name: str = "claude-3-5-sonnet-20241022", 
        api_key: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 512
    ):
        """Initialize the LLM client.
        
        Args:
            provider: LLM provider ('anthropic' or 'openai')
            model_name: Model name to use
            api_key: API key (provider-specific)
            temperature: Temperature for generation
            max_tokens: Maximum tokens to generate
        """
        self.provider = provider.lower()
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Validate provider
        if self.provider not in self.SUPPORTED_PROVIDERS:
            raise ValueError(
                f"Unknown provider: {provider}. Supported providers: {', '.join(self.SUPPORTED_PROVIDERS)}"
            )
        
        # Get and validate API key
        self.api_key = self._get_api_key(api_key)
        
        # Initialize chat model
        self.chat_model = self._initialize_chat_model()
        self.llm = self.chat_model
        
        print(f"Initialized {self.provider.upper()} with model: {self.model_name}")
    
    def _get_api_key(self, api_key: Optional[str]) -> str:
        """Get API key from parameter or environment variable.
        
        Args:
            api_key: API key provided by user
            
        Returns:
            str: API key
            
        Raises:
            ValueError: If API key is not found
        """
        if self.provider == 'anthropic':
            key = api_key or os.environ.get("ANTHROPIC_API_KEY")
            if not key:
                raise ValueError(
                    "Anthropic API key not provided and ANTHROPIC_API_KEY environment variable not set"
                )
        elif self.provider == 'openai':
            key = api_key or os.environ.get("OPENAI_API_KEY")
            if not key:
                raise ValueError(
                    "OpenAI API key not provided and OPENAI_API_KEY environment variable not set"
                )
        
        return key
    
    def _initialize_chat_model(self):
        """Initialize the appropriate chat model based on provider.
        
        Returns:
            Chat model instance
        """
        if self.provider == 'anthropic':
            return ChatAnthropic(
                model=self.model_name,
                anthropic_api_key=self.api_key,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
        elif self.provider == 'openai':
            return ChatOpenAI(
                model=self.model_name,
                openai_api_key=self.api_key,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
