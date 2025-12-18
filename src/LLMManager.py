from typing import Any, List, Optional
import os
from langchain_mistralai import ChatMistralAI
from langchain_anthropic import ChatAnthropic

class LLMManager:
    """Class for managing LLM providers (Mistral and Claude)."""

    def __init__(
        self, 
        provider: str = 'claude', 
        model_name: str = "claude-3-5-sonnet-20241022", 
        api_key: str = None,
        temperature: float = 0.1,
        max_tokens: int = 512
    ):
        """Initialize the LLM client.

        Args:
            provider: LLM provider ('mistral' or 'claude')
            model_name: Model name to use
            api_key: API key (provider-specific)
            temperature: Temperature for generation
            max_tokens: Maximum tokens to generate
        """
        self.provider = provider.lower()
        
        # Get API key based on provider
        if self.provider == 'mistral':
            self.api_key = api_key or os.environ.get("MISTRAL_API_KEY")
            if not self.api_key:
                raise ValueError(
                    "Mistral API key not provided and MISTRAL_API_KEY environment variable not set"
                )
        elif self.provider == 'claude':
            self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
            if not self.api_key:
                raise ValueError(
                    "Anthropic API key not provided and ANTHROPIC_API_KEY environment variable not set"
                )
        else:
            raise ValueError(f"Unknown provider: {provider}. Use 'mistral' or 'claude'")

        print(f"Initializing {self.provider.upper()} API with model: {model_name}")

        # Initialize the appropriate chat model
        if self.provider == 'mistral':
            self.chat_model = ChatMistralAI(
                model=model_name,
                mistral_api_key=self.api_key,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=0.95
            )
        elif self.provider == 'claude':
            self.chat_model = ChatAnthropic(
                model=model_name,
                anthropic_api_key=self.api_key,
                temperature=temperature,
                max_tokens=max_tokens
                
            )

        # For LangChain 1.0 compatibility, the chat model itself is the LLM
        self.llm = self.chat_model
