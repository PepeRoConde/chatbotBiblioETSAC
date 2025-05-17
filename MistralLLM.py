from typing import Any, List, Optional
import os
from langchain_mistralai import ChatMistralAI
from langchain.schema.output_parser import StrOutputParser


class MistralLLM:
    """Class for using the Mistral API."""
    
    def __init__(self, model_name: str = "mistral-medium", api_key: str = None):
        """Initialize the Mistral API client.
        
        Args:
            model_name: Mistral model name (e.g., "mistral-tiny", "mistral-small", "mistral-medium", "mistral-large")
            api_key: Mistral API key. If None, will look for MISTRAL_API_KEY environment variable
        """
        # Use provided API key or get from environment variables
        self.api_key = api_key or os.environ.get("MISTRAL_API_KEY")
        if not self.api_key:
            raise ValueError("Mistral API key not provided and MISTRAL_API_KEY environment variable not set")
        
        print(f"Initializing Mistral API with model: {model_name}")
        
        # Initialize the Mistral chat model
        self.chat_model = ChatMistralAI(
            model=model_name,
            mistral_api_key=self.api_key,
            temperature=0.1,
            max_tokens=512,
            top_p=0.95
        )
        
        # Create a simple wrapper to make it compatible with our existing code
        # that expects an LLM with an invoke method
        self.llm = self.chat_model | StrOutputParser()
