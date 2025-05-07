import os
from typing import Any, List, Mapping, Optional
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage

class MistralLLM(LLM):
    """LangChain implementation for the Mistral API."""
    
    client: MistralClient
    model_name: str = "mistral-large-latest"
    temperature: float = 0.1
    top_p: float = 0.95
    max_tokens: int = 512
    
    def __init__(
        self,
        api_key: Optional[str] = 'CCxNew5opgihFFCEJprQpgIQJJXJIc2T',
        model_name: str = "mistral-large-latest",
        temperature: float = 0.1,
        top_p: float = 0.95,
        max_tokens: int = 512,
        **kwargs
    ):
        """Initialize the Mistral LLM client.
        
        Args:
            api_key: Mistral API key (defaults to MISTRAL_API_KEY env var)
            model_name: Mistral model to use
            temperature: Sampling temperature
            top_p: Top-p sampling
            max_tokens: Maximum number of tokens to generate
        """
        super().__init__(**kwargs)
        
        # Use provided API key or get from environment
        if api_key is None:
            api_key = os.environ.get("MISTRAL_API_KEY")
            if api_key is None:
                raise ValueError(
                    "Mistral API key not provided and MISTRAL_API_KEY environment variable not set"
                )
        
        self.client = MistralClient(api_key=api_key)
        self.model_name = model_name
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        
        print(f"MistralLLM initialized with model: {model_name}")
    
    @property
    def _llm_type(self) -> str:
        return "mistral"
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs
    ) -> str:
        """Generate text using Mistral API.
        
        Args:
            prompt: Text prompt
            stop: Stop sequences
            run_manager: Callback manager
            
        Returns:
            Generated text
        """
        messages = [
            ChatMessage(role="user", content=prompt)
        ]
        
        chat_response = self.client.chat(
            model=self.model_name,
            messages=messages,
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens
        )
        
        return chat_response.choices[0].message.content
    
    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get identifying parameters."""
        return {
            "model_name": self.model_name,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens
        }
