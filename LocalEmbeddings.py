from typing import List, Dict, Tuple, Optional, Union, Any
from transformers import AutoTokenizer, AutoModel
import torch

class LocalEmbeddings:
    """Class for generating embeddings using local models."""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """Initialize embedding model.
        
        Args:
            model_name: The HuggingFace model name for embeddings
        """
        print(f"Loading embeddings model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
        print(f"Using device: {self.device}")
        self.model.to(self.device)
        
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of documents.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embedding vectors
        """
        embeddings = []
        batch_size = 8  # Process in smaller batches to save memory
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            inputs = self.tokenizer(batch, padding=True, truncation=True, 
                                    return_tensors="pt", max_length=512).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Use mean pooling to get document embeddings
                attention_mask = inputs["attention_mask"]
                embeddings_batch = self.mean_pooling(outputs.last_hidden_state, attention_mask)
                embeddings.extend(embeddings_batch.cpu().numpy())
                
        return embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """Generate embedding for a query.
        
        Args:
            text: Query text
            
        Returns:
            Embedding vector
        """
        return self.embed_documents([text])[0]
    
    @staticmethod
    def mean_pooling(token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Mean pooling operation to get sentence embeddings.
        
        Args:
            token_embeddings: Token-level embeddings
            attention_mask: Attention mask
            
        Returns:
            Pooled embeddings
        """
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    def __call__(self, text: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        """Make the class callable for LangChain compatibility.
        
        Args:
            text: Single string or list of strings
            
        Returns:
            Single embedding or list of embeddings
        """
        if isinstance(text, str):
            return self.embed_query(text)
        return self.embed_documents(text)