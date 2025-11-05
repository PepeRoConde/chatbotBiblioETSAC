import hashlib
import pickle
import os
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union, Any
from transformers import AutoTokenizer, AutoModel
import torch

class LocalEmbeddings:
    """Class for generating embeddings using local models with caching."""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", 
                 cache_dir: str = ".embeddings_cache"):
        """Initialize embedding model.
        
        Args:
            model_name: The HuggingFace model name for embeddings
            cache_dir: Directory to store cached embeddings
        """
        print(f"Loading embeddings model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
        print(f"Using device: {self.device}")
        self.model.to(self.device)
        
        # Setup cache
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.cache_file = self.cache_dir / "embeddings_cache.pkl"
        self.cache = self._load_cache()
        
    def _load_cache(self) -> Dict[str, List[float]]:
        """Load embeddings cache from disk."""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'rb') as f:
                    cache = pickle.load(f)
                print(f"Loaded {len(cache)} cached embeddings")
                return cache
            except Exception as e:
                print(f"Error loading cache: {e}")
                return {}
        return {}
    
    def _save_cache(self):
        """Save embeddings cache to disk."""
        try:
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.cache, f)
        except Exception as e:
            print(f"Error saving cache: {e}")
    
    @staticmethod
    def _get_text_hash(text: str) -> str:
        """Generate hash for a text string."""
        return hashlib.sha256(text.encode('utf-8')).hexdigest()
    
    def embed_documents(self, texts: List[str], use_cache: bool = True) -> List[List[float]]:
        """Generate embeddings for a list of documents with caching.
        
        Args:
            texts: List of text strings to embed
            use_cache: Whether to use cached embeddings
            
        Returns:
            List of embedding vectors
        """
        embeddings = []
        texts_to_embed = []
        text_indices = []
        
        # Check cache for each text
        for idx, text in enumerate(texts):
            text_hash = self._get_text_hash(text)
            
            if use_cache and text_hash in self.cache:
                embeddings.append(self.cache[text_hash])
            else:
                embeddings.append(None)  # Placeholder
                texts_to_embed.append(text)
                text_indices.append(idx)
        
        # Generate embeddings for uncached texts
        if texts_to_embed:
            print(f"Generating embeddings for {len(texts_to_embed)} new/modified documents")
            new_embeddings = self._generate_embeddings(texts_to_embed)
            
            # Update cache and results
            for idx, text, embedding in zip(text_indices, texts_to_embed, new_embeddings):
                text_hash = self._get_text_hash(text)
                self.cache[text_hash] = embedding
                embeddings[idx] = embedding
            
            # Save cache after generating new embeddings
            self._save_cache()
        else:
            print("All documents found in cache!")
        
        return embeddings
    
    def _generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings without caching logic.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embedding vectors
        """
        embeddings = []
        batch_size = 8
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            inputs = self.tokenizer(batch, padding=True, truncation=True, 
                                    return_tensors="pt", max_length=512).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                attention_mask = inputs["attention_mask"]
                embeddings_batch = self.mean_pooling(outputs.last_hidden_state, attention_mask)
                embeddings.extend(embeddings_batch.cpu().numpy().tolist())
                
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
        """Mean pooling operation to get sentence embeddings."""
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    def clear_cache(self):
        """Clear all cached embeddings."""
        self.cache = {}
        if self.cache_file.exists():
            self.cache_file.unlink()
        print("Cache cleared")
    
    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        return {
            "cached_documents": len(self.cache),
            "cache_size_mb": self.cache_file.stat().st_size / (1024*1024) if self.cache_file.exists() else 0
        }
    
    def __call__(self, text: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        """Make the class callable for LangChain compatibility."""
        if isinstance(text, str):
            return self.embed_query(text)
        return self.embed_documents(text)