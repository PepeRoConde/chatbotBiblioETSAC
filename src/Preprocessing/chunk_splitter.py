"""Document chunking functionality."""
from typing import List, Optional, Any
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain_text_splitters import RecursiveCharacterTextSplitter


class ChunkSplitter:
    """Handles splitting documents into chunks with optional prefixing."""
    
    def __init__(
        self,
        chunk_size: int = 300,
        chunk_overlap: int = 15,
        prefix_mode: str = "source",
        llm: Optional[Any] = None,
        max_workers: int = 7,
        verbose: bool = False
    ):
        """Initialize chunk splitter.
        
        Args:
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
            prefix_mode: How to prefix chunks ("none", "source", "llm")
            llm: LLM instance for prefix generation (required if prefix_mode="llm")
            max_workers: Maximum number of parallel workers
            verbose: Whether to show detailed information
        """
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        self.prefix_mode = prefix_mode
        self.llm = llm
        self.max_workers = max_workers
        self.verbose = verbose
        
        valid_modes = "none, source, llm"
        if self.prefix_mode not in {"none", "source", "llm"}:
            raise ValueError(f"prefix_mode must be one of {valid_modes}")
        
        # Use centralized Rich console utility
        from src.utils.rich_utils import get_console
        self.console = get_console()
    
    def _split_document_batch(self, docs: List) -> List:
        """Split a batch of documents into chunks."""
        return self.text_splitter.split_documents(docs)
    
    def _add_source_prefix(self, chunk) -> Any:
        """Add source-based prefix to a chunk with date information."""
        from pathlib import Path
        from datetime import datetime
        
        source_url = chunk.metadata.get("source_url", "URL descoñecida")
        text_path = chunk.metadata.get("text_path", "descoñecido")
        filename = Path(text_path).name
        file_type = chunk.metadata.get("original_format", "descoñecido")
        
        # Get date information - try different fields
        last_modified = chunk.metadata.get("last_modified")  # From server
        tipo_data = 'Data modificación'
        if not last_modified:
            last_modified = chunk.metadata.get("last_crawl")  # Fallback to crawl date
            tipo_data = 'Data Crawl'
        if not last_modified:
            last_modified = "Data descoñecida"
            tipo_data = 'No hai data'
        else:
            # Format date if it's in ISO format
            try:
                if 'T' in str(last_modified):  # ISO format
                    dt = datetime.fromisoformat(last_modified.replace('Z', '+00:00'))
                    last_modified = dt.strftime("%d/%m/%Y %H:%M")
                # If it's already formatted (e.g., "Sun, 14 Dec 2025 10:44:45 GMT"), keep as is
            except:
                pass  # Keep original format if parsing fails
        
        # Build prefix
        prefix = f"{filename}|{file_type.upper()}|{source_url}|{last_modified}|{tipo_data}\n"
 
        chunk.page_content = prefix + chunk.page_content
        return chunk
    
    def _generate_llm_prefix(self, chunk) -> Optional[str]:
        """Generate LLM-based prefix for a chunk."""
        if self.llm is None:
            return None
        
        from pathlib import Path
        source_file = Path(chunk.metadata.get("source_file", 'descoñecido')).name
        prompt = (
            f"Escribe unha breve frase introdutoria (máx. 1-2 oracións) que resuma o seguinte fragmento "
            f"do documento '{source_file}' e sirva como contexto:\n\n"
            f"---\n{chunk.page_content[:400]}\n---\n"
            f"Responde soamente coa frase introdutoria en galego, sen repetir o texto do fragmento."
        )
        
        try:
            if hasattr(self.llm, "invoke"):
                prefix_text = self.llm.invoke(prompt)
            elif hasattr(self.llm, "generate"):
                prefix_text = self.llm.generate(prompt)
            else:
                raise ValueError("O obxecto LLM non ten un método invoke() nin generate().")
            
            # Extract text from response
            if isinstance(prefix_text, dict) and "content" in prefix_text:
                prefix_text = prefix_text["content"]
            elif not isinstance(prefix_text, str):
                prefix_text = str(prefix_text)
            
            return prefix_text.strip()
            
        except Exception as e:
            self.log(f"[red]Erro xerando prefixo LLM: {e}[/red]")
            return None
    
    def split_documents(self, documents: List) -> List:
        """Split documents into chunks in parallel, with optional prefixing.
        
        Args:
            documents: List of documents to split
            
        Returns:
            List of split Document chunks
        """
        if not documents:
            self.log("[yellow]Non hai documentos para dividir[/yellow]")
            return []
        
        # Split documents in parallel batches
        num_docs = len(documents)
        num_batches = min(self.max_workers, num_docs)
        
        if self.verbose:
            self.log(f"[cyan]Dividindo {num_docs} documentos en {num_batches} lotes paralelos...[/cyan]")
        
        # Split documents into batches
        doc_batches = np.array_split(documents, num_batches)
        
        all_chunks = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [
                executor.submit(self._split_document_batch, batch.tolist())
                for batch in doc_batches if len(batch) > 0
            ]
            
            for future in as_completed(futures):
                all_chunks.extend(future.result())
        
        if self.verbose:
            self.log(f"[green]✓ Creados {len(all_chunks)} fragmentos[/green]")
        
        # Apply prefixes if needed
        if self.prefix_mode == "source":
            if self.verbose:
                self.log(f"[cyan]Aplicando prefixos de orixe en paralelo...[/cyan]")
            
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                all_chunks = list(executor.map(self._add_source_prefix, all_chunks))
            
            if self.verbose:
                self.log(f"[green]✓ Prefixos de orixe aplicados[/green]")
        
        elif self.prefix_mode == "llm":
            if self.llm is None:
                raise ValueError("Debe proporcionar un LLM se se usa prefix_mode='llm'")
            
            if self.verbose:
                self.log(f"[cyan]Xerando prefixos LLM en lotes (esto pode tardar)...[/cyan]")
            
            # Process in smaller batches to avoid overwhelming the LLM
            llm_batch_size = 10
            for i in range(0, len(all_chunks), llm_batch_size):
                batch = all_chunks[i:i + llm_batch_size]
                
                with ThreadPoolExecutor(max_workers=min(len(batch), self.max_workers)) as executor:
                    futures = {executor.submit(self._generate_llm_prefix, chunk): j 
                              for j, chunk in enumerate(batch)}
                    
                    for future in as_completed(futures):
                        j = futures[future]
                        prefix = future.result()
                        if prefix:
                            batch[j].page_content = prefix + "\n\n" + batch[j].page_content
                            batch[j].metadata["llm_prefix"] = prefix
                
                if self.verbose and i > 0:
                    progress = min(i + llm_batch_size, len(all_chunks))
                    self.log(f"[dim]Progreso LLM: {progress}/{len(all_chunks)}[/dim]")
            
            if self.verbose:
                self.log(f"[green]✓ Prefixos LLM xerados[/green]")
        
        return all_chunks
    
    def log(self, message: str, level: str = "info") -> None:
        """Log a message with appropriate styling based on level."""
        self.console.print(message)

