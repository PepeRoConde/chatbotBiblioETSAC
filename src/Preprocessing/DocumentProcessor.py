from typing import List, Dict, Tuple, Optional, Union, Any
from pathlib import Path
import os
import pickle
import json
import hashlib
import html
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import numpy as np


from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, BSHTMLLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from LocalEmbeddings import LocalEmbeddings
from Preprocessing.CleanHTMLLoader import CleanHTMLLoader
from Preprocessing.BM25 import BM25


class DocumentProcessor:
    """Class for processing multiple document types (PDF, HTML, TXT) with parallel processing."""
    
    def __init__(
        self, 
        docs_folder: str,
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        chunk_size: int = 300,
        chunk_overlap: int = 15,
        verbose: bool = False,
        cache_dir: str = ".doc_cache",
        prefix_mode: str = "source",
        llm: Optional[Any] = None,
        map_json: str = 'crawl/map.json',
        max_workers: int = 7,
        batch_size: int = 128,
        crawler_metadata_path: str = "crawl/metadata.json",
        text_folder: str = "crawl/text"
    ):
        """Initialize document processor with parallel processing support.
        
        Args:
            docs_folder: Folder containing documents (legacy, for compatibility)
            embedding_model_name: Name of embedding model
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
            verbose: Whether to show detailed information
            cache_dir: Directory for caching file metadata
            prefix_mode: How to prefix chunks ("none", "source", "llm")
            llm: LLM instance for prefix generation (required if prefix_mode="llm")
            map_json: Path to JSON file mapping filenames to URLs
            max_workers: Maximum number of parallel workers (default: 7)
            batch_size: Batch size for embedding generation (default: 128)
            crawler_metadata_path: Path to crawler's metadata.json
            text_folder: Folder containing plain text files from crawler
        """
        self.docs_folder = Path(docs_folder)
        self.text_folder = Path(text_folder)
        self.bm25_index = None
        self.bm25_documents = []
        self.crawler_metadata_path = Path(crawler_metadata_path)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        self.embeddings = LocalEmbeddings(
            model_name=embedding_model_name,
            cache_dir=str(self.cache_dir / "embeddings")
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        self.documents = []
        self.vectorstore = None
        self.verbose = verbose
        
        # File tracking - now uses crawler's metadata
        self.crawler_metadata = self._load_crawler_metadata()
        self.metadata_file = self.cache_dir / "processor_cache.json"
        self.file_metadata = self._load_file_metadata()
        
        # Parallel processing settings
        self.max_workers = max_workers
        self.batch_size = batch_size
        self.doc_lock = Lock()  # Thread-safe document list access
        
        # Use the global rich console if available
        try:
            self.console = __builtins__.rich_console
        except (AttributeError, NameError):
            from rich.console import Console
            self.console = Console()

        self.prefix_mode = prefix_mode
        self.map_json = map_json
        self.llm = llm
        
        valid_modes = "none, source, llm"
        if self.prefix_mode not in {"none", "source", "llm"}:
            raise ValueError(f"prefix_mode must be one of {valid_modes}")

        if self.verbose:
            mode_desc = {
                "none": "sen prefixo",
                "source": "co nome do documento",
                "llm": "xerado por LLM"
            }[self.prefix_mode]
            self.log(f"[cyan]Inicializado con {max_workers} workers paralelos[/cyan]")
            self.log(f"Dividindo documentos en fragmentos ({mode_desc})...")
    
    def _load_crawler_metadata(self) -> Dict[str, Any]:
        """Load metadata from crawler (shared metadata.json)."""
        if self.crawler_metadata_path.exists():
            try:
                with open(self.crawler_metadata_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                self.log(f"Could not load crawler metadata: {e}", "warning")
                return {}
        return {}
    
    def _save_crawler_metadata(self) -> None:
        """Save updated crawler metadata."""
        try:
            with open(self.crawler_metadata_path, 'w', encoding='utf-8') as f:
                json.dump(self.crawler_metadata, f, ensure_ascii=False, indent=2)
        except Exception as e:
            self.log(f"Could not save crawler metadata: {e}", "error")
    
    def _load_file_metadata(self) -> Dict[str, Dict]:
        """Load processor's internal cache."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                self.log(f"Could not load processor cache: {e}", "warning")
                return {}
        return {}
    
    def _save_file_metadata(self) -> None:
        """Save file metadata to disk."""
        try:
            with open(self.metadata_file, 'wb') as f:
                pickle.dump(self.file_metadata, f)
        except Exception as e:
            self.log(f"Could not save metadata: {e}", "error")
    
    def _get_file_hash(self, file_path: Path) -> str:
        """Calculate hash of a file's content."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    
    def _get_file_info(self, file_path: Path) -> Dict:
        """Get file information (hash, mod time, size)."""
        stat = file_path.stat()
        return {
            'hash': self._get_file_hash(file_path),
            'mtime': stat.st_mtime,
            'size': stat.st_size,
            'last_processed': datetime.now().isoformat()
        }
    
    def _file_has_changed(self, file_path: Path) -> bool:
        """Check if a file has changed since last processing."""
        file_key = str(file_path)
        
        if file_key not in self.file_metadata:
            return True
        
        current_mtime = file_path.stat().st_mtime
        stored_mtime = self.file_metadata[file_key].get('mtime', 0)
        
        if current_mtime > stored_mtime:
            current_hash = self._get_file_hash(file_path)
            stored_hash = self.file_metadata[file_key].get('hash', '')
            return current_hash != stored_hash
        
        return False
    
    def _get_deleted_files(self, current_files: List[Path]) -> List[str]:
        """Find files that were deleted since last run."""
        current_file_keys = {str(f) for f in current_files}
        stored_file_keys = set(self.file_metadata.keys())
        return list(stored_file_keys - current_file_keys)
    
    def log(self, message: str, level: str = "info") -> None:
        """Log a message with appropriate styling based on level."""
        self.console.print(message)
    
    def _check_for_changes(self) -> Dict[str, List]:
        """Check which files have changed without loading them."""
        all_files = self._get_all_files()
        
        new_files = []
        updated_files = []
        unchanged_files = []
        
        for file_path in all_files:
            file_key = str(file_path)
            
            if file_key not in self.file_metadata:
                new_files.append(file_key)
            elif self._file_has_changed(file_path):
                updated_files.append(file_key)
            else:
                unchanged_files.append(file_key)
        
        deleted_files = self._get_deleted_files(all_files)
        
        return {
            'new': new_files,
            'updated': updated_files,
            'unchanged': unchanged_files,
            'deleted': deleted_files
        }
    def _get_documents_needing_embeddings(self) -> List[Tuple[str, Dict]]:
        """Get documents that need embeddings based on crawler metadata.
        
        Returns:
            List of (url, metadata) tuples for documents needing processing
        """
        # Reload crawler metadata to get latest
        self.crawler_metadata = self._load_crawler_metadata()
        
        docs_needing_processing = []
        for url, meta in self.crawler_metadata.items():
            if isinstance(meta, dict) and meta.get('needs_embeddings', False):
                text_path = meta.get('text_path')
                if text_path:
                    # Convert relative path to absolute
                    full_path = self.crawler_metadata_path.parent / text_path
                    if full_path.exists():
                        docs_needing_processing.append((url, meta))
                    else:
                        self.log(f"[yellow]Text file not found: {text_path}[/yellow]", "warning")
        
        return docs_needing_processing
    
    # ========== PARALLEL LOADING METHODS ==========
    
    def _load_single_text_file(self, url: str, meta: Dict) -> Tuple[Optional[List], str, Optional[str]]:
        """Load a single plain text file and return documents, url, and error.
        
        This method is designed to be called by parallel workers.
        
        Args:
            url: Source URL for the document
            meta: Metadata dict from crawler
            
        Returns:
            (documents, url, error_message)
        """
        try:
            text_path_rel = meta.get('text_path')
            if not text_path_rel:
                return None, url, "No text_path in metadata"
            
            # Convert relative to absolute path
            text_path = self.crawler_metadata_path.parent / text_path_rel
            
            if not text_path.exists():
                return None, url, f"Text file not found: {text_path}"
            
            # Load plain text
            with open(text_path, 'r', encoding='utf-8') as f:
                text_content = f.read()
            
            # Create Document object
            from langchain_core.documents import Document
            doc = Document(
                page_content=text_content,
                metadata={
                    'source_url': url,
                    'text_path': str(text_path),
                    'text_hash': meta.get('text_hash', ''),
                    'original_format': meta.get('original_format', 'unknown'),
                    'last_crawl': meta.get('last_crawl', '')
                }
            )
            
            return [doc], url, None
            
        except Exception as e:
            return None, url, str(e)
    

    def _preserve_list_linebreaks(self, text: str) -> str:
        """
        Conserva saltos de línea solo para líneas de listas,
        y une todo el resto del texto en párrafos limpios.
        """
        import re
        lines = text.splitlines()
        processed_lines = []
        list_pattern = re.compile(r'^\s*([-*•]|\d+\.)\s+')
        
        buffer = []
        
        for line in lines:
            if list_pattern.match(line):
                # Vaciar buffer previo como párrafo unido
                if buffer:
                    processed_lines.append(" ".join(buffer).strip())
                    buffer = []
                processed_lines.append(line.strip())  # Línea de lista
            elif line.strip() == "":
                # Línea vacía indica final de párrafo
                if buffer:
                    processed_lines.append(" ".join(buffer).strip())
                    buffer = []
            else:
                buffer.append(line.strip())
        
        # Vaciar buffer final
        if buffer:
            processed_lines.append(" ".join(buffer).strip())
        
        return "\n".join(processed_lines)

   

    def load_documents(self, force_reload: bool = False) -> Dict[str, List]:
        """Load plain text files that need embeddings based on crawler metadata.
        
        Args:
            force_reload: If True, process all files (ignores needs_embeddings flag)
            
        Returns:
            Dictionary with 'processed' and 'skipped' document lists
        """
        # Get documents needing processing
        if force_reload:
            # Process all documents in text folder
            docs_to_process = []
            for url, meta in self.crawler_metadata.items():
                if isinstance(meta, dict) and 'text_path' in meta:
                    docs_to_process.append((url, meta))
            self.log(f"[yellow]Modo force_reload: procesando {len(docs_to_process)} documentos[/yellow]")
        else:
            docs_to_process = self._get_documents_needing_embeddings()
            self.log(f"[cyan]Documentos que necesitan embeddings: {len(docs_to_process)}[/cyan]")
        
        if not docs_to_process:
            self.log("[green]✓ Todos los documentos tienen embeddings actualizados[/green]")
            return {'processed': [], 'skipped': []}
        
        processed_docs = []
        skipped_docs = []
        
        # Process files in parallel
        self.log(f"[cyan]Cargando {len(docs_to_process)} textos planos en paralelo con {self.max_workers} workers...[/cyan]")
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_url = {
                executor.submit(self._load_single_text_file, url, meta): url
                for url, meta in docs_to_process
            }
            
            # Process completed tasks as they finish
            for future in as_completed(future_to_url):
                url = future_to_url[future]
                docs, _, error = future.result()
                
                if error:
                    self.log(f"[red]Error cargando {url}: {error}[/red]", "error")
                    skipped_docs.append(url)
                    continue
                
                # Thread-safe update of shared state
                with self.doc_lock:
                    self.documents.extend(docs)
                
                processed_docs.append(url)
                
                if self.verbose:
                    text_path = self.crawler_metadata[url].get('text_path', 'unknown')
                    self.log(f"[cyan]✓ Cargado: {Path(text_path).name}[/cyan]")
        
        # Summary
        if processed_docs:
            self.log(
                f"[green]Cargados {len(processed_docs)} textos planos, "
                f"{len(skipped_docs)} con errores[/green]",
                "success"
            )
        
        return {
            'processed': processed_docs,
            'skipped': skipped_docs
        }
    
    # ========== PARALLEL SPLITTING METHODS ==========
    
    def _split_document_batch(self, docs: List) -> List:
        """Split a batch of documents into chunks."""
        return self.text_splitter.split_documents(docs)
    
    def _add_source_prefix(self, chunk) -> Any:
        """Add source-based prefix to a chunk with date information."""
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
                from datetime import datetime
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
    
    def split_documents(self) -> List:
        """Split documents into chunks in parallel, with optional prefixing.
        
        Returns:
            List of split Document chunks
        """
        if not self.documents:
            self.log("[yellow]Non hai documentos para dividir[/yellow]")
            return []
        
        # Split documents in parallel batches
        num_docs = len(self.documents)
        num_batches = min(self.max_workers, num_docs)
        
        if self.verbose:
            self.log(f"[cyan]Dividindo {num_docs} documentos en {num_batches} lotes paralelos...[/cyan]")
        
        # Split documents into batches
        doc_batches = np.array_split(self.documents, num_batches)
        
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
    
    # ========== VECTORSTORE METHODS ==========
    
    def create_vectorstore(self, chunks: List) -> None:
        """Create a vector store from document chunks with batched processing.
        
        Args:
            chunks: List of document chunks to embed
        """
        if not chunks:
            self.log("[yellow]Non hai chunks para crear vectorstore[/yellow]")
            return
        
        if self.verbose:
            self.log(f"[cyan]Creando vectorstore con {len(chunks)} chunks (batch_size={self.batch_size})...[/cyan]")
        
        # Process in batches to manage memory and show progress
        for i in range(0, len(chunks), self.batch_size):
            batch = chunks[i:i + self.batch_size]
            
            if i == 0:
                # Create initial vectorstore
                self.vectorstore = FAISS.from_documents(batch, self.embeddings)
            else:
                # Merge additional batches
                batch_vectorstore = FAISS.from_documents(batch, self.embeddings)
                self.vectorstore.merge_from(batch_vectorstore)
            
            if self.verbose and i > 0:
                progress = min(i + self.batch_size, len(chunks))
                self.log(f"[dim]Progreso vectorstore: {progress}/{len(chunks)} chunks[/dim]")
        
        if self.verbose:
            self.log("[green]✓ Vectorstore creado![/green]", "success")
    
    
    def _build_bm25_index(self, path: str) -> None:
        """Build BM25 index from FAISS index.pkl file."""
        index_path = f"{path}/index.pkl"
        if not os.path.exists(index_path):
            if self.verbose:
                self.log(f"FAISS index not found at {index_path}, skipping BM25 build")
            return

        if self.verbose:
            self.log("Building BM25 index from FAISS index.pkl...")

        try:
            with open(index_path, 'rb') as f:
                data = pickle.load(f)

            docstore = data[0]
            index_to_id = data[1]

            # Extract texts and documents in FAISS order
            chunk_texts = []
            self.bm25_documents = []

            for idx in sorted(index_to_id.keys()):
                doc_id = index_to_id[idx]
                doc = docstore._dict[doc_id]
                chunk_texts.append(doc.page_content)
                self.bm25_documents.append(doc)

            # Build BM25 index
            self.bm25_index = BM25(b=0.75, k1=1.6)
            self.bm25_index.fit(chunk_texts)

            if self.verbose:
                self.log(f"BM25 index built on {len(chunk_texts)} chunks from FAISS")

        except Exception as e:
            self.log(f"Error building BM25 index: {e}", "error")
            self.bm25_index = None
            self.bm25_documents = []

    def process(self, force_reload: bool = False, incremental: bool = True) -> bool:
        """Process documents and rebuild vector store when embeddings change.
        
        Args:
            force_reload: Force reprocessing of all files
            incremental: Use incremental updates (only process docs with needs_embeddings=True)
            
        Returns:
            True if vectorstore was modified, False otherwise
        """
        vectorstore_modified = False
        
        # Load documents that need embeddings
        result = self.load_documents(force_reload=force_reload)
        processed_urls = result.get('processed', [])
        
        if not processed_urls:
            self.log("[green]✓ No hay cambios - vectorstore actualizado[/green]")
            return False
        
        self.log(f"[cyan]Procesando {len(processed_urls)} documentos con cambios...[/cyan]")
        
        # Generate chunks and embeddings
        if self.documents:
            chunks = self.split_documents()
            
            if chunks:
                # Generate embeddings (with cache)
                self.log("[cyan]Generando embeddings (usando caché para documentos sin cambios)...[/cyan]")
                
                # ALWAYS rebuild FAISS from scratch with all chunks
                self.log("[yellow]Reconstruyendo FAISS desde cero...[/yellow]")
                
                # Load ALL documents (not just changed ones) for complete FAISS rebuild
                _ = self.load_documents(force_reload=True)
                all_chunks = self.split_documents()
                
                if all_chunks:
                    self.create_vectorstore(all_chunks)
                    self.log("[green]✓ Vectorstore reconstruido exitosamente![/green]")
                    vectorstore_modified = True
                    
                    # Update crawler metadata: mark processed docs as embedded
                    for url in processed_urls:
                        if url in self.crawler_metadata and isinstance(self.crawler_metadata[url], dict):
                            self.crawler_metadata[url]['needs_embeddings'] = False
                            self.crawler_metadata[url]['last_embedded'] = datetime.now().isoformat()
                    
                    self._save_crawler_metadata()
                    self.log(f"[green]✓ Marcados {len(processed_urls)} documentos como embedded[/green]")
                else:
                    self.log("[yellow]⚠ Non se crearon chunks[/yellow]")
            else:
                self.log("[yellow]⚠ Non se crearon chunks[/yellow]")
        else:
            self.log("[yellow]⚠ Non hai documentos para procesar[/yellow]")
        
        # Clear documents from memory
        self.documents = []
        
        return vectorstore_modified

    def save_vectorstore(self, path: str) -> None:
        """Save the vector store to disk."""
        if self.vectorstore:
            self.vectorstore.save_local(path)
            self._build_bm25_index(path)
            self.save_bm25(path)
            if self.verbose:
                self.log(f"Vectorstore gardado en {path}", "success")
        else:
            self.log("Non hai vectorstore para gardar. Execute process() primeiro.", "warning")

    def save_bm25(self, path: str) -> None:
        """Save BM25 index to disk."""
        if self.bm25_index is not None:
            data = {
                'bm25_index': self.bm25_index,
                'documents': self.bm25_documents
            }
            with open(f"{path}/bm25_index.pkl", 'wb') as f:
                pickle.dump(data, f)
            if self.verbose:
                self.log(f"BM25 index saved to {path}/bm25_index.pkl")

    def load_vectorstore(self, path: str) -> None:
        """Load a vector store from disk."""
        if os.path.exists(path):
            self.vectorstore = FAISS.load_local(
                path, 
                self.embeddings, 
                allow_dangerous_deserialization=True
            )
            self.load_bm25(path)
            if self.verbose:
                self.log(f"Vectorstore cargado desde {path}", "success")
        else:
            if self.verbose:
                self.log(f"Non se atopou vectorstore en {path}", "warning")

    def load_bm25(self, path: str) -> None:
        """Load BM25 index from disk."""
        bm25_path = f"{path}/bm25_index.pkl"
        if os.path.exists(bm25_path):
            with open(bm25_path, 'rb') as f:
                data = pickle.load(f)
            self.bm25_index = data['bm25_index']
            self.bm25_documents = data['documents']
            if self.verbose:
                self.log(f"BM25 index loaded from {bm25_path}")
        else:
            if self.verbose:
                self.log(f"BM25 index not found at {bm25_path}")

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get statistics about cached data and parallel processing."""
        embedding_stats = self.embeddings.get_cache_stats()
        
        # Count documents by status
        total_docs = len(self.crawler_metadata)
        needs_embeddings = sum(1 for meta in self.crawler_metadata.values() 
                                if isinstance(meta, dict) and meta.get('needs_embeddings', False))
        embedded_docs = total_docs - needs_embeddings
        
        return {
            'total_documents': total_docs,
            'embedded_documents': embedded_docs,
            'needs_embeddings': needs_embeddings,
            'embedding_cache': embedding_stats,
            'cache_dir': str(self.cache_dir),
            'text_folder': str(self.text_folder),
            'max_workers': self.max_workers,
            'batch_size': self.batch_size
        }

    def _count_files_by_type(self) -> Dict[str, int]:
        """Count tracked files by type."""
        counts = {'pdf': 0, 'html': 0, 'txt': 0, 'other': 0}
        for file_path in self.file_metadata.keys():
            ext = Path(file_path).suffix.lower()
            if ext == '.pdf':
                counts['pdf'] += 1
            elif ext in ['.html', '.htm']:
                counts['html'] += 1
            elif ext == '.txt':
                counts['txt'] += 1
            else:
                counts['other'] += 1
        return counts

    def clear_cache(self) -> None:
        """Clear all caches."""
        self.file_metadata = {}
        self._save_file_metadata()
        self.embeddings.clear_cache()
        self.log("[green]Todos os cachés limpos[/green]", "success")
