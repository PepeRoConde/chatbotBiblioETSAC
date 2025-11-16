from typing import List, Dict, Tuple, Optional, Union, Any
from pathlib import Path
import os
import pickle
import json
import hashlib
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import numpy as np

from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, BSHTMLLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from LocalEmbeddings import LocalEmbeddings


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
        ocr_texts_folder: Optional[str] = "crawl/crawled_data/ocr_texts"
    ):
        """Initialize document processor with parallel processing support.
        
        Args:
            docs_folder: Folder containing documents
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
            ocr_texts_folder: Folder containing OCR text files (optional)
        """
        self.docs_folder = Path(docs_folder)
        self.ocr_texts_folder = Path(ocr_texts_folder) if ocr_texts_folder else None
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
        
        # File tracking
        self.metadata_file = self.cache_dir / "file_metadata.pkl"
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
    
    def _load_file_metadata(self) -> Dict[str, Dict]:
        """Load file metadata (hashes, modification times)."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                self.log(f"Could not load metadata: {e}", "warning")
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
    
    def _get_all_files(self) -> List[Path]:
        """Get all supported files from configured folders."""
        all_files = []
        
        # Files from main docs folder
        pdf_files = list(self.docs_folder.glob("*.pdf"))
        html_files = list(self.docs_folder.glob("*.html")) + list(self.docs_folder.glob("*.htm"))
        all_files.extend(pdf_files + html_files)
        
        # Files from OCR texts folder
        if self.ocr_texts_folder and self.ocr_texts_folder.exists():
            txt_files = list(self.ocr_texts_folder.glob("*.txt"))
            all_files.extend(txt_files)
        
        return all_files
    
    # ========== PARALLEL LOADING METHODS ==========
    
    def _load_single_file(self, file_path: Path) -> Tuple[Optional[List], str, Optional[Dict], bool, Optional[str]]:
        """Load a single file and return documents, key, metadata, is_new flag, and error.
        
        This method is designed to be called by parallel workers.
        """
        file_key = str(file_path)
        is_new = file_key not in self.file_metadata
        
        try:
            # Load based on file type
            if file_path.suffix.lower() == '.pdf':
                loader = PyPDFLoader(str(file_path))
            elif file_path.suffix.lower() == '.txt':
                loader = TextLoader(str(file_path), encoding='utf-8')
            else:  # HTML files
                loader = BSHTMLLoader(str(file_path), open_encoding="utf-8")
            
            docs = loader.load()
            file_hash = self._get_file_hash(file_path)
            
            # Tag documents with source file and type
            for doc in docs:
                doc.metadata['source_file'] = file_key
                doc.metadata['file_hash'] = file_hash
                doc.metadata['file_type'] = file_path.suffix.lower()[1:]  # Remove the dot
                
                # Add OCR flag if from OCR folder
                if self.ocr_texts_folder and str(self.ocr_texts_folder) in file_key:
                    doc.metadata['is_ocr'] = True
            
            # Get file info
            file_info = self._get_file_info(file_path)
            
            return docs, file_key, file_info, is_new, None
            
        except Exception as e:
            return None, file_key, None, is_new, str(e)
    
    def load_documents(self, force_reload: bool = False) -> Dict[str, List]:
        """Load all PDFs, HTML, and TXT files in parallel, only processing changed files.
        
        Args:
            force_reload: If True, reload all files regardless of changes
            
        Returns:
            Dictionary with 'new', 'updated', 'unchanged' document lists
        """
        all_files = self._get_all_files()
        
        # Count files by type for logging
        pdf_count = sum(1 for f in all_files if f.suffix.lower() == '.pdf')
        html_count = sum(1 for f in all_files if f.suffix.lower() in ['.html', '.htm'])
        txt_count = sum(1 for f in all_files if f.suffix.lower() == '.txt')
        
        if not force_reload:
            self.log(f"[cyan]Encontrados {pdf_count} PDFs, {html_count} HTMLs e {txt_count} TXTs[/cyan]")
        
        # Track file statuses
        new_docs = []
        updated_docs = []
        unchanged_docs = []
        
        # Check for deleted files
        deleted_files = self._get_deleted_files(all_files)
        if deleted_files:
            self.log(f"[yellow]Detectados {len(deleted_files)} arquivos eliminados[/yellow]")
            for deleted_file in deleted_files:
                del self.file_metadata[deleted_file]
        
        # Determine which files need processing
        files_to_process = []
        for file_path in all_files:
            file_key = str(file_path)
            is_changed = force_reload or self._file_has_changed(file_path)
            
            if not is_changed:
                unchanged_docs.append(file_key)
                if self.verbose:
                    self.log(f"[dim]Omitindo sen cambios: {file_path.name}[/dim]")
                continue
            
            files_to_process.append(file_path)
        
        if not files_to_process:
            self.log("[green]✓ Non se detectaron cambios[/green]")
            self._save_file_metadata()
            return {'new': new_docs, 'updated': updated_docs, 'unchanged': unchanged_docs}
        
        # Process files in parallel using ThreadPoolExecutor
        self.log(f"[cyan]Procesando {len(files_to_process)} arquivos en paralelo con {self.max_workers} workers...[/cyan]")
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_file = {
                executor.submit(self._load_single_file, file_path): file_path
                for file_path in files_to_process
            }
            
            # Process completed tasks as they finish
            for future in as_completed(future_to_file):
                file_path = future_to_file[future]
                docs, file_key, file_info, is_new, error = future.result()
                
                if error:
                    self.log(f"[red]Erro procesando {file_path.name}: {error}[/red]", "error")
                    continue
                
                # Thread-safe update of shared state
                with self.doc_lock:
                    self.documents.extend(docs)
                    self.file_metadata[file_key] = file_info
                
                if is_new:
                    new_docs.append(file_key)
                else:
                    updated_docs.append(file_key)
                
                if self.verbose:
                    status = "novo" if is_new else "actualizado"
                    self.log(f"[cyan]✓ Procesado {status}: {file_path.name}[/cyan]")
        
        # Save updated metadata
        self._save_file_metadata()
        
        # Summary
        if new_docs or updated_docs:
            self.log(
                f"[green]Procesados {len(new_docs)} novos, {len(updated_docs)} actualizados, "
                f"{len(unchanged_docs)} sen cambios[/green]",
                "success"
            )
        
        return {
            'new': new_docs,
            'updated': updated_docs,
            'unchanged': unchanged_docs
        }
    
    # ========== PARALLEL SPLITTING METHODS ==========
    
    def _split_document_batch(self, docs: List) -> List:
        """Split a batch of documents into chunks."""
        return self.text_splitter.split_documents(docs)
    
    def _add_source_prefix(self, chunk) -> Any:
        """Add source-based prefix to a chunk."""
        source_file = chunk.metadata.get("source_file", "descoñecido")
        filename = Path(source_file).name
        file_type = chunk.metadata.get("file_type", "descoñecido")
        is_ocr = chunk.metadata.get("is_ocr", False)
        
        # Load URL mapping if available
        mapping_path = Path(self.map_json)
        url = "URL descoñecida"
        
        if mapping_path.exists():
            try:
                with open(mapping_path, "r", encoding="utf-8") as f:
                    filename_to_url = json.load(f)
                url = filename_to_url.get(filename, "URL descoñecida")
            except Exception:
                pass
        
        # Build prefix based on file type
        if is_ocr:
            prefix = f"Este fragmento é de texto extraído por OCR do arquivo {filename} con url {url} :\n "
        else:
            prefix = f"Este fragmento é do documento {filename} ({file_type.upper()}) con url {url} :\n "
        
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
    
    def process(self, force_reload: bool = False, incremental: bool = True) -> bool:
        """Process all documents and create/update the vector store with parallel processing.
        
        Args:
            force_reload: Force reprocessing of all files
            incremental: Use incremental updates (only process changed files)
            
        Returns:
            True if vectorstore was modified, False otherwise
        """
        vectorstore_modified = False
        
        if incremental and not force_reload:
            # Check for changes first
            changes = self._check_for_changes()
            has_changes = changes['new'] or changes['updated'] or changes['deleted']
            
            if not has_changes:
                self.log("[green]✓ Non se detectaron cambios. Vectorstore actualizado![/green]")
                return False
            
            # Report changes
            change_summary = []
            if changes['new']:
                change_summary.append(f"{len(changes['new'])} novos")
            if changes['updated']:
                change_summary.append(f"{len(changes['updated'])} modificados")
            if changes['deleted']:
                change_summary.append(f"{len(changes['deleted'])} eliminados")
            
            self.log(f"[yellow]⚠ Cambios detectados: {', '.join(change_summary)}[/yellow]")
            self.log("[cyan]Reconstruíndo vectorstore con procesamento paralelo...[/cyan]")
            
            # Rebuild with all documents (but embeddings are cached!)
            self.documents = []
            self.load_documents(force_reload=True)
            
            if self.documents:
                chunks = self.split_documents()
                if chunks:
                    self.create_vectorstore(chunks)
                    self.log("[green]✓ Vectorstore reconstruído exitosamente![/green]")
                    vectorstore_modified = True
                else:
                    self.log("[yellow]⚠ Non se crearon chunks[/yellow]")
            else:
                self.log("[yellow]⚠ Non hai documentos para procesar[/yellow]")
            
            self.documents = []
            
        else:
            # Full rebuild
            self.log("[yellow]Realizando reconstrucción completa con procesamento paralelo...[/yellow]")
            self.documents = []
            self.load_documents(force_reload=True)
            
            if self.documents:
                chunks = self.split_documents()
                if chunks:
                    self.create_vectorstore(chunks)
                    vectorstore_modified = True
                else:
                    self.log("[yellow]⚠ Non se crearon chunks[/yellow]")
            else:
                self.log("[yellow]⚠ Non hai documentos para procesar[/yellow]")
            
            self.documents = []
        
        return vectorstore_modified
    
    def save_vectorstore(self, path: str) -> None:
        """Save the vector store to disk."""
        if self.vectorstore:
            self.vectorstore.save_local(path)
            if self.verbose:
                self.log(f"Vectorstore gardado en {path}", "success")
        else:
            self.log("Non hai vectorstore para gardar. Execute process() primeiro.", "warning")
    
    def load_vectorstore(self, path: str) -> None:
        """Load a vector store from disk."""
        if os.path.exists(path):
            self.vectorstore = FAISS.load_local(
                path, 
                self.embeddings, 
                allow_dangerous_deserialization=True
            )
            if self.verbose:
                self.log(f"Vectorstore cargado desde {path}", "success")
        else:
            if self.verbose:
                self.log(f"Non se atopou vectorstore en {path}", "warning")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get statistics about cached data and parallel processing."""
        embedding_stats = self.embeddings.get_cache_stats()
        
        return {
            'tracked_files': len(self.file_metadata),
            'files_by_type': self._count_files_by_type(),
            'embedding_cache': embedding_stats,
            'cache_dir': str(self.cache_dir),
            'max_workers': self.max_workers,
            'batch_size': self.batch_size,
            'ocr_texts_folder': str(self.ocr_texts_folder) if self.ocr_texts_folder else None
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
