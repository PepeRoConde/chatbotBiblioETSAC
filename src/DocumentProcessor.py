from typing import List, Dict, Tuple, Optional, Union, Any
from pathlib import Path
import os
import pickle
import json
import hashlib
from datetime import datetime
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, BSHTMLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from LocalEmbeddings import LocalEmbeddings

class DocumentProcessor:
    """Class for processing multiple document types (PDF, HTML) with incremental updates."""
    
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
        map_json: str = 'crawl/map.json' 
    ):
        """Initialize document processor.
        
        Args:
            docs_folder: Folder containing documents
            embedding_model_name: Name of embedding model
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
            verbose: Whether to show detailed information
            cache_dir: Directory for caching file metadata
        """
        self.docs_folder = Path(docs_folder)
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
        """Calculate hash of a file's content.
        
        Args:
            file_path: Path to the file
            
        Returns:
            SHA256 hash of the file
        """
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    
    def _get_file_info(self, file_path: Path) -> Dict:
        """Get file information (hash, mod time, size).
        
        Args:
            file_path: Path to the file
            
        Returns:
            Dictionary with file information
        """
        stat = file_path.stat()
        return {
            'hash': self._get_file_hash(file_path),
            'mtime': stat.st_mtime,
            'size': stat.st_size,
            'last_processed': datetime.now().isoformat()
        }
    
    def _file_has_changed(self, file_path: Path) -> bool:
        """Check if a file has changed since last processing.
        
        Args:
            file_path: Path to the file
            
        Returns:
            True if file is new or has changed
        """
        file_key = str(file_path)
        
        # New file
        if file_key not in self.file_metadata:
            return True
        
        # Check if file was modified
        current_mtime = file_path.stat().st_mtime
        stored_mtime = self.file_metadata[file_key].get('mtime', 0)
        
        # Quick check with modification time
        if current_mtime > stored_mtime:
            # Verify with hash to be sure
            current_hash = self._get_file_hash(file_path)
            stored_hash = self.file_metadata[file_key].get('hash', '')
            return current_hash != stored_hash
        
        return False
    
    def _get_deleted_files(self, current_files: List[Path]) -> List[str]:
        """Find files that were deleted since last run.
        
        Args:
            current_files: List of currently existing files
            
        Returns:
            List of deleted file paths
        """
        current_file_keys = {str(f) for f in current_files}
        stored_file_keys = set(self.file_metadata.keys())
        return list(stored_file_keys - current_file_keys)
        
    def log(self, message: str, level: str = "info") -> None:
        """Log a message with appropriate styling based on level."""
        self.console.print(message)
    
    def _check_for_changes(self) -> Dict[str, List]:
        """Check which files have changed without loading them.
        
        Returns:
            Dictionary with 'new', 'updated', 'unchanged', 'deleted' file lists
        """
        pdf_files = list(self.docs_folder.glob("*.pdf"))
        html_files = list(self.docs_folder.glob("*.html")) + list(self.docs_folder.glob("*.htm"))
        all_files = pdf_files + html_files
        
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
    
    def load_documents(self, force_reload: bool = False) -> Dict[str, List]:
        """Load all PDFs and HTML files, only processing changed files.
        
        Args:
            force_reload: If True, reload all files regardless of changes
            
        Returns:
            Dictionary with 'new', 'updated', 'unchanged' document lists
        """
        pdf_files = list(self.docs_folder.glob("*.pdf"))
        html_files = list(self.docs_folder.glob("*.html")) + list(self.docs_folder.glob("*.htm"))
        all_files = pdf_files + html_files
        self.log(os.getcwd())

        if not force_reload:
            self.log(f"[cyan]Encontrados {len(pdf_files)} PDFs e {len(html_files)} HTMLs[/cyan]")
        
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
        
        # Process each file
        for file_path in all_files:
            file_key = str(file_path)
            
            # Determine if file needs processing
            is_changed = force_reload or self._file_has_changed(file_path)
            is_new = file_key not in self.file_metadata
            
            if not is_changed:
                unchanged_docs.append(file_key)
                if self.verbose:
                    self.log(f"[dim]Omitindo sen cambios: {file_path.name}[/dim]")
                continue
            
            # Process the file
            try:
                if self.verbose:
                    status = "novo" if is_new else "actualizado"
                    self.log(f"[cyan]Procesando {status}: {file_path.name}[/cyan]")
                
                # Load based on file type
                if file_path.suffix.lower() == '.pdf':
                    loader = PyPDFLoader(str(file_path))
                else:
                    loader = BSHTMLLoader(str(file_path), open_encoding="utf-8")
                
                docs = loader.load()
                print(len(docs))
                # Tag documents with source file
                for doc in docs:
                    doc.metadata['source_file'] = file_key
                    doc.metadata['file_hash'] = self._get_file_hash(file_path)
                
                self.documents.extend(docs)
                
                # Update metadata
                self.file_metadata[file_key] = self._get_file_info(file_path)
                
                if is_new:
                    new_docs.append(file_key)
                else:
                    updated_docs.append(file_key)
                    
            except Exception as e:
                self.log(f"[red]Erro procesando {file_path}: {e}[/red]", "error")
        
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
    
    def split_documents(self) -> List:
        """Split documents into chunks, with optional prefixing behavior.
        
        Returns:
            List of split Document chunks
        """

        # Split using the text splitter
        chunks = self.text_splitter.split_documents(self.documents)

        if self.prefix_mode == "source":

            mapping_path = Path(self.map_json)
            if mapping_path.exists():
                try:
                    with open(mapping_path, "r", encoding="utf-8") as f:
                        filename_to_url = json.load(f)
                except Exception as e:
                    self.log(f"[yellow]Non se puido ler filename_to_url.json: {e}[/yellow]")
                    filename_to_url = {}
            else:
                filename_to_url = {}

            for chunk in chunks:
                source_file = chunk.metadata.get("source_file", "descoñecido")
                filename = Path(source_file).name
                url = filename_to_url.get(filename, "URL descoñecida")
                prefix = f"Este fragmento é do documento {filename} con url {url} :\n "
                chunk.page_content = prefix + chunk.page_content

        elif self.prefix_mode == "llm":
            if self.llm is None:
                raise ValueError("Debe proporcionar un LLM se se usa prefix_mode='llm'")

            for i, chunk in enumerate(chunks):
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

                    # if the LLM returns an object (e.g. from LangChain), extract text
                    if isinstance(prefix_text, dict) and "content" in prefix_text:
                        prefix_text = prefix_text["content"]
                    elif not isinstance(prefix_text, str):
                        prefix_text = str(prefix_text)

                    prefix = prefix_text.strip() + "\n\n"
                    chunk.page_content = prefix + chunk.page_content
                    chunk.metadata["llm_prefix"] = prefix_text.strip()

                except Exception as e:
                    self.log(f"[red]Erro xerando prefixo LLM para {source_file}: {e}[/red]")
                    continue

        if self.verbose:
            self.log(f"Creados {len(chunks)} fragmentos de documento", "success")

        return chunks
 
    def create_vectorstore(self, chunks: List) -> None:
        """Create a vector store from document chunks."""
        if self.verbose:
            self.log("Creando vectorstore...")
        self.vectorstore = FAISS.from_documents(chunks, self.embeddings)
        if self.verbose:
            self.log("Vectorstore creado!", "success")
    
    def process(self, force_reload: bool = False, incremental: bool = True) -> bool:
        """Process all documents and create/update the vector store.
        
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
                return False  # No cambios
            
            # Report changes
            change_summary = []
            if changes['new']:
                change_summary.append(f"{len(changes['new'])} novos")
            if changes['updated']:
                change_summary.append(f"{len(changes['updated'])} modificados")
            if changes['deleted']:
                change_summary.append(f"{len(changes['deleted'])} eliminados")
            
            self.log(f"[yellow]⚠ Cambios detectados: {', '.join(change_summary)}[/yellow]")
            print("el diablo en bicicleta")
            self.log("[cyan]Reconstruíndo vectorstore con caché de embeddings...[/cyan]")
            
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
                
            # Clear documents from memory
            self.documents = []
                
        else:
            # Full rebuild
            self.log("[yellow]Realizando reconstrucción completa...[/yellow]")
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
                self.log("[yellow]" + str(len(self.documents)))
            
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
        """Get statistics about cached data.
        
        Returns:
            Dictionary with cache statistics
        """
        embedding_stats = self.embeddings.get_cache_stats()
        
        return {
            'tracked_files': len(self.file_metadata),
            'files_by_type': self._count_files_by_type(),
            'embedding_cache': embedding_stats,
            'cache_dir': str(self.cache_dir)
        }
    
    def _count_files_by_type(self) -> Dict[str, int]:
        """Count tracked files by type."""
        counts = {'pdf': 0, 'html': 0, 'other': 0}
        for file_path in self.file_metadata.keys():
            ext = Path(file_path).suffix.lower()
            if ext == '.pdf':
                counts['pdf'] += 1
            elif ext in ['.html', '.htm']:
                counts['html'] += 1
            else:
                counts['other'] += 1
        return counts
    
    def clear_cache(self) -> None:
        """Clear all caches."""
        self.file_metadata = {}
        self._save_file_metadata()
        self.embeddings.clear_cache()
        self.log("[green]Todos os cachés limpos[/green]", "success")
