from typing import List, Dict, Tuple, Optional, Union, Any
from pathlib import Path
import os
import json
import hashlib
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import numpy as np

from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import (
    PyPDFLoader,
    BSHTMLLoader,
    TextLoader,
    UnstructuredWordDocumentLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter

from LocalEmbeddings import LocalEmbeddings


class DocumentProcessor:
    """Document processor with unified crawler metadata and parallel processing."""

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
        map_json: str = "crawl/map.json",
        metadata_json: str = "crawl/metadata.json",
        max_workers: int = 7,
        batch_size: int = 128,
        ocr_texts_folder: Optional[str] = "crawl/crawled_data/ocr_texts"
    ):

        self.docs_folder = Path(docs_folder)
        self.map_json = Path(map_json)
        self.metadata_file = Path(metadata_json)

        self.ocr_texts_folder = Path(ocr_texts_folder) if ocr_texts_folder else None
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

        # embeddings + splitter
        self.embeddings = LocalEmbeddings(
            model_name=embedding_model_name,
            cache_dir=str(self.cache_dir / "embeddings")
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )

        self.verbose = verbose
        self.documents = []
        self.vectorstore = None

        # unified metadata
        self.unified_metadata = self._load_unified_metadata()

        # parallel processing
        self.max_workers = max_workers
        self.batch_size = batch_size
        self.doc_lock = Lock()

        # logging
        try:
            self.console = __builtins__.rich_console
        except Exception:
            from rich.console import Console
            self.console = Console()

        # prefix
        self.prefix_mode = prefix_mode
        self.llm = llm

        if prefix_mode not in {"none", "source", "llm"}:
            raise ValueError("prefix_mode must be one of: none, source, llm")

    # -------------------------------------------------------------------
    # METADATA HELPERS
    # -------------------------------------------------------------------

    def _load_unified_metadata(self) -> Dict[str, Dict]:
        """Load unified crawler metadata."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                self.log(f"[red]Could not load unified metadata: {e}[/red]")
                return {}
        return {}

    def _save_unified_metadata(self):
        """Save unified metadata back to metadata.json."""
        try:
            with open(self.metadata_file, "w", encoding="utf-8") as f:
                json.dump(self.unified_metadata, f, ensure_ascii=False, indent=2)
        except Exception as e:
            self.log(f"[red]Could not save unified metadata: {e}[/red]")

    def _get_file_hash(self, file_path: Path) -> str:
        sha = hashlib.sha256()
        try:
            with open(file_path, "rb") as f:
                for blk in iter(lambda: f.read(4096), b""):
                    sha.update(blk)
            return sha.hexdigest()
        except Exception:
            return "error"

    # -------------------------------------------------------------------
    # MAP.JSON HELPERS
    # -------------------------------------------------------------------
    def _get_url_for_file(self, filename: str) -> Optional[str]:
        """Find URL for filename using map.json."""
        try:
            with open(self.map_json, "r", encoding="utf-8") as f:
                file_to_url = json.load(f)
            return file_to_url.get(filename)
        except Exception:
            return None

    # -------------------------------------------------------------------
    # CHANGE DETECTION
    # -------------------------------------------------------------------
    def _file_has_changed(self, file_path: Path) -> bool:
        """Detect file changes using unified metadata."""
        filename = file_path.name
        url = self._get_url_for_file(filename)
        if not url or url not in self.unified_metadata:
            return True

        info = self.unified_metadata[url].get("file_info", {})
        if not info:
            return True

        current_hash = self._get_file_hash(file_path)
        current_mtime = file_path.stat().st_mtime
        current_size = file_path.stat().st_size

        if current_hash != info.get("file_hash"):
            return True
        if current_mtime != info.get("file_mtime"):
            return True
        if current_size != info.get("file_size"):
            return True

        return False

    def _update_file_metadata(self, file_path: Path):
        filename = file_path.name
        url = self._get_url_for_file(filename)
        if not url:
            return

        if url not in self.unified_metadata:
            self.unified_metadata[url] = {}

        self.unified_metadata[url]["file_info"] = {
            "filename": filename,
            "file_hash": self._get_file_hash(file_path),
            "file_mtime": file_path.stat().st_mtime,
            "file_size": file_path.stat().st_size,
            "file_type": file_path.suffix.lower()[1:],
            "last_processed": datetime.now().isoformat(),
        }

    def _get_deleted_files(self, current_files: List[Path]) -> List[str]:
        current = {f.name for f in current_files}
        deleted = []

        for url, meta in self.unified_metadata.items():
            fname = meta.get("file_info", {}).get("filename")
            if fname and fname not in current:
                deleted.append(fname)

        return deleted

    def _get_all_files(self) -> List[Path]:
        files = []
        files += list(self.docs_folder.glob("*.pdf"))
        files += list(self.docs_folder.glob("*.html"))
        files += list(self.docs_folder.glob("*.htm"))
        files += list(self.docs_folder.glob("*.docx"))
        files += list(self.docs_folder.glob("*.doc"))
        files += list(self.docs_folder.glob("*.txt"))

        if self.ocr_texts_folder and self.ocr_texts_folder.exists():
            files += list(self.ocr_texts_folder.glob("*.txt"))

        return files

    # -------------------------------------------------------------------
    # LOADING FILES (PARALLEL)
    # -------------------------------------------------------------------
    def _load_single_file(self, file_path: Path):
        filename = file_path.name
        url = self._get_url_for_file(filename)
        is_new = not url or url not in self.unified_metadata

        try:
            ext = file_path.suffix.lower()
            if ext == ".pdf":
                loader = PyPDFLoader(str(file_path))
            elif ext in [".docx", ".doc"]:
                loader = UnstructuredWordDocumentLoader(str(file_path))
            elif ext == ".txt":
                loader = TextLoader(str(file_path), encoding="utf-8")
            else:
                loader = BSHTMLLoader(str(file_path), open_encoding="utf-8")

            docs = loader.load()
            file_hash = self._get_file_hash(file_path)

            for doc in docs:
                doc.metadata["source_file"] = str(file_path)
                doc.metadata["file_hash"] = file_hash
                doc.metadata["file_type"] = ext[1:]
                doc.metadata["source_url"] = url
                doc.metadata["is_ocr"] = (
                    self.ocr_texts_folder and str(self.ocr_texts_folder) in str(file_path)
                )

            # update metadata
            self._update_file_metadata(file_path)

            return docs, filename, is_new, None

        except Exception as e:
            return None, filename, is_new, str(e)

    def load_documents(self, force_reload=False):
        all_files = self._get_all_files()

        new_docs, updated_docs, unchanged_docs = [], [], []
        files_to_process = []

        for f in all_files:
            changed = force_reload or self._file_has_changed(f)
            if changed:
                files_to_process.append(f)
            else:
                unchanged_docs.append(f.name)

        if not files_to_process:
            self.log("[green]✓ No changes detected[/green]")
            return {"new": new_docs, "updated": updated_docs, "unchanged": unchanged_docs}

        self.log(
            f"[cyan]Processing {len(files_to_process)} files with {self.max_workers} workers...[/cyan]"
        )

        with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            futures = {ex.submit(self._load_single_file, f): f for f in files_to_process}

            for future in as_completed(futures):
                docs, fname, is_new, err = future.result()

                if err:
                    self.log(f"[red]Error processing {fname}: {err}[/red]")
                    continue

                if not docs:
                    continue

                with self.doc_lock:
                    self.documents.extend(docs)

                if is_new:
                    new_docs.append(fname)
                else:
                    updated_docs.append(fname)

        self._save_unified_metadata()

        return {"new": new_docs, "updated": updated_docs, "unchanged": unchanged_docs}

    # -------------------------------------------------------------------
    # SPLITTING + PREFIXES (PARALLEL)
    # -------------------------------------------------------------------
    def _split_document_batch(self, docs):
        return self.text_splitter.split_documents(docs)

    def _add_source_prefix(self, chunk):
        filename = Path(chunk.metadata.get("source_file", "")).name
        url = chunk.metadata.get("source_url", "URL descoñecida")
        is_ocr = chunk.metadata.get("is_ocr", False)
        ftype = chunk.metadata.get("file_type", "")

        if is_ocr:
            pref = f"Este fragmento é texto OCR do arquivo {filename} con url {url}:\n"
        else:
            pref = f"Este fragmento é do documento {filename} ({ftype.upper()}) con url {url}:\n"

        chunk.page_content = pref + chunk.page_content
        return chunk

    def split_documents(self):
        if not self.documents:
            return []

        batches = np.array_split(self.documents, min(len(self.documents), self.max_workers))

        chunks = []
        with ThreadPoolExecutor(self.max_workers) as ex:
            futs = [ex.submit(self._split_document_batch, b.tolist()) for b in batches if len(b)]
            for f in as_completed(futs):
                chunks.extend(f.result())

        if self.prefix_mode == "source":
            with ThreadPoolExecutor(self.max_workers) as ex:
                chunks = list(ex.map(self._add_source_prefix, chunks))

        elif self.prefix_mode == "llm":
            if not self.llm:
                raise ValueError("LLM required for prefix_mode='llm'")

            for i in range(0, len(chunks), 10):
                batch = chunks[i : i + 10]
                with ThreadPoolExecutor(self.max_workers) as ex:
                    results = list(ex.map(self._generate_llm_prefix, batch))
                    for chunk, pref in zip(batch, results):
                        if pref:
                            chunk.page_content = pref + "\n\n" + chunk.page_content

        return chunks

    def _generate_llm_prefix(self, chunk):
        prompt = (
            f"Escribe unha frase introdutoria de máximo dúas oracións sobre o fragmento "
            f"do documento {chunk.metadata.get('source_file')}:\n\n"
            f"{chunk.page_content[:400]}\n\n"
            "Só responde coa frase, en galego."
        )

        try:
            resp = self.llm.invoke(prompt)
            if isinstance(resp, dict):
                return resp.get("content")
            return str(resp)
        except Exception:
            return None

    # -------------------------------------------------------------------
    # VECTORSTORE CREATION
    # -------------------------------------------------------------------
    def create_vectorstore(self, chunks: List):
        if not chunks:
            return

        for i in range(0, len(chunks), self.batch_size):
            batch = chunks[i : i + self.batch_size]
            if i == 0:
                self.vectorstore = FAISS.from_documents(batch, self.embeddings)
            else:
                self.vectorstore.merge_from(FAISS.from_documents(batch, self.embeddings))

    # -------------------------------------------------------------------
    # PROCESS PIPELINE
    # -------------------------------------------------------------------
    def process(self, force_reload=False, incremental=True):
        if incremental and not force_reload:
            change = self._check_for_changes()
            if not (change["new"] or change["updated"] or change["deleted"]):
                self.log("[green]✓ Non se detectaron cambios[/green]")
                return False

            self.documents = []
            self.load_documents(force_reload=True)

        else:
            self.documents = []
            self.load_documents(force_reload=True)

        chunks = self.split_documents()
        if chunks:
            self.create_vectorstore(chunks)
            return True
        return False

    # -------------------------------------------------------------------
    # UTILS
    # -------------------------------------------------------------------
    def log(self, msg, level="info"):
        self.console.print(msg)

