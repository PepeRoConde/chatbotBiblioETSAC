# Código completo con soporte para force_recrawl mejorado
# -------------------------------------------------
# NOTA: Este archivo contiene la versión ampliada del crawler
# con el parámetro force_recrawl que permite volver a analizar
# URLs aunque aparezcan en visited_urls.

import os
import time
import json
import hashlib
import argparse
from urllib.parse import urljoin, urlparse
from pathlib import Path
from typing import Set, List, Dict, Any
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta


class CrawlerUDC:
    """
    Smart web crawler that downloads PDFs and HTML files,
    persists visited state, and refreshes only updated pages.
    Now includes: force_recrawl (forces complete recrawl ignoring visited state)
    """
    _metadata_lock = Lock()
    _visited_lock = Lock()
    _map_lock = Lock()

    def __init__(self, base_url: str,
                 output_dir: str = "crawled_data",
                 state_dir: str = "crawl",
                 keywords_file: str = "crawl/keywords.txt",
                 refresh_days: int = 30,
                 force_recrawl: bool = False):

        self.base_url = base_url.rstrip('/')
        self.domain = urlparse(base_url).netloc

        self.output_dir = Path(output_dir)
        self.state_dir = Path(state_dir)
        self.keywords_file = Path(keywords_file)
        self.visited_urls: Set[str] = set()
        self.downloaded_files: Set[str] = set()
        self.refresh_days = refresh_days
        self.force_recrawl = force_recrawl

        self.file_map: Dict[str, str] = {}

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.state_dir.mkdir(parents=True, exist_ok=True)

        self.meta_path = self.state_dir / "metadata.json"
        self.visited_path = self.state_dir / "visited_urls.txt"
        self.map_path = self.state_dir / "map.json"

        self.metadata: Dict[str, Dict[str, Any]] = self.load_metadata()

        self.headers = {
            'User-Agent': 'Mozilla/5.0 (compatible; UniversityCrawler/2.0; +research purposes)'
        }

        self.stats = {
            'pages_crawled': 0,
            'pdfs_downloaded': 0,
            'html_saved': 0,
            'errors': 0,
            'skipped_not_modified': 0,
            'force_recrawled': 0
        }

    # =================== Persistence ===================

    def load_metadata(self) -> Dict[str, Any]:
        with self._metadata_lock:
            if self.meta_path.exists():
                try:
                    with open(self.meta_path, "r", encoding="utf-8") as f:
                        return json.load(f)
                except Exception:
                    return {}
            return {}

    def save_metadata(self):
        with self._metadata_lock:
            existing = {}
            if self.meta_path.exists():
                try:
                    with open(self.meta_path, "r", encoding="utf-8") as f:
                        existing = json.load(f)
                except Exception:
                    pass

            existing.update(self.metadata)

            with open(self.meta_path, "w", encoding="utf-8") as f:
                json.dump(existing, f, ensure_ascii=False, indent=2)

    def save_file_map(self):
        with self._map_lock:
            existing_map = {}
            if self.map_path.exists():
                try:
                    with open(self.map_path, "r", encoding="utf-8") as f:
                        existing_map = json.load(f)
                except Exception:
                    pass

            existing_map.update(self.file_map)

            with open(self.map_path, "w", encoding="utf-8") as f:
                json.dump(existing_map, f, ensure_ascii=False, indent=2)

    def save_visited_urls(self):
        with self._visited_lock:
            existing_urls = set()
            if self.visited_path.exists():
                try:
                    with open(self.visited_path, "r", encoding="utf-8") as f:
                        existing_urls = set(line.strip() for line in f if line.strip())
                except Exception:
                    pass

            all_urls = existing_urls | self.visited_urls

            with open(self.visited_path, "w", encoding="utf-8") as f:
                for url in sorted(all_urls):
                    f.write(f"{url}\n")

    # =================== URL & file handling ===================

    def is_valid_url(self, url: str) -> bool:
        parsed = urlparse(url)
        return (
            parsed.netloc == self.domain and
            parsed.scheme in ['http', 'https'] and
            not any(ext in url.lower() for ext in [
                '.jpg', '.png', '.gif', '.css', '.js', '.ico', '.svg', '.woff'
            ])
        )

    def is_bureaucratic_pdf(self, url: str, text: str = "", keywords=None) -> bool:

        if keywords is None:
            keywords = [
                'regulation', 'reglamento', 'normativa',
                'procedure', 'procedimiento', 'proceso',
                'form', 'formulario', 'solicitud',
                'guideline', 'guia', 'manual',
                'policy', 'politica', 'norma',
                'enrollment', 'matricula', 'inscripcion',
                'administrative', 'administrativo',
                'academic', 'academico',
                'calendar', 'calendario',
                'syllabus', 'programa',
                'requirements', 'requisitos'
            ]
        elif isinstance(keywords, str):
            try:
                with open(keywords, 'r', encoding='utf-8') as f:
                    keywords = [line.strip() for line in f if line.strip()]
            except Exception:
                pass

        t = text.lower()
        u = url.lower()
        return any(k in u or k in t for k in keywords)

    def generate_filename(self, url: str, extension: str) -> str:
        parsed = urlparse(url)
        name = Path(parsed.path).stem or parsed.netloc.replace('.', '_')
        url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
        filename = f"{name[:50]}_{url_hash}.{extension}"
        return "".join(c for c in filename if c.isalnum() or c in '._-')

    # =================== Download logic ===================

    def should_refresh(self, url: str) -> bool:
        # Si force_recrawl está activado, siempre refrescar
        if self.force_recrawl:
            return True
            
        info = self.metadata.get(url)
        if not info:
            return True
        last_download = datetime.fromisoformat(info["last_download"])
        if datetime.now() - last_download > timedelta(days=self.refresh_days):
            return True
        return False

    def get_remote_metadata(self, url: str) -> Dict[str, str]:
        try:
            r = requests.head(url, headers=self.headers, timeout=15, allow_redirects=True)
            if r.status_code == 200:
                return {
                    "etag": r.headers.get("ETag"),
                    "last_modified": r.headers.get("Last-Modified")
                }
        except requests.RequestException:
            pass
        return {}

    def has_remote_changed(self, url: str) -> bool:
        # Si force_recrawl está activado, siempre considerar como cambiado
        if self.force_recrawl:
            return True
            
        old_meta = self.metadata.get(url, {})
        new_meta = self.get_remote_metadata(url)

        if not new_meta:
            return True

        if old_meta.get("etag") and new_meta.get("etag"):
            return old_meta["etag"] != new_meta["etag"]

        if old_meta.get("last_modified") and new_meta.get("last_modified"):
            return old_meta["last_modified"] != new_meta["last_modified"]

        return True

    def save_html(self, url: str, content: bytes):
        filename = self.generate_filename(url, 'html')
        filepath = self.output_dir / filename

        # Si force_recrawl está activado, sobrescribir siempre
        if filepath.exists() and self.force_recrawl:
            print(f"↻ Force overwriting HTML: {filename}")

        with open(filepath, 'wb') as f:
            f.write(content)

        self.file_map[filename] = url

        self.stats['html_saved'] += 1
        print(f"✓ Saved HTML: {filename}")

    def download_pdf(self, url: str):
        filename = self.generate_filename(url, 'pdf')
        filepath = self.output_dir / filename

        # Si force_recrawl está activado, descargar siempre
        if filepath.exists() and not self.force_recrawl and not self.should_refresh(url):
            print(f"⊘ Skipping existing PDF (fresh): {filename}")
            self.stats['skipped_not_modified'] += 1
            return

        try:
            r = requests.get(url, headers=self.headers, timeout=30)
            r.raise_for_status()
            
            # Verificar si el contenido realmente cambió (para evitar descargas innecesarias)
            if filepath.exists() and self.force_recrawl:
                with open(filepath, 'rb') as f:
                    existing_content = f.read()
                if existing_content == r.content:
                    print(f"⊘ PDF unchanged (force mode): {filename}")
                    return
            
            with open(filepath, 'wb') as f:
                f.write(r.content)

            remote_meta = self.get_remote_metadata(url)
            self.metadata[url] = {
                **remote_meta,
                "last_download": datetime.now().isoformat()
            }

            self.file_map[filename] = url

            self.downloaded_files.add(url)
            self.stats['pdfs_downloaded'] += 1
            if self.force_recrawl and filepath.exists():
                print(f"↻ Force re-downloaded PDF: {filename}")
            else:
                print(f"✓ Downloaded PDF: {filename}")
                
        except Exception as e:
            print(f"✗ Error downloading PDF {url}: {e}")
            self.stats['errors'] += 1

    # =================== Crawling ===================

    def crawl_page(self, url: str) -> List[str]:
        # ------ COMPORTAMIENTO MODIFICADO PARA FORCE_RECRAWL ------
        # Con force_recrawl: ignorar completamente el estado de visited_urls
        # Sin force_recrawl: comportamiento normal
        if url in self.visited_urls and not self.force_recrawl:
            return []

        # Solo añadir a visited_urls si no estamos en modo force_recrawl
        if not self.force_recrawl:
            self.visited_urls.add(url)
        else:
            self.stats['force_recrawled'] += 1
            print(f"↻ Force recrawling: {url}")
        # ------------------------------------------

        # Verificar si la página necesita refresh (si no estamos en force_recrawl)
        if not self.force_recrawl and not self.should_refresh(url) and not self.has_remote_changed(url):
            print(f"⊘ Skipping unchanged page: {url}")
            self.stats['skipped_not_modified'] += 1
            return []

        try:
            print(f"→ Crawling: {url}")
            response = requests.get(url, headers=self.headers, timeout=30)
            response.raise_for_status()
            
            # Guardar HTML (en force_recrawl siempre sobrescribe)
            self.save_html(url, response.content)
            
            soup = BeautifulSoup(response.content, 'lxml')
            self.stats['pages_crawled'] += 1

            # Procesar enlaces PDF
            for link in soup.find_all('a', href=True):
                href = urljoin(url, link['href'])
                if href.lower().endswith('.pdf') and self.is_bureaucratic_pdf(href, link.text):
                    self.download_pdf(href)

            # Extraer nuevos enlaces
            new_urls = [
                urljoin(url, link['href'])
                for link in soup.find_all('a', href=True)
                if self.is_valid_url(urljoin(url, link['href']))
            ]

            # Actualizar metadatos
            self.metadata[url] = {
                **self.get_remote_metadata(url),
                "last_download": datetime.now().isoformat()
            }

            time.sleep(1)
            return new_urls

        except Exception as e:
            print(f"✗ Error crawling {url}: {e}")
            self.stats['errors'] += 1
            return []

    def crawl(self, max_pages: int = 100, max_depth: int = 3):
        print(f"Starting crawl of {self.base_url}")
        print(f"Max pages: {max_pages}, Max depth: {max_depth}")
        print(f"Force recrawl: {self.force_recrawl}")
        print(f"Files will be saved to: {self.output_dir.absolute()}")
        print(f"State will be saved to: {self.state_dir.absolute()}\n")

        queue = deque([(self.base_url, 0)])

        while queue and (len(self.visited_urls) < max_pages or self.force_recrawl):
            url, depth = queue.popleft()
            if depth > max_depth:
                continue
                
            new_urls = self.crawl_page(url)
            for new_url in new_urls:
                if len(self.visited_urls) < max_pages or self.force_recrawl:
                    queue.append((new_url, depth + 1))

        self.save_metadata()
        self.save_file_map()
        self.save_visited_urls()

        self.print_summary()

    def print_summary(self):
        print("\n" + "="*50)
        print(f"CRAWL SUMMARY - {self.domain}")
        print("="*50)
        print(f"Pages crawled: {self.stats['pages_crawled']}")
        print(f"HTML files saved: {self.stats['html_saved']}")
        print(f"PDFs downloaded: {self.stats['pdfs_downloaded']}")
        print(f"Skipped (unchanged): {self.stats['skipped_not_modified']}")
        print(f"Errors encountered: {self.stats['errors']}")
        if self.force_recrawl:
            print(f"Pages force-recrawled: {self.stats['force_recrawled']}")
        print(f"Total URLs visited: {len(self.visited_urls)}")
        print(f"\nFiles saved to: {self.output_dir.absolute()}")
        print(f"State saved to: {self.state_dir.absolute()}")
        print("="*50)


# ================= CLI interface =================
def crawl_single_url(url: str, args) -> tuple:
    try:
        crawler = CrawlerUDC(
            base_url=url,
            output_dir=args.output_dir,
            state_dir=args.state_dir,
            keywords_file=args.keywords_file,
            refresh_days=args.refresh_days,
            force_recrawl=args.force
        )
        crawler.crawl(max_pages=args.max_pages, max_depth=args.max_depth)
        return (url, crawler.stats, None)
    except Exception as e:
        return (url, None, str(e))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Smart CrawlerUDC with metadata persistence.")
    parser.add_argument("--urls_file", "-f", type=str, default="crawl/urls.txt")
    parser.add_argument("--keywords_file", "-kf", type=str, default="crawl/keywords.txt")
    parser.add_argument("--max_pages", "-p", type=int, default=1000)
    parser.add_argument("--max_depth", "-d", type=int, default=2)
    parser.add_argument("--output_dir", "-o", type=str, default="crawl/crawled_data")
    parser.add_argument("--state_dir", "-s", type=str, default="crawl")
    parser.add_argument("--refresh_days", "-r", type=int, default=30)
    parser.add_argument("--workers", "-w", type=int, default=4)
    parser.add_argument("--force", action="store_true",
                        help="Force re-crawl even if URL already visited")

    args = parser.parse_args()

    urls_path = Path(args.urls_file)
    if not urls_path.exists():
        raise FileNotFoundError(f"URLs file not found: {urls_path}")

    with open(urls_path, "r", encoding="utf-8") as f:
        urls = [line.strip() for line in f if line.strip()]

    print(f"Found {len(urls)} URLs to crawl from {urls_path}")
    print(f"Using {args.workers} concurrent workers\n")

    start_time = time.time()
    all_stats = {}
    errors = {}

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_url = {
            executor.submit(crawl_single_url, url, args): url
            for url in urls
        }

        for future in as_completed(future_to_url):
            url = future_to_url[future]
            try:
                result_url, stats, error = future.result()
                if error:
                    errors[result_url] = error
                    print(f"\n✗ Error crawling {result_url}: {error}")
                else:
                    all_stats[result_url] = stats
                    print(f"\n✓ Completed: {result_url}")
            except Exception as e:
                errors[url] = str(e)
                print(f"\n✗ Exception for {url}: {e}")

    elapsed = time.time() - start_time

    print("\n" + "="*60)
    print("GLOBAL CRAWL SUMMARY")
    print("="*60)
    print(f"Total URLs processed: {len(urls)}")
    print(f"Successful: {len(all_stats)}")
    print(f"Failed: {len(errors)}")
    print(f"Time elapsed: {elapsed:.2f} seconds")
    print(f"Workers used: {args.workers}")

    if all_stats:
        total_pages = sum(s['pages_crawled'] for s in all_stats.values())
        total_pdfs = sum(s['pdfs_downloaded'] for s in all_stats.values())
        total_html = sum(s['html_saved'] for s in all_stats.values())
        total_errors = sum(s['errors'] for s in all_stats.values())
        total_force_recrawled = sum(s.get('force_recrawled', 0) for s in all_stats.values())

        print(f"\nAggregated stats:")
        print(f"  Total pages crawled: {total_pages}")
        print(f"  Total HTMLs saved: {total_html}")
        print(f"  Total PDFs downloaded: {total_pdfs}")
        print(f"  Total errors: {total_errors}")
        if args.force:
            print(f"  Total force-recrawled: {total_force_recrawled}")

    if errors:
        print(f"\nFailed URLs:")
        for url, error in errors.items():
            print(f"  - {url}: {error}")

    print("="*60)