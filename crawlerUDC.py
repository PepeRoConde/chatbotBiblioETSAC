import os
import time
import json
import hashlib
import argparse
from urllib.parse import urljoin, urlparse
from pathlib import Path
from typing import Set, List, Dict, Any
from collections import deque

import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta


class CrawlerUDC:
    """
    Smart web crawler that downloads PDFs and HTML files,
    persists visited state, and refreshes only updated pages.
    """

    def __init__(self, base_url: str, output_dir: str = "crawled_data",
                 refresh_days: int = 30):
        self.base_url = base_url.rstrip('/')
        self.domain = urlparse(base_url).netloc
        self.output_dir = Path(output_dir)
        self.visited_urls: Set[str] = set()
        self.downloaded_files: Set[str] = set()
        self.refresh_days = refresh_days

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Paths for persistent state
        self.meta_path = self.output_dir / "metadata.json"
        self.visited_path = self.output_dir / "visited_urls.txt"

        self.metadata: Dict[str, Dict[str, Any]] = self.load_metadata()

        self.headers = {
            'User-Agent': 'Mozilla/5.0 (compatible; UniversityCrawler/2.0; +research purposes)'
        }

        self.stats = {
            'pages_crawled': 0,
            'pdfs_downloaded': 0,
            'html_saved': 0,
            'errors': 0,
            'skipped_not_modified': 0
        }

    # =================== Persistence ===================

    def load_metadata(self) -> Dict[str, Any]:
        """Load metadata file if exists."""
        if self.meta_path.exists():
            try:
                with open(self.meta_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def save_metadata(self):
        """Save metadata JSON file."""
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)

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

    def is_bureaucratic_pdf(self, url: str, text: str = "") -> bool:
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
        """Check if we should re-download a URL (content update or old)."""
        info = self.metadata.get(url)
        if not info:
            return True  # never seen before

        last_download = datetime.fromisoformat(info["last_download"])
        if datetime.now() - last_download > timedelta(days=self.refresh_days):
            return True  # expired by age

        return False

    def get_remote_metadata(self, url: str) -> Dict[str, str]:
        """Make a HEAD request to check headers."""
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
        """Compare stored vs current ETag/Last-Modified."""
        old_meta = self.metadata.get(url, {})
        new_meta = self.get_remote_metadata(url)

        if not new_meta:
            # server didn't send headers — conservative approach: re-download
            return True

        if old_meta.get("etag") and new_meta.get("etag"):
            return old_meta["etag"] != new_meta["etag"]

        if old_meta.get("last_modified") and new_meta.get("last_modified"):
            return old_meta["last_modified"] != new_meta["last_modified"]

        # No comparable info, assume changed
        return True

    def save_html(self, url: str, content: bytes):
        filename = self.generate_filename(url, 'html')
        filepath = self.output_dir / filename
        with open(filepath, 'wb') as f:
            f.write(content)
        self.stats['html_saved'] += 1
        print(f"✓ Saved HTML: {filename}")

    def download_pdf(self, url: str):
        filename = self.generate_filename(url, 'pdf')
        filepath = self.output_dir / filename

        if filepath.exists() and not self.should_refresh(url):
            print(f" Skipping existing PDF (fresh): {filename}")
            self.stats['skipped_not_modified'] += 1
            return

        try:
            r = requests.get(url, headers=self.headers, timeout=30)
            r.raise_for_status()
            with open(filepath, 'wb') as f:
                f.write(r.content)

            remote_meta = self.get_remote_metadata(url)
            self.metadata[url] = {
                **remote_meta,
                "last_download": datetime.now().isoformat()
            }

            self.downloaded_files.add(url)
            self.stats['pdfs_downloaded'] += 1
            print(f"✓ Downloaded PDF: {filename}")
        except Exception as e:
            print(f"✗ Error downloading PDF {url}: {e}")
            self.stats['errors'] += 1

    # =================== Crawling ===================

    def crawl_page(self, url: str) -> List[str]:
        if url in self.visited_urls:
            return []
        self.visited_urls.add(url)

        if not self.should_refresh(url) and not self.has_remote_changed(url):
            print(f"Skipping unchanged page: {url}")
            self.stats['skipped_not_modified'] += 1
            return []

        try:
            print(f"Crawling: {url}")
            response = requests.get(url, headers=self.headers, timeout=30)
            response.raise_for_status()
            self.save_html(url, response.content)
            soup = BeautifulSoup(response.content, 'html.parser')
            self.stats['pages_crawled'] += 1

            for link in soup.find_all('a', href=True):
                href = urljoin(url, link['href'])
                if href.lower().endswith('.pdf') and self.is_bureaucratic_pdf(href, link.text):
                    self.download_pdf(href)

            new_urls = [
                urljoin(url, link['href'])
                for link in soup.find_all('a', href=True)
                if self.is_valid_url(urljoin(url, link['href']))
            ]

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
        print(f"Output directory: {self.output_dir.absolute()}\n")

        queue = deque([(self.base_url, 0)])

        while queue and len(self.visited_urls) < max_pages:
            url, depth = queue.popleft()
            if depth > max_depth:
                continue
            new_urls = self.crawl_page(url)
            for new_url in new_urls:
                if len(self.visited_urls) < max_pages:
                    queue.append((new_url, depth + 1))

        self.save_metadata()
        self.print_summary()

    def print_summary(self):
        print("\n" + "="*50)
        print("CRAWL SUMMARY")
        print("="*50)
        print(f"Pages crawled: {self.stats['pages_crawled']}")
        print(f"HTML files saved: {self.stats['html_saved']}")
        print(f"PDFs downloaded: {self.stats['pdfs_downloaded']}")
        print(f"Skipped (unchanged): {self.stats['skipped_not_modified']}")
        print(f"Errors encountered: {self.stats['errors']}")
        print(f"Total URLs visited: {len(self.visited_urls)}")
        print(f"\nAll files saved to: {self.output_dir.absolute()}")
        print("="*50)


# ================= CLI interface =================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Smart CrawlerUDC with metadata persistence.")
    parser.add_argument("--urls_file", "-f", type=str, default="urls.txt",
                        help="Path to file containing URLs to crawl (default: urls.txt)")
    parser.add_argument("--max_pages", "-p", type=int, default=100)
    parser.add_argument("--max_depth", "-d", type=int, default=3)
    parser.add_argument("--output_dir", "-o", type=str, default="crawled_data")
    parser.add_argument("--refresh_days", "-r", type=int, default=30,
                        help="Days before forcing re-check of a URL (default: 30).")

    args = parser.parse_args()

    urls_path = Path(args.urls_file)
    if not urls_path.exists():
        raise FileNotFoundError(f"URLs file not found: {urls_path}")

    with open(urls_path, "r", encoding="utf-8") as f:
        urls = [line.strip() for line in f if line.strip()]

    print(f"Found {len(urls)} URLs to crawl from {urls_path}\n")

    for i, url in enumerate(urls, start=1):
        print(f"\n=== [{i}/{len(urls)}] Crawling site: {url} ===")
        crawler = CrawlerUDC(
            base_url=url,
            output_dir=args.output_dir,
            refresh_days=args.refresh_days
        )
        crawler.crawl(max_pages=args.max_pages, max_depth=args.max_depth)
        print("\n--- Crawl completed for this site ---\n")
