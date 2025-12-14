import os
import time
import json
import hashlib
import argparse
from urllib.parse import urljoin, urlparse
from pathlib import Path
from typing import Set, List, Dict, Any, Tuple
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta

# Import our CleanHTMLLoader
from CleanHTMLLoader import CleanHTMLLoader

# PDF parsing imports
try:
    from pypdf import PdfReader
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
    print("Warning: pypdf not available. PDF text extraction will be disabled.")

# OCR imports
import sys

try:
    from PIL import Image
    import pytesseract

    if sys.platform.startswith("win"):
        pytesseract.pytesseract.tesseract_cmd = (
            r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        )

    OCR_AVAILABLE = True

except ImportError:
    OCR_AVAILABLE = False
    print("Warning: PIL/pytesseract not available. OCR will be disabled.")


def process_image_ocr(image_path: Path) -> str:
    """Perform OCR on a single image and return extracted text.
    Returns empty string if no text found or error occurs."""
    if not OCR_AVAILABLE:
        return ""
    
    try:
        img = Image.open(image_path)
        text = pytesseract.image_to_string(img, lang='eng+spa')
        return text.strip() if len(text) > args.min_char_ocr else ""
    except Exception as e:
        print(f"  OCR error for {image_path.name}: {e}")
        return ""


def extract_text_from_html(html_content: bytes) -> str:
    """Extract clean text from HTML using CleanHTMLLoader."""
    return CleanHTMLLoader.extract_text_from_html(html_content)


def extract_text_from_pdf(pdf_path: Path) -> str:
    """Extract text from PDF file."""
    if not PDF_AVAILABLE:
        return "[PDF parsing not available - pypdf not installed]"
    
    try:
        reader = PdfReader(str(pdf_path))
        text_parts = []
        
        for page in reader.pages:
            text = page.extract_text()
            if text:
                text_parts.append(text)
        
        return '\n\n'.join(text_parts)
    except Exception as e:
        return f"[Error extracting PDF text: {e}]"


def calculate_text_hash(text: str) -> str:
    """Calculate SHA256 hash of text content."""
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


class CrawlerUDC:
    """
    Smart web crawler that downloads PDFs, HTML files and images,
    persists visited state, and refreshes only updated pages.
    Now includes: force_recrawl, image downloading, and inline OCR processing
    """
    _metadata_lock = Lock()
    _visited_lock = Lock()
    _map_lock = Lock()

    def __init__(self, base_url: str,
                 output_dir: str = "crawled_data",
                 state_dir: str = "crawl",
                 keywords_file: str = "crawl/keywords.txt",
                 refresh_days: int = 30,
                 force_recrawl: bool = False,
                 download_images: bool = True,
                 enable_ocr: bool = True):

        self.base_url = base_url.rstrip('/')
        self.domain = urlparse(base_url).netloc

        self.output_dir = Path(output_dir)
        self.images_dir = self.output_dir / "images"
        self.state_dir = Path(state_dir)
        self.text_dir = self.state_dir / "text"  # Nueva carpeta para textos planos
        self.keywords_file = Path(keywords_file)
        self.visited_urls: Set[str] = set()
        self.downloaded_files: Set[str] = set()
        self.downloaded_images: Set[str] = set()
        self.refresh_days = refresh_days
        self.force_recrawl = force_recrawl
        self.download_images = download_images
        self.enable_ocr = enable_ocr and OCR_AVAILABLE

        self.file_map: Dict[str, str] = {}

        self.output_dir.mkdir(parents=True, exist_ok=True)
        if self.download_images:
            self.images_dir.mkdir(parents=True, exist_ok=True)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.text_dir.mkdir(parents=True, exist_ok=True)  # Crear carpeta de textos

        self.meta_path = self.state_dir / "metadata.json"
        self.visited_path = self.state_dir / "visited_urls.txt"
        self.map_path = self.state_dir / "map.json"

        self.metadata: Dict[str, Dict[str, Any]] = self.load_metadata()
        self.load_visited_urls()

        self.headers = {
            'User-Agent': 'Mozilla/5.0 (compatible; UniversityCrawler/2.0; +research purposes)'
        }

        self.stats = {
            'pages_crawled': 0,
            'pdfs_downloaded': 0,
            'html_saved': 0,
            'images_downloaded': 0,
            'ocr_processed': 0,
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

    def load_visited_urls(self):
        """Load previously visited URLs from file"""
        with self._visited_lock:
            if self.visited_path.exists():
                try:
                    with open(self.visited_path, "r", encoding="utf-8") as f:
                        self.visited_urls = set(line.strip() for line in f if line.strip())
                    print(f"Loaded {len(self.visited_urls)} previously visited URLs")
                except Exception as e:
                    print(f"Error loading visited URLs: {e}")
                    self.visited_urls = set()

    def append_visited_url(self, url: str):
        """Append a single URL to the visited file immediately"""
        with self._visited_lock:
            try:
                with open(self.visited_path, "a", encoding="utf-8") as f:
                    f.write(f"{url}\n")
            except Exception as e:
                print(f"Error appending visited URL {url}: {e}")

    def save_visited_urls(self):
        """Full save of all visited URLs (used as safety backup)"""
        with self._visited_lock:
            try:
                with open(self.visited_path, "w", encoding="utf-8") as f:
                    for url in sorted(self.visited_urls):
                        f.write(f"{url}\n")
            except Exception as e:
                print(f"Error saving visited URLs: {e}")

    # =================== URL & file handling ===================

    def extract_and_process_images_from_html(self, html_content: bytes, page_url: str) -> List[Tuple[str, str]]:
        """Extract images from HTML, download them, perform OCR, and return list of (image_name, ocr_text).
        
        Args:
            html_content: HTML content as bytes
            page_url: URL of the page (for resolving relative URLs and referer)
            
        Returns:
            List of tuples: (image_filename, ocr_text)
        """
        if not self.download_images or not self.enable_ocr:
            return []
        
        ocr_results = []
        
        try:
            soup = BeautifulSoup(html_content, 'lxml')
            image_tags = soup.find_all('img')
            
            if not image_tags:
                return []
            
            print(f"  Found {len(image_tags)} images to process")
            
            # Process images in parallel using ThreadPoolExecutor
            def process_single_image(img_tag) -> Tuple[str, str]:
                """Download image, perform OCR, return (filename, ocr_text)"""
                try:
                    # Get image URL (prefer srcset high-res, then src)
                    img_url = None
                    if img_tag.get('srcset'):
                        srcset = img_tag['srcset'].split(',')
                        for src in srcset:
                            parts = src.strip().split()
                            if len(parts) >= 1:
                                img_url = urljoin(page_url, parts[0])
                                break
                    
                    if not img_url and img_tag.get('data-src'):
                        img_url = urljoin(page_url, img_tag['data-src'])
                    elif not img_url and img_tag.get('src'):
                        img_url = urljoin(page_url, img_tag['src'])
                    
                    if not img_url or img_url in self.downloaded_images:
                        return None, ""
                    
                    # Download image
                    filepath = self._download_image_sync(img_url, page_url)
                    if not filepath or not filepath.exists():
                        return None, ""
                    
                    # Perform OCR
                    ocr_text = process_image_ocr(filepath)
                    
                    # Delete image after OCR
                    try:
                        os.remove(filepath)
                    except OSError:
                        pass
                    
                    if ocr_text:
                        self.stats['ocr_processed'] += 1
                        print(f"    OCR extracted {len(ocr_text)} chars from {filepath.name}")
                        return filepath.name, ocr_text
                    
                    return None, ""
                    
                except Exception as e:
                    print(f"    Error processing image: {e}")
                    return None, ""
            
            # Process images in parallel (max 4 concurrent downloads)
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = [executor.submit(process_single_image, img) for img in image_tags]
                
                for future in as_completed(futures):
                    img_name, ocr_text = future.result()
                    if img_name and ocr_text:
                        ocr_results.append((img_name, ocr_text))
            
        except Exception as e:
            print(f"  Error extracting images from HTML: {e}")
        
        return ocr_results

    def _download_image_sync(self, url: str, page_url: str = None) -> Path | None:
        """Synchronously download an image and return its path."""
        if not url or url in self.downloaded_images:
            return None
        
        try:
            headers = {
                'User-Agent': self.headers.get('User-Agent'),
                'Accept': 'image/avif,image/webp,image/apng,*/*',
                'Accept-Language': 'en-US,en;q=0.8',
                'Referer': page_url or self.base_url
            }
            
            r = requests.get(url, headers=headers, timeout=20)
            r.raise_for_status()
            
            content_type = r.headers.get('Content-Type', '')
            if not content_type.startswith("image/"):
                return None
            
            ext = content_type.split("/")[-1].split(";")[0].strip().lower()
            if ext == "":
                ext = "jpg"
            
            filename = self.generate_filename(url, ext)
            filepath = self.images_dir / filename
            
            with open(filepath, 'wb') as f:
                f.write(r.content)
            
            self.downloaded_images.add(url)
            self.file_map[filename] = url
            self.stats['images_downloaded'] += 1
            
            return filepath
            
        except Exception:
            return None

    # =================== URL & file handling ===================

    def is_valid_url(self, url: str) -> bool:
        parsed = urlparse(url)
        return (
            parsed.netloc == self.domain and
            parsed.scheme in ['http', 'https'] and
            not any(ext in url.lower() for ext in [
                '.css', '.js', '.ico', '.woff', '.woff2', '.ttf', '.eot'
            ])
        )

    def is_image_url(self, url: str) -> bool:
        """Detecta si una URL es una imagen"""
        image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.svg', '.ico']
        return any(url.lower().endswith(ext) for ext in image_extensions)

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

    def process_document_to_text(self, url: str, content: bytes, doc_type: str) -> bool:
        """Process document (HTML/PDF) to plain text with OCR integration.
        
        Args:
            url: Source URL
            content: Document content (bytes)
            doc_type: 'html' or 'pdf'
            
        Returns:
            True if processing succeeded and text was saved
        """
        filename_base = self.generate_filename(url, doc_type).rsplit('.', 1)[0]
        text_filename = filename_base + ".txt"
        text_path = self.text_dir / text_filename
        
        # Also save original file
        original_filename = self.generate_filename(url, doc_type)
        original_path = self.output_dir / original_filename
        
        # Check if needs reprocessing
        old_meta = self.metadata.get(url, {})
        remote_meta = self.get_remote_metadata(url)
        
        if not self.force_recrawl and text_path.exists():
            # Check if remote changed
            if old_meta.get("etag") and remote_meta.get("etag"):
                if old_meta["etag"] == remote_meta["etag"]:
                    print(f"Skipping unchanged document (ETag match): {text_filename}")
                    self.stats['skipped_not_modified'] += 1
                    return False
            if old_meta.get("last_modified") and remote_meta.get("last_modified"):
                if old_meta["last_modified"] == remote_meta["last_modified"]:
                    print(f"Skipping unchanged document (Last-Modified match): {text_filename}")
                    self.stats['skipped_not_modified'] += 1
                    return False
        
        try:
            # Save original file
            with open(original_path, 'wb') as f:
                f.write(content)
            
            # Extract base text
            if doc_type == 'html':
                base_text = extract_text_from_html(content)
                
                # Extract images and perform inline OCR
                ocr_results = self.extract_and_process_images_from_html(content, url)
                
                # Append OCR texts to base text
                if ocr_results:
                    print(f"  → {len(ocr_results)} images processed with OCR")
                    for img_name, ocr_text in ocr_results:
                        base_text += f"\n\n--- OCR de imagen: {img_name} ---\n{ocr_text}"
                
                final_text = base_text
                
            elif doc_type == 'pdf':
                # Save PDF first to extract text
                final_text = extract_text_from_pdf(original_path)
            else:
                print(f"Unknown document type: {doc_type}")
                return False
            
            # Calculate text hash
            text_hash = calculate_text_hash(final_text)
            
            # Check if text content changed
            needs_embedding = True
            if old_meta.get("text_hash") == text_hash:
                print(f"Text content unchanged: {text_filename}")
                needs_embedding = False
            
            # Save plain text
            with open(text_path, 'w', encoding='utf-8') as f:
                f.write(final_text)
            
            # Update metadata with new schema
            self.metadata[url] = {
                **remote_meta,
                "text_hash": text_hash,
                "needs_embeddings": needs_embedding,
                "last_crawl": datetime.now().isoformat(),
                "last_embedded": old_meta.get("last_embedded"),
                "text_path": str(text_path.relative_to(self.state_dir)),
                "original_path": str(original_path.relative_to(self.output_dir)),
                "original_format": doc_type
            }
            
            # Save metadata immediately so OCR can access it
            self.save_metadata()
            
            self.file_map[text_filename] = url
            self.file_map[original_filename] = url
            
            if doc_type == 'html':
                self.stats['html_saved'] += 1
            else:
                self.stats['pdfs_downloaded'] += 1
            
            status = "(necesita embeddings)" if needs_embedding else "(sin cambios en contenido)"
            print(f"Processed to text: {text_filename} {status}")
            return True
            
        except Exception as e:
            print(f"Error processing document {url} to text: {e}")
            self.stats['errors'] += 1
            return False
    
    def save_html(self, url: str, content: bytes):
        """Legacy method - now uses process_document_to_text"""
        self.process_document_to_text(url, content, 'html')

    def download_pdf(self, url: str):
        """Download PDF and process to plain text."""
        try:
            r = requests.get(url, headers=self.headers, timeout=30)
            r.raise_for_status()
            
            # Process to plain text with metadata tracking
            self.process_document_to_text(url, r.content, 'pdf')
            self.downloaded_files.add(url)
                
        except Exception as e:
            print(f"Error downloading PDF {url}: {e}")
            self.stats['errors'] += 1

    # =================== Crawling ===================

    def crawl_page(self, url: str) -> List[str]:
        if url in self.visited_urls and not self.force_recrawl:
            return []
        # Mark as visited and persist immediately
        if not self.force_recrawl:
            self.visited_urls.add(url)
            self.append_visited_url(url)
        else:
            self.stats['force_recrawled'] += 1
            print(f"Force recrawling: {url}")
        if not self.force_recrawl and not self.should_refresh(url) and not self.has_remote_changed(url):
            print(f"Skipping unchanged page: {url}")
            self.stats['skipped_not_modified'] += 1
            return []
        try:
            print(f"Crawling: {url}")
            response = requests.get(url, headers=self.headers, timeout=30)
            response.raise_for_status()
            
            # Process HTML to plain text (includes OCR)
            self.process_document_to_text(url, response.content, 'html')
            
            soup = BeautifulSoup(response.content, 'lxml')
            self.stats['pages_crawled'] += 1
            
            # Procesar enlaces PDF
            for link in soup.find_all('a', href=True):
                href = urljoin(url, link['href'])
                if href.lower().endswith('.pdf') and self.is_bureaucratic_pdf(href, link.text):
                    self.download_pdf(href)
                    if href not in self.metadata:
                        self.metadata[href] = {}
                    self.metadata[href]['discovered_from'] = url
            
            # Extraer nuevos enlaces
            new_urls = [
                urljoin(url, link['href'])
                for link in soup.find_all('a', href=True)
                if self.is_valid_url(urljoin(url, link['href']))
            ]
            
            for new_url in new_urls:
                if new_url not in self.metadata:
                    self.metadata[new_url] = {}
                self.metadata[new_url]['discovered_from'] = url
            
            # Actualizar metadatos (preservar campos de process_document_to_text)
            existing_meta = self.metadata.get(url, {})
            self.metadata[url] = {
                **existing_meta,  # Preservar needs_embeddings, text_hash, etc.
                **self.get_remote_metadata(url),
                "last_download": datetime.now().isoformat()
            }
            time.sleep(1)
            return new_urls
        except Exception as e:
            print('ai carai')
            print(f"Error crawling {url}: {e}")
            self.stats['errors'] += 1
            return []

    def crawl(self, max_pages: int = 100, max_depth: int = 3):
        print(f"Starting crawl of {self.base_url}")
        print(f"Max pages: {max_pages}, Max depth: {max_depth}")
        print(f"Force recrawl: {self.force_recrawl}")
        print(f"Download images: {self.download_images}")
        print(f"OCR enabled: {self.enable_ocr}")
        print(f"Files will be saved to: {self.output_dir.absolute()}")
        if self.download_images:
            print(f"Images will be saved to: {self.images_dir.absolute()}")
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
            # Save metadata and file map
            self.save_metadata()
            self.save_file_map()

        self.print_summary()

    def print_summary(self):
        print("\n" + "="*50)
        print(f"CRAWL SUMMARY - {self.domain}")
        print("="*50)
        print(f"Pages crawled: {self.stats['pages_crawled']}")
        print(f"HTML files saved: {self.stats['html_saved']}")
        print(f"PDFs downloaded: {self.stats['pdfs_downloaded']}")
        print(f"Images downloaded: {self.stats['images_downloaded']}")
        print(f"Skipped (unchanged): {self.stats['skipped_not_modified']}")
        print(f"Errors encountered: {self.stats['errors']}")
        if self.force_recrawl:
            print(f"Pages force-recrawled: {self.stats['force_recrawled']}")
        print(f"Total URLs visited: {len(self.visited_urls)}")
        print(f"\nFiles saved to: {self.output_dir.absolute()}")
        print(f"Plain texts saved to: {self.text_dir.absolute()}")
        if self.download_images:
            print(f"Images saved to: {self.images_dir.absolute()}")
        if self.enable_ocr:
            print(f"OCR texts saved to: {self.output_dir / 'ocr_texts'}")
        print(f"State saved to: {self.state_dir.absolute()}")
        
        # Count documents needing embeddings
        needs_embeddings = sum(1 for meta in self.metadata.values() 
                              if isinstance(meta, dict) and meta.get('needs_embeddings', False))
        print(f"\nDocuments needing embeddings: {needs_embeddings}")
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
            force_recrawl=args.force,
            download_images=args.download_images,
            enable_ocr=args.enable_ocr
        )
        crawler.crawl(max_pages=args.max_pages, max_depth=args.max_depth)
        return (url, crawler.stats, None)
    except Exception as e:
        return (url, None, str(e))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Smart CrawlerUDC with inline OCR support.")
    parser.add_argument("--urls_file", "-f", type=str, default="crawl/urls.txt")
    parser.add_argument("--keywords_file", "-kf", type=str, default="crawl/keywords.txt")
    parser.add_argument("--max_pages", "-p", type=int, default=2000)
    parser.add_argument("--max_depth", "-d", type=int, default=20)
    parser.add_argument("--min_char_ocr", "-c", type=int, default=25)
    parser.add_argument("--output_dir", "-o", type=str, default="crawl/crawled_data")
    parser.add_argument("--state_dir", "-s", type=str, default="crawl")
    parser.add_argument("--refresh_days", "-r", type=int, default=30)
    parser.add_argument("--workers", "-w", type=int, default=8)
    parser.add_argument("--force", action="store_true",
                        help="Force re-crawl even if URL already visited")
    parser.add_argument("--download_images", "-i", action="store_true", default=True,
                        help="Download all images found (default: True)")
    parser.add_argument("--no_images", action="store_false", dest="download_images",
                        help="Skip image downloading")
    parser.add_argument("--enable_ocr", action="store_true", default=True,
                        help="Enable OCR processing of images (default: True)")
    parser.add_argument("--no_ocr", action="store_false", dest="enable_ocr",
                        help="Disable OCR processing")

    args = parser.parse_args()

    if args.enable_ocr and not OCR_AVAILABLE:
        print("Error: OCR is enabled but required libraries are not installed.")
        print("   Please install: pip install pillow pytesseract")
        print("   And ensure tesseract-ocr is installed on your system.")
        exit(1)

    urls_path = Path(args.urls_file)
    if not urls_path.exists():
        raise FileNotFoundError(f"URLs file not found: {urls_path}")

    with open(urls_path, "r", encoding="utf-8") as f:
        urls = [line.strip() for line in f if line.strip()]

    print(f"Found {len(urls)} URLs to crawl from {urls_path}")
    print(f"Using {args.workers} concurrent crawler workers")
    print(f"Image downloading: {'enabled' if args.download_images else 'disabled'}")
    print(f"OCR processing: {'enabled (inline)' if args.enable_ocr else 'disabled'}\n")

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
                    print(f"\nError crawling {result_url}: {error}")
                else:
                    all_stats[result_url] = stats
                    print(f"\nCompleted: {result_url}")
            except Exception as e:
                errors[url] = str(e)
                print(f"\nException for {url}: {e}")

    elapsed = time.time() - start_time 

    print("\n" + "="*60)
    print("GLOBAL CRAWL SUMMARY")
    print("="*60)
    print(f"Total URLs processed: {len(urls)}")
    print(f"Successful: {len(all_stats)}")
    print(f"Failed: {len(errors)}")
    print(f"Time elapsed: {elapsed:.2f} seconds")
    print(f"Crawler workers used: {args.workers}")

    if all_stats:
        total_pages = sum(s['pages_crawled'] for s in all_stats.values())
        total_pdfs = sum(s['pdfs_downloaded'] for s in all_stats.values())
        total_html = sum(s['html_saved'] for s in all_stats.values())
        total_images = sum(s['images_downloaded'] for s in all_stats.values())
        total_ocr = sum(s.get('ocr_processed', 0) for s in all_stats.values())
        total_errors = sum(s['errors'] for s in all_stats.values())
        total_force_recrawled = sum(s.get('force_recrawled', 0) for s in all_stats.values())

        print(f"\nAggregated stats:")
        print(f"  Total pages crawled: {total_pages}")
        print(f"  Total HTMLs saved: {total_html}")
        print(f"  Total PDFs downloaded: {total_pdfs}")
        print(f"  Total images downloaded: {total_images}")
        if args.enable_ocr:
            print(f"  Total OCR processed: {total_ocr}")
        print(f"  Total errors: {total_errors}")
        if args.force:
            print(f"  Total force-recrawled: {total_force_recrawled}")

    if errors:
        print(f"\nFailed URLs:")
        for url, error in errors.items():
            print(f"  - {url}: {error}")

    print("="*60)
