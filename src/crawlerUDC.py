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
from multiprocessing import Process, Queue, Event
from queue import Empty

import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta

# OCR imports
try:
    from PIL import Image
    import pytesseract
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False
    print("⚠ Warning: PIL/pytesseract not available. OCR will be disabled.")


class OCRProcessor:
    """
    Separate process that consumes images from a queue and performs OCR.
    """
    def __init__(self, image_queue: Queue, output_dir: Path, stop_event: Event):
        self.image_queue = image_queue
        self.output_dir = output_dir
        self.ocr_dir = output_dir / "ocr_texts"
        self.ocr_dir.mkdir(parents=True, exist_ok=True)
        self.stop_event = stop_event
        self.stats = {
            'processed': 0,
            'errors': 0
        }

    def process_image(self, image_path: Path) -> bool:
        """Perform OCR on a single image and save as .txt"""
        try:
            txt_filename = image_path.stem + ".txt"
            txt_path = self.ocr_dir / txt_filename

            # Skip if already processed
            if txt_path.exists():
                print(f"⊘ OCR already exists: {txt_filename}")
                return True

            # Open and perform OCR
            img = Image.open(image_path)
            text = pytesseract.image_to_string(img, lang='eng+spa')  # English + Spanish

            if len(text) > 1:
                # Save OCR result
                with open(txt_path, 'w', encoding='utf-8') as f:
                    f.write(text)

                self.stats['processed'] += 1
                print(f"📝 OCR completed: {txt_filename} ({len(text)} chars)")
                os.remove(image_path)
                return True

            else:
                print(f'📝 OCR completado pero {image_path.stem} no se guarda como .txt porque no tiene texto.')
                os.remove(image_path)
                return True



        except Exception as e:
            print(f"✗ OCR error for {image_path.name}: {e}")
            self.stats['errors'] += 1
            os.remove(image_path)
            return False

    def run(self):
        """Main loop for OCR processor"""
        print(f"🔍 OCR Processor started (PID: {os.getpid()})")
        print(f"   Output directory: {self.ocr_dir.absolute()}")

        while not self.stop_event.is_set() or not self.image_queue.empty():
            try:
                # Get image path from queue (timeout to check stop_event periodically)
                image_path = self.image_queue.get(timeout=1)
                
                if image_path is None:  # Poison pill
                    break
                    
                self.process_image(image_path)

            except Empty:
                continue
            except Exception as e:
                print(f"✗ OCR processor error: {e}")
                self.stats['errors'] += 1

        print(f"\n🔍 OCR Processor finished:")
        print(f"   Processed: {self.stats['processed']}")
        print(f"   Errors: {self.stats['errors']}")


def start_ocr_worker(image_queue: Queue, output_dir: Path, stop_event: Event):
    """Function to start OCR processor in a separate process"""
    processor = OCRProcessor(image_queue, output_dir, stop_event)
    processor.run()


class CrawlerUDC:
    """
    Smart web crawler that downloads PDFs, HTML files and images,
    persists visited state, and refreshes only updated pages.
    Now includes: force_recrawl, image downloading, and parallel OCR
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
                 enable_ocr: bool = True,
                 ocr_workers: int = 2):

        self.base_url = base_url.rstrip('/')
        self.domain = urlparse(base_url).netloc

        self.output_dir = Path(output_dir)
        self.images_dir = self.output_dir / "images"
        self.state_dir = Path(state_dir)
        self.keywords_file = Path(keywords_file)
        self.visited_urls: Set[str] = set()
        self.downloaded_files: Set[str] = set()
        self.downloaded_images: Set[str] = set()
        self.refresh_days = refresh_days
        self.force_recrawl = force_recrawl
        self.download_images = download_images
        self.enable_ocr = enable_ocr and OCR_AVAILABLE
        self.ocr_workers = ocr_workers

        self.file_map: Dict[str, str] = {}

        self.output_dir.mkdir(parents=True, exist_ok=True)
        if self.download_images:
            self.images_dir.mkdir(parents=True, exist_ok=True)
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
            'images_downloaded': 0,
            'errors': 0,
            'skipped_not_modified': 0,
            'force_recrawled': 0,
            'new_urls': 0,
            'changed_urls': 0,
            'deleted': 0
        }

        # OCR process management
        self.ocr_queue = None
        self.ocr_processes = []
        self.ocr_stop_event = None

    # =================== Persistence ===================

    def load_metadata(self) -> Dict[str, Any]:
        with self._metadata_lock:
            if self.meta_path.exists():
                try:
                    with open(self.meta_path, "r", encoding="utf-8") as f:
                        metadata = json.load(f)
                        # Initialize change_status for existing entries if not present
                        for url, meta in metadata.items():
                            if 'change_status' not in meta:
                                meta['change_status'] = 0  # Assume no changes initially
                        return metadata
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

    # =================== OCR Management ===================

    def start_ocr_workers(self):
        """Start OCR worker processes"""
        if not self.enable_ocr:
            return

        from multiprocessing import Manager
        manager = Manager()
        self.ocr_queue = manager.Queue()
        self.ocr_stop_event = manager.Event()

        print(f"\n🔍 Starting {self.ocr_workers} OCR worker process(es)...")
        for i in range(self.ocr_workers):
            p = Process(
                target=start_ocr_worker,
                args=(self.ocr_queue, self.output_dir, self.ocr_stop_event),
                daemon=False
            )
            p.start()
            self.ocr_processes.append(p)
            print(f"   Worker {i+1} started (PID: {p.pid})")

    def stop_ocr_workers(self):
        """Stop OCR worker processes gracefully"""
        if not self.enable_ocr or not self.ocr_processes:
            return

        print("\n🔍 Stopping OCR workers...")
        
        # Send stop signal
        self.ocr_stop_event.set()
        
        # Send poison pills
        for _ in self.ocr_processes:
            self.ocr_queue.put(None)

        # Wait for all processes to finish
        for i, p in enumerate(self.ocr_processes):
            p.join(timeout=30)
            if p.is_alive():
                print(f"   ⚠ Worker {i+1} did not stop gracefully, terminating...")
                p.terminate()
                p.join()
            else:
                print(f"   Worker {i+1} stopped")

        self.ocr_processes.clear()

    def queue_image_for_ocr(self, image_path: Path):
        """Add image to OCR queue"""
        if self.enable_ocr and self.ocr_queue is not None:
            self.ocr_queue.put(image_path)

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

    def has_remote_changed(self, url: str) -> int:
        """
        Returns change status:
        0 - no changes
        1 - changed (content modified)
        2 - new (not in metadata)
        3 - deleted (URL no longer accessible)
        """
        if self.force_recrawl:
            return 1  # Treat as changed
        
        old_meta = self.metadata.get(url, {})
        
        # New URL (not in metadata)
        if not old_meta:
            return 2
        
        # Check if URL is still accessible
        try:
            response = requests.head(url, headers=self.headers, timeout=15, allow_redirects=True)
            if response.status_code == 404 or response.status_code == 410:
                return 3  # Explicitly deleted
                
            new_meta = self.get_remote_metadata(url)
            # If we can't get metadata, treat as potentially deleted
            if not new_meta:
                return 3
        except requests.exceptions.RequestException as e:
            # If request fails completely, treat as deleted
            print(f"✗ Request failed for {url}: {e}")
            return 3
        
        # Check for content changes
        content_changed = False
        if old_meta.get("etag") and new_meta.get("etag"):
            if old_meta["etag"] != new_meta["etag"]:
                content_changed = True
        
        if old_meta.get("last_modified") and new_meta.get("last_modified"):
            if old_meta["last_modified"] != new_meta["last_modified"]:
                content_changed = True
        
        return 1 if content_changed else 0

    def detect_deleted_urls(self) -> List[str]:
        """Compare current file_map with previous map.json to find deleted URLs"""
        deleted_urls = []

        # Load previous file map if it exists
        previous_map = {}
        if self.map_path.exists():
            try:
                with open(self.map_path, "r", encoding="utf-8") as f:
                    previous_map = json.load(f)
            except Exception:
                pass

        # Find URLs that were in previous map but not in current file_map
        current_urls = set(self.file_map.values())
        for filename, url in previous_map.items():
            if url not in current_urls and url not in self.visited_urls:
                deleted_urls.append(url)
                # Mark as deleted in metadata
                if url in self.metadata:
                    self.metadata[url]['change_status'] = 3
                    self.metadata[url]['deleted_detected'] = datetime.now().isoformat()

        return deleted_urls

    def save_html(self, url: str, content: bytes):
        filename = self.generate_filename(url, 'html')
        filepath = self.output_dir / filename

        if filepath.exists() and self.force_recrawl:
            print(f"↻ Force overwriting HTML: {filename}")

        with open(filepath, 'wb') as f:
            f.write(content)

        self.file_map[filename] = url

        self.stats['html_saved'] += 1
        print(f"✓ Saved HTML: {filename}")

    def download_image(self, url: str, page_url: str = None):
        """Download high-quality images with srcset/lazy-loading support."""

        if not url or url in self.downloaded_images and not self.force_recrawl:
            return

        try:
            # More realistic browser-like headers
            headers = {
                'User-Agent': self.headers.get('User-Agent'),
                'Accept': 'image/avif,image/webp,image/apng,*/*',
                'Accept-Language': 'en-US,en;q=0.8',
                'Referer': page_url or self.base_url
            }

            r = requests.get(url, headers=headers, timeout=20)
            r.raise_for_status()

            # Validate it's actually an image
            content_type = r.headers.get('Content-Type', '')
            if not content_type.startswith("image/"):
                print(f"✗ Skipped non-image content: {url}")
                return

            # Extract extension based on Content-Type
            ext = content_type.split("/")[-1].split(";")[0].strip().lower()
            if ext == "":
                ext = "jpg"   # fallback but rarely needed

            # Build filename based on real extension
            filename = self.generate_filename(url, ext)
            filepath = self.images_dir / filename

            # Skip if exists and not in force mode
            if filepath.exists() and not self.force_recrawl:
                self.downloaded_images.add(url)
                return

            # Save the image
            with open(filepath, 'wb') as f:
                f.write(r.content)

            self.downloaded_images.add(url)
            self.file_map[filename] = url
            self.stats['images_downloaded'] += 1

            print(f"🖼 High-quality image saved: {filename}")

            # Queue for OCR processing
            self.queue_image_for_ocr(filepath)

        except Exception as e:
            print(f"✗ Error downloading image {url}: {e}")
            self.stats['errors'] += 1

    def download_pdf(self, url: str):
        filename = self.generate_filename(url, 'pdf')
        filepath = self.output_dir / filename

        if filepath.exists() and not self.force_recrawl and not self.should_refresh(url):
            print(f"⊘ Skipping existing PDF (fresh): {filename}")
            self.stats['skipped_not_modified'] += 1
            return

        try:
            r = requests.get(url, headers=self.headers, timeout=30)
            r.raise_for_status()
            
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
        if url in self.visited_urls and not self.force_recrawl:
            return []
    
        # Get change status and make programmatic decisions
        change_status = self.has_remote_changed(url)
        
        # Store change status in metadata for persistence
        if url in self.metadata:
            self.metadata[url]['change_status'] = change_status
        else:
            self.metadata[url] = {'change_status': change_status}
        
        # Track statistics based on change status
        if change_status == 2:  # New URL
            self.stats['new_urls'] = self.stats.get('new_urls', 0) + 1
            print(f"🆕 New URL discovered: {url}")
        elif change_status == 1:  # Changed
            self.stats['changed_urls'] = self.stats.get('changed_urls', 0) + 1
            print(f"📝 Changed URL: {url}")
        elif change_status == 3:  # Deleted
            self.stats['deleted'] = self.stats.get('deleted', 0) + 1
            print(f"🗑️ URL appears deleted: {url}")
            # Remove from metadata if it exists and is deleted
            if url in self.metadata:
                del self.metadata[url]
            return []
        
        # Programmatic decision: skip unchanged URLs unless force recrawl
        if change_status == 0 and not self.force_recrawl:
            print(f"⊘ Skipping unchanged page: {url}")
            self.stats['skipped_not_modified'] += 1
            return []
    
        if not self.force_recrawl:
            self.visited_urls.add(url)
        else:
            self.stats['force_recrawled'] += 1
            print(f"↻ Force recrawling: {url}")
    
        try:
            print(f"→ Crawling: {url}")
            response = requests.get(url, headers=self.headers, timeout=30)
            response.raise_for_status()
            
            self.save_html(url, response.content)
            
            soup = BeautifulSoup(response.content, 'lxml')
            self.stats['pages_crawled'] += 1
    
            # Procesar imágenes si está habilitado
            if self.download_images:
                for img in soup.find_all('img'):
                    # Pick best possible URL (lazy-loading support)
                    src = (
                        img.get("src")
                        or img.get("data-src")
                        or img.get("data-original")
                        or img.get("data-lazy")
                    )
    
                    # Handle srcset for high-resolution selection
                    srcset = img.get("srcset")
                    if srcset:
                        try:
                            # pick last (usually highest-res)
                            candidates = [s.strip().split()[0] for s in srcset.split(",")]
                            if candidates:
                                src = candidates[-1]
                        except Exception:
                            pass
    
                    if not src:
                        continue
    
                    img_url = urljoin(url, src)
    
                    # Only download valid resources
                    if self.is_image_url(img_url) or self.is_valid_url(img_url):
                        self.download_image(img_url, page_url=url)
    
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
    
            # Actualizar metadatos con información completa
            remote_meta = self.get_remote_metadata(url)
            self.metadata[url] = {
                **remote_meta,
                "last_download": datetime.now().isoformat(),
                "change_status": change_status,
                "content_length": len(response.content),
                "last_crawled": datetime.now().isoformat()
            }
    
            time.sleep(1)
            return new_urls
    
        except Exception as e:
            print(f"✗ Error crawling {url}: {e}")
            self.stats['errors'] += 1
            
            # Update metadata even for failed crawls to track the error
            if url in self.metadata:
                self.metadata[url]['last_error'] = str(e)
                self.metadata[url]['last_error_time'] = datetime.now().isoformat()
            
            return []

    def crawl(self, max_pages: int = 100, max_depth: int = 3):
        print(f"Starting crawl of {self.base_url}")
        print(f"Max pages: {max_pages}, Max depth: {max_depth}")
        print(f"Force recrawl: {self.force_recrawl}")
        print(f"Download images: {self.download_images}")
        print(f"OCR enabled: {self.enable_ocr}")
        if self.enable_ocr:
            print(f"OCR workers: {self.ocr_workers}")
        print(f"Files will be saved to: {self.output_dir.absolute()}")
        if self.download_images:
            print(f"Images will be saved to: {self.images_dir.absolute()}")
        print(f"State will be saved to: {self.state_dir.absolute()}\n")

        # Start OCR workers
        self.start_ocr_workers()

        try:
            queue = deque([(self.base_url, 0)])

            while queue and (len(self.visited_urls) < max_pages or self.force_recrawl):
                url, depth = queue.popleft()
                if depth > max_depth:
                    continue
                    
                new_urls = self.crawl_page(url)
                for new_url in new_urls:
                    if len(self.visited_urls) < max_pages or self.force_recrawl:
                        queue.append((new_url, depth + 1))

        finally:
            # Detect deletions by comparing with previous file map
            deleted_urls = self.detect_deleted_urls()
            if deleted_urls:
                print(f"\n🗑️ Detected {len(deleted_urls)} deleted URLs from previous crawl")
                for url in deleted_urls[:5]:  # Show first 5
                    print(f"   - {url}")
                if len(deleted_urls) > 5:
                    print(f"   ... and {len(deleted_urls) - 5} more")
            
            # Always stop OCR workers
            self.stop_ocr_workers()

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
        print(f"Images downloaded: {self.stats['images_downloaded']}")
        print(f"Skipped (unchanged): {self.stats['skipped_not_modified']}")
        print(f"Errors encountered: {self.stats['errors']}")
        
        # New change tracking stats
        if 'new_urls' in self.stats:
            print(f"New URLs discovered: {self.stats['new_urls']}")
        if 'changed_urls' in self.stats:
            print(f"Changed URLs: {self.stats['changed_urls']}")
        if 'deleted' in self.stats:
            print(f"Deleted URLs detected: {self.stats['deleted']}")
            
        if self.force_recrawl:
            print(f"Pages force-recrawled: {self.stats['force_recrawled']}")
        print(f"Total URLs visited: {len(self.visited_urls)}")
        print(f"\nFiles saved to: {self.output_dir.absolute()}")
        if self.download_images:
            print(f"Images saved to: {self.images_dir.absolute()}")
        if self.enable_ocr:
            print(f"OCR texts saved to: {self.output_dir / 'ocr_texts'}")
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
            force_recrawl=args.force,
            download_images=args.download_images,
            enable_ocr=args.enable_ocr,
            ocr_workers=args.ocr_workers
        )
        crawler.crawl(max_pages=args.max_pages, max_depth=args.max_depth)
        return (url, crawler.stats, None)
    except Exception as e:
        return (url, None, str(e))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Smart CrawlerUDC with parallel OCR support.")
    parser.add_argument("--urls_file", "-f", type=str, default="crawl/urls.txt")
    parser.add_argument("--keywords_file", "-kf", type=str, default="crawl/keywords.txt")
    parser.add_argument("--max_pages", "-p", type=int, default=200)
    parser.add_argument("--max_depth", "-d", type=int, default=4)
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
    parser.add_argument("--ocr_workers", type=int, default=2,
                        help="Number of parallel OCR worker processes (default: 2)")

    args = parser.parse_args()

    if args.enable_ocr and not OCR_AVAILABLE:
        print("⚠ Error: OCR is enabled but required libraries are not installed.")
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
    print(f"OCR processing: {'enabled' if args.enable_ocr else 'disabled'}")
    if args.enable_ocr:
        print(f"OCR workers per crawler: {args.ocr_workers}\n")

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
    print(f"Crawler workers used: {args.workers}")
    if args.enable_ocr:
        print(f"OCR workers per crawler: {args.ocr_workers}")

    if all_stats:
        total_pages = sum(s['pages_crawled'] for s in all_stats.values())
        total_pdfs = sum(s['pdfs_downloaded'] for s in all_stats.values())
        total_html = sum(s['html_saved'] for s in all_stats.values())
        total_images = sum(s['images_downloaded'] for s in all_stats.values())
        total_errors = sum(s['errors'] for s in all_stats.values())
        total_force_recrawled = sum(s.get('force_recrawled', 0) for s in all_stats.values())
        total_new_urls = sum(s.get('new_urls', 0) for s in all_stats.values())
        total_changed_urls = sum(s.get('changed_urls', 0) for s in all_stats.values())
        total_deleted = sum(s.get('deleted', 0) for s in all_stats.values())

        print(f"\nAggregated stats:")
        print(f"  Total pages crawled: {total_pages}")
        print(f"  Total HTMLs saved: {total_html}")
        print(f"  Total PDFs downloaded: {total_pdfs}")
        print(f"  Total images downloaded: {total_images}")
        print(f"  Total errors: {total_errors}")
        print(f"  New URLs discovered: {total_new_urls}")
        print(f"  Changed URLs: {total_changed_urls}")
        print(f"  Deleted URLs detected: {total_deleted}")
        if args.force:
            print(f"  Total force-recrawled: {total_force_recrawled}")

    if errors:
        print(f"\nFailed URLs:")
        for url, error in errors.items():
            print(f"  - {url}: {error}")

    print("="*60)
