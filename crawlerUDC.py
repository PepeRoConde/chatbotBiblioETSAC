import os
import time
import hashlib
from urllib.parse import urljoin, urlparse
from pathlib import Path
from typing import Set, List
from collections import deque

import requests
from bs4 import BeautifulSoup


class CrawlerUDC:
    """
    Simple web crawler that downloads PDFs and HTML files to a single folder.
    """
    
    def __init__(self, base_url: str, output_dir: str = "crawled_data"):
        self.base_url = base_url.rstrip('/')
        self.domain = urlparse(base_url).netloc
        self.output_dir = Path(output_dir)
        self.visited_urls: Set[str] = set()
        self.downloaded_files: Set[str] = set()
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure request headers
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (compatible; UniversityCrawler/1.0; +research purposes)'
        }
        
        # Crawl statistics
        self.stats = {
            'pages_crawled': 0,
            'pdfs_downloaded': 0,
            'html_saved': 0,
            'errors': 0
        }
    
    def is_valid_url(self, url: str) -> bool:
        """Check if URL belongs to the university domain and is valid."""
        parsed = urlparse(url)
        return (
            parsed.netloc == self.domain and
            parsed.scheme in ['http', 'https'] and
            not any(ext in url.lower() for ext in ['.jpg', '.png', '.gif', '.css', '.js', '.ico', '.svg', '.woff'])
        )
    
    def is_bureaucratic_pdf(self, url: str, text: str = "") -> bool:
        """
        Heuristic to identify bureaucratic/administrative PDFs.
        Customize keywords based on your university's structure.
        """
        bureaucratic_keywords = [
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
        
        url_lower = url.lower()
        text_lower = text.lower()
        
        return any(keyword in url_lower or keyword in text_lower 
                   for keyword in bureaucratic_keywords)
    
    def generate_filename(self, url: str, extension: str) -> str:
        """Generate a unique filename from URL."""
        # Try to get a meaningful name from the URL
        parsed = urlparse(url)
        path_parts = [p for p in parsed.path.split('/') if p]
        
        if path_parts:
            # Use the last part of the path
            name = path_parts[-1]
            # Remove extension if present
            name = name.rsplit('.', 1)[0]
            # Clean the name
            name = name[:50]  # Limit length
        else:
            name = parsed.netloc.replace('.', '_')
        
        # Add hash to ensure uniqueness
        url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
        filename = f"{name}_{url_hash}.{extension}"
        
        # Sanitize filename
        filename = "".join(c for c in filename if c.isalnum() or c in '._-')
        
        return filename
    
    def download_pdf(self, url: str) -> bool:
        """Download a PDF file."""
        try:
            response = requests.get(url, headers=self.headers, timeout=30)
            response.raise_for_status()
            
            filename = self.generate_filename(url, 'pdf')
            filepath = self.output_dir / filename
            
            with open(filepath, 'wb') as f:
                f.write(response.content)
            
            self.downloaded_files.add(url)
            self.stats['pdfs_downloaded'] += 1
            print(f"✓ Downloaded PDF: {filename}")
            return True
            
        except Exception as e:
            print(f"✗ Error downloading PDF {url}: {e}")
            self.stats['errors'] += 1
            return False
    
    def save_html(self, url: str, content: bytes) -> bool:
        """Save HTML content to file."""
        try:
            filename = self.generate_filename(url, 'html')
            filepath = self.output_dir / filename
            
            with open(filepath, 'wb') as f:
                f.write(content)
            
            self.stats['html_saved'] += 1
            print(f"✓ Saved HTML: {filename}")
            return True
            
        except Exception as e:
            print(f"✗ Error saving HTML {url}: {e}")
            self.stats['errors'] += 1
            return False
    
    def crawl_page(self, url: str) -> List[str]:
        """Crawl a single page and return new URLs to visit."""
        if url in self.visited_urls:
            return []
        
        self.visited_urls.add(url)
        new_urls = []
        
        try:
            print(f"Crawling: {url}")
            response = requests.get(url, headers=self.headers, timeout=30)
            response.raise_for_status()
            
            # Save the HTML file
            self.save_html(url, response.content)
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            self.stats['pages_crawled'] += 1
            
            # Find and download PDFs
            for link in soup.find_all('a', href=True):
                href = urljoin(url, link['href'])
                
                if href.lower().endswith('.pdf') and href not in self.downloaded_files:
                    # Check if it's a bureaucratic PDF
                    link_text = link.get_text(strip=True)
                    if self.is_bureaucratic_pdf(href, link_text):
                        self.download_pdf(href)
            
            # Collect new URLs to crawl
            for link in soup.find_all('a', href=True):
                href = urljoin(url, link['href'])
                if self.is_valid_url(href) and href not in self.visited_urls:
                    new_urls.append(href)
            
            # Be respectful - add delay between requests
            time.sleep(1)
            
        except Exception as e:
            print(f"✗ Error crawling {url}: {e}")
            self.stats['errors'] += 1
        
        return new_urls
    
    def crawl(self, max_pages: int = 100, max_depth: int = 3):
        """
        Start crawling from the base URL.
        
        Args:
            max_pages: Maximum number of pages to crawl
            max_depth: Maximum depth to crawl from base URL
        """
        print(f"Starting crawl of {self.base_url}")
        print(f"Max pages: {max_pages}, Max depth: {max_depth}")
        print(f"Output directory: {self.output_dir.absolute()}\n")
        
        # BFS crawl with depth tracking
        queue = deque([(self.base_url, 0)])  # (url, depth)
        
        while queue and len(self.visited_urls) < max_pages:
            url, depth = queue.popleft()
            
            if depth > max_depth:
                continue
            
            new_urls = self.crawl_page(url)
            
            # Add new URLs to queue with incremented depth
            for new_url in new_urls:
                if len(self.visited_urls) < max_pages:
                    queue.append((new_url, depth + 1))
        
        self.print_summary()
    
    def print_summary(self):
        """Print crawl statistics."""
        print("\n" + "="*50)
        print("CRAWL SUMMARY")
        print("="*50)
        print(f"Pages crawled: {self.stats['pages_crawled']}")
        print(f"HTML files saved: {self.stats['html_saved']}")
        print(f"PDFs downloaded: {self.stats['pdfs_downloaded']}")
        print(f"Errors encountered: {self.stats['errors']}")
        print(f"Total URLs visited: {len(self.visited_urls)}")
        print(f"\nAll files saved to: {self.output_dir.absolute()}")
        print("="*50)


# Example usage
if __name__ == "__main__":
    # Replace with your university's homepage
    UNIVERSITY_URL = "https://www.udc.es/"
    
    crawler = CrawlerUDC(
        base_url=UNIVERSITY_URL,
        output_dir="crawled_data"
    )
    
    # Start crawling
    crawler.crawl(max_pages=1000, max_depth=10000)
    
    print("\nCrawl complete! All PDFs and HTML files are in the same folder.")
    print("You can process them later for your RAG system.")