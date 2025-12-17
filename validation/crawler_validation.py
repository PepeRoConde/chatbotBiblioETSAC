"""
Crawler de validación - Procesa solo URLs específicas de url_validation.txt
Basado en CrawlerUDC pero sin crawling recursivo, solo procesamiento directo.
"""

import sys
import time
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Tuple


# OCR imports y forzar ruta igual que en crawlerUDC.py
import sys as _sys
try:
    import pytesseract
    import platform
    from PIL import Image
    # if _sys.platform.startswith("win"):
    #     pytesseract.pytesseract.tesseract_cmd = (
    #         r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
    #     )
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False
    print("Warning: PIL/pytesseract not available. OCR will be disabled.")

# Añadir directorio src al path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from crawlerUDC import (
    CrawlerUDC,
    extract_text_from_html,
    extract_text_from_pdf,
    process_image_ocr,
    calculate_text_hash
)

import requests
import json
from bs4 import BeautifulSoup
from urllib.parse import urlparse
from datetime import datetime


class ValidationCrawler:
    """Crawler simplificado para procesar solo URLs de validación."""
    
    def __init__(
        self,
        output_dir: str = "validation/crawled_validation",
        state_dir: str = "validation/validation_state",
        download_images: bool = True,
        enable_ocr: bool = True
    ):
        self.output_dir = Path(output_dir)
        self.images_dir = self.output_dir / "images"
        self.text_dir = Path(state_dir) / "text"
        self.state_dir = Path(state_dir)
        
        self.download_images = download_images
        self.enable_ocr = enable_ocr
        
        # Crear directorios
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.text_dir.mkdir(parents=True, exist_ok=True)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        
        # Archivos de estado (como CrawlerUDC)
        self.metadata_path = self.state_dir / "metadata.json"
        self.map_path = self.state_dir / "map.json"
        
        # Cargar metadatos si existen
        self.metadata = self.load_metadata()
        self.file_map = self.load_file_map()
        
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (compatible; ValidationCrawler/1.0; +research purposes)'
        }
        
        self.stats = {
            'processed': 0,
            'html': 0,
            'pdf': 0,
            'images': 0,
            'ocr_processed': 0,
            'errors': 0
        }
        
        self.results: Dict[str, Dict[str, Any]] = {}
    
    def load_metadata(self) -> Dict[str, Any]:
        """Carga metadata.json (formato CrawlerUDC)."""
        if self.metadata_path.exists():
            try:
                with open(self.metadata_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}
    
    def save_metadata(self):
        """Guarda metadata.json (formato CrawlerUDC)."""
        with open(self.metadata_path, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False)
    
    def load_file_map(self) -> Dict[str, str]:
        """Carga map.json (formato CrawlerUDC)."""
        if self.map_path.exists():
            try:
                with open(self.map_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}
    
    def save_file_map(self):
        """Guarda map.json (formato CrawlerUDC)."""
        with open(self.map_path, 'w', encoding='utf-8') as f:
            json.dump(self.file_map, f, indent=2, ensure_ascii=False)
    
    def get_remote_metadata(self, url: str) -> Dict[str, str]:
        """Obtiene metadatos remotos (ETag, Last-Modified)."""
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
    
    def generate_filename(self, url: str, extension: str) -> str:
        """Genera nombre de archivo único basado en URL."""
        parsed = urlparse(url)
        name = Path(parsed.path).stem or parsed.netloc.replace('.', '_')
        url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
        filename = f"{name[:50]}_{url_hash}.{extension}"
        return "".join(c for c in filename if c.isalnum() or c in '._-')
    
    def process_url(self, url: str) -> Dict[str, Any]:
        """Procesa una URL individual (HTML o PDF)."""
        print(f"\n{'='*60}")
        print(f"Procesando: {url}")
        print(f"{'='*60}")
        
        result = {
            'url': url,
            'success': False,
            'type': None,
            'text_file': None,
            'original_file': None,
            'images': [],
            'ocr_texts': [],
            'error': None
        }
        
        try:
            response = requests.get(url, headers=self.headers, timeout=30)
            response.raise_for_status()
            
            content_type = response.headers.get('content-type', '').lower()
            content = response.content
            
            # Determinar tipo
            if 'pdf' in content_type or url.lower().endswith('.pdf'):
                result['type'] = 'pdf'
                self._process_pdf(url, content, result)
                self.stats['pdf'] += 1
            else:
                result['type'] = 'html'
                self._process_html(url, content, result)
                self.stats['html'] += 1
            
            result['success'] = True
            self.stats['processed'] += 1
            
        except Exception as e:
            result['error'] = str(e)
            self.stats['errors'] += 1
            print(f"  ❌ Error: {e}")
        
        return result
    
    def _process_pdf(self, url: str, content: bytes, result: Dict):
        """Procesa un PDF (formato CrawlerUDC)."""
        print("  📄 Tipo: PDF")
        
        # Generar nombres de archivo
        pdf_filename = self.generate_filename(url, 'pdf')
        text_filename = pdf_filename.replace('.pdf', '.txt')
        
        pdf_path = self.output_dir / pdf_filename
        text_path = self.text_dir / text_filename
        
        # Guardar PDF original
        pdf_path.write_bytes(content)
        result['original_file'] = str(pdf_path)
        print(f"  ✓ PDF guardado: {pdf_filename}")
        
        # Extraer texto
        text = extract_text_from_pdf(pdf_path)
        
        # Calcular hash del texto
        text_hash = calculate_text_hash(text)
        
        # Guardar texto
        text_path.write_text(text, encoding='utf-8')
        result['text_file'] = str(text_path)
        print(f"  ✓ Texto extraído: {len(text)} caracteres")
        
        # Metadatos formato CrawlerUDC
        remote_meta = self.get_remote_metadata(url)
        self.metadata[url] = {
            **remote_meta,
            "text_hash": text_hash,
            "needs_embeddings": True,
            "last_crawl": datetime.now().isoformat(),
            "last_embedded": None,
            "text_path": str(text_path.relative_to(self.state_dir)),
            "original_path": str(pdf_path.relative_to(self.output_dir)),
            "original_format": "pdf"
        }
        
        # Actualizar mapa
        self.file_map[text_filename] = url
        self.file_map[pdf_filename] = url
    
    def _process_html(self, url: str, content: bytes, result: Dict):
        """Procesa una página HTML (formato CrawlerUDC)."""
        print("  🌐 Tipo: HTML")
        
        # Generar nombres de archivo
        html_filename = self.generate_filename(url, 'html')
        text_filename = html_filename.replace('.html', '.txt')
        
        html_path = self.output_dir / html_filename
        text_path = self.text_dir / text_filename
        
        # Guardar HTML original
        html_path.write_bytes(content)
        result['original_file'] = str(html_path)
        print(f"  ✓ HTML guardado: {html_filename}")
        
        # Extraer texto limpio
        base_text = extract_text_from_html(content)
        
        # Procesar imágenes si está habilitado
        if self.download_images and self.enable_ocr:
            ocr_results = self._extract_images_and_ocr(content, url)
            result['images'] = [img for img, _ in ocr_results]
            result['ocr_texts'] = [ocr for _, ocr in ocr_results]
            
            # Agregar textos OCR al texto principal
            if ocr_results:
                ocr_combined = "\n\n=== TEXTO DE IMÁGENES (OCR) ===\n\n"
                for img_name, ocr_text in ocr_results:
                    if ocr_text:
                        ocr_combined += f"[Imagen: {img_name}]\n{ocr_text}\n\n"
                base_text += ocr_combined
        
        # Calcular hash del texto
        text_hash = calculate_text_hash(base_text)
        
        # Guardar texto completo
        text_path.write_text(base_text, encoding='utf-8')
        result['text_file'] = str(text_path)
        print(f"  ✓ Texto extraído: {len(base_text)} caracteres")
        
        # Metadatos formato CrawlerUDC
        remote_meta = self.get_remote_metadata(url)
        self.metadata[url] = {
            **remote_meta,
            "text_hash": text_hash,
            "needs_embeddings": True,
            "last_crawl": datetime.now().isoformat(),
            "last_embedded": None,
            "text_path": str(text_path.relative_to(self.state_dir)),
            "original_path": str(html_path.relative_to(self.output_dir)),
            "original_format": "html"
        }
        
        # Actualizar mapa
        self.file_map[text_filename] = url
        self.file_map[html_filename] = url
    
    def _extract_images_and_ocr(self, html_content: bytes, page_url: str) -> List[Tuple[str, str]]:
        """Extrae imágenes del HTML y aplica OCR."""
        from urllib.parse import urljoin
        
        ocr_results = []
        
        try:
            soup = BeautifulSoup(html_content, 'lxml')
            image_tags = soup.find_all('img')
            
            if not image_tags:
                return []
            
            print(f"  🖼️  Encontradas {len(image_tags)} imágenes")
            
            for idx, img_tag in enumerate(image_tags, 1):
                img_url = img_tag.get('src') or img_tag.get('data-src')
                if not img_url:
                    continue
                
                # Resolver URL relativa
                if not img_url.startswith('http'):
                    img_url = urljoin(page_url, img_url)
                
                # Descargar imagen
                try:
                    img_response = requests.get(img_url, headers=self.headers, timeout=10)
                    img_response.raise_for_status()
                    
                    # Guardar imagen
                    img_name = f"img_{idx}_{hashlib.md5(img_url.encode()).hexdigest()[:8]}.jpg"
                    img_path = self.images_dir / img_name
                    img_path.write_bytes(img_response.content)
                    
                    self.stats['images'] += 1
                    
                    # Aplicar OCR
                    ocr_text, is_table = process_image_ocr(img_path)
                    if ocr_text:
                        ocr_results.append((img_name, ocr_text))
                        self.stats['ocr_processed'] += 1
                        print(f"    ✓ OCR imagen {idx}: {len(ocr_text)} chars (tabla: {is_table})")
                    
                except Exception as e:
                    print(f"    ⚠️  Error procesando imagen {idx}: {e}")
            
        except Exception as e:
            print(f"  ⚠️  Error extrayendo imágenes: {e}")
        
        return ocr_results
    
    def process_urls_from_file(self, urls_file: str) -> Dict[str, Dict[str, Any]]:
        """Procesa todas las URLs del archivo (formato CrawlerUDC)."""
        urls_path = Path(urls_file)
        
        if not urls_path.exists():
            raise FileNotFoundError(f"Archivo no encontrado: {urls_file}")
        
        # Leer URLs
        with open(urls_path, 'r', encoding='utf-8') as f:
            urls = [line.strip() for line in f if line.strip() and not line.startswith('#')]
        
        print(f"\n{'='*60}")
        print(f"VALIDACIÓN DE CRAWLER")
        print(f"{'='*60}")
        print(f"URLs a procesar: {len(urls)}")
        print(f"Download images: {self.download_images}")
        print(f"OCR enabled: {self.enable_ocr}")
        print(f"Output dir: {self.output_dir.absolute()}")
        print(f"Text dir: {self.text_dir.absolute()}\n")
        
        start_time = time.time()
        
        # Procesar cada URL
        for url in urls:
            result = self.process_url(url)
            time.sleep(0.5)  # Pausa para no saturar el servidor
        
        # Guardar metadata y map al finalizar
        self.save_metadata()
        self.save_file_map()
        
        elapsed = time.time() - start_time
        
        # Resumen
        self.print_summary(elapsed)
        
        print(f"\n📝 Metadata guardado: {self.metadata_path}")
        print(f"🗺️  Map guardado: {self.map_path}")
        
        return {}  # Ya no usamos self.results
    
    def print_summary(self, elapsed: float):
        """Imprime resumen de la ejecución."""
        print(f"\n{'='*60}")
        print("RESUMEN DE VALIDACIÓN")
        print(f"{'='*60}")
        print(f"URLs procesadas exitosamente: {self.stats['processed']}")
        print(f"  - HTML: {self.stats['html']}")
        print(f"  - PDF: {self.stats['pdf']}")
        print(f"Imágenes descargadas: {self.stats['images']}")
        print(f"Imágenes con OCR: {self.stats['ocr_processed']}")
        print(f"Errores: {self.stats['errors']}")
        print(f"Tiempo total: {elapsed:.2f} segundos")
        print(f"\nArchivos guardados en:")
        print(f"  - Originales: {self.output_dir.absolute()}")
        print(f"  - Textos: {self.text_dir.absolute()}")
        if self.download_images:
            print(f"  - Imágenes: {self.images_dir.absolute()}")
        print(f"{'='*60}\n")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Crawler de validación - Procesa URLs específicas'
    )
    parser.add_argument(
        '--urls_file', '-f',
        type=str,
        default='validation/url_validation.txt',
        help='Archivo con URLs a procesar'
    )
    parser.add_argument(
        '--output_dir', '-o',
        type=str,
        default='validation/crawled_validation',
        help='Directorio para archivos originales'
    )
    parser.add_argument(
        '--state_dir', '-s',
        type=str,
        default='validation/validation_state',
        help='Directorio para textos y estado'
    )
    parser.add_argument(
        '--no_images',
        action='store_false',
        dest='download_images',
        help='No descargar imágenes'
    )
    parser.add_argument(
        '--no_ocr',
        action='store_false',
        dest='enable_ocr',
        help='No aplicar OCR'
    )
    
    args = parser.parse_args()
    
    # Crear crawler
    crawler = ValidationCrawler(
        output_dir=args.output_dir,
        state_dir=args.state_dir,
        download_images=args.download_images,
        enable_ocr=args.enable_ocr
    )
    
    # Procesar URLs
    results = crawler.process_urls_from_file(args.urls_file)
    
    # Guardar resultados
    import json
    results_path = Path(args.state_dir) / "processing_results.json"
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"Resultados guardados en: {results_path}")


if __name__ == "__main__":
    main()
