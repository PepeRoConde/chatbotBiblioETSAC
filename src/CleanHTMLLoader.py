from bs4 import BeautifulSoup
from langchain_core.documents import Document  




class CleanHTMLLoader:
    def __init__(self, file_path: str = None, html_content: bytes = None, encoding: str = "utf-8"):
        """Initialize CleanHTMLLoader.
        
        Args:
            file_path: Path to HTML file (optional if html_content provided)
            html_content: HTML content as bytes (optional if file_path provided)
            encoding: Encoding to use
        """
        self.file_path = file_path
        self.html_content = html_content
        self.encoding = encoding

    def load(self) -> list[Document]:
        """Load and parse HTML to extract clean text."""
        if self.html_content is not None:
            html = self.html_content.decode(self.encoding)
        elif self.file_path is not None:
            with open(self.file_path, encoding=self.encoding) as f:
                html = f.read()
        else:
            raise ValueError("Either file_path or html_content must be provided")

        soup = BeautifulSoup(html, "html.parser")

        # Extraer solo texto de <p>, <li>, <h1-h6>
        elements = soup.find_all(["p", "li", "h1", "h2", "h3", "h4", "h5", "h6"])
        texts = [el.get_text(separator=" ", strip=True) for el in elements]

        # Crear un documento por bloque de texto
        docs = [Document(page_content=text, metadata={"source_file": self.file_path or "<bytes>"}) for text in texts if text]
        return docs
    
    @staticmethod
    def extract_text_from_html(html_content: bytes, encoding: str = "utf-8") -> str:
        """Extract clean plain text from HTML content.
        
        Args:
            html_content: HTML content as bytes
            encoding: Encoding to use
            
        Returns:
            Clean plain text extracted from HTML
        """
        loader = CleanHTMLLoader(html_content=html_content, encoding=encoding)
        docs = loader.load()
        # Join all text blocks with newlines
        return '\n'.join(doc.page_content for doc in docs)
