from bs4 import BeautifulSoup
from langchain_core.documents import Document


class CleanHTMLLoader:
    def __init__(self, file_path: str = None, html_content: bytes = None, encoding: str = "utf-8"):
        """
        Args:
            file_path: Path to HTML file (optional if html_content provided)
            html_content: HTML content as bytes (optional if file_path provided)
            encoding: Encoding to use
        """
        self.file_path = file_path
        self.html_content = html_content
        self.encoding = encoding

    def load(self) -> list[Document]:
        if self.html_content is not None:
            html = self.html_content.decode(self.encoding)
        elif self.file_path is not None:
            with open(self.file_path, encoding=self.encoding) as f:
                html = f.read()
        else:
            raise ValueError("Either file_path or html_content must be provided")

        soup = BeautifulSoup(html, "html.parser")

        texts = []

        # -----------------------------
        # 1. Texto normal (p, h*, li)
        # -----------------------------
        for el in soup.find_all(["p", "li", "h1", "h2", "h3", "h4", "h5", "h6"]):
            txt = el.get_text(" ", strip=True)
            if txt:
                texts.append(txt)

        # -----------------------------
        # 2. Tablas (fila completa)
        # -----------------------------
        for table in soup.find_all("table"):
            for row in table.find_all("tr"):
                cells = [
                    cell.get_text(" ", strip=True)
                    for cell in row.find_all(["th", "td"])
                ]
                if cells:
                    texts.append(" | ".join(cells))

        # -----------------------------
        # 3. Crear documentos
        # -----------------------------
        docs = [
            Document(
                page_content=text,
                metadata={
                    "source_file": self.file_path or "<bytes>",
                    "type": "html"
                }
            )
            for text in texts
        ]

        return docs

    @staticmethod
    def extract_text_from_html(html_content: bytes, encoding: str = "utf-8") -> str:
        loader = CleanHTMLLoader(html_content=html_content, encoding=encoding)
        docs = loader.load()
        return "\n".join(doc.page_content for doc in docs)
