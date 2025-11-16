from bs4 import BeautifulSoup
from langchain_core.documents import Document  




class CleanHTMLLoader:
    def __init__(self, file_path: str, encoding: str = "utf-8"):
        self.file_path = file_path
        self.encoding = encoding

    def load(self) -> list[Document]:
        with open(self.file_path, encoding=self.encoding) as f:
            html = f.read()

        soup = BeautifulSoup(html, "html.parser")

        # Extraer solo texto de <p>, <li>, <h1-h6>
        elements = soup.find_all(["p", "li", "h1", "h2", "h3", "h4", "h5", "h6"])
        texts = [el.get_text(separator=" ", strip=True) for el in elements]

        # Crear un documento por bloque de texto
        docs = [Document(page_content=text, metadata={"source_file": self.file_path}) for text in texts if text]
        return docs
