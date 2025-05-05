from typing import List, Dict, Tuple, Optional, Union, Any
from pathlib import Path
import os
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, BSHTMLLoader
from LocalEmbeddings import LocalEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

class DocumentProcessor:
    """Class for processing multiple document types (PDF, HTML)."""
    
    def __init__(
        self, 
        docs_folder: str,
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        chunk_size: int = 300,
        chunk_overlap: int = 15
    ):
        """Initialize document processor.
        
        Args:
            docs_folder: Folder containing documents
            embedding_model_name: Name of embedding model
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
        """
        self.docs_folder = Path(docs_folder)
        self.embeddings = LocalEmbeddings(model_name=embedding_model_name)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        self.documents = []
        self.vectorstore = None
        
    def load_documents(self) -> None:
        """Load all PDFs and HTML files from the specified folder."""
        pdf_files = list(self.docs_folder.glob("*.pdf"))
        html_files = list(self.docs_folder.glob("*.html")) + list(self.docs_folder.glob("*.htm"))
        
        print(f"Found {len(pdf_files)} PDF files and {len(html_files)} HTML files")
        
        # Process PDF files
        for pdf_path in pdf_files:
            try:
                print(f"Processing PDF: {pdf_path}")
                loader = PyPDFLoader(str(pdf_path))
                self.documents.extend(loader.load())
            except Exception as e:
                print(f"Error processing PDF {pdf_path}: {e}")
        
        # Process HTML files
        for html_path in html_files:
            try:
                print(f"Processing HTML: {html_path}")
                loader = BSHTMLLoader(str(html_path))
                self.documents.extend(loader.load())
            except Exception as e:
                print(f"Error processing HTML {html_path}: {e}")
            
        print(f"Loaded {len(self.documents)} documents in total")
        
    def split_documents(self) -> List:
        """Split documents into chunks.
        
        Returns:
            List of document chunks
        """
        print("Splitting documents into chunks...")
        return self.text_splitter.split_documents(self.documents)
    
    def create_vectorstore(self, chunks: List) -> None:
        """Create a vector store from document chunks.
        
        Args:
            chunks: List of document chunks
        """
        print("Creating vector store...")
        self.vectorstore = FAISS.from_documents(chunks, self.embeddings)
        
    def process(self) -> None:
        """Process all documents and create the vector store."""
        self.load_documents()
        chunks = self.split_documents()
        self.create_vectorstore(chunks)
        print("Vector store creation complete!")
        
    def save_vectorstore(self, path: str) -> None:
        """Save the vector store to disk.
        
        Args:
            path: Path to save the vector store
        """
        if self.vectorstore:
            self.vectorstore.save_local(path)
            print(f"Vector store saved to {path}")
        else:
            print("No vector store to save. Run process() first.")
            
    def load_vectorstore(self, path: str) -> None:
        """Load a vector store from disk.
        
        Args:
            path: Path to load the vector store from
        """
        if os.path.exists(path):
            self.vectorstore = FAISS.load_local(path, self.embeddings, allow_dangerous_deserialization=True)
            print(f"Vector store loaded from {path}")
        else:
            print(f"No vector store found at {path}")
