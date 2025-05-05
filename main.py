import os
import argparse
from pathlib import Path
import numpy as np
from typing import List, Dict, Tuple, Optional, Union, Any
import torch
from bs4 import BeautifulSoup
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, pipeline
from langchain_community.document_loaders import PyPDFLoader, BSHTMLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document

from LocalEmbeddings import LocalEmbeddings
from DocumentProcessor import DocumentProcessor
from RAGSystem import RAGSystem 


        



def main():
    """Main function to run the RAG system."""
    parser = argparse.ArgumentParser(description='Fully Local RAG System for PDF and HTML documents')
    parser.add_argument('--docs_folder', type=str, help='Folder containing PDF and HTML files', default='/Users/pepe/OneDrive - Universidade da Coruña/documentacion_y_normativa')
    parser.add_argument('--vector_store', type=str, default='local_vectorstore', help='Path to save/load vector store')
    parser.add_argument('--rebuild', action='store_true', help='Rebuild vector store even if it exists')
    parser.add_argument('--language', type=str, default='english', choices=['english', 'spanish', 'galician'], 
                       help='Language for the prompt template')
    parser.add_argument('--chunk_size', type=int, default=300, help='Size of text chunks')
    parser.add_argument('--chunk_overlap', type=int, default=15, help='Overlap between chunks')
    parser.add_argument('--k', type=int, default=4, help='Number of documents to retrieve')
    parser.add_argument('--model', type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0", 
                       help='HuggingFace model to use for language generation')
    parser.add_argument('--embedding_model', type=str, default="sentence-transformers/all-MiniLM-L6-v2",
                       help='HuggingFace model to use for embeddings')
    args = parser.parse_args()
    
    # Initialize the document processor
    processor = DocumentProcessor(
        docs_folder=args.docs_folder,
        embedding_model_name=args.embedding_model,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap
    )
    
    # Check if vector store exists and if we need to rebuild
    if not os.path.exists(args.vector_store) or args.rebuild:
        processor.process()
        processor.save_vectorstore(args.vector_store)
    else:
        processor.load_vectorstore(args.vector_store)
    
    # Initialize the RAG system
    rag = RAGSystem(
        vectorstore=processor.vectorstore,
        k=args.k,
        language=args.language
    )
    
    # Interactive query loop
    print("\nLocal RAG System Ready! Type 'exit' to quit.")
    while True:
        question = input("\nEnter your question: ")
        if question.lower() == 'exit':
            break
            
        try:
            answer, sources = rag.query(question)
            
            print("\nAnswer:", answer)
            print("\nSources:")
            for i, doc in enumerate(sources[:3]):  # Show top 3 sources
                print(f"Source {i+1}:\n{doc.page_content[:200]}...\n")
        except Exception as e:
            print(f"Error processing query: {e}")


if __name__ == "__main__":
    main()
