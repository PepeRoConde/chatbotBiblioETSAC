import os
import argparse
from pathlib import Path
import numpy as np
from typing import List, Dict, Tuple
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, pipeline
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA


class LocalEmbeddings:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        """Initialize with a smaller, faster model for local use"""
        print(f"Loading embeddings model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.device = 'mps'
        self.model.to(self.device)
        
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of documents"""
        embeddings = []
        batch_size = 8  # Process in smaller batches to save memory
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            inputs = self.tokenizer(batch, padding=True, truncation=True, 
                                    return_tensors="pt", max_length=512).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Use mean pooling to get document embeddings
                attention_mask = inputs["attention_mask"]
                embeddings_batch = self.mean_pooling(outputs.last_hidden_state, attention_mask)
                embeddings.extend(embeddings_batch.cpu().numpy())
                
        return embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """Generate embedding for a query"""
        return self.embed_documents([text])[0]
    
    def mean_pooling(self, token_embeddings, attention_mask):
        """Mean pooling operation to get sentence embeddings"""
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    def __call__(self, text):
        """Make the class callable for LangChain compatibility"""
        if isinstance(text, str):
            return self.embed_query(text)
        return self.embed_documents(text)


class PDFProcessor:
    def __init__(self, pdf_folder: str):
        self.pdf_folder = pdf_folder
        self.embeddings = LocalEmbeddings()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,  # Smaller chunks for better performance
            chunk_overlap=100
        )
        self.documents = []
        self.vectorstore = None
        
    def load_pdfs(self) -> None:
        """Load all PDFs from the specified folder"""
        pdf_files = list(Path(self.pdf_folder).glob("*.pdf"))
        print(f"Found {len(pdf_files)} PDF files")
        
        for pdf_path in pdf_files:
            print(f"Processing {pdf_path}")
            loader = PyPDFLoader(str(pdf_path))
            self.documents.extend(loader.load())
            
        print(f"Loaded {len(self.documents)} pages in total")
        
    def split_documents(self) -> List:
        """Split documents into chunks"""
        print("Splitting documents into chunks...")
        return self.text_splitter.split_documents(self.documents)
    
    def create_vectorstore(self, chunks: List) -> None:
        """Create a vector store from document chunks"""
        print("Creating vector store...")
        # Create FAISS index directly from documents
        self.vectorstore = FAISS.from_documents(chunks, self.embeddings)
        
    def process(self) -> None:
        """Process all PDFs and create the vector store"""
        self.load_pdfs()
        chunks = self.split_documents()
        self.create_vectorstore(chunks)
        print("Vector store creation complete!")
        
    def save_vectorstore(self, path: str) -> None:
        """Save the vector store to disk"""
        if self.vectorstore:
            self.vectorstore.save_local(path)
            print(f"Vector store saved to {path}")
        else:
            print("No vector store to save. Run process() first.")
            
    def load_vectorstore(self, path: str) -> None:
        """Load a vector store from disk"""
        if os.path.exists(path):
            self.vectorstore = FAISS.load_local(path, self.embeddings, allow_dangerous_deserialization=True)
            print(f"Vector store loaded from {path}")
        else:
            print(f"No vector store found at {path}")


class LocalLLM:
    #def __init__(self, model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
    #def __init__(self, model_name="meta-llama/llama-2-7b-chat-hf"):
        """Initialize with a small, fast model for local use"""
        print(f"Loading LLM: {model_name}")
        self.device = 'mps'
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            torch_dtype=torch.float16,  # Quantized models often use float16 precision
            low_cpu_mem_usage=True  # This is to help with memory efficiency in local setups
        )
        self.model.to(self.device)
        
        self.pipeline = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=512,
            temperature=0.8,
            top_p=0.95,
            repetition_penalty=1.1,
            device=self.device,
            do_sample=True
        )
        
        self.llm = HuggingFacePipeline(pipeline=self.pipeline)
        

class RAGSystem:
    def __init__(
        self, 
        vectorstore,
        k: int = 4
    ):
        self.vectorstore = vectorstore
        
        # Initialize the local LLM
        self.local_llm = LocalLLM()
        self.llm = self.local_llm.llm
        
        # Create the retriever
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": k})
        
        # Create the prompt template
        template = """Usa a seguinte informacion para responder á pregunta. Se conciso, extrae informacion importante do texto. Si non sabes, di educadamente que non sabes, non intentes inventar a resposta. A resposta debe ser agradable e clara.

        Contexto:
        {context}

        Pregunta: {question}
        
        Resposta:"""
        
        self.prompt = PromptTemplate.from_template(template)
        
        # Create the QA chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": self.prompt}
        )
        
    def query(self, question: str) -> Tuple[str, List]:
        """
        Query the RAG system with a question
        Returns the answer and the source documents used
        """
        # Use invoke instead of __call__ to avoid deprecation warning
        result = self.qa_chain.invoke({"query": question})
        
        # Extract answer and sources based on the returned structure
        if "result" in result:
            answer = result["result"]
        else:
            answer = result.get("answer", "No answer found")
            
        source_docs = result.get("source_documents", [])
        
        return answer, source_docs


def main():
    parser = argparse.ArgumentParser(description='Fully Local RAG System for PDF documents')
    parser.add_argument('--pdf_folder', type=str, required=True, help='Folder containing PDF files')
    parser.add_argument('--vector_store', type=str, default='local_vectorstore', help='Path to save/load vector store')
    parser.add_argument('--rebuild', action='store_true', help='Rebuild vector store even if it exists')
    parser.add_argument('--model', type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0", 
                       help='HuggingFace model to use for language generation')
    parser.add_argument('--embedding_model', type=str, default="sentence-transformers/all-MiniLM-L6-v2",
                       help='HuggingFace model to use for embeddings')
    args = parser.parse_args()
    
    # Initialize the PDF processor
    processor = PDFProcessor(args.pdf_folder)
    
    # Check if vector store exists and if we need to rebuild
    if not os.path.exists(args.vector_store) or args.rebuild:
        processor.process()
        processor.save_vectorstore(args.vector_store)
    else:
        processor.load_vectorstore(args.vector_store)
    
    # Initialize the RAG system
    rag = RAGSystem(processor.vectorstore)
    
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
