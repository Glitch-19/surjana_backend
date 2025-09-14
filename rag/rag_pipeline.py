"""
RAG Pipeline for NCERT Science Textbook
This module creates embeddings and a FAISS vector database from the NCERT Class 6 Science textbook.
"""
import os
import logging
from typing import List, Optional
from pathlib import Path

import faiss
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('rag_pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class RAGPipeline:
    """
    RAG Pipeline for creating and managing vector embeddings from text documents.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the RAG pipeline with Google API credentials.
        
        Args:
            api_key: Google API key for embeddings. If None, loads from environment.
        """
        load_dotenv()
        
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("Google API key not found. Set GOOGLE_API_KEY in environment or pass api_key parameter.")
        
        logger.info("Initializing RAG Pipeline")
        
        # Initialize embeddings model
        try:
            self.embeddings = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001",
                google_api_key=self.api_key
            )
            logger.info("Google Generative AI Embeddings initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize embeddings model: {e}")
            raise
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        logger.info("Text splitter initialized with chunk_size=1000, chunk_overlap=200")
        
        self.vector_store: Optional[FAISS] = None
        
    def load_and_process_document(self, file_path: str) -> List[Document]:
        """
        Load and process a text document into chunks.
        
        Args:
            file_path: Path to the text file to process
            
        Returns:
            List of Document chunks
        """
        logger.info(f"Loading document from: {file_path}")
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        try:
            # Load the document
            loader = TextLoader(file_path, encoding='utf-8')
            documents = loader.load()
            logger.info(f"Loaded document with {len(documents)} pages")
            
            # Split into chunks
            chunks = self.text_splitter.split_documents(documents)
            logger.info(f"Created {len(chunks)} chunks from document")
            
            return chunks
            
        except Exception as e:
            logger.error(f"Error processing document: {e}")
            raise
    
    def create_vector_store(self, documents: List[Document]) -> FAISS:
        """
        Create FAISS vector store from document chunks.
        
        Args:
            documents: List of document chunks to embed
            
        Returns:
            FAISS vector store
        """
        logger.info(f"Creating vector store from {len(documents)} documents")
        
        try:
            # Create embeddings and FAISS vector store
            vector_store = FAISS.from_documents(
                documents=documents,
                embedding=self.embeddings
            )
            logger.info("FAISS vector store created successfully")
            
            self.vector_store = vector_store
            return vector_store
            
        except Exception as e:
            logger.error(f"Error creating vector store: {e}")
            raise
    
    def save_vector_store(self, save_path: str) -> None:
        """
        Save the FAISS vector store to disk.
        
        Args:
            save_path: Directory path to save the vector store
        """
        if not self.vector_store:
            raise ValueError("No vector store to save. Create vector store first.")
        
        logger.info(f"Saving vector store to: {save_path}")
        
        try:
            os.makedirs(save_path, exist_ok=True)
            self.vector_store.save_local(save_path)
            logger.info("Vector store saved successfully")
            
        except Exception as e:
            logger.error(f"Error saving vector store: {e}")
            raise
    
    def load_vector_store(self, load_path: str) -> FAISS:
        """
        Load FAISS vector store from disk.
        
        Args:
            load_path: Directory path to load the vector store from
            
        Returns:
            Loaded FAISS vector store
        """
        logger.info(f"Loading vector store from: {load_path}")
        
        try:
            vector_store = FAISS.load_local(
                load_path,
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            self.vector_store = vector_store
            logger.info("Vector store loaded successfully")
            
            return vector_store
            
        except Exception as e:
            logger.error(f"Error loading vector store: {e}")
            raise
    
    def build_rag_pipeline(self, text_file_path: str, save_path: str) -> FAISS:
        """
        Complete RAG pipeline: load document, create embeddings, save vector store.
        
        Args:
            text_file_path: Path to the input text file
            save_path: Directory to save the vector store
            
        Returns:
            Created FAISS vector store
        """
        logger.info("Starting complete RAG pipeline build")
        
        # Load and process document
        documents = self.load_and_process_document(text_file_path)
        
        # Create vector store
        vector_store = self.create_vector_store(documents)
        
        # Save vector store
        self.save_vector_store(save_path)
        
        logger.info("RAG pipeline build completed successfully")
        return vector_store


def main():
    """
    Main function to build the RAG pipeline for NCERT Science textbook.
    """
    # Define file paths
    current_dir = Path(__file__).parent
    text_file_path = current_dir / "ncert_class6_science.txt"
    vector_store_path = current_dir / "vector_store"
    
    try:
        # Initialize RAG pipeline
        rag = RAGPipeline()
        
        # Build the complete pipeline
        vector_store = rag.build_rag_pipeline(
            text_file_path=str(text_file_path),
            save_path=str(vector_store_path)
        )
        
        logger.info("RAG pipeline setup completed successfully!")
        logger.info(f"Vector store saved to: {vector_store_path}")
        
    except Exception as e:
        logger.error(f"Failed to build RAG pipeline: {e}")
        raise


if __name__ == "__main__":
    main()