"""
RAG Pipeline for NCERT Science Textbook with Google Gemini Embeddings
This module creates embeddings and a FAISS vector database from the NCERT Class 6 Science textbook.
Uses Google's Gemini embedding model for high-quality embeddings.
"""
import os
import logging
from typing import List, Optional
from pathlib import Path

import faiss
import numpy as np
from dotenv import load_dotenv
from google import genai
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

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

class GeminiEmbeddings(Embeddings):
    """
    Custom embeddings class using Google Gemini embedding model.
    """
    
    def __init__(self, api_key: str, model: str = "gemini-embedding-001"):
        """
        Initialize Gemini embeddings.
        
        Args:
            api_key: Google API key
            model: Gemini embedding model name
        """
        self.api_key = api_key
        self.model = model
        
        # Initialize genai client with API key
        self.client = genai.Client(api_key=api_key)
        
        logger.info(f"Gemini embeddings initialized with model: {model}")
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of documents.
        
        Args:
            texts: List of text documents to embed
            
        Returns:
            List of embeddings
        """
        logger.info(f"Embedding {len(texts)} documents")
        
        embeddings = []
        for i, text in enumerate(texts):
            try:
                result = self.client.models.embed_content(
                    model=self.model,
                    contents=text
                )
                # Extract the values from the first embedding
                embeddings.append(result.embeddings[0].values)
                
                if (i + 1) % 10 == 0:
                    logger.info(f"Embedded {i + 1}/{len(texts)} documents")
                    
            except Exception as e:
                logger.error(f"Error embedding document {i}: {e}")
                # Use zero vector as fallback
                embeddings.append([0.0] * 3072)  # Gemini embeddings are 3072-dimensional
        
        logger.info(f"Successfully embedded {len(embeddings)} documents")
        return embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """
        Embed a single query.
        
        Args:
            text: Query text to embed
            
        Returns:
            Embedding vector
        """
        try:
            result = self.client.models.embed_content(
                model=self.model,
                contents=text
            )
            # Extract the values from the first embedding
            return result.embeddings[0].values
        except Exception as e:
            logger.error(f"Error embedding query: {e}")
            return [0.0] * 3072  # Gemini embeddings are 3072-dimensional

class RAGPipeline:
    """
    RAG Pipeline for creating and managing vector embeddings from text documents.
    Uses Google Gemini embeddings for high-quality semantic understanding.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the RAG pipeline with Gemini embeddings.
        
        Args:
            api_key: Google API key for embeddings. If None, loads from environment.
        """
        load_dotenv()
        
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("Google API key not found. Set GOOGLE_API_KEY in environment or pass api_key parameter.")
        
        logger.info("Initializing RAG Pipeline with Gemini embeddings")
        
        # Initialize embeddings model
        try:
            self.embeddings = GeminiEmbeddings(api_key=self.api_key)
            logger.info("Gemini Embeddings initialized successfully")
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
    
    def similarity_search(self, query: str, k: int = 5) -> List[Document]:
        """
        Perform similarity search in the vector store.
        
        Args:
            query: Search query
            k: Number of similar documents to return
            
        Returns:
            List of similar documents
        """
        if not self.vector_store:
            raise ValueError("No vector store loaded. Load or create vector store first.")
        
        logger.info(f"Performing similarity search for query: '{query[:50]}...', k={k}")
        
        try:
            results = self.vector_store.similarity_search(query, k=k)
            logger.info(f"Found {len(results)} similar documents")
            return results
            
        except Exception as e:
            logger.error(f"Error during similarity search: {e}")
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
    vector_store_path = current_dir / "vector_store_gemini"
    
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
        
        # Test similarity search
        test_query = "What is photosynthesis?"
        results = rag.similarity_search(test_query, k=3)
        
        logger.info(f"Test search results for '{test_query}':")
        for i, doc in enumerate(results):
            logger.info(f"Result {i+1}: {doc.page_content[:100]}...")
        
    except Exception as e:
        logger.error(f"Failed to build RAG pipeline: {e}")
        raise


if __name__ == "__main__":
    main()