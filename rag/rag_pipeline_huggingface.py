"""
RAG Pipeline for NCERT Science Textbook with HuggingFace Embeddings
This module creates embeddings and a FAISS vector database from the NCERT Class 6 Science textbook.
Uses local sentence-transformers for embeddings to avoid API key issues.
"""
import os
import logging
from typing import List, Optional
from pathlib import Path

import faiss
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
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
    Uses HuggingFace sentence-transformers for local embeddings.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the RAG pipeline with HuggingFace embeddings.
        
        Args:
            model_name: HuggingFace model name for embeddings
        """
        logger.info("Initializing RAG Pipeline with HuggingFace embeddings")
        
        # Initialize embeddings model
        try:
            self.embeddings = HuggingFaceEmbeddings(
                model_name=f"sentence-transformers/{model_name}",
                model_kwargs={'device': 'cpu'}
            )
            logger.info(f"HuggingFace Embeddings initialized successfully with model: {model_name}")
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