"""
RAG Query Interface
This module provides a query interface for the RAG pipeline that retrieves relevant context
and generates answers using Google's Gemini LLM and Gemini embeddings.
"""
import os
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_core.embeddings import Embeddings
import google.generativeai as genai

# Configure logging - log to file only, no console output
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('rag_query.log')
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
        
        # Configure genai with API key
        genai.configure(api_key=api_key)
        
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
                result = genai.embed_content(
                    model=self.model,
                    content=text
                )
                # Extract the embedding values
                embeddings.append(result['embedding'])
                
            except Exception as e:
                logger.error(f"Error embedding document {i}: {e}")
                # Use zero vector as fallback
                embeddings.append([0.0] * 3072)  # Gemini embeddings are 3072-dimensional
        
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
            result = genai.embed_content(
                model=self.model,
                content=text
            )
            # Extract the embedding values
            return result['embedding']
        except Exception as e:
            logger.error(f"Error embedding query: {e}")
            return [0.0] * 3072  # Gemini embeddings are 3072-dimensional

class RAGQueryEngine:
    """
    RAG Query Engine for answering questions using retrieved context from NCERT Science textbook.
    """
    
    def __init__(self, vector_store_path: str, api_key: Optional[str] = None):
        """
        Initialize the RAG Query Engine.
        
        Args:
            vector_store_path: Path to the saved FAISS vector store
            api_key: Google API key for LLM and embeddings. If None, loads from environment.
        """
        load_dotenv()
        
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("Google API key not found. Set GOOGLE_API_KEY in environment or pass api_key parameter.")
        
        logger.info("Initializing RAG Query Engine with Gemini embeddings")
        
        # Initialize embeddings (using Gemini to match the vector store)
        try:
            self.embeddings = GeminiEmbeddings(api_key=self.api_key)
            logger.info("Gemini Embeddings initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize embeddings: {e}")
            raise
        
        # Load vector store
        try:
            self.vector_store = FAISS.load_local(
                vector_store_path,
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            logger.info(f"Vector store loaded from: {vector_store_path}")
        except Exception as e:
            logger.error(f"Failed to load vector store from {vector_store_path}: {e}")
            raise
        
        # Initialize LLM
        self.llm = None
        if self.api_key:
            try:
                self.llm = ChatGoogleGenerativeAI(
                    model="models/gemini-2.0-flash",
                    google_api_key=self.api_key,
                    temperature=0.3
                )
                logger.info("Google Gemini LLM initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize Google LLM: {e}. Using retrieval-only mode.")
        else:
            logger.warning("No Google API key provided. Using retrieval-only mode.")
        
        # Setup retrieval chain
        self._setup_retrieval_chain()
    
    def _setup_retrieval_chain(self):
        """Setup the retrieval chain with custom prompt template."""
        
        # Create custom prompt template
        template = """You are an AI assistant specializing in NCERT Class 6 Science education. 
Use the following pieces of context from the NCERT Science textbook to answer the question. 
If you don't know the answer based on the context, just say that you don't know.

Context:
{context}

Question: {question}

Instructions:
1. Provide accurate, educational answers suitable for Class 6 students
2. Use simple, clear language that students can understand
3. Include relevant scientific concepts from the context
4. If possible, provide examples to help explain concepts
5. Base your answer primarily on the provided context

Answer:"""

        self.prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )
        
        # Setup retrieval chain if LLM is available
        if self.llm:
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=self.vector_store.as_retriever(search_kwargs={"k": 5}),
                chain_type_kwargs={"prompt": self.prompt},
                return_source_documents=True
            )
            logger.info("RetrievalQA chain setup completed")
        else:
            self.qa_chain = None
    
    def retrieve_context(self, query: str, k: int = 5) -> List[Document]:
        """
        Retrieve relevant context documents for a query.
        
        Args:
            query: The question or search query
            k: Number of documents to retrieve
            
        Returns:
            List of relevant documents
        """
        logger.info(f"Retrieving context for query: '{query[:50]}...', k={k}")
        
        try:
            results = self.vector_store.similarity_search(query, k=k)
            logger.info(f"Retrieved {len(results)} relevant documents")
            return results
        except Exception as e:
            logger.error(f"Error retrieving context: {e}")
            raise
    
    def answer_query(self, question: str) -> Dict[str, Any]:
        """
        Answer a question using RAG pipeline.
        
        Args:
            question: The question to answer
            
        Returns:
            Dictionary containing answer, source documents, and metadata
        """
        logger.info(f"Processing question: '{question}'")
        
        try:
            if self.qa_chain and self.llm:
                # Use full RAG pipeline with LLM
                result = self.qa_chain({"query": question})
                
                response = {
                    "question": question,
                    "answer": result["result"],
                    "source_documents": [
                        {
                            "content": doc.page_content,
                            "metadata": doc.metadata
                        }
                        for doc in result["source_documents"]
                    ],
                    "mode": "rag_with_llm"
                }
                
                logger.info("Question answered using RAG with LLM")
                
            else:
                # Retrieval-only mode
                docs = self.retrieve_context(question, k=5)
                
                # Combine context for basic response
                context = "\n\n".join([doc.page_content for doc in docs])
                
                response = {
                    "question": question,
                    "answer": f"Based on the retrieved context from NCERT Science textbook:\n\n{context[:1000]}{'...' if len(context) > 1000 else ''}",
                    "source_documents": [
                        {
                            "content": doc.page_content,
                            "metadata": doc.metadata
                        }
                        for doc in docs
                    ],
                    "mode": "retrieval_only",
                    "note": "LLM not available. Showing retrieved context only."
                }
                
                logger.info("Question processed using retrieval-only mode")
            
            return response
            
        except Exception as e:
            logger.error(f"Error answering question: {e}")
            raise
    
    def batch_query(self, questions: List[str]) -> List[Dict[str, Any]]:
        """
        Process multiple questions in batch.
        
        Args:
            questions: List of questions to process
            
        Returns:
            List of response dictionaries
        """
        logger.info(f"Processing batch of {len(questions)} questions")
        
        results = []
        for i, question in enumerate(questions):
            try:
                result = self.answer_query(question)
                results.append(result)
                logger.info(f"Processed question {i+1}/{len(questions)}")
            except Exception as e:
                logger.error(f"Error processing question {i+1}: {e}")
                results.append({
                    "question": question,
                    "answer": f"Error processing question: {e}",
                    "source_documents": [],
                    "mode": "error"
                })
        
        return results
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the vector store and query engine.
        
        Returns:
            Dictionary with statistics
        """
        try:
            # Get vector store info
            index_size = self.vector_store.index.ntotal if hasattr(self.vector_store, 'index') else 0
            
            stats = {
                "vector_store_size": index_size,
                "embedding_model": "gemini-embedding-001",
                "llm_available": self.llm is not None,
                "llm_model": "models/gemini-2.0-flash" if self.llm else None,
                "mode": "rag_with_llm" if self.llm else "retrieval_only"
            }
            
            logger.info(f"Generated statistics: {stats}")
            return stats
            
        except Exception as e:
            logger.error(f"Error generating statistics: {e}")
            return {"error": str(e)}


def main():
    """
    Interactive RAG Query Engine - Ask questions and get answers from NCERT Science textbook.
    """
    # Define paths
    current_dir = Path(__file__).parent
    vector_store_path = current_dir / "vector_store_gemini"
    
    try:
        # Check if vector store exists
        if not vector_store_path.exists():
            print(f"❌ Vector store not found at {vector_store_path}")
            print("Please run rag_pipeline_gemini.py first to create the vector store")
            return
        
        # Initialize query engine
        print("🔧 Initializing RAG Query Engine...")
        logger.info("Initializing RAG Query Engine...")
        query_engine = RAGQueryEngine(str(vector_store_path))
        
        # Get statistics (log only)
        stats = query_engine.get_statistics()
        logger.info(f"Query engine statistics: {stats}")
        
        print("✅ RAG Query Engine initialized successfully!")
        print(f"📚 Loaded {stats['vector_store_size']} documents from NCERT Science textbook")
        print(f"🤖 Using {stats['llm_model']} for answer generation")
        print("\n" + "="*60)
        print("🎓 NCERT Science RAG Query System")
        print("Ask me any question about science topics from the NCERT textbook!")
        print("Type 'quit', 'exit', or 'bye' to stop.")
        print("="*60)
        
        # Interactive query loop
        while True:
            try:
                # Get user input
                question = input("\n🤔 Your question: ").strip()
                
                # Check for exit commands
                if question.lower() in ['quit', 'exit', 'bye', 'q']:
                    print("👋 Thank you for using the RAG Query System! Goodbye!")
                    logger.info("User ended session")
                    break
                
                # Skip empty questions
                if not question:
                    print("⚠  Please enter a question.")
                    continue
                
                print("🔍 Searching for relevant information...")
                logger.info(f"User question: {question}")
                
                # Process the question
                result = query_engine.answer_query(question)
                
                # Log the full result
                logger.info(f"Question processed: {question}")
                logger.info(f"Answer: {result['answer']}")
                logger.info(f"Mode: {result['mode']}")
                logger.info(f"Sources: {len(result['source_documents'])} documents")
                logger.info("-" * 80)
                
                # Display only the answer to user
                print("\n📖 Answer:")
                print("-" * 50)
                print(result['answer'])
                print("-" * 50)
                print(f"📋 Sources: {len(result['source_documents'])} relevant documents found")
                
            except KeyboardInterrupt:
                print("\n\n👋 Session interrupted. Goodbye!")
                logger.info("Session interrupted by user")
                break
            except Exception as e:
                print(f"❌ Error processing your question: {e}")
                logger.error(f"Error processing question '{question}': {e}")
                print("Please try asking your question differently.")
        
    except Exception as e:
        print(f"❌ Failed to initialize RAG system: {e}")
        logger.error(f"Failed to run query engine: {e}")
        raise


if __name__ == "__main__":
    main()