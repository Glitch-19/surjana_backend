"""
Demo RAG System with Smaller Chunks to Work within Gemini Free Tier Limits
"""

import os
from pathlib import Path
from typing import List, Dict, Any, Optional
import google.generativeai as genai
from dotenv import load_dotenv

# LangChain imports
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Load environment variables from edu-game-backend directory
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'edu-game-backend', '.env'))

class GeminiRAGDemo:
    def __init__(self, google_api_key: Optional[str] = None):
        """Initialize Gemini RAG demo system with API key"""
        self.google_api_key = google_api_key or os.getenv("GOOGLE_API_KEY")
        if not self.google_api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment or not provided")
        
        # Configure Gemini
        genai.configure(api_key=self.google_api_key)
        
        # Initialize embeddings
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=self.google_api_key
        )
        
        # Initialize LLM
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=self.google_api_key,
            temperature=0.3
        )
        
        self.vectorstore = None
        self.retrieval_chain = None

    def load_and_chunk_pdf_sample(self, pdf_path: str, max_pages: int = 5) -> List[Dict[str, Any]]:
        """Load PDF and create a small sample of chunks to stay within quota"""
        print(f"Loading first {max_pages} pages from PDF: {pdf_path}")
        
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
            
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        
        # Take only first few pages
        sample_documents = documents[:max_pages]
        print(f"Using {len(sample_documents)} pages from PDF")
        
        # Create text splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        
        # Split documents into chunks
        chunks = text_splitter.split_documents(sample_documents)
        
        # Limit to even fewer chunks to stay within quota
        max_chunks = min(10, len(chunks))
        chunks = chunks[:max_chunks]
        
        print(f"Created {len(chunks)} text chunks for processing")
        return chunks

    def create_embeddings_and_store(self, chunks: List[Dict[str, Any]], persist_directory: str):
        """Create embeddings and store in ChromaDB"""
        print("Creating embeddings with Gemini (limited batch)...")
        
        # Create ChromaDB vector store
        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=persist_directory
        )
        
        print(f"Successfully created and stored {len(chunks)} embeddings in ChromaDB")

    def setup_retrieval_chain(self):
        """Setup the retrieval chain for Q&A"""
        if not self.vectorstore:
            raise ValueError("Vector store not initialized. Call create_embeddings_and_store first.")
        
        print("Setting up retrieval chain...")
        
        # Create retriever
        retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}
        )
        
        # Create prompt template
        prompt_template = ChatPromptTemplate.from_template("""
        You are an AI assistant helping students understand science concepts from NCERT textbooks.
        
        Use the following context to answer the question. If you cannot find the answer in the context, say so clearly.
        
        Context: {context}
        
        Question: {question}
        
        Answer: Provide a clear, educational answer based on the context above.
        """)
        
        # Create retrieval chain
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
        
        self.retrieval_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt_template
            | self.llm
            | StrOutputParser()
        )
        
        print("Retrieval chain setup complete!")

    def query(self, question: str) -> str:
        """Query the RAG system"""
        if not self.retrieval_chain:
            raise ValueError("Retrieval chain not setup. Call setup_retrieval_chain first.")
        
        print(f"\nProcessing question: {question}")
        result = self.retrieval_chain.invoke(question)
        return result

    def build_demo_system(self, pdf_path: str, persist_directory: str):
        """Build the complete demo RAG system with limited data"""
        print("Building Gemini RAG Demo System (limited for free tier)...")
        
        # Load and chunk PDF (limited)
        chunks = self.load_and_chunk_pdf_sample(pdf_path, max_pages=3)
        
        # Create embeddings and store
        self.create_embeddings_and_store(chunks, persist_directory)
        
        # Setup retrieval
        self.setup_retrieval_chain()
        
        print("✅ Demo RAG system built successfully!")

def main():
    """Main demo function"""
    # Configuration
    PDF_PATH = os.path.join(os.path.dirname(__file__), "6science.pdf")
    CHROMA_DIR = os.path.join(os.path.dirname(__file__), "chroma_db_demo")
    
    try:
        print("=== Gemini RAG Demo System ===")
        print("(Limited version for free tier quota)")
        
        # Create RAG system
        rag_system = GeminiRAGDemo()
        
        # Build the demo system
        rag_system.build_demo_system(PDF_PATH, CHROMA_DIR)
        
        # Demo questions
        demo_questions = [
            "What is photosynthesis?",
            "How do plants make their food?",
            "What are the parts of a flower?"
        ]
        
        print("\n" + "="*50)
        print("DEMO QUERIES")
        print("="*50)
        
        for question in demo_questions:
            try:
                answer = rag_system.query(question)
                print(f"\n🔍 Question: {question}")
                print(f"📚 Answer: {answer}")
                print("-" * 50)
            except Exception as e:
                print(f"Error processing question '{question}': {e}")
        
        # Interactive mode
        print("\n" + "="*50)
        print("INTERACTIVE MODE")
        print("="*50)
        print("Ask questions about the NCERT Science content!")
        print("Type 'quit' to exit")
        
        while True:
            try:
                user_question = input("\n🔍 Your question: ").strip()
                if user_question.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye!")
                    break
                
                if user_question:
                    answer = rag_system.query(user_question)
                    print(f"📚 Answer: {answer}")
                    
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")
    
    except Exception as e:
        print(f"Error initializing RAG system: {e}")
        return

if __name__ == "__main__":
    main()