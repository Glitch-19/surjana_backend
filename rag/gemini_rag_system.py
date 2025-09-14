"""
Comprehensive Gemini RAG System
Loads PDF, chunks text, creates embeddings with Google Gemini, stores in ChromaDB
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

class GeminiRAGSystem:
    def __init__(self, google_api_key: Optional[str] = None):
        """Initialize Gemini RAG system with API key"""
        self.google_api_key = google_api_key or os.getenv("GOOGLE_API_KEY")
        if not self.google_api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment or not provided")
        
        # Configure Gemini
        genai.configure(api_key=self.google_api_key)
        
        # Initialize embeddings and LLM
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=self.google_api_key
        )
        
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-pro",
            google_api_key=self.google_api_key,
            temperature=0.3,
            max_tokens=1024
        )
        
        self.vectorstore = None
        self.retriever = None
        self.rag_chain = None
        
    def load_and_process_pdf(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Load PDF and split into chunks"""
        print(f"Loading PDF: {pdf_path}")
        
        # Load PDF
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        
        print(f"Loaded {len(documents)} pages from PDF")
        
        # Split into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            separators=["\n\n", "\n", " ", ""]
        )
        
        chunks = text_splitter.split_documents(documents)
        print(f"Created {len(chunks)} text chunks")
        
        return chunks
    
    def create_embeddings_and_store(self, chunks: List[Any], persist_directory: str = "./chroma_db") -> None:
        """Create embeddings and store in ChromaDB"""
        print("Creating embeddings with Gemini...")
        
        # Create vector store
        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=persist_directory
        )
        
        print(f"Stored {len(chunks)} embeddings in ChromaDB at {persist_directory}")
        
    def setup_retriever(self, k: int = 5) -> None:
        """Setup retriever for RAG"""
        if not self.vectorstore:
            raise ValueError("Vector store not initialized. Run create_embeddings_and_store first.")
        
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": k})
        print(f"Retriever setup to return top {k} relevant chunks")
        
    def setup_rag_chain(self) -> None:
        """Setup the complete RAG chain"""
        if not self.retriever:
            raise ValueError("Retriever not setup. Run setup_retriever first.")
        
        # Define prompt template
        prompt = ChatPromptTemplate.from_template("""
You are an AI tutor specialized in NCERT curriculum. Use the provided context to answer questions clearly and accurately.

Context:
{context}

Question: {question}

Answer the question based on the context provided. If the context doesn't contain enough information, say so clearly. 
Provide explanations suitable for students and include examples when helpful.

Answer:
""")
        
        # Create RAG chain
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
        
        self.rag_chain = (
            {"context": self.retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )
        
        print("RAG chain setup complete!")
        
    def query(self, question: str) -> Dict[str, Any]:
        """Query the RAG system"""
        if not self.rag_chain:
            raise ValueError("RAG chain not setup. Run setup_rag_chain first.")
        
        print(f"Querying: {question}")
        
        # Get relevant documents
        relevant_docs = self.retriever.get_relevant_documents(question)
        
        # Generate answer
        answer = self.rag_chain.invoke(question)
        
        return {
            "question": question,
            "answer": answer,
            "source_chunks": len(relevant_docs),
            "sources": [{"page": doc.metadata.get("page", "unknown"), 
                        "content_preview": doc.page_content[:200] + "..."} 
                       for doc in relevant_docs[:3]]
        }
    
    def build_complete_system(self, pdf_path: str, persist_directory: str = "./chroma_db") -> None:
        """Build the complete RAG system from PDF"""
        print("Building complete Gemini RAG system...")
        
        # Step 1: Load and process PDF
        chunks = self.load_and_process_pdf(pdf_path)
        
        # Step 2: Create embeddings and store
        self.create_embeddings_and_store(chunks, persist_directory)
        
        # Step 3: Setup retriever
        self.setup_retriever()
        
        # Step 4: Setup RAG chain
        self.setup_rag_chain()
        
        print("✅ Complete RAG system ready!")
        
    def load_existing_vectorstore(self, persist_directory: str = "./chroma_db") -> None:
        """Load existing vector store from disk"""
        if Path(persist_directory).exists():
            print(f"Loading existing vector store from {persist_directory}")
            self.vectorstore = Chroma(
                persist_directory=persist_directory,
                embedding_function=self.embeddings
            )
            self.setup_retriever()
            self.setup_rag_chain()
            print("✅ Existing RAG system loaded!")
        else:
            raise ValueError(f"No existing vector store found at {persist_directory}")

def main():
    """Demo the Gemini RAG system"""
    
    # Paths
    PDF_PATH = r"C:\Users\91885\OneDrive\Desktop\surjana\rag\6science.pdf"
    CHROMA_DIR = r"C:\Users\91885\OneDrive\Desktop\surjana\rag\chroma_ncert_science"
    
    # Initialize system
    rag_system = GeminiRAGSystem()
    
    # Check if vector store exists
    if Path(CHROMA_DIR).exists():
        print("Found existing vector store, loading...")
        rag_system.load_existing_vectorstore(CHROMA_DIR)
    else:
        print("Creating new RAG system...")
        rag_system.build_complete_system(PDF_PATH, CHROMA_DIR)
    
    # Demo queries
    demo_questions = [
        "What is photosynthesis?",
        "Explain the parts of a plant",
        "What are the different types of food?",
        "How do plants make their food?",
        "What is the role of leaves in plants?"
    ]
    
    print("\n" + "="*50)
    print("DEMO: Querying NCERT Class 6 Science")
    print("="*50)
    
    for question in demo_questions:
        print(f"\n🤔 Q: {question}")
        try:
            result = rag_system.query(question)
            print(f"📖 A: {result['answer']}")
            print(f"📚 Sources: {result['source_chunks']} relevant chunks found")
        except Exception as e:
            print(f"❌ Error: {e}")
        print("-" * 40)
    
    print("\n✨ RAG System Demo Complete!")
    
    # Interactive mode
    print("\n🔄 Enter your questions (type 'quit' to exit):")
    while True:
        user_question = input("\nYour question: ").strip()
        if user_question.lower() in ['quit', 'exit', 'q']:
            break
        
        if user_question:
            try:
                result = rag_system.query(user_question)
                print(f"\n📖 Answer: {result['answer']}")
            except Exception as e:
                print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()