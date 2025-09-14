"""
RAG Query System - Uses Existing Vector Database
Queries pre-built embeddings without creating new ones
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from typing import Optional, List
import google.generativeai as genai

# LangChain imports
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Load environment variables
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'edu-game-backend', '.env'))

class RAGQuerySystem:
    def __init__(self, google_api_key: Optional[str] = None):
        """Initialize RAG query system with existing vector database"""
        self.google_api_key = google_api_key or os.getenv("GOOGLE_API_KEY")
        if not self.google_api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment")
        
        # Configure Gemini
        genai.configure(api_key=self.google_api_key)
        
        # Initialize embeddings (for queries only)
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

    def load_existing_vectorstore(self, persist_directory: str):
        """Load existing ChromaDB vector store"""
        if not os.path.exists(persist_directory):
            raise FileNotFoundError(f"Vector database not found: {persist_directory}")
        
        print(f"Loading existing vector database from: {persist_directory}")
        
        # Load existing ChromaDB
        self.vectorstore = Chroma(
            persist_directory=persist_directory,
            embedding_function=self.embeddings
        )
        
        # Get collection info
        try:
            collection = self.vectorstore._collection
            count = collection.count()
            print(f"✅ Loaded vector database with {count} documents")
            return True
        except Exception as e:
            print(f"❌ Error accessing vector database: {e}")
            return False

    def setup_retrieval_chain(self):
        """Setup the retrieval chain for Q&A"""
        if not self.vectorstore:
            raise ValueError("Vector store not loaded. Call load_existing_vectorstore first.")
        
        print("Setting up retrieval chain...")
        
        # Create retriever
        retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4}
        )
        
        # Create prompt template
        prompt_template = ChatPromptTemplate.from_template("""
        You are an AI assistant helping students understand science concepts from NCERT Class 6 Science textbook.
        
        Use the following context from the textbook to answer the question. Provide a clear, educational answer based on the context.
        If the context doesn't contain enough information to answer fully, say so and provide what information you can.
        
        Context from NCERT Science textbook:
        {context}
        
        Student Question: {question}
        
        Educational Answer: Provide a clear, student-friendly explanation based on the textbook content above.
        """)
        
        # Create retrieval chain
        def format_docs(docs):
            return "\n\n".join([f"Page {i+1}: {doc.page_content}" for i, doc in enumerate(docs)])
        
        self.retrieval_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt_template
            | self.llm
            | StrOutputParser()
        )
        
        print("✅ Retrieval chain setup complete!")

    def query(self, question: str) -> str:
        """Query the RAG system"""
        if not self.retrieval_chain:
            raise ValueError("Retrieval chain not setup. Call setup_retrieval_chain first.")
        
        print(f"\n🔍 Processing: {question}")
        
        try:
            result = self.retrieval_chain.invoke(question)
            return result
        except Exception as e:
            return f"Error processing query: {e}"

def run_demo_queries():
    """Run demo queries on existing vector database"""
    
    # Try to find existing vector databases
    possible_dbs = [
        os.path.join(os.path.dirname(__file__), "chroma_ncert_science"),
        os.path.join(os.path.dirname(__file__), "chroma_db_demo"),
        os.path.join(os.path.dirname(__file__), "chroma_db")
    ]
    
    vectordb_path = None
    for db_path in possible_dbs:
        if os.path.exists(db_path):
            vectordb_path = db_path
            break
    
    if not vectordb_path:
        print("❌ No existing vector database found!")
        print("Available directories:")
        for db_path in possible_dbs:
            print(f"  - {db_path} {'✅' if os.path.exists(db_path) else '❌'}")
        return
    
    try:
        print("="*60)
        print("RAG QUERY SYSTEM - DEMO")
        print("="*60)
        
        # Initialize system
        rag_system = RAGQuerySystem()
        
        # Load existing database
        if not rag_system.load_existing_vectorstore(vectordb_path):
            print("Failed to load vector database")
            return
        
        # Setup retrieval
        rag_system.setup_retrieval_chain()
        
        # Demo questions about NCERT Science
        demo_questions = [
            "What is photosynthesis?",
            "How do plants make their food?", 
            "What are the parts of a plant?",
            "What do plants need to grow?",
            "How do leaves help plants?",
            "What is the function of roots?",
            "How do plants get water?",
            "What is the importance of sunlight for plants?"
        ]
        
        print("\n" + "="*60)
        print("DEMO QUERIES - NCERT SCIENCE QUESTIONS")
        print("="*60)
        
        for i, question in enumerate(demo_questions, 1):
            try:
                print(f"\n📚 Question {i}: {question}")
                answer = rag_system.query(question)
                print(f"🤖 Answer: {answer}")
                print("-" * 50)
            except Exception as e:
                print(f"❌ Error with question {i}: {e}")
                continue
        
        # Interactive mode
        print("\n" + "="*60)
        print("INTERACTIVE MODE")
        print("="*60)
        print("Ask your own questions about NCERT Science!")
        print("Type 'quit' to exit")
        
        while True:
            try:
                user_question = input("\n🔍 Your question: ").strip()
                if user_question.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye! 👋")
                    break
                
                if user_question:
                    answer = rag_system.query(user_question)
                    print(f"🤖 Answer: {answer}")
                    
            except KeyboardInterrupt:
                print("\nGoodbye! 👋")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
    
    except Exception as e:
        print(f"❌ System error: {e}")

if __name__ == "__main__":
    run_demo_queries()