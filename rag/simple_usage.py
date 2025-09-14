"""
Simple RAG Usage Example
This script shows how to use the RAG system to query NCERT Science content.
"""
import logging
from pathlib import Path

# Import the query engine
from query import RAGQueryEngine

# Configure simple logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Simple example of using the RAG system."""
    
    # Path to the vector store (must be built first using rag_pipeline_huggingface.py)
    vector_store_path = Path(__file__).parent / "vector_store"
    
    if not vector_store_path.exists():
        logger.error("Vector store not found!")
        logger.error("Please run 'python rag_pipeline_huggingface.py' first to build the vector database.")
        return
    
    # Initialize the query engine
    logger.info("Loading RAG system...")
    query_engine = RAGQueryEngine(str(vector_store_path))
    
    # Example questions
    questions = [
        "What is photosynthesis?",
        "How do plants make food?",
        "What are the parts of a leaf?"
    ]
    
    # Process each question
    for question in questions:
        logger.info(f"\nQuestion: {question}")
        
        result = query_engine.answer_query(question)
        
        logger.info(f"Answer: {result['answer']}")
        logger.info(f"Number of source documents: {len(result['source_documents'])}")
        
        # Show first source snippet
        if result['source_documents']:
            first_source = result['source_documents'][0]['content']
            logger.info(f"Source snippet: {first_source[:200]}...")

if __name__ == "__main__":
    main()