"""
RAG Query Interface
This module provides a query interface for the RAG pipeline that retrieves relevant context
and generates answers using Google's Gemini LLM and Gemini embeddings.
"""
import os
import logging
from ty    def generate_quiz_question(self, topic: str, context: str) -> Dict[str, Any]:
        """
        Generate a quiz question based on the topic and context to assess understanding.
        
        Args:
            topic: The main topic being discussed
            context: The context from which to generate the quiz
            
        Returns:
            Dictionary containing quiz question and correct answer
        """
        quiz_prompt = f"""Based on this content about {topic}:

{context}

Generate a simple multiple-choice question to test basic understanding of the key concept. Make it appropriate for Class 6 students.

Format your response as:
QUESTION: [Your question here]
A) [Option A]
B) [Option B] 
C) [Option C]
D) [Option D]
CORRECT: [Letter of correct answer]
EXPLANATION: [Why this answer is correct in 1-2 sentences]
"""

        try:
            if self.llm:
                response = self.llm.invoke(quiz_prompt)
                return {"quiz_content": response.content, "topic": topic}
            else:
                return {"quiz_content": "Quiz generation requires LLM connection", "topic": topic}
        except Exception as e:
            logger.error(f"Error generating quiz: {e}")
            return {"quiz_content": f"Could not generate quiz: {e}", "topic": topic}

    def evaluate_quiz_response(self, student_answer: str, correct_answer: str, topic: str, context: str) -> Dict[str, Any]:
        """
        Evaluate student's quiz response and provide adaptive feedback.
        
        Args:
            student_answer: Student's answer choice
            correct_answer: The correct answer
            topic: The topic being tested
            context: Original context for re-explanation
            
        Returns:
            Dictionary with evaluation results and adaptive feedback
        """
        is_correct = student_answer.upper().strip() == correct_answer.upper().strip()
        
        if is_correct:
            feedback_prompt = f"""The student correctly answered a quiz about {topic}. 
            
Provide encouraging feedback and suggest the next learning step. Keep it brief and motivating."""
        else:
            feedback_prompt = f"""The student incorrectly answered a quiz about {topic}. They chose {student_answer} but the correct answer was {correct_answer}.

Based on this context:
{context}

Provide a re-explanation using a DIFFERENT approach:
- Use a different analogy or example
- Break it down differently 
- Focus on the part they misunderstood
- Keep it simple and encouraging for Class 6 students

Start with "Let me explain this differently..." """

        try:
            if self.llm:
                feedback = self.llm.invoke(feedback_prompt)
                return {
                    "is_correct": is_correct,
                    "feedback": feedback.content,
                    "needs_reinforcement": not is_correct
                }
            else:
                return {
                    "is_correct": is_correct, 
                    "feedback": "Great job!" if is_correct else "Let's try again with a different explanation.",
                    "needs_reinforcement": not is_correct
                }
        except Exception as e:
            logger.error(f"Error generating feedback: {e}")
            return {
                "is_correct": is_correct,
                "feedback": "Unable to generate feedback at this time.",
                "needs_reinforcement": not is_correct
            }

    def adaptive_learning_session(self, question: str) -> Dict[str, Any]:
        """
        Conduct a full adaptive learning session with quiz and feedback.
        
        Args:
            question: Initial student question
            
        Returns:
            Dictionary containing the full learning session results
        """
        logger.info(f"Starting adaptive learning session for: '{question}'")
        
        # Step 1: Get initial answer
        initial_response = self.answer_query(question)
        
        # Step 2: Generate quiz based on the content
        if initial_response.get('source_documents'):
            context = "\n".join([doc['content'][:500] for doc in initial_response['source_documents'][:2]])
            quiz_data = self.generate_quiz_question(question, context)
        else:
            quiz_data = {"quiz_content": "No quiz available", "topic": question}
        
        return {
            "initial_answer": initial_response,
            "quiz": quiz_data,
            "session_id": f"session_{hash(question) % 10000}"
        }

    def answer_query(self, question: str) -> Dict[str, Any]:
        """
        Answer a question using the RAG pipeline.
        
        Args:
            question: The question to answer
            
        Returns:
            Dictionary containing answer and metadata
        """
        logger.info(f"Processing question: '{question}'")
        
        try:
            if self.qa_chain and self.llm:
                # Use full RAG pipeline with LLM
                result = self.qa_chain.invoke({"query": question}), Dict, Any, Optional
from pathlib import Path

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_core.embeddings import Embeddings
from google import genai

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
            result = self.client.models.embed_content(
                model=self.model,
                contents=text
            )
            # Extract the values from the first embedding
            return result.embeddings[0].values
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
        template = """You are an adaptive learning AI that specializes in breaking down complex study materials into bite-sized, digestible chunks for Class 6 Science students.

Your mission: Transform overwhelming content into engaging, manageable learning pieces that boost retention and keep students motivated.

Context from NCERT Science:
{context}

Student Question: {question}

LEARNING APPROACH:
🎯 Break It Down: Divide complex concepts into 2-3 simple, connected ideas
📝 Bite-Sized Format: Use short paragraphs, bullet points, or numbered steps
🧠 Memory Boosters: Include easy-to-remember keywords or phrases
⚡ Quick Wins: Highlight the most important point first for immediate understanding
🔄 Build Connections: Link new concepts to familiar everyday examples

RESPONSE STRUCTURE:
1. **Key Point** (1 sentence): The main idea in simple terms
2. **Mini-Explanation** (2-3 sentences): Break down the concept step-by-step  
3. **Real-Life Connection** (1 sentence): Connect to student's daily experience
4. **Memory Tip** (1 phrase): A simple way to remember this concept

Keep responses concise but complete. If the context doesn't contain enough information, say "I need more information from your textbook to give you the complete answer."

Digestible Answer:
also ask a counter quiz whether the student understood the concept. if the student answers wrong, explain the concept again in a different way.
"""

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
                result = self.qa_chain.invoke({"query": question})
                
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
                    print("⚠️  Please enter a question.")
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
                print(f"📋 Sources: {len(result['source_documents'])} relevant documents found from NCERT Science textbook")
                
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