"""
Adaptive RAG Query Interface
This module provides an adaptive learning system that assesses student understanding
and adjusts explanations based on quiz responses.
"""
import os
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
import hashlib

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_core.embeddings import Embeddings
from google import genai

# Configure logging to file only (no console output)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('adaptive_rag.log')
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
            texts: List of document texts to embed
            
        Returns:
            List of embedding vectors
        """
        logger.info(f"Embedding {len(texts)} documents")
        embeddings = []
        
        for i, text in enumerate(texts):
            try:
                result = self.client.models.embed_content(
                    model=self.model,
                    content=text
                )
                embeddings.append(result.embeddings[0].values)
                
                if (i + 1) % 10 == 0:
                    logger.info(f"Embedded {i + 1}/{len(texts)} documents")
                    
            except Exception as e:
                logger.error(f"Error embedding document {i}: {e}")
                # Use zero vector as fallback
                embeddings.append([0.0] * 3072)  # Gemini embeddings are 3072-dimensional
        
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """
        Embed a single query text.
        
        Args:
            text: Query text to embed
            
        Returns:
            Embedding vector
        """
        try:
            result = self.client.models.embed_content(
                model=self.model,
                content=text
            )
            return result.embeddings[0].values
        except Exception as e:
            logger.error(f"Error embedding query '{text}': {e}")
            return [0.0] * 3072  # Gemini embeddings are 3072-dimensional

class AdaptiveRAGEngine:
    """
    Adaptive RAG Query Engine that assesses understanding and adapts explanations.
    """
    
    def __init__(self, vector_store_path: str, api_key: Optional[str] = None):
        """
        Initialize the Adaptive RAG Engine.
        
        Args:
            vector_store_path: Path to the saved FAISS vector store
            api_key: Google API key for LLM and embeddings. If None, loads from environment.
        """
        load_dotenv()
        
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("Google API key not found. Set GOOGLE_API_KEY in environment or pass api_key parameter.")
        
        logger.info("Initializing Adaptive RAG Engine with Gemini embeddings")
        
        # Initialize embeddings
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
        
        # Create custom prompt template for adaptive learning
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

Digestible Answer:"""

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
            logger.warning("QA chain not available - LLM initialization failed")

    def generate_quiz_question(self, topic: str, context: str) -> Dict[str, Any]:
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

Format your response EXACTLY as shown:
QUESTION: [Your question here]
A) [Option A]
B) [Option B] 
C) [Option C]
D) [Option D]
CORRECT: [Letter of correct answer]
EXPLANATION: [Why this answer is correct in 1-2 sentences]"""

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
            student_answer: Student's answer choice (A, B, C, or D)
            correct_answer: The correct answer
            topic: The topic being tested
            context: Original context for re-explanation
            
        Returns:
            Dictionary with evaluation results and adaptive feedback
        """
        is_correct = student_answer.upper().strip() == correct_answer.upper().strip()
        
        if is_correct:
            feedback_prompt = f"""The student correctly answered a quiz about {topic}. 

Provide encouraging feedback and suggest the next learning step. Keep it brief and motivating for Class 6 students."""
        else:
            feedback_prompt = f"""The student incorrectly answered a quiz about {topic}. They chose {student_answer} but the correct answer was {correct_answer}.

Based on this context:
{context}

Provide a re-explanation using a DIFFERENT approach than before:
- Use a different analogy or example
- Break it down differently 
- Focus on the part they misunderstood
- Keep it simple and encouraging for Class 6 students
- Use everyday examples they can relate to

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
                    "feedback": "Great job! 🎉" if is_correct else "Let's try again with a different explanation.",
                    "needs_reinforcement": not is_correct
                }
        except Exception as e:
            logger.error(f"Error generating feedback: {e}")
            return {
                "is_correct": is_correct,
                "feedback": "Unable to generate feedback at this time.",
                "needs_reinforcement": not is_correct
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
                return response
                
            else:
                # Fallback to similarity search only
                docs = self.vector_store.similarity_search(question, k=5)
                
                response = {
                    "question": question,
                    "answer": "LLM not available. Here are relevant excerpts from the textbook:\n\n" + 
                             "\n\n---\n\n".join([doc.page_content for doc in docs]),
                    "source_documents": [
                        {
                            "content": doc.page_content,
                            "metadata": doc.metadata
                        }
                        for doc in docs
                    ],
                    "mode": "retrieval_only"
                }
                
                logger.info("Question answered using retrieval only")
                return response
                
        except Exception as e:
            logger.error(f"Error answering question: {e}")
            return {
                "question": question,
                "answer": f"Error processing your question: {e}",
                "source_documents": [],
                "mode": "error"
            }

    def adaptive_learning_session(self, question: str) -> Dict[str, Any]:
        """
        Conduct a full adaptive learning session with initial answer and quiz.
        
        Args:
            question: Initial student question
            
        Returns:
            Dictionary containing the learning session data
        """
        logger.info(f"Starting adaptive learning session for: '{question}'")
        
        # Step 1: Get initial answer
        initial_response = self.answer_query(question)
        
        # Step 2: Generate quiz based on the content
        if initial_response.get('source_documents') and len(initial_response['source_documents']) > 0:
            # Use first 2 source documents for context
            context_parts = []
            for doc in initial_response['source_documents'][:2]:
                context_parts.append(doc['content'][:400])  # Limit context size
            
            context = "\n".join(context_parts)
            quiz_data = self.generate_quiz_question(question, context)
        else:
            quiz_data = {
                "quiz_content": "No quiz available - insufficient context found", 
                "topic": question
            }
        
        session_id = str(abs(hash(question)) % 100000)
        
        return {
            "session_id": session_id,
            "initial_answer": initial_response,
            "quiz": quiz_data,
            "status": "awaiting_quiz_response"
        }

def main():
    """Main function to run the adaptive RAG query interface."""
    
    print("🚀 Welcome to the Adaptive Learning Assistant!")
    print("I'll help you understand science concepts step by step.")
    print("After each explanation, I'll ask a quick question to check your understanding.")
    print("Type 'quit' to exit.\n")
    
    try:
        # Initialize query engine
        print("🔧 Initializing learning system...")
        
        vector_store_path = Path(__file__).parent / "vector_store_gemini"
        query_engine = AdaptiveRAGEngine(str(vector_store_path))
        
        print("✅ System ready! Let's start learning!\n")
        logger.info("Adaptive RAG system initialized successfully")
        
        # Store active sessions for quiz follow-ups
        active_sessions = {}
        
        # Interactive learning loop
        while True:
            try:
                # Check if we're expecting a quiz answer
                if active_sessions:
                    print("\n📝 Please answer the quiz question above first!")
                    print("Enter your choice (A, B, C, or D):")
                    student_answer = input("Your answer: ").strip().upper()
                    
                    if student_answer in ['A', 'B', 'C', 'D']:
                        # Process the quiz answer
                        session_data = list(active_sessions.values())[0]
                        session_id = list(active_sessions.keys())[0]
                        
                        # Extract correct answer from quiz content
                        quiz_content = session_data['quiz']['quiz_content']
                        correct_answer = "A"  # Default fallback
                        
                        try:
                            # Parse the quiz content to find correct answer
                            for line in quiz_content.split('\n'):
                                if line.startswith('CORRECT:'):
                                    correct_answer = line.split(':')[1].strip()
                                    break
                        except:
                            pass
                        
                        # Get context for re-explanation if needed
                        context = ""
                        if session_data['initial_answer']['source_documents']:
                            context = session_data['initial_answer']['source_documents'][0]['content'][:500]
                        
                        # Evaluate the response
                        evaluation = query_engine.evaluate_quiz_response(
                            student_answer, 
                            correct_answer, 
                            session_data['quiz']['topic'], 
                            context
                        )
                        
                        # Provide feedback
                        print("\n" + "="*50)
                        if evaluation['is_correct']:
                            print("🎉 Correct! Well done!")
                        else:
                            print(f"❌ Not quite right. The correct answer was {correct_answer}.")
                        
                        print("\n💡 Feedback:")
                        print("-" * 30)
                        print(evaluation['feedback'])
                        print("="*50)
                        
                        # Log the interaction
                        logger.info(f"Quiz completed - Student: {student_answer}, Correct: {correct_answer}, Result: {evaluation['is_correct']}")
                        
                        # Clear the session
                        active_sessions = {}
                        continue
                    
                    elif student_answer.lower() in ['quit', 'exit', 'q']:
                        break
                    else:
                        print("⚠️ Please enter A, B, C, or D (or 'quit' to exit)")
                        continue
                
                # Get new question from user
                question = input("\n🤔 What would you like to learn about? ").strip()
                
                # Check for exit commands
                if question.lower() in ['quit', 'exit', 'bye', 'q']:
                    print("👋 Thank you for learning with us! Keep exploring science! 🔬")
                    logger.info("User ended session")
                    break
                
                # Skip empty questions
                if not question:
                    print("⚠️ Please enter a question.")
                    continue
                
                print("\n🔍 Let me find the best explanation for you...")
                logger.info(f"User question: {question}")
                
                # Start adaptive learning session
                session = query_engine.adaptive_learning_session(question)
                
                # Display the initial answer
                print("\n📖 Here's what I found:")
                print("="*50)
                print(session['initial_answer']['answer'])
                print("="*50)
                
                # Display quiz if available
                if 'quiz_content' in session['quiz'] and 'Quiz generation requires LLM' not in session['quiz']['quiz_content']:
                    print("\n🧠 Quick Understanding Check:")
                    print("-" * 40)
                    print(session['quiz']['quiz_content'])
                    print("-" * 40)
                    
                    # Store session for quiz processing
                    active_sessions[session['session_id']] = session
                    
                else:
                    print(f"\n📚 Source: Found {len(session['initial_answer']['source_documents'])} relevant sections from NCERT Science textbook")
                
                # Log the session
                logger.info(f"Session {session['session_id']} started for question: {question}")
                
            except KeyboardInterrupt:
                print("\n\n👋 Learning session interrupted. Goodbye!")
                logger.info("Session interrupted by user")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                logger.error(f"Error in main loop: {e}")
                print("Please try asking your question differently.")
        
    except Exception as e:
        print(f"❌ Failed to initialize learning system: {e}")
        logger.error(f"Failed to run adaptive query engine: {e}")
        raise


if __name__ == "__main__":
    main()