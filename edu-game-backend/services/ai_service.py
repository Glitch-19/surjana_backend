import requests
import json
from config import Config

class AIService:
    def __init__(self):
        self.groq_api_key = Config.GROQ_API_KEY
        self.groq_url = Config.GROQ_API_URL
        self.ncert_context = Config.NCERT_CONTEXT
    
    def get_hint(self, question_data, user_attempt=None):
        """
        Generate a helpful hint for a specific question
        """
        prompt = f"""
        {self.ncert_context}
        
        Student is stuck on this Class {question_data['class']} {question_data['subject']} question:
        Chapter: {question_data['chapter']}
        Question: {question_data['question']}
        Correct Answer: {question_data['answer']}
        
        {"Student's attempt: " + user_attempt if user_attempt else "Student hasn't attempted yet."}
        
        Provide a helpful hint that:
        1. Doesn't give away the complete answer
        2. Guides them to the right approach
        3. Uses simple language appropriate for Class {question_data['class']}
        4. Includes a memory trick or mnemonic if possible
        5. Encourages them to keep trying
        
        Keep the hint under 100 words.
        """
        
        return self._call_ai_api(prompt)
    
    def get_mnemonic(self, question_data):
        """
        Generate a mnemonic or memory device for remembering the concept
        """
        prompt = f"""
        {self.ncert_context}
        
        Create a fun and memorable mnemonic for this Class {question_data['class']} {question_data['subject']} concept:
        Chapter: {question_data['chapter']}
        Question: {question_data['question']}
        Answer: {question_data['answer']}
        
        Make the mnemonic:
        1. Easy to remember for Class {question_data['class']} students
        2. Fun and engaging
        3. Directly related to the concept
        4. Under 50 words
        
        Format: Provide the mnemonic and a brief explanation of how it helps remember the concept.
        """
        
        return self._call_ai_api(prompt)
    
    def get_step_by_step_solution(self, question_data):
        """
        Generate step-by-step solution guidance
        """
        prompt = f"""
        {self.ncert_context}
        
        Break down this Class {question_data['class']} {question_data['subject']} problem into simple steps:
        Chapter: {question_data['chapter']}
        Question: {question_data['question']}
        Answer: {question_data['answer']}
        
        Provide a step-by-step approach that:
        1. Uses simple language for Class {question_data['class']} level
        2. Explains the reasoning behind each step
        3. Connects to NCERT concepts they've learned
        4. Includes tips to avoid common mistakes
        5. Builds confidence
        
        Format as numbered steps, each step under 30 words.
        """
        
        return self._call_ai_api(prompt)
    
    def get_gamified_encouragement(self, question_data, is_correct=True):
        """
        Generate gamified encouragement messages
        """
        if is_correct:
            prompt = f"""
            Create a fun, game-style success message for a Class {question_data['class']} student who just solved:
            "{question_data['question']}"
            
            Make it:
            1. Celebratory and encouraging
            2. Reference the specific subject/concept
            3. Game-themed (treasure found, level up, etc.)
            4. Under 40 words
            
            Example style: "🎉 Excellent! You've unlocked the secrets of photosynthesis! Your plant kingdom knowledge grows stronger. Ready for the next challenge, nature explorer?"
            """
        else:
            prompt = f"""
            Create a supportive, game-style encouragement message for a Class {question_data['class']} student who needs help with:
            "{question_data['question']}"
            
            Make it:
            1. Encouraging, not discouraging
            2. Motivating to try again
            3. Game-themed (keep exploring, try different path, etc.)
            4. Under 40 words
            
            Example style: "🤔 Hmm, not quite there yet, explorer! Every great scientist learns from experiments. Let's try a different approach to crack this code!"
            """
        
        return self._call_ai_api(prompt)
    
    def _call_ai_api(self, prompt):
        """
        Make API call to Groq (or fallback to a simple response)
        """
        if not self.groq_api_key:
            # Fallback response when API key is not available
            return {
                "success": False,
                "message": "AI service not configured. Please add GROQ_API_KEY to .env file.",
                "content": "Keep trying! Break the problem into smaller parts and think step by step."
            }
        
        try:
            headers = {
                "Authorization": f"Bearer {self.groq_api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "model": "llama3-8b-8192",  # Groq's fast model
                "temperature": 0.7,
                "max_tokens": 300
            }
            
            response = requests.post(self.groq_url, headers=headers, json=data, timeout=10)
            
            if response.status_code == 200:
                result = response.json()
                content = result['choices'][0]['message']['content']
                return {
                    "success": True,
                    "content": content
                }
            else:
                return {
                    "success": False,
                    "message": f"API Error: {response.status_code}",
                    "content": "Try breaking the problem into smaller steps!"
                }
                
        except Exception as e:
            return {
                "success": False,
                "message": f"Error: {str(e)}",
                "content": "Remember to use the concepts you've learned in this chapter!"
            }

# Global function for easy access
def get_ai_hint(question, options):
    """
    Simple function to get AI hint for a question with options
    """
    ai_service = AIService()
    
    # Create simplified question data for hint generation
    question_data = {
        'question': question,
        'options': options,
        'class': '6-10',
        'subject': 'General',
        'chapter': 'Mixed Topics',
        'answer': 'Check options carefully'
    }
    
    prompt = f"""
    Help a student with this question:
    Question: {question}
    Options: {', '.join(options) if isinstance(options, list) else options}
    
    Provide a helpful hint that:
    1. Doesn't give away the answer directly
    2. Guides them to think about the concept
    3. Uses encouraging language
    4. Keeps it under 60 words
    
    Example: "Think about what you learned about this topic. Look for key words in the question that connect to the concept."
    """
    
    result = ai_service._call_ai_api(prompt)
    return result.get('content', 'Take your time and think through each option carefully!')

def get_step_by_step_guidance(question, subject, student_level="beginner"):
    """
    Get detailed step-by-step guidance for solving a problem
    """
    ai_service = AIService()
    
    prompt = f"""
    As a friendly AI tutor, provide step-by-step guidance for this {subject} question:
    
    Question: {question}
    Student Level: {student_level}
    
    Break down the solution into 3-4 simple steps:
    1. First, identify what the question is asking
    2. Then, explain the key concept involved
    3. Show how to approach the problem
    4. Give a final tip for similar questions
    
    Use encouraging, game-like language. Make it feel like unlocking levels in a video game!
    Keep each step under 25 words.
    
    Example format:
    🎯 Step 1: [identification step]
    🔍 Step 2: [concept explanation]
    ⚡ Step 3: [solution approach]
    🏆 Pro Tip: [general advice]
    """
    
    result = ai_service._call_ai_api(prompt)
    return result.get('content', 'Break the problem into smaller parts and solve step by step!')

def get_memory_technique(concept, subject, grade_level):
    """
    Generate fun memory techniques and mnemonics for learning concepts
    """
    ai_service = AIService()
    
    prompt = f"""
    Create a fun, memorable learning technique for this concept:
    
    Subject: {subject}
    Grade: {grade_level}
    Concept: {concept}
    
    Provide:
    1. A catchy mnemonic or memory trick
    2. A simple rhyme or song (optional)
    3. A visual association or story
    4. A game-based way to remember it
    
    Make it:
    - Fun and engaging for students
    - Easy to remember
    - Appropriate for the grade level
    - Creative and unique
    
    Example format:
    🎵 Mnemonic: [memory device]
    📖 Story: [visual/story association]
    🎮 Game Trick: [gamified memory method]
    
    Keep total response under 150 words.
    """
    
    result = ai_service._call_ai_api(prompt)
    return result.get('content', 'Try creating a story or song to remember this concept!')

def get_gamified_explanation(question, correct_answer, subject):
    """
    Explain the answer in a fun, game-themed way
    """
    ai_service = AIService()
    
    prompt = f"""
    Explain this {subject} answer like you're a game character giving a quest reward explanation:
    
    Question: {question}
    Correct Answer: {correct_answer}
    
    Make the explanation:
    - Exciting and game-themed
    - Clear and educational
    - Celebratory (like winning a level)
    - Include power-up or skill-gained metaphors
    
    Use gaming language like:
    - "You've unlocked..."
    - "Level up! You now know..."
    - "Achievement earned..."
    - "New skill acquired..."
    
    Keep it under 100 words and make learning feel like gaming!
    """
    
    result = ai_service._call_ai_api(prompt)
    return result.get('content', f'Excellent! You have mastered this {subject} concept!')
