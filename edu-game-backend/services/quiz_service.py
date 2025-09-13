import re
from difflib import SequenceMatcher

class QuizService:
    def __init__(self):
        self.difficulty_multipliers = {
            'easy': 1.0,
            'medium': 1.5,
            'hard': 2.0
        }
        
        self.subject_multipliers = {
            'Mathematics': 1.2,
            'Science': 1.1,
            'History': 1.0
        }
    
    def check_answer(self, question, user_answer):
        """
        Check if user answer is correct and calculate accuracy
        """
        correct_answer = question['answer'].strip()
        user_answer = user_answer.strip()
        
        # Handle different question types
        if question['type'] == 'mcq' and 'options' in question:
            # For MCQ, exact match required
            is_correct = user_answer.lower() == correct_answer.lower()
            accuracy = 1.0 if is_correct else 0.0
            
        elif question['type'] == 'theory':
            # For theory questions, use similarity matching
            accuracy = self._calculate_text_similarity(correct_answer, user_answer)
            is_correct = accuracy >= 0.6  # 60% similarity threshold
            
        else:
            # Default: exact match
            is_correct = user_answer.lower().strip() == correct_answer.lower().strip()
            accuracy = 1.0 if is_correct else 0.0
        
        # Generate feedback
        feedback = self._generate_feedback(is_correct, accuracy, question)
        
        return {
            'is_correct': is_correct,
            'accuracy': accuracy,
            'feedback': feedback
        }
    
    def _calculate_text_similarity(self, correct, user_input):
        """
        Calculate similarity between correct answer and user input
        """
        # Clean and normalize text
        correct = re.sub(r'[^\w\s]', '', correct.lower())
        user_input = re.sub(r'[^\w\s]', '', user_input.lower())
        
        # Use sequence matcher for similarity
        similarity = SequenceMatcher(None, correct, user_input).ratio()
        
        # Check for key terms
        correct_words = set(correct.split())
        user_words = set(user_input.split())
        
        if len(correct_words) > 0:
            common_words = correct_words.intersection(user_words)
            keyword_score = len(common_words) / len(correct_words)
            
            # Combine similarity and keyword score
            final_score = (similarity * 0.6) + (keyword_score * 0.4)
        else:
            final_score = similarity
        
        return min(final_score, 1.0)
    
    def calculate_score(self, question, accuracy):
        """
        Calculate score based on difficulty, subject, and accuracy
        """
        base_score = 100
        
        # Apply difficulty multiplier
        difficulty = question.get('difficulty', 'medium')
        difficulty_mult = self.difficulty_multipliers.get(difficulty, 1.0)
        
        # Apply subject multiplier
        subject = question.get('subject', '')
        subject_mult = self.subject_multipliers.get(subject, 1.0)
        
        # Calculate final score
        score = int(base_score * accuracy * difficulty_mult * subject_mult)
        
        return score
    
    def _generate_feedback(self, is_correct, accuracy, question):
        """
        Generate contextual feedback based on performance
        """
        if is_correct:
            if accuracy >= 0.9:
                return "🌟 Excellent! Perfect understanding of the concept!"
            elif accuracy >= 0.8:
                return "✅ Great job! You got the main idea right!"
            else:
                return "👍 Good work! You're on the right track!"
        
        else:
            if accuracy >= 0.4:
                return f"🤔 Close! You have some understanding. Review the concept of {question.get('chapter', 'this topic')} again."
            elif accuracy >= 0.2:
                return f"📚 You're getting there! Focus on the key concepts in {question.get('chapter', 'this chapter')}."
            else:
                return f"💪 Keep learning! Review {question.get('chapter', 'this topic')} and try again."
    
    def calculate_tokens_earned(self, score, streak_count=0):
        """
        Calculate tokens earned based on score and streak
        """
        base_tokens = max(1, score // 50)  # 1 token per 50 points
        
        # Streak bonus
        streak_bonus = min(streak_count // 3, 5)  # Max 5 bonus tokens
        
        total_tokens = base_tokens + streak_bonus
        
        return total_tokens
    
    def get_achievement_status(self, user_stats):
        """
        Check for achievements based on user statistics
        """
        achievements = []
        
        # Score-based achievements
        if user_stats.get('total_score', 0) >= 1000:
            achievements.append({
                'name': 'Scholar',
                'description': 'Earned 1000+ points',
                'icon': '🎓'
            })
        
        # Streak-based achievements
        if user_stats.get('current_streak', 0) >= 5:
            achievements.append({
                'name': 'On Fire',
                'description': '5+ correct answers in a row',
                'icon': '🔥'
            })
        
        # Subject mastery
        for subject in ['Science', 'Mathematics', 'History']:
            subject_score = user_stats.get(f'{subject.lower()}_score', 0)
            if subject_score >= 500:
                achievements.append({
                    'name': f'{subject} Expert',
                    'description': f'500+ points in {subject}',
                    'icon': '🏆'
                })
        
        return achievements
