import json
import os

class DataLoader:
    def __init__(self):
        self.questions_file = os.path.join(os.path.dirname(__file__), '..', 'data', 'questions.json')
        self.questions_data = None
        self.load_questions()
    
    def load_questions(self):
        """Load questions from JSON file"""
        try:
            with open(self.questions_file, 'r', encoding='utf-8') as f:
                self.questions_data = json.load(f)
            print(f"Loaded {len(self.questions_data)} questions from dataset")
        except FileNotFoundError:
            print("Questions file not found. Creating empty dataset.")
            self.questions_data = []
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON: {e}")
            self.questions_data = []
    
    def get_question_by_id(self, question_id):
        """Get a specific question by ID"""
        if not self.questions_data:
            return None
        
        for question in self.questions_data:
            if question.get('id') == question_id:
                return question
        return None
    
    def get_questions_by_class(self, class_num):
        """Get all questions for a specific class"""
        if not self.questions_data:
            return []
        
        return [q for q in self.questions_data if q.get('class') == class_num]
    
    def get_questions_by_subject(self, subject):
        """Get all questions for a specific subject"""
        if not self.questions_data:
            return []
        
        return [q for q in self.questions_data if q.get('subject', '').lower() == subject.lower()]
    
    def get_questions_by_difficulty(self, difficulty):
        """Get all questions for a specific difficulty level"""
        if not self.questions_data:
            return []
        
        return [q for q in self.questions_data if q.get('difficulty', '').lower() == difficulty.lower()]
    
    def get_filtered_questions(self, class_num=None, subject=None, difficulty=None, limit=None):
        """Get questions with multiple filters"""
        if not self.questions_data:
            return []
        
        filtered = self.questions_data
        
        if class_num:
            filtered = [q for q in filtered if q.get('class') == class_num]
        
        if subject:
            filtered = [q for q in filtered if q.get('subject', '').lower() == subject.lower()]
        
        if difficulty:
            filtered = [q for q in filtered if q.get('difficulty', '').lower() == difficulty.lower()]
        
        if limit:
            filtered = filtered[:limit]
        
        return filtered
    
    def get_random_questions(self, count=5, class_num=None, subject=None):
        """Get random questions for quiz generation"""
        import random
        
        filtered = self.get_filtered_questions(class_num=class_num, subject=subject)
        
        if len(filtered) <= count:
            return filtered
        
        return random.sample(filtered, count)
    
    def search_questions(self, query):
        """Search questions by keyword in question text or chapter"""
        if not self.questions_data:
            return []
        
        query = query.lower()
        results = []
        
        for question in self.questions_data:
            if (query in question.get('question', '').lower() or 
                query in question.get('chapter', '').lower() or
                query in question.get('answer', '').lower()):
                results.append(question)
        
        return results

# Global instance for easy access
_data_loader = DataLoader()

def get_questions_by_filters(class_num=None, subject=None, difficulty=None, limit=None):
    """Global function to get filtered questions"""
    return _data_loader.get_filtered_questions(class_num=class_num, subject=subject, difficulty=difficulty, limit=limit)

def get_all_questions():
    """Get all questions"""
    return _data_loader.questions_data if _data_loader.questions_data else []
