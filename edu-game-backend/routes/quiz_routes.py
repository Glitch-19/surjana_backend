from flask import Blueprint, request, jsonify
from utils.data_loader import DataLoader
from services.quiz_service import QuizService

quiz_bp = Blueprint('quiz', __name__)
data_loader = DataLoader()
quiz_service = QuizService()

@quiz_bp.route('/questions', methods=['GET'])
def get_questions():
    """
    Get questions based on filters
    Query parameters:
    - class: class number (6-10)
    - subject: subject name
    - difficulty: difficulty level
    - limit: number of questions
    """
    try:
        class_num = request.args.get('class', type=int)
        subject = request.args.get('subject')
        difficulty = request.args.get('difficulty')
        limit = request.args.get('limit', type=int)
        
        questions = data_loader.get_filtered_questions(
            class_num=class_num,
            subject=subject,
            difficulty=difficulty,
            limit=limit
        )
        
        # Remove answers from questions for quiz mode
        quiz_questions = []
        for q in questions:
            quiz_q = {
                'id': q['id'],
                'class': q['class'],
                'subject': q['subject'],
                'chapter': q['chapter'],
                'question': q['question'],
                'type': q['type'],
                'difficulty': q['difficulty'],
                'gamify': q.get('gamify', '')
            }
            # Only include options for MCQ type questions
            if 'options' in q and q['options']:
                quiz_q['options'] = q['options']
            
            quiz_questions.append(quiz_q)
        
        return jsonify({
            "success": True,
            "questions": quiz_questions,
            "count": len(quiz_questions)
        })
        
    except Exception as e:
        return jsonify({
            "error": str(e),
            "success": False
        }), 500

@quiz_bp.route('/question/<int:question_id>', methods=['GET'])
def get_question(question_id):
    """Get a single question by ID"""
    try:
        question = data_loader.get_question_by_id(question_id)
        
        if not question:
            return jsonify({
                "error": "Question not found",
                "success": False
            }), 404
        
        # Remove answer for quiz mode
        quiz_question = {
            'id': question['id'],
            'class': question['class'],
            'subject': question['subject'],
            'chapter': question['chapter'],
            'question': question['question'],
            'type': question['type'],
            'difficulty': question['difficulty'],
            'gamify': question.get('gamify', '')
        }
        
        if 'options' in question and question['options']:
            quiz_question['options'] = question['options']
        
        return jsonify({
            "success": True,
            "question": quiz_question
        })
        
    except Exception as e:
        return jsonify({
            "error": str(e),
            "success": False
        }), 500

@quiz_bp.route('/submit', methods=['POST'])
def submit_answer():
    """
    Submit answer and get feedback
    Expected payload: {
        "question_id": int,
        "user_answer": string,
        "user_id": string (optional)
    }
    """
    try:
        data = request.get_json()
        question_id = data.get('question_id')
        user_answer = data.get('user_answer')
        user_id = data.get('user_id', 'anonymous')
        
        # Get correct answer
        question = data_loader.get_question_by_id(question_id)
        if not question:
            return jsonify({
                "error": "Question not found",
                "success": False
            }), 404
        
        # Check answer and calculate score
        result = quiz_service.check_answer(question, user_answer)
        
        # Update user progress (you can extend this for persistent storage)
        if result['is_correct']:
            score = quiz_service.calculate_score(question, result['accuracy'])
        else:
            score = 0
        
        return jsonify({
            "success": True,
            "is_correct": result['is_correct'],
            "accuracy": result['accuracy'],
            "score": score,
            "feedback": result['feedback'],
            "correct_answer": question['answer'] if not result['is_correct'] else None,
            "question_id": question_id
        })
        
    except Exception as e:
        return jsonify({
            "error": str(e),
            "success": False
        }), 500

@quiz_bp.route('/random', methods=['GET'])
def get_random_quiz():
    """
    Get random questions for a quick quiz
    Query parameters:
    - count: number of questions (default: 5)
    - class: class number
    - subject: subject name
    """
    try:
        count = request.args.get('count', 5, type=int)
        class_num = request.args.get('class', type=int)
        subject = request.args.get('subject')
        
        questions = data_loader.get_random_questions(
            count=count,
            class_num=class_num,
            subject=subject
        )
        
        # Remove answers from questions
        quiz_questions = []
        for q in questions:
            quiz_q = {
                'id': q['id'],
                'class': q['class'],
                'subject': q['subject'],
                'chapter': q['chapter'],
                'question': q['question'],
                'type': q['type'],
                'difficulty': q['difficulty'],
                'gamify': q.get('gamify', '')
            }
            if 'options' in q and q['options']:
                quiz_q['options'] = q['options']
            
            quiz_questions.append(quiz_q)
        
        return jsonify({
            "success": True,
            "questions": quiz_questions,
            "count": len(quiz_questions),
            "quiz_id": f"quiz_{class_num}_{subject}_{count}"
        })
        
    except Exception as e:
        return jsonify({
            "error": str(e),
            "success": False
        }), 500
