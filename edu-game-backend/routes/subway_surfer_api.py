"""
Enhanced Game API for Subway Surfer Educational Integration
Provides dynamic question fetching and adaptive learning mechanics
"""
from flask import Blueprint, request, jsonify
from services.quiz_service import QuizService
from services.ai_service import get_ai_hint
from utils.data_loader import get_questions_by_filters
import random
import time

game_api = Blueprint('game_api', __name__)
quiz_service = QuizService()

# Track active game sessions
active_sessions = {}

class GameSession:
    def __init__(self, player_name, game_type="subway_surfer"):
        self.player_name = player_name
        self.game_type = game_type
        self.start_time = time.time()
        self.score = 0
        self.coins = 0
        self.distance = 0
        self.questions_answered = 0
        self.correct_answers = 0
        self.streak = 0
        self.max_streak = 0
        self.difficulty_level = "easy"
        self.question_history = []
        self.performance_metrics = {
            'average_response_time': 0,
            'subject_performance': {},
            'difficulty_progression': []
        }
    
    def get_adaptive_difficulty(self):
        """Determine difficulty based on recent performance"""
        if self.questions_answered < 3:
            return "easy"
        
        recent_performance = self.correct_answers / self.questions_answered if self.questions_answered > 0 else 0
        
        if recent_performance >= 0.8 and self.streak >= 3:
            return "hard"
        elif recent_performance >= 0.6:
            return "medium" 
        else:
            return "easy"
    
    def get_recommended_subject(self):
        """Get subject that needs more practice"""
        if not self.performance_metrics['subject_performance']:
            return None
        
        # Find subject with lowest accuracy
        subjects = self.performance_metrics['subject_performance']
        return min(subjects, key=lambda x: subjects[x]['accuracy']) if subjects else None

@game_api.route('/subway-surfer/start', methods=['POST'])
def start_subway_surfer_session():
    """Start a new Subway Surfer educational session"""
    try:
        data = request.json
        player_name = data.get('player_name', 'Anonymous')
        
        session = GameSession(player_name, "subway_surfer")
        session_id = f"ss_{int(time.time())}_{random.randint(1000, 9999)}"
        active_sessions[session_id] = session
        
        # Get initial question
        questions = get_questions_by_filters()
        initial_question = random.choice(questions) if questions else None
        
        return jsonify({
            'success': True,
            'session_id': session_id,
            'player_name': player_name,
            'initial_question': format_question_for_game(initial_question) if initial_question else None,
            'game_config': {
                'question_interval': 15000,  # 15 seconds
                'speed_multiplier': 1.0,
                'coin_bonus': 10,
                'streak_bonus': 50
            }
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@game_api.route('/subway-surfer/question', methods=['POST'])
def get_adaptive_question():
    """Get next question based on player performance"""
    try:
        data = request.json
        session_id = data.get('session_id')
        
        if session_id not in active_sessions:
            return jsonify({'success': False, 'error': 'Invalid session'}), 400
        
        session = active_sessions[session_id]
        
        # Determine adaptive parameters
        difficulty = session.get_adaptive_difficulty()
        recommended_subject = session.get_recommended_subject()
        
        # Get filtered questions
        filters = {
            'difficulty': difficulty,
            'subject': recommended_subject if recommended_subject else None
        }
        
        questions = get_questions_by_filters(**filters)
        
        # Avoid recent questions
        available_questions = [q for q in questions if q.get('id') not in session.question_history[-5:]]
        
        if not available_questions:
            available_questions = questions
        
        question = random.choice(available_questions) if available_questions else None
        
        if question:
            session.question_history.append(question.get('id', len(session.question_history)))
        
        return jsonify({
            'success': True,
            'question': format_question_for_game(question) if question else None,
            'adaptive_info': {
                'difficulty': difficulty,
                'recommended_subject': recommended_subject,
                'performance_hint': get_performance_hint(session)
            }
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@game_api.route('/subway-surfer/answer', methods=['POST'])
def submit_game_answer():
    """Submit answer and update game state"""
    try:
        data = request.json
        session_id = data.get('session_id')
        answer_index = data.get('answer_index')
        question_id = data.get('question_id')
        response_time = data.get('response_time', 0)
        
        if session_id not in active_sessions:
            return jsonify({'success': False, 'error': 'Invalid session'}), 400
        
        session = active_sessions[session_id]
        
        # Get the question to check answer
        questions = get_questions_by_filters()
        question = next((q for q in questions if str(q.get('id', '')) == str(question_id)), None)
        
        if not question:
            return jsonify({'success': False, 'error': 'Question not found'}), 400
        
        correct_index = question.get('correct_answer', 0)
        is_correct = answer_index == correct_index
        
        # Update session stats
        session.questions_answered += 1
        if is_correct:
            session.correct_answers += 1
            session.streak += 1
            session.max_streak = max(session.max_streak, session.streak)
        else:
            session.streak = 0
        
        # Update subject performance
        subject = question.get('subject', 'General')
        if subject not in session.performance_metrics['subject_performance']:
            session.performance_metrics['subject_performance'][subject] = {
                'total': 0, 'correct': 0, 'accuracy': 0
            }
        
        session.performance_metrics['subject_performance'][subject]['total'] += 1
        if is_correct:
            session.performance_metrics['subject_performance'][subject]['correct'] += 1
        
        session.performance_metrics['subject_performance'][subject]['accuracy'] = (
            session.performance_metrics['subject_performance'][subject]['correct'] / 
            session.performance_metrics['subject_performance'][subject]['total']
        )
        
        # Calculate rewards
        base_score = 100 if is_correct else 0
        streak_bonus = session.streak * 25 if session.streak >= 2 else 0
        time_bonus = max(0, 50 - response_time // 1000) if is_correct and response_time < 10000 else 0
        
        total_score = base_score + streak_bonus + time_bonus
        coin_reward = (10 + session.streak * 2) if is_correct else 0
        
        session.score += total_score
        session.coins += coin_reward
        
        # Get AI hint for wrong answers
        ai_hint = None
        if not is_correct:
            try:
                ai_hint = get_ai_hint(question.get('question', ''), question.get('options', []))
            except:
                ai_hint = "Keep trying! Review the concept and you'll get it next time."
        
        return jsonify({
            'success': True,
            'correct': is_correct,
            'explanation': question.get('explanation', ''),
            'ai_hint': ai_hint,
            'rewards': {
                'base_score': base_score,
                'streak_bonus': streak_bonus,
                'time_bonus': time_bonus,
                'total_score': total_score,
                'coins': coin_reward
            },
            'session_stats': {
                'total_score': session.score,
                'total_coins': session.coins,
                'questions_answered': session.questions_answered,
                'accuracy': round(session.correct_answers / session.questions_answered * 100, 1) if session.questions_answered > 0 else 0,
                'current_streak': session.streak,
                'max_streak': session.max_streak,
                'difficulty': session.get_adaptive_difficulty()
            },
            'game_effects': {
                'speed_boost': is_correct and session.streak >= 3,
                'shield_activate': session.streak >= 5,
                'coin_magnet': session.streak >= 2
            }
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@game_api.route('/subway-surfer/stats/<session_id>', methods=['GET'])
def get_session_stats(session_id):
    """Get detailed session statistics"""
    try:
        if session_id not in active_sessions:
            return jsonify({'success': False, 'error': 'Invalid session'}), 400
        
        session = active_sessions[session_id]
        
        return jsonify({
            'success': True,
            'session_id': session_id,
            'player_name': session.player_name,
            'duration': int(time.time() - session.start_time),
            'stats': {
                'score': session.score,
                'coins': session.coins,
                'questions_answered': session.questions_answered,
                'correct_answers': session.correct_answers,
                'accuracy': round(session.correct_answers / session.questions_answered * 100, 1) if session.questions_answered > 0 else 0,
                'current_streak': session.streak,
                'max_streak': session.max_streak,
                'current_difficulty': session.get_adaptive_difficulty()
            },
            'performance': session.performance_metrics,
            'achievements': get_achievements(session)
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@game_api.route('/subway-surfer/leaderboard', methods=['GET'])
def get_leaderboard():
    """Get top players leaderboard"""
    try:
        # Sort active sessions by score
        leaderboard = []
        for session_id, session in active_sessions.items():
            if session.questions_answered > 0:  # Only include players who answered questions
                leaderboard.append({
                    'player_name': session.player_name,
                    'score': session.score,
                    'accuracy': round(session.correct_answers / session.questions_answered * 100, 1),
                    'questions_answered': session.questions_answered,
                    'max_streak': session.max_streak,
                    'coins': session.coins
                })
        
        # Sort by score descending
        leaderboard.sort(key=lambda x: x['score'], reverse=True)
        
        return jsonify({
            'success': True,
            'leaderboard': leaderboard[:10],  # Top 10
            'total_players': len(leaderboard)
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

def format_question_for_game(question):
    """Format question for game display"""
    if not question:
        return None
    
    return {
        'id': question.get('id', ''),
        'question': question.get('question', ''),
        'options': question.get('options', []),
        'subject': question.get('subject', 'General'),
        'class': question.get('class', ''),
        'difficulty': question.get('difficulty', 'medium'),
        'gamify': question.get('gamify', {})
    }

def get_performance_hint(session):
    """Get performance-based hint for player"""
    accuracy = session.correct_answers / session.questions_answered if session.questions_answered > 0 else 0
    
    if accuracy >= 0.9:
        return "🔥 Excellent! You're a learning champion!"
    elif accuracy >= 0.7:
        return "👍 Great job! Keep up the good work!"
    elif accuracy >= 0.5:
        return "📚 You're improving! Focus on understanding concepts."
    else:
        return "💪 Don't give up! Practice makes perfect!"

def get_achievements(session):
    """Calculate achievements for the session"""
    achievements = []
    
    if session.questions_answered >= 10:
        achievements.append("📚 Knowledge Seeker")
    
    if session.max_streak >= 5:
        achievements.append("🔥 Streak Master")
    
    if session.correct_answers / session.questions_answered >= 0.9 and session.questions_answered >= 5:
        achievements.append("🎯 Accuracy Expert")
    
    if session.coins >= 100:
        achievements.append("💰 Coin Collector")
    
    if session.score >= 1000:
        achievements.append("⭐ High Scorer")
    
    return achievements