from flask import Blueprint, request, jsonify
from services.game_mechanics import GameMechanics
from utils.data_loader import DataLoader
import json
import uuid

game_bp = Blueprint('game', __name__)
game_mechanics = GameMechanics()
data_loader = DataLoader()

# In-memory game sessions (in production, use Redis or database)
game_sessions = {}

@game_bp.route('/start', methods=['POST'])
def start_game():
    """
    Start a new game session
    Expected payload: {
        "game_type": "subway_surfer|car_race|treasure_hunt|quiz_master",
        "player_name": "string",
        "class": 6-10,
        "subject": "Science|Mathematics|History" (optional),
        "difficulty": "easy|medium|hard" (optional)
    }
    """
    try:
        data = request.get_json()
        game_type = data.get('game_type')
        player_name = data.get('player_name', 'Anonymous')
        class_num = data.get('class', 8)
        subject = data.get('subject')
        difficulty = data.get('difficulty', 'medium')
        
        # Generate unique player ID
        player_id = str(uuid.uuid4())[:8]
        
        # Initialize game session
        game_session = game_mechanics.start_game_session(
            game_type=game_type,
            player_id=player_id,
            difficulty=difficulty
        )
        
        if 'error' in game_session:
            return jsonify(game_session), 400
        
        # Store session
        session_id = game_session['session']['session_id']
        game_sessions[session_id] = game_session
        
        # Get first question
        first_question = data_loader.get_random_questions(
            count=1, 
            class_num=class_num, 
            subject=subject
        )[0] if data_loader.get_filtered_questions(class_num=class_num, subject=subject) else None
        
        if first_question:
            # Remove answer from question
            game_question = {
                'id': first_question['id'],
                'question': first_question['question'],
                'chapter': first_question['chapter'],
                'subject': first_question['subject'],
                'difficulty': first_question['difficulty'],
                'type': first_question['type']
            }
            if 'options' in first_question and first_question['options']:
                game_question['options'] = first_question['options']
            
            game_session['session']['current_question'] = first_question['id']
        else:
            game_question = None
        
        return jsonify({
            "success": True,
            "session_id": session_id,
            "player_id": player_id,
            "game_state": game_session,
            "current_question": game_question,
            "instructions": game_session.get('instructions', 'Answer questions to play!')
        })
        
    except Exception as e:
        return jsonify({
            "error": str(e),
            "success": False
        }), 500

@game_bp.route('/answer', methods=['POST'])
def submit_game_answer():
    """
    Submit answer in game context
    Expected payload: {
        "session_id": "string",
        "answer": "string",
        "question_id": int
    }
    """
    try:
        data = request.get_json()
        session_id = data.get('session_id')
        user_answer = data.get('answer')
        question_id = data.get('question_id')
        
        # Get session
        if session_id not in game_sessions:
            return jsonify({
                "error": "Game session not found",
                "success": False
            }), 404
        
        session = game_sessions[session_id]
        game_type = session['session']['game_type']
        
        # Get question data
        question_data = data_loader.get_question_by_id(question_id)
        if not question_data:
            return jsonify({
                "error": "Question not found",
                "success": False
            }), 404
        
        # Process answer through game mechanics
        game_result = game_mechanics.process_answer(
            session_id=session_id,
            game_type=game_type,
            answer=user_answer,
            question_data=question_data
        )
        
        # Update session stats
        session['session']['questions_answered'] += 1
        
        if game_result.get('success'):
            session['session']['correct_streak'] += 1
            session['session']['score'] += game_result.get('rewards', {}).get('coins', 0)
        else:
            session['session']['correct_streak'] = 0
            if 'lives' in session['session']:
                session['session']['lives'] += game_result.get('penalties', {}).get('lives', 0)
        
        # Check if game should continue
        game_over = False
        if session['session'].get('lives', 3) <= 0:
            game_over = True
            session['session']['game_state'] = 'ended'
        
        # Get next question if game continues
        next_question = None
        if not game_over:
            next_q = data_loader.get_random_questions(count=1)[0] if data_loader.questions_data else None
            if next_q:
                next_question = {
                    'id': next_q['id'],
                    'question': next_q['question'],
                    'chapter': next_q['chapter'],
                    'subject': next_q['subject'],
                    'difficulty': next_q['difficulty'],
                    'type': next_q['type']
                }
                if 'options' in next_q and next_q['options']:
                    next_question['options'] = next_q['options']
                
                session['session']['current_question'] = next_q['id']
        
        return jsonify({
            "success": True,
            "game_result": game_result,
            "updated_session": session['session'],
            "next_question": next_question,
            "game_over": game_over,
            "correct_answer": question_data['answer'] if not game_result.get('success') else None
        })
        
    except Exception as e:
        return jsonify({
            "error": str(e),
            "success": False
        }), 500

@game_bp.route('/session/<session_id>', methods=['GET'])
def get_game_session(session_id):
    """Get current game session state"""
    try:
        if session_id not in game_sessions:
            return jsonify({
                "error": "Game session not found",
                "success": False
            }), 404
        
        session = game_sessions[session_id]
        
        return jsonify({
            "success": True,
            "session": session['session'],
            "game_state": session
        })
        
    except Exception as e:
        return jsonify({
            "error": str(e),
            "success": False
        }), 500

@game_bp.route('/leaderboard', methods=['GET'])
def get_leaderboard():
    """Get leaderboard for different game types"""
    try:
        game_type = request.args.get('game_type', 'all')
        
        # Extract scores from all sessions
        leaderboard = []
        for session_id, session_data in game_sessions.items():
            session = session_data['session']
            if game_type == 'all' or session['game_type'] == game_type:
                leaderboard.append({
                    'player_id': session['player_id'],
                    'game_type': session['game_type'],
                    'score': session['score'],
                    'level': session.get('level', 1),
                    'questions_answered': session['questions_answered'],
                    'correct_streak': session['correct_streak']
                })
        
        # Sort by score descending
        leaderboard.sort(key=lambda x: x['score'], reverse=True)
        
        return jsonify({
            "success": True,
            "leaderboard": leaderboard[:10],  # Top 10
            "game_type": game_type
        })
        
    except Exception as e:
        return jsonify({
            "error": str(e),
            "success": False
        }), 500

@game_bp.route('/achievements/<session_id>', methods=['GET'])
def get_achievements(session_id):
    """Get achievements for a game session"""
    try:
        if session_id not in game_sessions:
            return jsonify({
                "error": "Game session not found",
                "success": False
            }), 404
        
        session = game_sessions[session_id]['session']
        
        # Calculate achievements
        achievements = []
        
        # Score-based achievements
        score = session.get('score', 0)
        if score >= 1000:
            achievements.append({
                'name': 'High Scorer',
                'description': 'Scored 1000+ points',
                'icon': '🏆',
                'rarity': 'gold'
            })
        elif score >= 500:
            achievements.append({
                'name': 'Good Player',
                'description': 'Scored 500+ points', 
                'icon': '🥈',
                'rarity': 'silver'
            })
        elif score >= 100:
            achievements.append({
                'name': 'Getting Started',
                'description': 'Scored 100+ points',
                'icon': '🥉', 
                'rarity': 'bronze'
            })
        
        # Streak-based achievements
        streak = session.get('correct_streak', 0)
        if streak >= 10:
            achievements.append({
                'name': 'Unstoppable',
                'description': '10+ correct answers in a row',
                'icon': '🔥',
                'rarity': 'legendary'
            })
        elif streak >= 5:
            achievements.append({
                'name': 'On Fire',
                'description': '5+ correct answers in a row',
                'icon': '⚡',
                'rarity': 'epic'
            })
        
        # Questions-based achievements
        questions = session.get('questions_answered', 0)
        if questions >= 50:
            achievements.append({
                'name': 'Knowledge Seeker',
                'description': 'Answered 50+ questions',
                'icon': '📚',
                'rarity': 'rare'
            })
        
        return jsonify({
            "success": True,
            "achievements": achievements,
            "session_stats": {
                'score': score,
                'streak': streak,
                'questions_answered': questions,
                'level': session.get('level', 1)
            }
        })
        
    except Exception as e:
        return jsonify({
            "error": str(e),
            "success": False
        }), 500

@game_bp.route('/power-ups/<session_id>', methods=['POST'])
def use_power_up():
    """Use a power-up in game"""
    try:
        data = request.get_json()
        session_id = data.get('session_id', session_id)
        power_up_type = data.get('power_up_type')
        
        if session_id not in game_sessions:
            return jsonify({
                "error": "Game session not found",
                "success": False
            }), 404
        
        session = game_sessions[session_id]['session']
        
        # Check if player has the power-up
        if power_up_type not in session.get('power_ups', []):
            return jsonify({
                "error": "Power-up not available",
                "success": False
            }), 400
        
        # Apply power-up effects
        effects = {}
        if power_up_type == '50-50':
            effects = {
                'name': '50-50 Lifeline',
                'description': 'Eliminates 2 wrong options',
                'effect': 'remove_wrong_options'
            }
        elif power_up_type == 'extra_time':
            effects = {
                'name': 'Extra Time',
                'description': 'Adds 30 seconds to timer',
                'effect': 'time_bonus'
            }
        elif power_up_type == 'double_coins':
            effects = {
                'name': 'Double Coins',
                'description': 'Next correct answer gives double coins',
                'effect': 'coin_multiplier'
            }
        
        # Remove power-up from inventory
        session['power_ups'].remove(power_up_type)
        
        return jsonify({
            "success": True,
            "power_up_used": power_up_type,
            "effects": effects,
            "remaining_power_ups": session['power_ups']
        })
        
    except Exception as e:
        return jsonify({
            "error": str(e),
            "success": False
        }), 500