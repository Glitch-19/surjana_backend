"""
Duolingo-Style Learning API Routes
Handles questions, progress, achievements, and AI integration
"""
from flask import Blueprint, request, jsonify
import json
import os
import sys
from datetime import datetime
import random

# Add the data directory to the path for importing questions
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'data'))

try:
    from ncert_questions import (
        NCERT_QUESTION_DATABASE, 
        get_questions_by_lesson,
        get_questions_by_subject_class,
        get_mixed_questions,
        DAILY_CHALLENGES,
        DIFFICULTY_LEVELS,
        get_user_lesson_questions,
        get_user_subject_class_questions,
        get_user_mixed_questions,
        reset_user_rotations,
        # Added for auto-reset when lesson exhausted
        reset_question_rotations,
        # New imports
        fetch_lesson_questions, fetch_subject_class_questions, fetch_mixed_questions,
        fetch_user_lesson_questions, fetch_user_subject_class_questions, fetch_user_mixed_questions,
        get_lesson_status, get_subject_class_status, get_mixed_status,
        get_user_lesson_status, get_user_subject_class_status, get_user_mixed_status,
        update_answer_tracking, get_review_items, adaptive_reset_user_lesson
    )
except ImportError:
    # Fallback if import fails (reduced capability)
    NCERT_QUESTION_DATABASE = {}
    DAILY_CHALLENGES = []
    DIFFICULTY_LEVELS = {'easy': {'xp': 10, 'tokens': 1}, 'medium': {'xp': 15, 'tokens': 2}, 'hard': {'xp': 25, 'tokens': 3}}

duolingo_api = Blueprint('duolingo_api', __name__)

# Data storage paths
PROGRESS_DB = "data/student_progress.json"
ACHIEVEMENTS_DB = "data/achievements.json"
SESSIONS_DB = "data/learning_sessions.json"

def load_data(file_path):
    """Load data from JSON file"""
    if not os.path.exists(file_path):
        return {}
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except:
        return {}

def save_data(file_path, data):
    """Save data to JSON file"""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)

@duolingo_api.route('/questions/lesson', methods=['POST'])
def get_lesson_questions():
    """Get questions for a specific lesson"""
    try:
        data = request.json
        lesson_id = data.get('lesson_id')
        count = data.get('count', 10)
        
        if not lesson_id:
            return jsonify({'success': False, 'error': 'Lesson ID required'}), 400
        
        # Get questions from database (rotation-based, may exhaust)
        questions = get_questions_by_lesson(lesson_id, count)

        # If exhausted (empty) but lesson exists, auto-reset rotation once
        if not questions and lesson_id in NCERT_QUESTION_DATABASE:
            try:
                reset_question_rotations(scope='lesson', lesson_id=lesson_id)
                questions = get_questions_by_lesson(lesson_id, count)
            except Exception:
                pass

        used_fallback = False
        # Final fallback generation if still empty (e.g., invalid lesson id or import failure)
        if not questions:
            used_fallback = True
            questions = generate_fallback_questions(lesson_id, count)
        
        # Remove correct answer from response (send separately for security)
        safe_questions = []
        for q in questions:
            safe_q = {
                'id': q['id'],
                'text': q['text'],
                'options': q['options'],
                'concept': q['concept'],
                'difficulty': q['difficulty'],
                'source': q.get('source', 'NCERT')
            }
            safe_questions.append(safe_q)
        
        return jsonify({
            'success': True,
            'questions': safe_questions,
            'lesson_id': lesson_id,
            'count': len(safe_questions),
            'fallback': used_fallback,
            'has_ncert_source': bool(NCERT_QUESTION_DATABASE.get(lesson_id))
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@duolingo_api.route('/questions/subject', methods=['POST'])
def get_subject_questions():
    """Get questions filtered by subject and class"""
    try:
        data = request.json
        subject = data.get('subject')
        class_level = data.get('class', 6)
        count = data.get('count', 10)
        
        if not subject:
            return jsonify({'success': False, 'error': 'Subject required'}), 400
        
        questions = get_questions_by_subject_class(subject, class_level, count)
        
        # Remove correct answers for security
        safe_questions = [
            {
                'id': q['id'],
                'text': q['text'], 
                'options': q['options'],
                'concept': q['concept'],
                'difficulty': q['difficulty']
            }
            for q in questions
        ]
        
        return jsonify({
            'success': True,
            'questions': safe_questions,
            'subject': subject,
            'class': class_level
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@duolingo_api.route('/questions/mixed', methods=['POST'])
def get_mixed_questions_api():
    """Get random mix of questions"""
    try:
        data = request.json
        count = data.get('count', 10)
        
        questions = get_mixed_questions(count)
        
        safe_questions = [
            {
                'id': q['id'],
                'text': q['text'],
                'options': q['options'], 
                'concept': q['concept'],
                'difficulty': q['difficulty']
            }
            for q in questions
        ]
        
        return jsonify({
            'success': True,
            'questions': safe_questions,
            'count': len(safe_questions)
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@duolingo_api.route('/answer/check', methods=['POST'])
def check_answer():
    """Check if submitted answer is correct (with tracking & progress meta)"""
    try:
        data = request.json
        question_id = data.get('question_id')
        lesson_id = data.get('lesson_id')
        selected_option = data.get('selected_option')
        student_id = data.get('student_id', 'default')
        if question_id is None or selected_option is None:
            return jsonify({'success': False, 'error': 'Missing required fields'}), 400
        # Because question IDs are reused per lesson (id starts at 1 for each lesson),
        # we must scope lookup to the provided lesson_id first to avoid mismatching concepts.
        correct_answer = None
        explanation = ""
        question_data = None
        search_order = []
        if lesson_id and lesson_id in NCERT_QUESTION_DATABASE:
            search_order.append((lesson_id, NCERT_QUESTION_DATABASE.get(lesson_id, [])))
        # Fallback global search if not found in specified lesson (robustness for older clients)
        search_order.extend([(lid, qs) for lid, qs in NCERT_QUESTION_DATABASE.items() if lid != lesson_id])
        for lesson, questions in search_order:
            for q in questions:
                if q.get('id') == question_id:
                    correct_answer = q.get('correct')
                    explanation = q.get('explanation')
                    question_data = q
                    break
            if correct_answer is not None:
                break
        if correct_answer is None:
            return jsonify({'success': False, 'error': 'Question not found'}), 404
        is_correct = selected_option == correct_answer
        difficulty = question_data.get('difficulty', 'easy')
        rewards = DIFFICULTY_LEVELS.get(difficulty, DIFFICULTY_LEVELS['easy'])
        xp_earned = rewards['xp'] if is_correct else 0
        tokens_earned = rewards['tokens'] if is_correct else 0
        if is_correct:
            update_student_progress(student_id, xp_earned, tokens_earned, lesson_id)
        # Track answer for review/adaptive
        if lesson_id:
            try:
                update_answer_tracking(student_id, lesson_id, question_id, is_correct)
            except Exception:
                pass
        response = {
            'success': True,
            'correct': is_correct,
            'correct_option': correct_answer,
            'explanation': explanation,
            'concept': question_data.get('concept', ''),
            'xp_earned': xp_earned,
            'tokens_earned': tokens_earned,
            'difficulty': difficulty
        }
        # Provide optional rich tutoring layer
        if data.get('rich_explanation') or str(request.args.get('rich_explanation','')).lower() == 'true':
            try:
                from services.ai_service import AIService
                ai_service = AIService()
                rich = ai_service.get_rich_hint(question_data, user_attempt=data.get('user_attempt'), selected_option=selected_option)
                response['rich_tutoring'] = rich
                if not is_correct and selected_option is not None:
                    response['wrong_analysis'] = ai_service.get_wrong_answer_analysis(question_data, selected_option)
            except Exception:
                pass
        return jsonify(response)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@duolingo_api.route('/progress/student/<student_id>', methods=['GET'])
def get_student_progress(student_id):
    """Get detailed student progress"""
    try:
        progress_data = load_data(PROGRESS_DB)
        
        if student_id not in progress_data:
            # Create new student profile
            progress_data[student_id] = {
                'xp': 0,
                'tokens': 0,
                'streak': 0,
                'level': 1,
                'lessons_completed': {},
                'subjects_progress': {'science': 0, 'maths': 0},
                'daily_progress': 0,
                'last_activity': datetime.now().isoformat(),
                'achievements': [],
                'total_questions_answered': 0,
                'correct_answers': 0
            }
            save_data(PROGRESS_DB, progress_data)
        
        student_data = progress_data[student_id]
        
        # Calculate additional stats
        accuracy = 0
        if student_data['total_questions_answered'] > 0:
            accuracy = (student_data['correct_answers'] / student_data['total_questions_answered']) * 100
        
        response = {
            'success': True,
            'student_id': student_id,
            'progress': {
                **student_data,
                'accuracy': round(accuracy, 1),
                'next_level_xp': calculate_next_level_xp(student_data['level']),
                'overall_progress': calculate_overall_progress(student_data['lessons_completed'])
            }
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@duolingo_api.route('/progress/update', methods=['POST'])
def update_progress():
    """Update student progress after lesson completion"""
    try:
        data = request.json
        student_id = data.get('student_id', 'default')
        lesson_id = data.get('lesson_id')
        score = data.get('score', 0)
        total_questions = data.get('total_questions', 10)
        time_spent = data.get('time_spent', 0)
        
        progress_data = load_data(PROGRESS_DB)
        
        if student_id not in progress_data:
            progress_data[student_id] = {
                'xp': 0, 'tokens': 0, 'streak': 0, 'level': 1,
                'lessons_completed': {}, 'subjects_progress': {'science': 0, 'maths': 0},
                'daily_progress': 0, 'achievements': [],
                'total_questions_answered': 0, 'correct_answers': 0
            }
        
        student_data = progress_data[student_id]
        
        # Update lesson completion
        if lesson_id:
            student_data['lessons_completed'][lesson_id] = {
                'score': score,
                'total': total_questions,
                'accuracy': (score / total_questions) * 100,
                'completed_at': datetime.now().isoformat(),
                'time_spent': time_spent
            }
        
        # Update question stats
        student_data['total_questions_answered'] += total_questions
        student_data['correct_answers'] += score
        
        # Update daily progress
        student_data['daily_progress'] += 1
        
        # Check for level up
        old_level = student_data['level']
        new_level = calculate_level_from_xp(student_data['xp'])
        
        level_up = new_level > old_level
        if level_up:
            student_data['level'] = new_level
        
        # Check for new achievements
        new_achievements = check_achievements(student_data)
        student_data['achievements'].extend(new_achievements)
        
        # Update last activity
        student_data['last_activity'] = datetime.now().isoformat()
        
        save_data(PROGRESS_DB, progress_data)
        
        return jsonify({
            'success': True,
            'progress': student_data,
            'level_up': level_up,
            'new_achievements': new_achievements
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@duolingo_api.route('/leaderboard', methods=['GET'])
def get_leaderboard():
    """Get leaderboard with top students"""
    try:
        progress_data = load_data(PROGRESS_DB)
        
        # Create leaderboard from all students
        leaderboard = []
        for student_id, data in progress_data.items():
            leaderboard.append({
                'student_id': student_id,
                'xp': data['xp'],
                'level': data['level'],
                'streak': data['streak'],
                'lessons_completed': len(data['lessons_completed']),
                'accuracy': calculate_accuracy(data)
            })
        
        # Sort by XP (descending)
        leaderboard.sort(key=lambda x: x['xp'], reverse=True)
        
        # Add rank
        for i, student in enumerate(leaderboard):
            student['rank'] = i + 1
        
        return jsonify({
            'success': True,
            'leaderboard': leaderboard[:20],  # Top 20
            'total_students': len(leaderboard)
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@duolingo_api.route('/achievements/<student_id>', methods=['GET'])
def get_achievements(student_id):
    """Get student achievements"""
    try:
        achievements_data = load_data(ACHIEVEMENTS_DB)
        
        student_achievements = achievements_data.get(student_id, [])
        
        # Define all possible achievements
        all_achievements = [
            {'id': 'first_lesson', 'name': 'First Steps', 'description': 'Complete your first lesson', 'icon': '🌟'},
            {'id': 'science_explorer', 'name': 'Science Explorer', 'description': 'Complete 5 science lessons', 'icon': '🧪'},
            {'id': 'math_wizard', 'name': 'Math Wizard', 'description': 'Complete 5 math lessons', 'icon': '📊'},
            {'id': 'perfect_score', 'name': 'Perfectionist', 'description': 'Get 100% in any lesson', 'icon': '💯'},
            {'id': 'streak_master', 'name': 'Streak Master', 'description': '7 day learning streak', 'icon': '🔥'},
            {'id': 'xp_collector', 'name': 'XP Collector', 'description': 'Earn 1000 XP', 'icon': '⭐'},
            {'id': 'knowledge_seeker', 'name': 'Knowledge Seeker', 'description': 'Answer 100 questions', 'icon': '🎯'},
            {'id': 'class_topper', 'name': 'Class Topper', 'description': 'Rank #1 on leaderboard', 'icon': '🏆'}
        ]
        
        return jsonify({
            'success': True,
            'earned_achievements': student_achievements,
            'all_achievements': all_achievements,
            'progress': len(student_achievements)
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@duolingo_api.route('/daily-challenge', methods=['GET'])
def get_daily_challenge():
    """Get today's daily challenge"""
    try:
        today = datetime.now().strftime('%Y-%m-%d')
        
        # Select challenge based on date (same challenge for everyone on same day)
        challenge_index = hash(today) % len(DAILY_CHALLENGES)
        challenge = DAILY_CHALLENGES[challenge_index]
        
        return jsonify({
            'success': True,
            'challenge': {
                **challenge,
                'date': today,
                'expires_at': f"{today} 23:59:59"
            }
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@duolingo_api.route('/hint/ai', methods=['POST'])
def get_ai_hint():
    """Get AI-powered hint for a question. Supports simple or rich modes.
    Payload fields: question (str), options (list), concept (str), question_id (int optional), mode ('simple'|'rich'), user_attempt, selected_option.
    """
    try:
        data = request.json or {}
        question_text = data.get('question')
        options = data.get('options', [])
        concept = data.get('concept', '')
        question_id = data.get('question_id')
        mode = (data.get('mode') or request.args.get('mode') or 'simple').lower()
        user_attempt = data.get('user_attempt')
        selected_option = data.get('selected_option')

        from services.ai_service import AIService
        ai_service = AIService()

        # Retrieve full question record if question_id provided
        record = None
        if question_id:
            for lesson, qlist in NCERT_QUESTION_DATABASE.items():
                for q in qlist:
                    if q['id'] == question_id:
                        record = q
                        break
                if record:
                    break
        if not record:
            record = {
                'question': question_text,
                'text': question_text,
                'options': options,
                'concept': concept or 'General Concept',
                'explanation': '',
                'difficulty': 'easy',
                'correct': None
            }

        if mode == 'rich':
            rich = ai_service.get_rich_hint(record, user_attempt=user_attempt, selected_option=selected_option)
            return jsonify({'success': True, 'mode': 'rich', **rich})
        else:
            basic = ai_service.get_hint({
                'question': record.get('question'),
                'class': '6-10',
                'subject': 'General',
                'chapter': concept or 'Concept',
                'answer': 'Not disclosed'
            }, user_attempt=user_attempt)
            return jsonify({'success': True, 'mode': 'simple', 'hint': basic.get('content'), 'source': 'ai' if basic.get('success') else 'fallback'})
    except Exception as e:
        # Fallback generic
        return jsonify({'success': False, 'error': str(e), 'hint': f'Think about the core principle of {concept}.'}), 500

# ------------------ Per-User Unique Question Endpoints ---------------------
@duolingo_api.route('/questions/user/lesson', methods=['POST'])
def get_user_lesson_questions_api():
    data = request.json or {}
    user_id = data.get('student_id') or data.get('user_id') or 'anonymous'
    lesson_id = data.get('lesson_id')
    count = int(data.get('count', 10))
    if not lesson_id:
        return jsonify({'success': False, 'error': 'lesson_id required'}), 400
    qs = get_user_lesson_questions(user_id, lesson_id, count)
    safe = [
        {
            'id': q['id'],
            'text': q['text'],
            'options': q['options'],
            'concept': q.get('concept'),
            'difficulty': q.get('difficulty'),
            'reward_preview': DIFFICULTY_LEVELS.get(q.get('difficulty', 'easy'), {})
        } for q in qs
    ]
    return jsonify({'success': True, 'user_id': user_id, 'lesson_id': lesson_id, 'questions': safe, 'count': len(safe)})

@duolingo_api.route('/questions/user/subject', methods=['POST'])
def get_user_subject_class_questions_api():
    data = request.json or {}
    user_id = data.get('student_id') or data.get('user_id') or 'anonymous'
    subject = data.get('subject')
    class_level = data.get('class') or data.get('class_level') or 6
    count = int(data.get('count', 10))
    if not subject:
        return jsonify({'success': False, 'error': 'subject required'}), 400
    qs = get_user_subject_class_questions(user_id, subject, class_level, count)
    safe = [
        {
            'id': q['id'],
            'text': q['text'],
            'options': q['options'],
            'concept': q.get('concept'),
            'difficulty': q.get('difficulty'),
            'reward_preview': DIFFICULTY_LEVELS.get(q.get('difficulty', 'easy'), {})
        } for q in qs
    ]
    return jsonify({'success': True, 'user_id': user_id, 'subject': subject, 'class': class_level, 'questions': safe, 'count': len(safe)})

@duolingo_api.route('/questions/user/mixed', methods=['POST'])
def get_user_mixed_questions_api():
    data = request.json or {}
    user_id = data.get('student_id') or data.get('user_id') or 'anonymous'
    count = int(data.get('count', 10))
    qs = get_user_mixed_questions(user_id, count)
    safe = [
        {
            'id': q['id'],
            'text': q['text'],
            'options': q['options'],
            'concept': q.get('concept'),
            'difficulty': q.get('difficulty'),
            'reward_preview': DIFFICULTY_LEVELS.get(q.get('difficulty', 'easy'), {})
        } for q in qs
    ]
    return jsonify({'success': True, 'user_id': user_id, 'questions': safe, 'count': len(safe)})

@duolingo_api.route('/questions/user/reset', methods=['POST'])
def reset_user_rotations_api():
    data = request.json or {}
    user_id = data.get('student_id') or data.get('user_id') or 'anonymous'
    scope = data.get('scope', 'all')
    lesson_id = data.get('lesson_id')
    subject = data.get('subject')
    class_level = data.get('class') or data.get('class_level')
    try:
        reset_user_rotations(user_id, scope=scope, lesson_id=lesson_id, subject=subject, class_level=class_level)
        return jsonify({'success': True, 'user_id': user_id, 'scope': scope})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400
# ---------------------------------------------------------------------------

# Helper Functions
def update_student_progress(student_id, xp, tokens, lesson_id):
    """Update student progress after correct answer"""
    progress_data = load_data(PROGRESS_DB)
    
    if student_id not in progress_data:
        progress_data[student_id] = {
            'xp': 0, 'tokens': 0, 'streak': 0, 'level': 1,
            'lessons_completed': {}, 'subjects_progress': {'science': 0, 'maths': 0},
            'daily_progress': 0, 'achievements': [],
            'total_questions_answered': 0, 'correct_answers': 0
        }
    
    student_data = progress_data[student_id]
    student_data['xp'] += xp
    student_data['tokens'] += tokens
    
    # Update level based on XP
    student_data['level'] = calculate_level_from_xp(student_data['xp'])
    
    save_data(PROGRESS_DB, progress_data)

def calculate_level_from_xp(xp):
    """Calculate level based on XP (100 XP per level)"""
    return max(1, xp // 100 + 1)

def calculate_next_level_xp(current_level):
    """Calculate XP needed for next level"""
    return current_level * 100

def calculate_overall_progress(lessons_completed):
    """Calculate overall progress percentage"""
    total_lessons = len(NCERT_QUESTION_DATABASE)
    completed_count = len(lessons_completed)
    return (completed_count / total_lessons * 100) if total_lessons > 0 else 0

def calculate_accuracy(student_data):
    """Calculate student's overall accuracy"""
    total = student_data.get('total_questions_answered', 0)
    correct = student_data.get('correct_answers', 0)
    return round((correct / total * 100), 1) if total > 0 else 0

def check_achievements(student_data):
    """Check for new achievements"""
    new_achievements = []
    current_achievements = student_data.get('achievements', [])
    
    # First lesson achievement
    if len(student_data['lessons_completed']) >= 1 and 'first_lesson' not in current_achievements:
        new_achievements.append('first_lesson')
    
    # Science explorer
    science_lessons = sum(1 for lesson in student_data['lessons_completed'] if 'science' in lesson)
    if science_lessons >= 5 and 'science_explorer' not in current_achievements:
        new_achievements.append('science_explorer')
    
    # Math wizard  
    math_lessons = sum(1 for lesson in student_data['lessons_completed'] if 'math' in lesson)
    if math_lessons >= 5 and 'math_wizard' not in current_achievements:
        new_achievements.append('math_wizard')
    
    # Perfect score
    for lesson_data in student_data['lessons_completed'].values():
        if lesson_data['accuracy'] >= 100 and 'perfect_score' not in current_achievements:
            new_achievements.append('perfect_score')
            break
    
    # XP milestones
    if student_data['xp'] >= 1000 and 'xp_collector' not in current_achievements:
        new_achievements.append('xp_collector')
    
    # Question milestone
    if student_data['total_questions_answered'] >= 100 and 'knowledge_seeker' not in current_achievements:
        new_achievements.append('knowledge_seeker')
    
    return new_achievements

def generate_fallback_questions(lesson_id, count):
    """Generate basic questions when database is empty"""
    topics = {
        'light-class6': ['luminous objects', 'shadows', 'reflection'],
        'integers-class6': ['positive numbers', 'negative numbers', 'addition'],
        'heat-class7': ['temperature', 'heat transfer', 'conduction'],
        'algebra-class7': ['variables', 'equations', 'solving']
    }
    
    topic_list = topics.get(lesson_id, ['general concepts'])
    
    questions = []
    for i in range(min(count, 3)):  # Generate up to 3 fallback questions
        questions.append({
            'id': i + 1,
            'text': f'Which concept is most important in understanding {topic_list[i % len(topic_list)]}?',
            'options': [f'Concept A about {topic_list[0]}', f'Concept B about {topic_list[-1]}', 'Basic understanding', 'All of the above'],
            'correct': 2,
            'explanation': 'Basic understanding is fundamental to learning any concept.',
            'concept': topic_list[i % len(topic_list)],
            'difficulty': 'easy'
        })
    
    return questions

@duolingo_api.route('/questions/status', methods=['GET'])
def question_status():
    scope = request.args.get('scope', 'mixed')
    lesson_id = request.args.get('lesson_id')
    subject = request.args.get('subject')
    class_level = request.args.get('class', type=int)
    difficulties_param = request.args.get('difficulties')  # e.g. easy,medium
    difficulties = [d.strip() for d in difficulties_param.split(',')] if difficulties_param else None

    if scope == 'lesson' and lesson_id:
        return jsonify({'success': True, 'status': get_lesson_status(lesson_id, difficulties)})
    if scope == 'subject_class' and subject and class_level is not None:
        return jsonify({'success': True, 'status': get_subject_class_status(subject, class_level, difficulties)})
    if scope == 'mixed':
        return jsonify({'success': True, 'status': get_mixed_status(difficulties)})
    return jsonify({'success': False, 'error': 'Invalid parameters'}), 400

@duolingo_api.route('/questions/user/status', methods=['GET'])
def user_question_status():
    user_id = request.args.get('student_id') or request.args.get('user_id') or 'anonymous'
    scope = request.args.get('scope', 'mixed')
    lesson_id = request.args.get('lesson_id')
    subject = request.args.get('subject')
    class_level = request.args.get('class', type=int)
    difficulties_param = request.args.get('difficulties')
    difficulties = [d.strip() for d in difficulties_param.split(',')] if difficulties_param else None

    if scope == 'lesson' and lesson_id:
        return jsonify({'success': True, 'status': get_user_lesson_status(user_id, lesson_id, difficulties)})
    if scope == 'subject_class' and subject and class_level is not None:
        return jsonify({'success': True, 'status': get_user_subject_class_status(user_id, subject, class_level, difficulties)})
    if scope == 'mixed':
        return jsonify({'success': True, 'status': get_user_mixed_status(user_id, difficulties)})
    return jsonify({'success': False, 'error': 'Invalid parameters'}), 400

@duolingo_api.route('/questions/user/lesson/filtered', methods=['POST'])
def get_user_lesson_filtered():
    data = request.json or {}
    user_id = data.get('student_id') or data.get('user_id') or 'anonymous'
    lesson_id = data.get('lesson_id')
    difficulties = data.get('difficulties')  # list or comma string
    if isinstance(difficulties, str):
        difficulties = [d.strip() for d in difficulties.split(',') if d.strip()]
    count = int(data.get('count', 10))
    if not lesson_id:
        return jsonify({'success': False, 'error': 'lesson_id required'}), 400
    qs, status_obj = fetch_user_lesson_questions(user_id, lesson_id, count, difficulties)
    safe = [{
        'id': q['id'], 'text': q['text'], 'options': q['options'],
        'concept': q.get('concept'), 'difficulty': q.get('difficulty'),
        'reward_preview': DIFFICULTY_LEVELS.get(q.get('difficulty','easy'), {})
    } for q in qs]
    return jsonify({'success': True, 'questions': safe, 'status': status_obj})

@duolingo_api.route('/questions/user/subject/filtered', methods=['POST'])
def get_user_subject_filtered():
    data = request.json or {}
    user_id = data.get('student_id') or data.get('user_id') or 'anonymous'
    subject = data.get('subject')
    class_level = data.get('class') or data.get('class_level')
    difficulties = data.get('difficulties')
    if isinstance(difficulties, str):
        difficulties = [d.strip() for d in difficulties.split(',') if d.strip()]
    count = int(data.get('count', 10))
    if not subject:
        return jsonify({'success': False, 'error': 'subject required'}), 400
    qs, status_obj = fetch_user_subject_class_questions(user_id, subject, class_level, count, difficulties)
    safe = [{
        'id': q['id'], 'text': q['text'], 'options': q['options'],
        'concept': q.get('concept'), 'difficulty': q.get('difficulty'),
        'reward_preview': DIFFICULTY_LEVELS.get(q.get('difficulty','easy'), {})
    } for q in qs]
    return jsonify({'success': True, 'questions': safe, 'status': status_obj})

@duolingo_api.route('/questions/user/mixed/filtered', methods=['POST'])
def get_user_mixed_filtered():
    data = request.json or {}
    user_id = data.get('student_id') or data.get('user_id') or 'anonymous'
    difficulties = data.get('difficulties')
    if isinstance(difficulties, str):
        difficulties = [d.strip() for d in difficulties.split(',') if d.strip()]
    count = int(data.get('count', 10))
    qs, status_obj = fetch_user_mixed_questions(user_id, count, difficulties)
    safe = [{
        'id': q['id'], 'text': q['text'], 'options': q['options'],
        'concept': q.get('concept'), 'difficulty': q.get('difficulty'),
        'reward_preview': DIFFICULTY_LEVELS.get(q.get('difficulty','easy'), {})
    } for q in qs]
    return jsonify({'success': True, 'questions': safe, 'status': status_obj})

@duolingo_api.route('/review/<student_id>', methods=['GET'])
def review_all(student_id):
    wrong_only = request.args.get('wrong_only', 'false').lower() == 'true'
    lesson_id = request.args.get('lesson_id')
    items = get_review_items(student_id, lesson_id, wrong_only)
    return jsonify({'success': True, 'count': len(items), 'items': items})

@duolingo_api.route('/adaptive/reset', methods=['POST'])
def adaptive_reset():
    data = request.json or {}
    user_id = data.get('student_id') or data.get('user_id')
    lesson_id = data.get('lesson_id')
    if not user_id or not lesson_id:
        return jsonify({'success': False, 'error': 'student_id and lesson_id required'}), 400
    ok = adaptive_reset_user_lesson(user_id, lesson_id)
    return jsonify({'success': True, 'adaptive': ok})

# Hook answer tracking inside existing check_answer response (wrap existing)
# (For brevity we patch after definition if available)

# Macro-Synthesis feature removed per request (routes deleted intentionally)