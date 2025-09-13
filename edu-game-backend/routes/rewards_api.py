"""
Token Reward System and Student Shop
Manages earning tokens through learning and redeeming them for real-world rewards
"""
from flask import Blueprint, request, jsonify
import json
import os
from datetime import datetime
import uuid

rewards_api = Blueprint('rewards_api', __name__)

# Student rewards catalog
REWARDS_CATALOG = {
    "stationery": {
        "pencil_set": {
            "name": "Colorful Pencil Set (12 pieces)",
            "description": "Vibrant colored pencils perfect for creative learning!",
            "cost": 50,
            "image": "🖍️",
            "category": "Art Supplies",
            "age_group": "6-12",
            "stock": 100
        },
        "notebook_set": {
            "name": "Learning Champion Notebook",
            "description": "Premium quality notebook with inspiring quotes and learning tips",
            "cost": 75,
            "image": "📔",
            "category": "Writing",
            "age_group": "8-16",
            "stock": 50
        },
        "pencil_box": {
            "name": "Smart Student Pencil Box",
            "description": "Multi-compartment pencil box with calculator and ruler",
            "cost": 100,
            "image": "📦",
            "category": "Organization",
            "age_group": "10-16",
            "stock": 30
        }
    },
    "books": {
        "story_book": {
            "name": "Educational Adventure Stories",
            "description": "Collection of fun stories that teach science and math concepts",
            "cost": 125,
            "image": "📚",
            "category": "Educational Books",
            "age_group": "8-14",
            "stock": 25
        },
        "workbook": {
            "name": "Interactive Learning Workbook",
            "description": "Hands-on activities and puzzles for extra practice",
            "cost": 90,
            "image": "📝",
            "category": "Practice Books",
            "age_group": "6-12",
            "stock": 40
        }
    },
    "games": {
        "math_cards": {
            "name": "Math Challenge Cards",
            "description": "Fun card games that make math practice enjoyable",
            "cost": 60,
            "image": "🃏",
            "category": "Educational Games",
            "age_group": "8-14",
            "stock": 60
        },
        "science_kit": {
            "name": "Junior Scientist Kit",
            "description": "Safe, fun experiments to explore science concepts",
            "cost": 200,
            "image": "🔬",
            "category": "STEM Kits",
            "age_group": "10-16",
            "stock": 15
        }
    },
    "digital": {
        "course_access": {
            "name": "Premium Learning Course Access (1 month)",
            "description": "Access to advanced interactive courses and AI tutoring",
            "cost": 150,
            "image": "💻",
            "category": "Digital Learning",
            "age_group": "6-18",
            "stock": 999
        },
        "certificate": {
            "name": "Digital Achievement Certificate",
            "description": "Personalized certificate recognizing learning achievements",
            "cost": 25,
            "image": "🏆",
            "category": "Recognition",
            "age_group": "6-18",
            "stock": 999
        }
    }
}

# File paths
TOKENS_DB = "data/student_tokens.json"
ORDERS_DB = "data/student_orders.json"

def load_student_data(db_file):
    """Load student data from JSON file"""
    if not os.path.exists(db_file):
        return {}
    try:
        with open(db_file, 'r') as f:
            return json.load(f)
    except:
        return {}

def save_student_data(db_file, data):
    """Save student data to JSON file"""
    os.makedirs(os.path.dirname(db_file), exist_ok=True)
    with open(db_file, 'w') as f:
        json.dump(data, f, indent=2)

@rewards_api.route('/tokens/earn', methods=['POST'])
def earn_tokens():
    """Award tokens for learning achievements"""
    try:
        data = request.json
        student_id = data.get('student_id')
        achievement_type = data.get('achievement_type')  # quiz_perfect, streak_5, level_complete, etc.
        subject = data.get('subject', 'General')
        score = data.get('score', 0)
        
        if not student_id or not achievement_type:
            return jsonify({'success': False, 'error': 'Missing required fields'}), 400
        
        # Load student tokens
        tokens_data = load_student_data(TOKENS_DB)
        
        if student_id not in tokens_data:
            tokens_data[student_id] = {
                'total_tokens': 0,
                'available_tokens': 0,
                'earned_history': [],
                'achievements': []
            }
        
        # Calculate token reward based on achievement
        token_rewards = {
            'quiz_perfect': 10,
            'quiz_good': 5,
            'streak_3': 5,
            'streak_5': 15,
            'streak_10': 30,
            'level_complete': 20,
            'daily_goal': 10,
            'weekly_challenge': 25,
            'subject_mastery': 50,
            'first_attempt_correct': 8,
            'improvement_milestone': 15
        }
        
        tokens_earned = token_rewards.get(achievement_type, 1)
        
        # Bonus tokens for high scores
        if score >= 95:
            tokens_earned += 5
        elif score >= 85:
            tokens_earned += 3
        
        # Update student tokens
        tokens_data[student_id]['total_tokens'] += tokens_earned
        tokens_data[student_id]['available_tokens'] += tokens_earned
        
        # Record achievement
        achievement_record = {
            'timestamp': datetime.now().isoformat(),
            'achievement_type': achievement_type,
            'tokens_earned': tokens_earned,
            'subject': subject,
            'score': score,
            'description': get_achievement_description(achievement_type, score)
        }
        
        tokens_data[student_id]['earned_history'].append(achievement_record)
        tokens_data[student_id]['achievements'].append(achievement_type)
        
        # Save data
        save_student_data(TOKENS_DB, tokens_data)
        
        return jsonify({
            'success': True,
            'tokens_earned': tokens_earned,
            'total_tokens': tokens_data[student_id]['total_tokens'],
            'available_tokens': tokens_data[student_id]['available_tokens'],
            'achievement': achievement_record
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@rewards_api.route('/shop/catalog', methods=['GET'])
def get_shop_catalog():
    """Get the rewards catalog"""
    try:
        # Add availability status to each item
        catalog_with_availability = {}
        
        for category, items in REWARDS_CATALOG.items():
            catalog_with_availability[category] = {}
            for item_id, item_data in items.items():
                catalog_with_availability[category][item_id] = {
                    **item_data,
                    'available': item_data['stock'] > 0,
                    'item_id': item_id
                }
        
        return jsonify({
            'success': True,
            'catalog': catalog_with_availability,
            'categories': list(REWARDS_CATALOG.keys())
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@rewards_api.route('/shop/purchase', methods=['POST'])
def purchase_reward():
    """Purchase a reward with tokens"""
    try:
        data = request.json
        student_id = data.get('student_id')
        item_id = data.get('item_id')
        category = data.get('category')
        delivery_info = data.get('delivery_info', {})
        
        if not all([student_id, item_id, category]):
            return jsonify({'success': False, 'error': 'Missing required fields'}), 400
        
        # Find the item
        if category not in REWARDS_CATALOG or item_id not in REWARDS_CATALOG[category]:
            return jsonify({'success': False, 'error': 'Item not found'}), 404
        
        item = REWARDS_CATALOG[category][item_id]
        
        # Check stock
        if item['stock'] <= 0:
            return jsonify({'success': False, 'error': 'Item out of stock'}), 400
        
        # Load student tokens
        tokens_data = load_student_data(TOKENS_DB)
        
        if student_id not in tokens_data:
            return jsonify({'success': False, 'error': 'Student not found'}), 404
        
        student_tokens = tokens_data[student_id]
        
        # Check if student has enough tokens
        if student_tokens['available_tokens'] < item['cost']:
            return jsonify({
                'success': False, 
                'error': 'Insufficient tokens',
                'needed': item['cost'],
                'available': student_tokens['available_tokens']
            }), 400
        
        # Process purchase
        order_id = str(uuid.uuid4())[:8]
        
        # Deduct tokens
        tokens_data[student_id]['available_tokens'] -= item['cost']
        
        # Record purchase in tokens history
        purchase_record = {
            'timestamp': datetime.now().isoformat(),
            'order_id': order_id,
            'item': item['name'],
            'cost': item['cost'],
            'type': 'purchase'
        }
        
        tokens_data[student_id]['earned_history'].append(purchase_record)
        
        # Save tokens data
        save_student_data(TOKENS_DB, tokens_data)
        
        # Create order record
        orders_data = load_student_data(ORDERS_DB)
        
        if student_id not in orders_data:
            orders_data[student_id] = []
        
        order = {
            'order_id': order_id,
            'timestamp': datetime.now().isoformat(),
            'item': {
                'id': item_id,
                'name': item['name'],
                'category': category,
                'cost': item['cost'],
                'description': item['description']
            },
            'delivery_info': delivery_info,
            'status': 'confirmed',
            'estimated_delivery': get_estimated_delivery(category)
        }
        
        orders_data[student_id].append(order)
        
        # Save orders data
        save_student_data(ORDERS_DB, orders_data)
        
        # Update stock (in real implementation, this would be in a database)
        REWARDS_CATALOG[category][item_id]['stock'] -= 1
        
        return jsonify({
            'success': True,
            'order': order,
            'remaining_tokens': tokens_data[student_id]['available_tokens'],
            'message': f'🎉 Purchase successful! Your {item["name"]} will be delivered soon!'
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@rewards_api.route('/student/<student_id>/tokens', methods=['GET'])
def get_student_tokens(student_id):
    """Get student's token balance and history"""
    try:
        tokens_data = load_student_data(TOKENS_DB)
        
        if student_id not in tokens_data:
            return jsonify({
                'success': True,
                'tokens': {
                    'total_tokens': 0,
                    'available_tokens': 0,
                    'earned_history': [],
                    'achievements': []
                }
            })
        
        return jsonify({
            'success': True,
            'tokens': tokens_data[student_id]
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@rewards_api.route('/student/<student_id>/orders', methods=['GET'])
def get_student_orders(student_id):
    """Get student's order history"""
    try:
        orders_data = load_student_data(ORDERS_DB)
        
        student_orders = orders_data.get(student_id, [])
        
        return jsonify({
            'success': True,
            'orders': student_orders,
            'total_orders': len(student_orders)
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@rewards_api.route('/leaderboard/tokens', methods=['GET'])
def get_token_leaderboard():
    """Get top students by tokens earned"""
    try:
        tokens_data = load_student_data(TOKENS_DB)
        
        leaderboard = []
        for student_id, data in tokens_data.items():
            leaderboard.append({
                'student_id': student_id,
                'total_tokens': data['total_tokens'],
                'achievements_count': len(set(data['achievements']))
            })
        
        # Sort by total tokens
        leaderboard.sort(key=lambda x: x['total_tokens'], reverse=True)
        
        return jsonify({
            'success': True,
            'leaderboard': leaderboard[:20],  # Top 20
            'total_students': len(leaderboard)
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

def get_achievement_description(achievement_type, score=0):
    """Get human-readable achievement description"""
    descriptions = {
        'quiz_perfect': f'Perfect quiz score! ({score}%)',
        'quiz_good': f'Great quiz performance! ({score}%)',
        'streak_3': 'Answered 3 questions correctly in a row!',
        'streak_5': 'Amazing 5-question streak!',
        'streak_10': 'Incredible 10-question streak!',
        'level_complete': 'Completed a learning level!',
        'daily_goal': 'Met daily learning goal!',
        'weekly_challenge': 'Completed weekly challenge!',
        'subject_mastery': 'Mastered a subject!',
        'first_attempt_correct': 'Got it right on the first try!',
        'improvement_milestone': 'Showed significant improvement!'
    }
    return descriptions.get(achievement_type, 'Learning achievement unlocked!')

def get_estimated_delivery(category):
    """Get estimated delivery time based on item category"""
    delivery_times = {
        'stationery': '3-5 business days',
        'books': '5-7 business days',
        'games': '7-10 business days',
        'digital': 'Instant access'
    }
    return delivery_times.get(category, '5-7 business days')