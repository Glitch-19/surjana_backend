from flask import Blueprint, request, jsonify
from services.ai_service import AIService
from utils.data_loader import DataLoader
import json

ai_bp = Blueprint('ai', __name__)
ai_service = AIService()
data_loader = DataLoader()

@ai_bp.route('/hint', methods=['POST'])
def get_hint():
    """AI-generated hint for a question.
    Modes:
      - default: simple free-text hint
      - rich: structured JSON with multi-layer tutoring (set payload {"mode":"rich"} or query ?mode=rich)
    Expected payload: { question_id:int, user_attempt?:str, mode?:"rich" }
    """
    try:
        data = request.get_json() or {}
        question_id = data.get('question_id')
        user_attempt = data.get('user_attempt')
        mode = data.get('mode') or request.args.get('mode','default')

        if not question_id:
            return jsonify({"success": False, "error": "question_id required"}), 400

        question_data = data_loader.get_question_by_id(question_id)
        if not question_data:
            return jsonify({"success": False, "error": "Question not found"}), 404

        if mode == 'rich':
            rich = ai_service.get_rich_hint(question_data, user_attempt)
            return jsonify({
                'success': True,
                'mode': 'rich',
                'question_id': question_id,
                **rich
            })
        else:
            hint_response = ai_service.get_hint(question_data, user_attempt)
            return jsonify({
                'success': True,
                'mode': 'simple',
                'hint': hint_response.get('content'),
                'question_id': question_id,
                'gamify_suggestion': question_data.get('gamify','')
            })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@ai_bp.route('/mnemonic', methods=['POST'])
def get_mnemonic():
    """
    Get AI-generated mnemonic for a concept
    Expected payload: {
        "question_id": int
    }
    """
    try:
        data = request.get_json()
        question_id = data.get('question_id')
        
        # Get question data from dataset
        question_data = data_loader.get_question_by_id(question_id)
        if not question_data:
            return jsonify({
                "error": "Question not found",
                "success": False
            }), 404
        
        # Generate mnemonic using AI service
        mnemonic_response = ai_service.get_mnemonic(question_data)
        
        return jsonify({
            "success": True,
            "mnemonic": mnemonic_response['content'],
            "question_id": question_id
        })
        
    except Exception as e:
        return jsonify({
            "error": str(e),
            "success": False
        }), 500

@ai_bp.route('/solution-steps', methods=['POST'])
def get_solution_steps():
    """
    Get AI-generated step-by-step solution guidance
    Expected payload: {
        "question_id": int
    }
    """
    try:
        data = request.get_json()
        question_id = data.get('question_id')
        
        # Get question data from dataset
        question_data = data_loader.get_question_by_id(question_id)
        if not question_data:
            return jsonify({
                "error": "Question not found",
                "success": False
            }), 404
        
        # Generate solution steps using AI service
        solution_response = ai_service.get_step_by_step_solution(question_data)
        
        return jsonify({
            "success": True,
            "solution_steps": solution_response['content'],
            "question_id": question_id
        })
        
    except Exception as e:
        return jsonify({
            "error": str(e),
            "success": False
        }), 500

@ai_bp.route('/encouragement', methods=['POST'])
def get_encouragement():
    """
    Get AI-generated gamified encouragement
    Expected payload: {
        "question_id": int,
        "is_correct": boolean
    }
    """
    try:
        data = request.get_json()
        question_id = data.get('question_id')
        is_correct = data.get('is_correct', False)
        
        # Get question data from dataset
        question_data = data_loader.get_question_by_id(question_id)
        if not question_data:
            return jsonify({
                "error": "Question not found",
                "success": False
            }), 404
        
        # Generate encouragement using AI service
        encouragement_response = ai_service.get_gamified_encouragement(question_data, is_correct)
        
        return jsonify({
            "success": True,
            "encouragement": encouragement_response['content'],
            "question_id": question_id,
            "is_correct": is_correct
        })
        
    except Exception as e:
        return jsonify({
            "error": str(e),
            "success": False
        }), 500
