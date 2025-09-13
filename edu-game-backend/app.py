from flask import Flask, jsonify, send_file
from flask_cors import CORS
from config import Config
import os

# Import routes
from routes.ai_routes import ai_bp
from routes.quiz_routes import quiz_bp
from routes.game_routes import game_bp
from routes.subway_surfer_api import game_api
from routes.rewards_api import rewards_api
from routes.duolingo_api import duolingo_api

def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)
    
    # Enable CORS for frontend integration
    CORS(app)
    
    # Register blueprints
    app.register_blueprint(ai_bp, url_prefix='/api/ai')
    app.register_blueprint(quiz_bp, url_prefix='/api/quiz')
    app.register_blueprint(game_bp, url_prefix='/api/game')
    app.register_blueprint(game_api, url_prefix='/api/subway-surfer')
    app.register_blueprint(rewards_api, url_prefix='/api/rewards')
    app.register_blueprint(duolingo_api, url_prefix='/api/duolingo')
    
    # Serve the Duolingo-style app
    @app.route('/duolingo')
    def duolingo_app():
        return send_file('duolingo_style_app.html')
    
    # Health check endpoint
    @app.route('/')
    def health_check():
        return jsonify({
            "message": "Edu Game Backend is running!",
            "version": "1.0.0",
            "apps": {
                "duolingo_learning": "/duolingo"
            },
            "endpoints": {
                "ai_hints": "/api/ai/hint",
                "ai_mnemonics": "/api/ai/mnemonic", 
                "ai_solutions": "/api/ai/solution-steps",
                "ai_encouragement": "/api/ai/encouragement",
                "quiz_questions": "/api/quiz/questions",
                "quiz_submit": "/api/quiz/submit",
                "game_start": "/api/game/start",
                "game_answer": "/api/game/answer",
                "game_leaderboard": "/api/game/leaderboard",
                "rewards_earn": "/api/rewards/tokens/earn",
                "rewards_shop": "/api/rewards/shop/catalog",
                "rewards_purchase": "/api/rewards/shop/purchase",
                "rewards_tokens": "/api/rewards/student/{id}/tokens",
                "rewards_orders": "/api/rewards/student/{id}/orders",
                "duolingo_questions": "/api/duolingo/questions/lesson",
                "duolingo_check": "/api/duolingo/answer/check",
                "duolingo_progress": "/api/duolingo/progress/student/{id}",
                "duolingo_leaderboard": "/api/duolingo/leaderboard",
                "duolingo_hints": "/api/duolingo/hint/ai",
                "duolingo_user_lesson": "/api/duolingo/questions/user/lesson",
                "duolingo_user_subject": "/api/duolingo/questions/user/subject",
                "duolingo_user_mixed": "/api/duolingo/questions/user/mixed",
                "duolingo_user_reset": "/api/duolingo/questions/user/reset"
            }
        })
    
    return app

if __name__ == '__main__':
    app = create_app()
    print("🚀 Starting Edu Game Backend...")
    print("📚 NCERT AI Tutor Ready!")
    print("🎮 Gamified Learning Engine Active!")
    app.run(debug=True, host='0.0.0.0', port=5000)
