import random
import time
from datetime import datetime, timedelta

class GameMechanics:
    def __init__(self):
        self.game_types = {
            'subway_surfer': SubwaySurferMechanics(),
            'car_race': CarRaceMechanics(),
            'treasure_hunt': TreasureHuntMechanics(),
            'quiz_master': QuizMasterMechanics()
        }
    
    def start_game_session(self, game_type, player_id, difficulty='medium'):
        """Initialize a new game session"""
        if game_type not in self.game_types:
            return {"error": "Game type not supported"}
        
        session = {
            'session_id': f"{game_type}_{player_id}_{int(time.time())}",
            'game_type': game_type,
            'player_id': player_id,
            'level': 1,
            'score': 0,
            'lives': 3,
            'coins': 0,
            'power_ups': [],
            'achievements': [],
            'current_question': None,
            'questions_answered': 0,
            'correct_streak': 0,
            'start_time': datetime.now().isoformat(),
            'game_state': 'active'
        }
        
        return self.game_types[game_type].initialize_game(session, difficulty)
    
    def process_answer(self, session_id, game_type, answer, question_data):
        """Process player answer and update game state"""
        if game_type not in self.game_types:
            return {"error": "Game type not supported"}
        
        return self.game_types[game_type].process_answer(session_id, answer, question_data)

class SubwaySurferMechanics:
    """Endless runner where learning helps you survive longer"""
    
    def initialize_game(self, session, difficulty):
        session['distance'] = 0
        session['speed'] = 1.0
        session['obstacles_dodged'] = 0
        session['power_ups'] = ['magnet', 'jetpack', 'multiplier']
        
        game_state = {
            'session': session,
            'current_obstacle': self.generate_obstacle(),
            'next_power_up_distance': 100,
            'instructions': "🏃‍♂️ Answer questions to dodge obstacles and collect coins! Wrong answers slow you down!"
        }
        
        return game_state
    
    def generate_obstacle(self):
        """Generate random obstacles that represent learning challenges"""
        obstacles = [
            {'type': 'train', 'difficulty': 'hard', 'damage': 2},
            {'type': 'barrier', 'difficulty': 'medium', 'damage': 1},
            {'type': 'pit', 'difficulty': 'easy', 'damage': 1}
        ]
        return random.choice(obstacles)
    
    def process_answer(self, session_id, answer, question_data):
        """Process answer in context of subway surfer game"""
        is_correct = self.check_answer(answer, question_data['answer'])
        
        if is_correct:
            result = {
                'success': True,
                'action': 'dodge_obstacle',
                'rewards': {
                    'coins': 20 + (question_data['difficulty'] == 'hard' and 10 or 0),
                    'speed_boost': 0.2,
                    'distance': 50
                },
                'message': f"🎯 Perfect dodge! You jumped over the {self.generate_obstacle()['type']}!",
                'power_up': random.choice(['coin_magnet', 'shield', 'double_coins']) if random.random() < 0.3 else None
            }
        else:
            result = {
                'success': False,
                'action': 'hit_obstacle',
                'penalties': {
                    'lives': -1,
                    'speed_reduction': 0.3,
                    'coins': -5
                },
                'message': "💥 Ouch! You hit an obstacle. Study harder to run faster!",
                'hint': f"Think about: {question_data.get('chapter', 'the concept')}"
            }
        
        return result
    
    def check_answer(self, user_answer, correct_answer):
        return user_answer.lower().strip() == correct_answer.lower().strip()

class CarRaceMechanics:
    """Racing game where knowledge gives you speed advantages"""
    
    def initialize_game(self, session, difficulty):
        session['lap'] = 1
        session['position'] = 8  # Start at 8th position out of 8 cars
        session['fuel'] = 100
        session['nitro'] = 0
        session['track_progress'] = 0
        
        game_state = {
            'session': session,
            'current_lap_length': 1000,
            'opponents': self.generate_opponents(),
            'next_question_checkpoint': 200,
            'instructions': "🏎️ Answer questions at checkpoints to get nitro boosts and overtake opponents!"
        }
        
        return game_state
    
    def generate_opponents(self):
        """Generate AI opponents with educational themes"""
        opponents = [
            {'name': 'Newton Speedster', 'subject': 'Science', 'position': 1, 'speed': 0.9},
            {'name': 'Algebra Ace', 'subject': 'Mathematics', 'position': 2, 'speed': 0.85},
            {'name': 'History Hunter', 'subject': 'History', 'position': 3, 'speed': 0.8},
            {'name': 'Chemistry Chaser', 'subject': 'Science', 'position': 4, 'speed': 0.75},
            {'name': 'Geometry Genius', 'subject': 'Mathematics', 'position': 5, 'speed': 0.7},
            {'name': 'Geography Guide', 'subject': 'Geography', 'position': 6, 'speed': 0.65},
            {'name': 'Biology Bullet', 'subject': 'Science', 'position': 7, 'speed': 0.6}
        ]
        return opponents
    
    def process_answer(self, session_id, answer, question_data):
        is_correct = self.check_answer(answer, question_data['answer'])
        
        if is_correct:
            # Calculate position improvement based on difficulty
            position_boost = {'easy': 1, 'medium': 2, 'hard': 3}.get(question_data['difficulty'], 1)
            
            result = {
                'success': True,
                'action': 'nitro_boost',
                'rewards': {
                    'nitro': 50,
                    'position_improvement': position_boost,
                    'fuel': 20,
                    'track_progress': 100
                },
                'message': f"🚀 NITRO BOOST! You overtook {position_boost} cars with your knowledge!",
                'effect': 'speed_lines_effect'
            }
        else:
            result = {
                'success': False,
                'action': 'engine_trouble',
                'penalties': {
                    'fuel': -15,
                    'position_loss': 1,
                    'speed_reduction': 0.5
                },
                'message': "⚠️ Engine trouble! Wrong answer slowed you down!",
                'hint': f"Study {question_data.get('subject')} Chapter: {question_data.get('chapter', 'this topic')}"
            }
        
        return result
    
    def check_answer(self, user_answer, correct_answer):
        return user_answer.lower().strip() == correct_answer.lower().strip()

class TreasureHuntMechanics:
    """Adventure game where each answer reveals clues and treasures"""
    
    def initialize_game(self, session, difficulty):
        session['map_explored'] = 0  # Percentage of map explored
        session['treasures_found'] = 0
        session['clues_collected'] = []
        session['current_location'] = {'x': 0, 'y': 0}
        session['inventory'] = []
        
        game_state = {
            'session': session,
            'map_size': {'width': 10, 'height': 10},
            'treasure_locations': self.generate_treasure_map(),
            'current_clue': None,
            'instructions': "🗺️ Answer questions to unlock clues and find hidden treasures!"
        }
        
        return game_state
    
    def generate_treasure_map(self):
        """Generate treasure locations based on educational themes"""
        treasures = [
            {'location': {'x': 3, 'y': 2}, 'name': 'Newton\'s Apple', 'subject': 'Science', 'value': 100},
            {'location': {'x': 7, 'y': 5}, 'name': 'Pythagoras\' Theorem Scroll', 'subject': 'Mathematics', 'value': 150},
            {'location': {'x': 2, 'y': 8}, 'name': 'Ashoka\'s Edict', 'subject': 'History', 'value': 120},
            {'location': {'x': 9, 'y': 3}, 'name': 'DNA Helix Crystal', 'subject': 'Science', 'value': 200},
            {'location': {'x': 5, 'y': 7}, 'name': 'Golden Ratio Compass', 'subject': 'Mathematics', 'value': 180}
        ]
        return treasures
    
    def process_answer(self, session_id, answer, question_data):
        is_correct = self.check_answer(answer, question_data['answer'])
        
        if is_correct:
            # Generate clue based on question subject
            clue = self.generate_clue(question_data)
            
            result = {
                'success': True,
                'action': 'clue_revealed',
                'rewards': {
                    'clue': clue,
                    'map_reveal': 15,  # Reveal 15% more of the map
                    'treasure_hint': self.get_treasure_hint(question_data['subject']),
                    'exploration_points': 25
                },
                'message': f"🔍 Clue discovered! Your knowledge of {question_data['subject']} reveals secrets!",
                'visual_effect': 'map_reveal_animation'
            }
        else:
            result = {
                'success': False,
                'action': 'false_path',
                'penalties': {
                    'energy': -10,
                    'wrong_direction': True
                },
                'message': "🚫 Wrong path! The clue remains hidden.",
                'hint': f"Study more about {question_data.get('chapter')} to find the right path!"
            }
        
        return result
    
    def generate_clue(self, question_data):
        """Generate subject-specific treasure clues"""
        clue_templates = {
            'Science': [
                "The treasure lies where Newton's laws are strongest...",
                "Follow the path of photosynthesis to find green gold...",
                "Where elements combine, precious metals shine..."
            ],
            'Mathematics': [
                "Count the golden ratio steps from the ancient pillar...",
                "The treasure forms a perfect geometric pattern...",
                "Solve the equation of the ancient mathematician..."
            ],
            'History': [
                "In the shadow of the great emperor's pillar...",
                "Where ancient trade routes crossed the mighty river...",
                "The freedom fighter's secret hideout holds the key..."
            ]
        }
        
        subject = question_data.get('subject', 'History')
        return random.choice(clue_templates.get(subject, clue_templates['History']))
    
    def get_treasure_hint(self, subject):
        """Get direction hint based on subject"""
        hints = {
            'Science': "Look near the laboratory ruins to the east...",
            'Mathematics': "The geometric temple holds ancient wisdom...",
            'History': "Ancient monuments mark the spot..."
        }
        return hints.get(subject, "Explore the mysterious ruins...")
    
    def check_answer(self, user_answer, correct_answer):
        return user_answer.lower().strip() == correct_answer.lower().strip()

class QuizMasterMechanics:
    """Classic quiz show with dramatic effects and prizes"""
    
    def initialize_game(self, session, difficulty):
        session['round'] = 1
        session['prize_money'] = 0
        session['lifelines'] = ['50-50', 'phone_friend', 'audience_poll']
        session['risk_level'] = 'safe'
        
        game_state = {
            'session': session,
            'prize_ladder': [100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000],
            'safe_havens': [1000, 10000],
            'current_prize': 100,
            'instructions': "🎯 Answer questions to climb the prize ladder! Use lifelines wisely!"
        }
        
        return game_state
    
    def process_answer(self, session_id, answer, question_data):
        is_correct = self.check_answer(answer, question_data['answer'])
        
        if is_correct:
            result = {
                'success': True,
                'action': 'advance_ladder',
                'rewards': {
                    'prize_won': self.calculate_prize_money(),
                    'confidence_boost': 20,
                    'knowledge_points': 50
                },
                'message': "🏆 Correct! You're moving up the ladder of knowledge!",
                'dramatic_effect': 'spotlight_and_applause'
            }
        else:
            result = {
                'success': False,
                'action': 'game_over',
                'penalties': {
                    'prize_lost': True,
                    'fall_to_safe_haven': True
                },
                'message': "❌ Wrong answer! But learning never stops!",
                'consolation': f"You still earned valuable knowledge about {question_data.get('subject')}!"
            }
        
        return result
    
    def calculate_prize_money(self):
        # Simple prize calculation - can be made more sophisticated
        return random.randint(100, 1000)
    
    def check_answer(self, user_answer, correct_answer):
        return user_answer.lower().strip() == correct_answer.lower().strip()