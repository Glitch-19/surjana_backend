import requests
import json
from config import Config
from typing import List, Dict, Any, Optional

class AIService:
    def __init__(self):
        self.groq_api_key = Config.GROQ_API_KEY
        self.groq_url = Config.GROQ_API_URL
        self.ncert_context = Config.NCERT_CONTEXT

    # ---------------- Rich Hint & Explanation Layer -----------------
    def get_rich_hint(self, question_data: Dict[str, Any], user_attempt: Optional[str] = None, selected_option: Optional[int] = None) -> Dict[str, Any]:
        """Return a structured multi-layer tutoring hint.

        question_data expected keys: text/question, options, correct, explanation, concept, difficulty
        Returns dict with: hint, concept_recap, misconception, steps (list), mnemonic, encouragement,
        wrong_option_explanation (if attempt wrong), answer_explanation, source ('ai'|'fallback').
        """
        base_question = question_data.get('question') or question_data.get('text') or ''
        options = question_data.get('options', [])
        correct_index = question_data.get('correct')
        concept = question_data.get('concept') or 'Concept'
        explanation = question_data.get('explanation') or ''

        # Determine if we can call AI
        if not self.groq_api_key:
            return self._fallback_rich_hint(base_question, options, correct_index, concept, explanation, user_attempt, selected_option)

        # Build JSON-only prompt for deterministic parsing
        attempt_text = user_attempt or ''
        selected_opt_text = None
        if selected_option is not None and 0 <= selected_option < len(options):
            selected_opt_text = options[selected_option]
        correct_opt_text = None
        if isinstance(correct_index, int) and 0 <= correct_index < len(options):
            correct_opt_text = options[correct_index]

        # Shorter, stricter prompt specification
        prompt = f"""
You are an NCERT-aligned tutoring assistant for Classes 6-10.
STRICT OUTPUT: Return ONLY valid JSON (no markdown) with keys:
{{
  "hint": "concise scaffolding (<=25 words)",
  "concept_recap": "very short recap (<=25 words)",
  "common_misconception": "a misconception students often have",
  "steps": ["step 1", "step 2", "step 3"],
  "mnemonic": "memory aid (<=15 words)",
  "encouragement": "motivational phrase",
  "wrong_option_explanation": "why the chosen wrong option is incorrect",
  "answer_explanation": "why correct is right (<=35 words)",
  "answer_validation_rationale": "concise logical reasoning chain"
}}

Question: {base_question}
Options: {options}
Concept: {concept}
Correct Answer Index: {correct_index}
Correct Answer Text: {correct_opt_text}
Official Explanation: {explanation}
Student Attempt Text: {attempt_text}
Selected Option Index: {selected_option}
Selected Option Text: {selected_opt_text}

Rules:
- If no student attempt, set wrong_option_explanation to "".
- steps length 2-4; each step <=10 words.
- Never reveal the correct answer explicitly inside hint.
- Keep JSON valid (double quotes escaped) with UTF-8.
"""

        ai_result = self._call_ai_api(prompt, expect_json=True)
        if ai_result.get('success') and isinstance(ai_result.get('content'), dict):
            data = ai_result['content']
            # Fill missing keys gracefully
            result = {
                'source': 'ai',
                'hint': data.get('hint') or self._fallback_hint_sentence(concept, base_question),
                'concept_recap': data.get('concept_recap') or self._short_concept_recap(concept, explanation),
                'misconception': data.get('common_misconception') or self._generic_misconception(concept),
                'steps': data.get('steps') or self._fallback_steps(concept, base_question, options),
                'mnemonic': data.get('mnemonic') or self._mnemonic_from_options(options, concept),
                'encouragement': data.get('encouragement') or self._encouragement_line(),
                'wrong_option_explanation': data.get('wrong_option_explanation') or ('' if selected_option is None or selected_option == correct_index else self._explain_wrong_option(selected_option, correct_index, options, concept)),
                'answer_explanation': data.get('answer_explanation') or explanation,
                'answer_validation_rationale': data.get('answer_validation_rationale') or self._validation_chain(concept, base_question, correct_opt_text, explanation)
            }
            # Enforce brevity
            result['hint'] = self._trim_words(result['hint'], 25)
            result['concept_recap'] = self._trim_words(result['concept_recap'], 25)
            result['mnemonic'] = self._trim_words(result['mnemonic'], 15)
            result['answer_explanation'] = self._trim_words(result['answer_explanation'], 35)
            if isinstance(result.get('steps'), list):
                trimmed = []
                for s in result['steps'][:4]:
                    trimmed.append(self._trim_words(str(s), 10))
                result['steps'] = trimmed
            return result
        # fallback parsing failed
        return self._fallback_rich_hint(base_question, options, correct_index, concept, explanation, user_attempt, selected_option)

    def get_wrong_answer_analysis(self, question_data: Dict[str, Any], selected_option: int) -> Dict[str, Any]:
        """Return targeted explanation for a wrong selected option."""
        options = question_data.get('options', [])
        correct_index = question_data.get('correct')
        concept = question_data.get('concept') or 'Concept'
        explanation = question_data.get('explanation') or ''
        question_text = question_data.get('question') or question_data.get('text') or ''
        if selected_option == correct_index:
            return {'message': 'Selected option is correct.', 'analysis': explanation, 'source': 'local'}
        if not self.groq_api_key:
            return {
                'message': 'Analysis generated (fallback).',
                'analysis': self._explain_wrong_option(selected_option, correct_index, options, concept),
                'source': 'fallback'
            }
        wrong_opt_text = options[selected_option] if 0 <= selected_option < len(options) else 'Unknown'
        correct_opt_text = options[correct_index] if isinstance(correct_index, int) and 0 <= correct_index < len(options) else 'Unknown'
        prompt = f"""
You are an NCERT tutor. Explain why the student's chosen option is wrong.
Return ONLY JSON with keys: {{"wrong_option_explanation":"...","contrast_with_correct":"..."}}
Question: {question_text}
Options: {options}
Concept: {concept}
Chosen Wrong Option: {wrong_opt_text}
Correct Option: {correct_opt_text}
Official Explanation: {explanation}
Keep each value <=60 words and do not restate entire question.
"""
        result = self._call_ai_api(prompt, expect_json=True)
        if result.get('success') and isinstance(result.get('content'), dict):
            return {'source': 'ai', **result['content']}
        return {
            'source': 'fallback',
            'wrong_option_explanation': self._explain_wrong_option(selected_option, correct_index, options, concept),
            'contrast_with_correct': explanation
        }

    # ---------------- Fallback Helpers -----------------
    def _fallback_rich_hint(self, question: str, options: List[str], correct_index: Any, concept: str, explanation: str, user_attempt: Optional[str], selected_option: Optional[int]):
        import random
        # Slight variation pool so every fallback hint isn't identical, kept short
        variant_phrases = [
            f"Focus on the core rule in {concept.lower()}.",
            f"Which option matches {concept.lower()}?",
            "Eliminate off-topic choices.",
            f"What must be true for {concept.lower()}?",
        ]
        base_hint = self._fallback_hint_sentence(concept, question)
        variation = random.choice(variant_phrases)
        combined_hint_full = base_hint if variation.lower() in base_hint.lower() else f"{base_hint} {variation}"
        combined_hint = self._trim_words(combined_hint_full, 25)
        return {
            'source': 'fallback',
            'hint': combined_hint,
            'concept_recap': self._short_concept_recap(concept, explanation),
            'misconception': self._generic_misconception(concept),
            'steps': self._fallback_steps(concept, question, options),
            'mnemonic': self._mnemonic_from_options(options, concept),
            'encouragement': self._encouragement_line(),
            'wrong_option_explanation': '' if selected_option is None or selected_option == correct_index else self._explain_wrong_option(selected_option, correct_index, options, concept),
            'answer_explanation': self._trim_words(explanation, 35),
            'answer_validation_rationale': self._validation_chain(concept, question, (options[correct_index] if isinstance(correct_index, int) and 0<=correct_index<len(options) else ''), explanation)
        }

    def _fallback_hint_sentence(self, concept: str, question: str):
        return f"Clarify the core idea in {concept.lower()} before scanning options."

    def _short_concept_recap(self, concept: str, explanation: str):
        if not explanation:
            return f"{concept} relies on its core rule."[:120]
        return explanation.split('.')[:1][0][:140]

    def _generic_misconception(self, concept: str):
        return f"Students confuse surface details with the rule in {concept.lower()}."[:140]

    def _fallback_steps(self, concept: str, question: str, options: List[str]):
        steps = [
            f"Identify concept {concept}.",
            "Remove unrelated options.",
            "Match definition precisely.",
            "Select best fit.",
        ]
        return [self._trim_words(s, 10) for s in steps][:4]

    def _mnemonic_from_options(self, options: List[str], concept: str):
        if not options:
            return f"Link {concept} to a story."[:60]
        initials = ''.join(o[0] for o in options[:4]).upper()
        return f"{concept}:{initials}"[:60]

    def _encouragement_line(self):
        return "Nice focus—keep going!"

    def _explain_wrong_option(self, selected_index: int, correct_index: Any, options: List[str], concept: str):
        if not options or not isinstance(selected_index, int) or selected_index < 0 or selected_index >= len(options):
            return "That choice misses the core rule—recheck the concept."
        chosen = options[selected_index]
        if isinstance(correct_index, int) and 0 <= correct_index < len(options):
            correct_text = options[correct_index]
            return f"'{chosen}' misses a requirement that '{correct_text}' satisfies for {concept.lower()}."[:180]
        return f"'{chosen}' doesn't fully match {concept.lower()}."[:160]

    def _validation_chain(self, concept: str, question: str, correct_text: Optional[str], explanation: str):
        base = explanation or f"Apply the rule of {concept.lower()} logically."
        return base[:220]

    # Utility to trim by word count
    def _trim_words(self, text: str, max_words: int) -> str:
        if not text:
            return text
        words = str(text).split()
        if len(words) <= max_words:
            return text.strip()
        return ' '.join(words[:max_words]).rstrip(',.;:') + '…'

# Global function for easy access
def get_ai_hint(question, options):
    """
    Simple function to get AI hint for a question with options
    """
    ai_service = AIService()
    
    # Create simplified question data for hint generation
    question_data = {
        'question': question,
        'options': options,
        'class': '6-10',
        'subject': 'General',
        'chapter': 'Mixed Topics',
        'answer': 'Check options carefully'
    }
    
    prompt = f"""
    Help a student with this question:
    Question: {question}
    Options: {', '.join(options) if isinstance(options, list) else options}
    
    Provide a helpful hint that:
    1. Doesn't give away the answer directly
    2. Guides them to think about the concept
    3. Uses encouraging language
    4. Keeps it under 60 words
    
    Example: "Think about what you learned about this topic. Look for key words in the question that connect to the concept."
    """
    
    result = ai_service._call_ai_api(prompt)
    return result.get('content', 'Take your time and think through each option carefully!')

def get_step_by_step_guidance(question, subject, student_level="beginner"):
    """
    Get detailed step-by-step guidance for solving a problem
    """
    ai_service = AIService()
    
    prompt = f"""
    As a friendly AI tutor, provide step-by-step guidance for this {subject} question:
    
    Question: {question}
    Student Level: {student_level}
    
    Break down the solution into 3-4 simple steps:
    1. First, identify what the question is asking
    2. Then, explain the key concept involved
    3. Show how to approach the problem
    4. Give a final tip for similar questions
    
    Use encouraging, game-like language. Make it feel like unlocking levels in a video game!
    Keep each step under 25 words.
    
    Example format:
    🎯 Step 1: [identification step]
    🔍 Step 2: [concept explanation]
    ⚡ Step 3: [solution approach]
    🏆 Pro Tip: [general advice]
    """
    
    result = ai_service._call_ai_api(prompt)
    return result.get('content', 'Break the problem into smaller parts and solve step by step!')

def get_memory_technique(concept, subject, grade_level):
    """
    Generate fun memory techniques and mnemonics for learning concepts
    """
    ai_service = AIService()
    
    prompt = f"""
    Create a fun, memorable learning technique for this concept:
    
    Subject: {subject}
    Grade: {grade_level}
    Concept: {concept}
    
    Provide:
    1. A catchy mnemonic or memory trick
    2. A simple rhyme or song (optional)
    3. A visual association or story
    4. A game-based way to remember it
    
    Make it:
    - Fun and engaging for students
    - Easy to remember
    - Appropriate for the grade level
    - Creative and unique
    
    Example format:
    🎵 Mnemonic: [memory device]
    📖 Story: [visual/story association]
    🎮 Game Trick: [gamified memory method]
    
    Keep total response under 150 words.
    """
    
    result = ai_service._call_ai_api(prompt)
    return result.get('content', 'Try creating a story or song to remember this concept!')

def get_gamified_explanation(question, correct_answer, subject):
    """
    Explain the answer in a fun, game-themed way
    """
    ai_service = AIService()
    
    prompt = f"""
    Explain this {subject} answer like you're a game character giving a quest reward explanation:
    
    Question: {question}
    Correct Answer: {correct_answer}
    
    Make the explanation:
    - Exciting and game-themed
    - Clear and educational
    - Celebratory (like winning a level)
    - Include power-up or skill-gained metaphors
    
    Use gaming language like:
    - "You've unlocked..."
    - "Level up! You now know..."
    - "Achievement earned..."
    - "New skill acquired..."
    
    Keep it under 100 words and make learning feel like gaming!
    """
    
    result = ai_service._call_ai_api(prompt)
    return result.get('content', f'Excellent! You have mastered this {subject} concept!')
