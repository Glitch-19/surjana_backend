import os
import json
from datetime import datetime, timedelta
from typing import List, Dict, Any


MACRO_DB_PATH = os.path.join('data', 'macro_synthesis_progress.json')


def _load_db() -> Dict[str, Any]:
    if not os.path.exists(MACRO_DB_PATH):
        return {}
    try:
        with open(MACRO_DB_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return {}


def _save_db(data: Dict[str, Any]):
    os.makedirs(os.path.dirname(MACRO_DB_PATH), exist_ok=True)
    with open(MACRO_DB_PATH, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)


class MacroSynthesisService:
    """Tracks completion of micro-chunks (flashcards/questions) and emits macro-synthesis prompts.

    Terminology:
      micro-chunk: smallest learning unit (flashcard, single question, concept snippet)
      macro-synthesis window: batch of N related micro-chunks that once completed triggers reflective prompt
    """

    def __init__(self, threshold: int = 2, cooldown_minutes: int = 20):
        self.threshold = threshold
        self.cooldown = timedelta(minutes=cooldown_minutes)

    def record_micro_chunk(self, student_id: str, topic: str, chunk_id: str, concept: str = None):
        db = _load_db()
        student = db.setdefault(student_id, {})
        topic_block = student.setdefault(topic, {
            'recent_chunks': [],  # list of {id, concept, at}
            'last_macro_prompt_at': None,
            'macro_history': []  # list of {generated_at, chunk_ids, prompt, student_response}
        })
        # prevent duplicate immediate entries
        if any(c['id'] == chunk_id for c in topic_block['recent_chunks']):
            return self._macro_status(topic_block)
        topic_block['recent_chunks'].append({
            'id': chunk_id,
            'concept': concept,
            'at': datetime.utcnow().isoformat()
        })
        # keep only last 25 to avoid growth
        if len(topic_block['recent_chunks']) > 25:
            topic_block['recent_chunks'] = topic_block['recent_chunks'][-25:]
        _save_db(db)
        return self._macro_status(topic_block)

    def _macro_status(self, topic_block: Dict[str, Any]):
        last_prompt_at = topic_block.get('last_macro_prompt_at')
        last_dt = datetime.fromisoformat(last_prompt_at) if last_prompt_at else None
        now = datetime.utcnow()
        eligible = len(topic_block['recent_chunks']) >= self.threshold and (not last_dt or (now - last_dt) >= self.cooldown)
        return {
            'recent_count': len(topic_block['recent_chunks']),
            'threshold': self.threshold,
            'eligible': eligible,
            'cooldown_remaining_seconds': 0 if not last_dt else max(0, int((self.cooldown - (now - last_dt)).total_seconds())) if last_dt else 0
        }

    def maybe_generate_prompt(self, student_id: str, topic: str):
        db = _load_db()
        topic_block = db.get(student_id, {}).get(topic)
        if not topic_block:
            return {'eligible': False, 'reason': 'no_progress'}
        status = self._macro_status(topic_block)
        if not status['eligible']:
            return {'eligible': False, **status}
        chunk_subset = topic_block['recent_chunks'][-self.threshold:]
        concepts = [c.get('concept') for c in chunk_subset if c.get('concept')]
        concept_part = ''
        if concepts:
            unique_concepts = sorted(set(concepts))
            concept_part = f" They covered micro-concepts: {', '.join(unique_concepts)}."
        prompt = (
            f"Macro-Synthesis: You've just worked through {self.threshold} micro-points about {topic}." \
            f" Now synthesize: Describe how these pieces connect to explain the bigger idea of {topic}." \
            f" Include cause-effect links, sequences, and why this matters in real life.{concept_part}"
        )
        # record macro prompt
        record = {
            'generated_at': datetime.utcnow().isoformat(),
            'chunk_ids': [c['id'] for c in chunk_subset],
            'prompt': prompt,
            'student_response': None
        }
        topic_block['macro_history'].append(record)
        topic_block['last_macro_prompt_at'] = record['generated_at']
        _save_db(db)
        return {'eligible': True, 'prompt': prompt, 'history_index': len(topic_block['macro_history']) - 1}

    def submit_response(self, student_id: str, topic: str, history_index: int, response_text: str):
        db = _load_db()
        topic_block = db.get(student_id, {}).get(topic)
        if not topic_block:
            return {'success': False, 'error': 'not_found'}
        if history_index < 0 or history_index >= len(topic_block['macro_history']):
            return {'success': False, 'error': 'invalid_index'}
        topic_block['macro_history'][history_index]['student_response'] = {
            'text': response_text,
            'submitted_at': datetime.utcnow().isoformat()
        }
        _save_db(db)
        return {'success': True}

    def get_status(self, student_id: str, topic: str):
        db = _load_db()
        topic_block = db.get(student_id, {}).get(topic)
        if not topic_block:
            return {'recent_count': 0, 'threshold': self.threshold, 'eligible': False}
        status = self._macro_status(topic_block)
        status['history_length'] = len(topic_block['macro_history'])
        return status


macro_synthesis_service = MacroSynthesisService()
