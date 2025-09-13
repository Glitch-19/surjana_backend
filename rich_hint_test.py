import sys, json
sys.path.append('edu-game-backend')
from app import create_app
app = create_app()
client = app.test_client()
# assume question id 1 exists in dataset (NCERT loader)
resp = client.post('/api/ai/hint?mode=rich', json={'question_id':1, 'mode':'rich'})
print('STATUS', resp.status_code)
print('BODY', json.dumps(resp.json, indent=2))
