import sys
sys.path.append('edu-game-backend')
from app import create_app
app = create_app()
client = app.test_client()
# Simulate 5 micro-chunk completions
for i in range(5):
    r = client.post('/api/duolingo/macro/micro-chunk/complete', json={'student_id':'s1','topic':'photosynthesis','chunk_id':f'card{i}','concept':f'step{i}'} )
print('After adds status:', r.json)
# Try to get prompt
p = client.post('/api/duolingo/macro/prompt', json={'student_id':'s1','topic':'photosynthesis'})
print('Prompt attempt:', p.json)
if p.json.get('prompt'):
    idx = p.json.get('history_index')
    resp = client.post('/api/duolingo/macro/response', json={'student_id':'s1','topic':'photosynthesis','history_index':idx,'response':'Plants convert light to chemical energy.'})
    print('Submit response:', resp.json)
status = client.get('/api/duolingo/macro/status?student_id=s1&topic=photosynthesis')
print('Final status:', status.json)
