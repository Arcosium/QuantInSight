"""Fixed-host model adapters. Only the user's strategy text leaves this server."""
import json
import re
import urllib.request
import urllib.error
from fastapi import HTTPException

ENDPOINTS = {
    'openai': 'https://api.openai.com/v1/chat/completions',
    'deepseek': 'https://api.deepseek.com/chat/completions',
    'openrouter': 'https://openrouter.ai/api/v1/chat/completions',
    'anthropic': 'https://api.anthropic.com/v1/messages',
}

class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        raise HTTPException(502, 'API 제공사가 예상하지 못한 주소로 응답했습니다.')


def build_request(provider, key, model, system, prompt):
    if '..' in model or not re.fullmatch(r'[A-Za-z0-9_./:\-]{1,160}', model):
        raise HTTPException(422, '모델 ID를 확인해 주세요.')
    headers={'Content-Type':'application/json'}
    if provider=='gemini':
        url='https://generativelanguage.googleapis.com/v1beta/models/'+model.removeprefix('models/')+':generateContent'
        headers['x-goog-api-key']=key
        body=dict(systemInstruction=dict(parts=[dict(text=system)]),contents=[dict(role='user',parts=[dict(text=prompt)])],generationConfig=dict(maxOutputTokens=4000,responseMimeType='application/json'))
    elif provider=='anthropic':
        url=ENDPOINTS[provider]
        headers.update({'x-api-key':key,'anthropic-version':'2023-06-01'})
        body=dict(model=model,max_tokens=4000,system=system,messages=[dict(role='user',content=prompt)])
    elif provider in ENDPOINTS or provider=='local':
        url=ENDPOINTS.get(provider,'http://127.0.0.1:11434/v1/chat/completions')
        if provider!='local':headers['Authorization']='Bearer '+key
        body=dict(model=model,messages=[dict(role='system',content=system),dict(role='user',content=prompt)],stream=False,response_format=dict(type='json_object'))
        body['max_completion_tokens' if provider=='openai' else 'max_tokens']=4000
        if provider=='local':body['chat_template_kwargs']=dict(enable_thinking=False)
    else:raise HTTPException(422,'지원하지 않는 API 제공사입니다.')
    return urllib.request.Request(url,data=json.dumps(body).encode(),headers=headers)


def generate(provider,key,model,system,prompt):
    request=build_request(provider,key,model,system,prompt)
    try:
        with urllib.request.build_opener(NoRedirect).open(request,timeout=180) as response:
            raw=response.read(1_000_001)
        if len(raw)>1_000_000:raise ValueError('response too large')
        result=json.loads(raw)
        if provider=='anthropic':content=''.join(c['text'] for c in result['content'] if c.get('type')=='text')
        elif provider=='gemini':content=''.join(c.get('text','') for c in result['candidates'][0]['content']['parts'] if not c.get('thought'))
        else:content=result['choices'][0]['message']['content']
        content=content.strip()
        if content.startswith('```') and content.endswith('```'):content=content.split('\n',1)[1].rsplit('```',1)[0].strip()
        return json.loads(content)
    except urllib.error.HTTPError as exc:
        status=exc.code
        raise HTTPException(502, f'API 요청 실패 ({status}). 키·모델 ID·사용 한도를 확인해 주세요.') from None
    except HTTPException:raise
    except Exception:raise HTTPException(502,'AI 응답을 처리하지 못했습니다. 모델과 전략 조건을 확인해 주세요.') from None
