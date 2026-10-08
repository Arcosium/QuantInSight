"""Same-origin view of the existing AutoCrypto paper books; no model changes."""
import hashlib
import base64
import re
import urllib.request
import urllib.error
from urllib.parse import quote
from fastapi import APIRouter,HTTPException,Request
from fastapi.responses import HTMLResponse,Response,FileResponse
from .config import HOME
from .auth import user
router=APIRouter()
SOURCE=HOME/'projects/AutoCrypto/web/static'

@router.get('/crypto/view')
def view(request:Request):
    user(request)
    html=(SOURCE/'index.html').read_text()
    html=html.replace('/api/','/crypto/api/').replace('/static/','/crypto/static/')
    html=re.sub(r'<link[^>]+(?:fonts.googleapis.com|fonts.gstatic.com)[^>]*>','',html)
    hashes=[]
    for script in re.findall(r'<script>([\s\S]*?)</script>',html):
        hashes.append("'sha256-"+base64.b64encode(hashlib.sha256(script.encode()).digest()).decode()+"'")
    csp="default-src 'self'; script-src 'self' https://static.cloudflareinsights.com "+' '.join(hashes)+"; style-src 'self' 'unsafe-inline'; font-src 'self'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'self'"
    return HTMLResponse(html,headers={'Content-Security-Policy':csp})

@router.get('/crypto/static/{filename:path}')
def static(filename:str,request:Request):
    user(request)
    path=(SOURCE/filename).resolve()
    if not path.is_relative_to(SOURCE.resolve()) or not path.is_file():raise HTTPException(404)
    return FileResponse(path)

@router.get('/crypto/api/{endpoint:path}')
def api(endpoint:str,request:Request):
    user(request)
    candle=endpoint.removeprefix('candles/') if endpoint.startswith('candles/') else ''
    if not (endpoint in {'summary','alarms','log','book/cnn','book/cnn_equal','book/v1','book/v2','book/v2d'} or candle.isalnum() and len(candle)<=16):raise HTTPException(404)
    url='http://127.0.0.1:8996/api/'+quote(endpoint,safe='/')
    if request.url.query:url+='?'+request.url.query
    try:
        with urllib.request.urlopen(url,timeout=20) as r:return Response(r.read(),media_type='application/json')
    except (OSError,urllib.error.URLError):raise HTTPException(503,'크립토 모의매매 서비스에 연결하지 못했습니다.')
