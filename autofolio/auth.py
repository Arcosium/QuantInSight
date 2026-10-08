"""Application accounts and encrypted, user-scoped generation credentials."""
import hashlib
import json
import re
import hmac
import os
import secrets
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import urlsplit

from cryptography.fernet import Fernet
from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from pydantic import BaseModel, ConfigDict, Field, model_validator
from starlette.middleware.base import BaseHTTPMiddleware

router = APIRouter()
COOKIE = 'qis_session'
SESSION_SECONDS = 12 * 3600
PROVIDERS = {'openai', 'anthropic', 'gemini', 'deepseek', 'openrouter'}
ROOT = Path(__file__).resolve().parents[1]


def auth_dir():
    path = Path(os.environ.get('QUANTINSIGHT_AUTH_DIR', Path.home() / 'vault/QuantInSight/auth'))
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    path.chmod(0o700)
    return path


@contextmanager
def connect():
    path = auth_dir() / 'accounts.sqlite3'
    db = sqlite3.connect(path, timeout=15)
    path.chmod(0o600)
    db.row_factory = sqlite3.Row
    db.executescript('''
      CREATE TABLE IF NOT EXISTS users(id INTEGER PRIMARY KEY, username TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL, role TEXT NOT NULL CHECK(role IN ('admin','user')), created REAL NOT NULL);
      CREATE TABLE IF NOT EXISTS sessions(token TEXT PRIMARY KEY, user_id INTEGER NOT NULL, expires REAL NOT NULL);
      CREATE TABLE IF NOT EXISTS credentials(user_id INTEGER PRIMARY KEY, provider TEXT NOT NULL,
        model TEXT NOT NULL, secret BLOB NOT NULL);
      CREATE TABLE IF NOT EXISTS connections(user_id INTEGER NOT NULL, kind TEXT NOT NULL,
        secret BLOB NOT NULL, PRIMARY KEY(user_id,kind));
      CREATE TABLE IF NOT EXISTS attempts(bucket TEXT PRIMARY KEY, count INTEGER NOT NULL, expires REAL NOT NULL);
    ''')
    try:
        yield db
        db.commit()
    finally:
        db.close()


def password_hash(value):
    salt = secrets.token_bytes(16)
    digest = hashlib.scrypt(value.encode(), salt=salt, n=16384, r=8, p=1)
    return salt.hex() + ':' + digest.hex()


def password_matches(value, encoded):
    salt, digest = encoded.split(':')
    actual = hashlib.scrypt(value.encode(), salt=bytes.fromhex(salt), n=16384, r=8, p=1)
    return hmac.compare_digest(actual.hex(), digest)


def bootstrap_admin(username, password):
    """Provision once from a local secret input; never a web-accessible role change."""
    username = username.lower()
    with connect() as db:
        existing = db.execute('SELECT role FROM users WHERE username=?', (username,)).fetchone()
        if existing:
            if existing['role'] != 'admin':
                raise RuntimeError('Administrator username is already registered')
            return
        if db.execute("SELECT 1 FROM users WHERE role='admin'").fetchone():
            raise RuntimeError('Administrator already provisioned')
        db.execute('INSERT INTO users(username,password,role,created) VALUES(?,?,?,?)',
                   (username, password_hash(password), 'admin', time.time()))


def mutation_guard(request):
    if request.headers.get('x-requested-with') != 'QuantInSight':
        raise HTTPException(403, '이 화면에서 다시 요청해 주세요.')
    origin = request.headers.get('origin')
    if origin:
        parsed = urlsplit(origin)
        if parsed.scheme not in ('http', 'https') or parsed.netloc != request.headers.get('host'):
            raise HTTPException(403, '다른 사이트의 요청은 허용하지 않습니다.')


def user(request):
    value = getattr(request.state, 'user', None)
    if not value:
        raise HTTPException(401, '로그인이 필요합니다.')
    return value


def admin_required(request):
    value = user(request)
    if value['role'] != 'admin':
        raise HTTPException(403, '관리자만 사용할 수 있습니다.')
    return value


class AuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        request.state.user = None
        token = request.cookies.get(COOKIE)
        if token and len(token) <= 128:
            with connect() as db:
                row = db.execute('SELECT u.id,u.username,u.role FROM sessions s JOIN users u ON u.id=s.user_id '
                                 'WHERE s.token=? AND s.expires>?',
                                 (hashlib.sha256(token.encode()).hexdigest(), time.time())).fetchone()
            if row:
                request.state.user = dict(row)
        public = request.url.path in {'/login', '/api/auth/login', '/api/auth/register',
                                     '/static/login.html', '/static/login.js', '/static/login.css',
                                     '/static/observatory.png'} or request.url.path.startswith('/static/fonts/')
        if not public and not request.state.user:
            if request.url.path.startswith('/api/'):
                return JSONResponse({'detail': '로그인이 필요합니다.'}, status_code=401)
            return RedirectResponse('/login', status_code=303)
        if request.method not in {'GET', 'HEAD', 'OPTIONS'}:
            try:
                mutation_guard(request)
            except HTTPException as exc:
                return JSONResponse({'detail': exc.detail}, status_code=exc.status_code)
        response = await call_next(request)
        response.headers['Cache-Control'] = 'no-store'
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['Referrer-Policy'] = 'same-origin'
        response.headers['X-Frame-Options'] = 'SAMEORIGIN'
        return response


class Login(BaseModel):
    model_config = ConfigDict(extra='forbid')
    username: str = Field(min_length=3, max_length=40, pattern=r'^[A-Za-z0-9_\-]+$')
    password: str = Field(min_length=10, max_length=200)


class KISConnection(BaseModel):
    model_config = ConfigDict(extra='forbid')
    app_key: str = Field(default='', max_length=1000)
    app_secret: str = Field(default='', max_length=2000)
    account_no: str = Field(default='', max_length=8)
    account_product: str = Field(default='', max_length=2)
    mode: str = 'live'

    @model_validator(mode='after')
    def validate_connection(self):
        for field in ('app_key', 'app_secret', 'account_no', 'account_product'):
            setattr(self, field, getattr(self, field).strip())
        if self.mode != 'live':
            raise HTTPException(422, '한국투자증권 계좌 유형을 확인해 주세요.')
        if self.app_key or self.app_secret or self.account_no or self.account_product:
            if not all((self.app_key, self.app_secret, self.account_no, self.account_product)):
                raise HTTPException(422, '한국투자증권 연결 항목을 모두 입력해 주세요.')
            if not re.fullmatch(r'[0-9]{8}', self.account_no) or not re.fullmatch(r'[0-9]{2}', self.account_product):
                raise HTTPException(422, '계좌번호 8자리와 상품코드 2자리를 확인해 주세요.')
        return self


class TimefolioConnection(BaseModel):
    model_config = ConfigDict(extra='forbid')
    username: str = Field(default='', max_length=100)
    password: str = Field(default='', max_length=200)

    @model_validator(mode='after')
    def validate_connection(self):
        self.username = self.username.strip()
        if bool(self.username) != bool(self.password):
            raise HTTPException(422, '타임폴리오 아이디와 비밀번호를 모두 입력해 주세요.')
        return self


class Connections(BaseModel):
    model_config = ConfigDict(extra='forbid')
    kis: KISConnection | None = None
    timefolio: TimefolioConnection | None = None


class Registration(Login):
    kis: KISConnection | None = None
    timefolio: TimefolioConnection | None = None


def store_connections(db, user_id, value):
    entries = []
    if value.kis and value.kis.app_key:
        entries.append(('kis-' + value.kis.mode, value.kis.model_dump()))
    if value.timefolio and value.timefolio.username:
        entries.append(('timefolio', value.timefolio.model_dump()))
    for kind, body in entries:
        encrypted = cipher().encrypt(json.dumps(body).encode())
        db.execute('INSERT INTO connections VALUES(?,?,?) ON CONFLICT(user_id,kind) '
                   'DO UPDATE SET secret=excluded.secret', (user_id, kind, encrypted))


def get_connection(user_id, kind):
    if kind not in {'kis-live', 'timefolio'}:
        return None
    with connect() as db:
        row = db.execute('SELECT secret FROM connections WHERE user_id=? AND kind=?', (user_id, kind)).fetchone()
    return json.loads(cipher().decrypt(row['secret'])) if row else None


def connection_metadata(user_id):
    with connect() as db:
        kinds = {row['kind'] for row in db.execute('SELECT kind FROM connections WHERE user_id=?', (user_id,))}
    return {'connections': [{'kind': kind, 'configured': kind in kinds}
                            for kind in ('kis-live', 'timefolio')]}


@router.get('/api/auth/connections')
def connections_status(request: Request):
    return connection_metadata(user(request)['id'])


@router.post('/api/auth/connections')
def save_connections(value: Connections, request: Request):
    current = user(request)
    with connect() as db:
        store_connections(db, current['id'], value)
    return connection_metadata(current['id'])


def throttle(request, action):
    # Do not trust a client-supplied forwarding header. Account and peer buckets
    # survive worker restarts and cap expensive password hashing attempts.
    peer = request.client.host if request.client else 'unknown'
    bucket = hashlib.sha256((action + ':' + peer).encode()).hexdigest()
    now = time.time()
    with connect() as db:
        db.execute('DELETE FROM attempts WHERE expires<?', (now,))
        db.execute('INSERT INTO attempts VALUES(?,1,?) ON CONFLICT(bucket) DO UPDATE SET count=count+1',
                   (bucket, now + 900))
        count = db.execute('SELECT count FROM attempts WHERE bucket=?', (bucket,)).fetchone()[0]
    if count > (30 if action == 'login' else 10):
        raise HTTPException(429, '요청이 많습니다. 15분 뒤 다시 시도해 주세요.')


@router.get('/login', include_in_schema=False)
def login_page(request: Request):
    if getattr(request.state, 'user', None):
        return RedirectResponse('/', status_code=303)
    return FileResponse(ROOT / 'static/login.html')


@router.post('/api/auth/register')
def register(value: Registration, request: Request):
    throttle(request, 'register')
    with connect() as db:
        if not db.execute("SELECT 1 FROM users WHERE role='admin'").fetchone():
            raise HTTPException(503, '관리자 계정 설정 중입니다.')
        try:
            cursor = db.execute('INSERT INTO users(username,password,role,created) VALUES(?,?,?,?)',
                       (value.username.lower(), password_hash(value.password), 'user', time.time()))
            store_connections(db, cursor.lastrowid, value)
        except sqlite3.IntegrityError:
            raise HTTPException(409, '사용할 수 없는 아이디입니다.')
    return {'registered': True}


@router.post('/api/auth/login')
def login(value: Login, request: Request, response: Response):
    throttle(request, 'login')
    with connect() as db:
        row = db.execute('SELECT * FROM users WHERE username=?', (value.username.lower(),)).fetchone()
        # Use the same work factor even for an unknown account.
        encoded = row['password'] if row else ('00' * 16 + ':' + '00' * 64)
        if not password_matches(value.password, encoded) or row is None:
            raise HTTPException(401, '아이디 또는 비밀번호를 확인해 주세요.')
        token = secrets.token_urlsafe(32)
        db.execute('DELETE FROM sessions WHERE expires<?', (time.time(),))
        db.execute('INSERT INTO sessions VALUES(?,?,?)',
                   (hashlib.sha256(token.encode()).hexdigest(), row['id'], time.time() + SESSION_SECONDS))
    response.set_cookie(COOKIE, token, httponly=True, secure=request.url.hostname not in {'localhost','127.0.0.1','::1','testserver'},
                        samesite='lax', max_age=SESSION_SECONDS, path='/')
    return {key: row[key] for key in ('id', 'username', 'role')}


@router.post('/api/auth/logout')
def logout(request: Request, response: Response):
    token = request.cookies.get(COOKIE, '')
    with connect() as db:
        db.execute('DELETE FROM sessions WHERE token=?', (hashlib.sha256(token.encode()).hexdigest(),))
    response.delete_cookie(COOKIE, path='/')
    return {'logged_out': True}


@router.get('/api/auth/me')
def me(request: Request):
    return user(request)


def cipher():
    path = auth_dir() / 'credentials.key'
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        pass
    else:
        with os.fdopen(fd, 'wb') as out:
            out.write(Fernet.generate_key())
    return Fernet(path.read_bytes())


class Credentials(BaseModel):
    model_config = ConfigDict(extra='forbid')
    provider: str
    model: str = Field(min_length=1, max_length=120)
    api_key: str = Field(min_length=8, max_length=1000)


def get_credentials(user_id):
    with connect() as db:
        row = db.execute('SELECT provider,model,secret FROM credentials WHERE user_id=?', (user_id,)).fetchone()
    if not row:
        return None
    return {'provider': row['provider'], 'model': row['model'], 'api_key': cipher().decrypt(row['secret']).decode()}


@router.get('/api/auth/credentials')
def credentials_status(request: Request):
    current = user(request)
    with connect() as db:
        row = db.execute('SELECT provider,model FROM credentials WHERE user_id=?', (current['id'],)).fetchone()
    return {'configured': bool(row), **(dict(row) if row else {})}


@router.post('/api/auth/credentials')
def save_credentials(value: Credentials, request: Request):
    current = user(request)
    if value.provider not in PROVIDERS or not value.api_key.strip() or not value.model.strip():
        raise HTTPException(422, 'API 제공자와 모델, 키를 확인해 주세요.')
    secret = cipher().encrypt(value.api_key.strip().encode())
    with connect() as db:
        db.execute('INSERT INTO credentials VALUES(?,?,?,?) ON CONFLICT(user_id) DO UPDATE SET '
                   'provider=excluded.provider,model=excluded.model,secret=excluded.secret',
                   (current['id'], value.provider, value.model.strip(), secret))
    return {'configured': True, 'provider': value.provider, 'model': value.model.strip()}


@router.delete('/api/auth/credentials')
def delete_credentials(request: Request):
    with connect() as db:
        db.execute('DELETE FROM credentials WHERE user_id=?', (user(request)['id'],))
    return {'configured': False}
