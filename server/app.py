"""Arquant v1.0 - FastAPI Server"""
import asyncio, logging, os, signal
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, Literal
import aiohttp
from fastapi import FastAPI, Request, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from infra import auth_store
from infra.rate_limit import SlidingWindowLimiter
from infra.user_context import REGISTRY

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger("APP")
# 좀비 자식 자동 수거 — ops_support_worker / weekly_review / restart_server 는 전부
# fire-and-forget Popen 이라 아무도 wait() 하지 않는다. 서버가 무재부팅으로 며칠 돌면
# <defunct> python3.11 이 그대로 남는다(2026-08-02 관측: 1일 6시간 상주).
# 이 프로세스는 자식의 종료코드를 읽는 코드가 하나도 없으므로(subprocess.run/wait 미사용)
# 커널에 수거를 맡기는 것이 가장 짧고 확실하다. 자식 결과를 읽어야 하는 코드가 생기면
# 이 줄을 지우고 해당 spawn 만 wait 스레드로 바꿔라.
signal.signal(signal.SIGCHLD, signal.SIG_IGN)
# KIS 실전 OpenAPI 기본 URL — 등록/검증 기본값. 한 곳에서 관리해 불일치를 막는다.
DEFAULT_KIS_BASE_URL = "https://openapi.koreainvestment.com:9443"
MOCK_KIS_BASE_URL = "https://openapivts.koreainvestment.com:29443"
ALLOWED_KIS_BASE_URLS = frozenset({DEFAULT_KIS_BASE_URL, MOCK_KIS_BASE_URL})
app = FastAPI(title="QuantInSight", version="2.0.0")
# 계정 프로필 디렉토리 루트. 모듈 상수로 둬서 테스트가 격리(monkeypatch)할 수 있게 한다.
# (과거 엔드포인트가 실경로를 직접 계산해 테스트가 실데이터 profiles/<uid> 를 삭제하던 버그 방지)
_PROFILES_DIR = Path(__file__).resolve().parent.parent / "data" / "profiles"
# 사장 피드백 2026-05-20: 배포 전 보안 점검 — wildcard + credentials 동시 허용은 안티패턴.
# 프로덕션 도메인 + 로컬 개발(에뮬레이터/localhost)만 허용. 추가 origin은 ARQUANT_EXTRA_ORIGINS(콤마 구분)로 주입.
_ALLOWED_ORIGINS = [
    "https://quantinsight.ai-ve.uk",  # 주 도메인 (ArQuant→QuantInSight 개명, 2026-07-20)
    "http://localhost:8500", "http://127.0.0.1:8500",
    "http://10.0.2.2:8500",  # Android emulator host loopback
]
_extra = os.getenv("ARQUANT_EXTRA_ORIGINS", "").strip()
if _extra:
    _ALLOWED_ORIGINS += [o.strip() for o in _extra.split(",") if o.strip()]
app.add_middleware(CORSMiddleware, allow_origins=_ALLOWED_ORIGINS, allow_credentials=True,
                   allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
                   allow_headers=["Content-Type", "Cf-Access-Jwt-Assertion"])


@app.exception_handler(auth_store.FernetKeyLost)
async def _fernet_lost_handler(request: Request, exc: auth_store.FernetKeyLost):
    # 멀티유저 계정 보호: 키 분실 시 전 유저 계정이 위험 — 새 키 생성 대신 503으로 명확히 안내.
    return JSONResponse(status_code=503, content={
        "error": str(exc), "code": "fernet_key_lost",
        "hint": "data/.fernet.key 백업을 복구하거나 ARQUANT_FERNET_KEY 환경변수로 키를 주입한 뒤 서버를 재시작하세요."})

# Cloudflare Access owns sign-in. The origin verifies signature, audience and owner.
from infra import cloudflare_access

@app.middleware("http")
async def app_auth(request: Request, call_next):
    if request.url.path == "/health":
        return await call_next(request)
    if request.method not in ("GET", "HEAD", "OPTIONS"):
        origin = request.headers.get("origin")
        if origin and origin != "https://quantinsight.ai-ve.uk":
            return JSONResponse(status_code=403, content={"error": "허용되지 않은 요청 출처"})
    try:
        uid, _ = await asyncio.to_thread(cloudflare_access.verify,
            request.headers.get("Cf-Access-Jwt-Assertion", ""))
    except cloudflare_access.AccessNotConfigured:
        return JSONResponse(status_code=503, content={"error": "접속 보호 설정 중입니다"})
    except PermissionError:
        return JSONResponse(status_code=401, content={"error": "Cloudflare Access 인증이 필요합니다"})
    request.state.user_id = uid
    return await call_next(request)


# ─── 자격증명 검증 ────────────────────────────────────────────────────────────
async def _validate_kis(app_key: str, app_secret: str, base_url: str) -> tuple[bool, str]:
    base_url = (base_url or DEFAULT_KIS_BASE_URL).rstrip("/")
    if base_url not in ALLOWED_KIS_BASE_URLS:
        return False, "KIS 거래 환경은 실전투자 또는 모의투자만 선택할 수 있습니다."
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=15)) as s:
            async with s.post(f"{base_url}/oauth2/tokenP",
                              json={"grant_type": "client_credentials",
                                    "appkey": app_key, "appsecret": app_secret}) as r:
                d = await r.json(content_type=None)
        if isinstance(d, dict) and d.get("access_token"):
            return True, "ok"
        return False, f"KIS 인증 실패: {(d or {}).get('error_description') or (d or {}).get('msg1') or '응답에 access_token 없음'}"
    except Exception as e:
        return False, f"KIS 연결 실패: {e}"




# ─── Phase 2 멀티테넌트 — 유저별 매매 루프 라이프사이클 ─────────────────────────
# 전역 단일 _task / 활성 계정 개념 폐지. 각 유저는 자신의 UserContext.task(asyncio.Task)로
# 독립 매매 루프를 돈다. 한 유저의 start/stop 이 다른 유저에게 영향을 주지 않는다.
async def _start_uid(uid: int, directive=None) -> None:
    ctx = REGISTRY.get_or_create(uid)
    from config import PAPER_ONLY
    kind = auth_store.profile_kind_of(ctx.creds)
    if kind not in auth_store.PROFILE_KIND_ORDER:
        raise HTTPException(403, "KIS 실전 또는 모의 프로필이 필요합니다")
    if PAPER_ONLY and kind == auth_store.PROFILE_KIS_REAL:
        raise HTTPException(409, "실전 매매는 정지되어 있습니다. KIS 모의투자 프로필을 선택하세요")
    if ctx.task and not ctx.task.done():
        raise HTTPException(409, "이미 감시 중")
    from infra import user_paths
    # 부팅 자동재개 마커 — 서버 재시작 후에도 이 유저 루프가 돌고 있었으면 다시 켠다.
    user_paths.running_marker(uid).write_text("1", encoding="utf-8")
    ctx.task = asyncio.create_task(_supervised_loop(ctx, directive))


async def _supervised_loop(ctx, directive):
    try:
        await ctx.swarm.start_continuous(directive)
    except asyncio.CancelledError:
        raise
    except Exception as e:
        logger.exception("uid=%s 매매 루프 비정상 종료: %s", ctx.uid, e)
        try:
            await ws_mgr.send_to_uid(ctx.uid, {"type": "status", "state": "ERROR",
                                               "detail": f"매매 루프 중단: {e}"})
        except Exception:
            pass


async def _stop_uid(uid: int) -> None:
    ctx = REGISTRY.get(uid)
    from infra import user_paths
    _marker = user_paths.running_marker(uid)
    # pytest 가드 — 테스트가 실 .running 을 지워 재시작 자동재개를 끊지 않게 (2026-06-11).
    if not _pytest_live_path(_marker):
        _marker.unlink(missing_ok=True)
    if not ctx:
        return
    ctx.swarm.stop()
    if ctx.task and not ctx.task.done():
        ctx.task.cancel()
    ctx.task = None


def _pytest_live_path(path) -> bool:
    """pytest 실행 중인데 path 가 라이브 저장소 하위인가 — 삭제류 호출 차단용.
    (테스트가 tmp 로 monkeypatch 한 경로는 False → 정상 동작.)"""
    import os
    from pathlib import Path as _P
    try:
        if not os.environ.get("PYTEST_CURRENT_TEST"):
            return False
        live_root = _P(__file__).resolve().parent.parent
        return str(_P(path).resolve()).startswith(str(live_root))
    except Exception:
        return False


def _rmtree_pytest_guarded(path) -> None:
    """rmtree 전 pytest-라이브 가드 — 운영호스트에서 pytest 가 실 data/<uid>·profiles/<uid> 를
    지우는 참사 차단 (2026-06-11 확정: test_admin_members 멤버삭제 테스트가 tmp 격리 없이
    _decommission_uid 를 태워 실 data/1·data/2 가 전부 삭제됐고, 과거 '거래기록 소실'
    사건들의 근본 원인이었다). 테스트는 tmp 경로로 monkeypatch 하면 정상 삭제된다."""
    import shutil
    if _pytest_live_path(path):
        logging.getLogger("AUTH").error("pytest 가드 — 라이브 경로 삭제 차단: %s", path)
        return
    shutil.rmtree(path, ignore_errors=True)


async def _decommission_uid(uid: int) -> None:
    """유저 삭제/탈퇴 시 실행 주체를 완전히 종료한다 — 매매 루프 정지 → 컨텍스트 제거 →
    프로필·데이터 디렉터리 정리. 루프를 '먼저' 멈춰야, rmtree 후 살아있는 루프가
    profiles/<uid> 를 재생성(고아 부활)하거나 삭제된 유저 명의로 실주문을 내는 것을 막는다."""
    from infra import user_paths
    await _stop_uid(uid)
    try:
        REGISTRY.drop(uid)
    except Exception:
        pass
    _rmtree_pytest_guarded(_PROFILES_DIR / str(uid))
    _rmtree_pytest_guarded(user_paths.user_dir(uid))

# 사장 지시 2026-05-21: 모바일 푸시 알림 4종 ↔ 프로필 알림설정 키 매핑.
# 모바일(네이티브 WsManager) 연결은 이 4종 이벤트만, 그것도 프로필 설정이 ON 인 것만 받는다.
# 웹 대시보드(client=web) 연결은 설정과 무관하게 모든 이벤트를 받아 통신로그에 전부 표시한다.
_NOTIF_EVENT_KEY = {"order_submitted": "order_submitted",
                    "trade_executed": "trade", "trade_failed": "trade",
                    "cycle_complete": "cycle", "market_close": "market_close"}


def _should_send(meta: dict, msg: dict) -> bool:
    if (meta.get("client") or "web") != "mobile":
        return True   # 웹: 전부 수신
    nk = _NOTIF_EVENT_KEY.get((msg or {}).get("type"))
    if nk is None:
        return False  # 모바일: 알림성 이벤트만 (잡음 차단)
    try:
        import runtime
        return bool(runtime.notif_settings(meta.get("uid")).get(nk, True))
    except Exception:
        return True


class WS:
    def __init__(self):
        self.conns: list[WebSocket] = []
        self.meta: dict = {}  # ws -> {"uid": int|None, "client": "web"|"mobile"}
    async def connect(self, ws, uid=None, client="web", view_uid=None):
        await ws.accept(); self.conns.append(ws)
        self.meta[ws] = {"uid": uid, "view_uid": view_uid if view_uid is not None else uid,
                         "client": client}
    def disconnect(self, ws):
        if ws in self.conns: self.conns.remove(ws)
        self.meta.pop(ws, None)
    async def broadcast(self, msg):
        dead=[]
        for c in self.conns:
            try:
                if _should_send(self.meta.get(c) or {}, msg):
                    await c.send_json(msg)
            except: dead.append(c)
        for d in dead: self.disconnect(d)
    async def send_to_uid(self, uid, msg):
        """특정 유저의 연결에만 송신 (per-uid 사이클 이벤트·피드백 답글).
        버그수정 2026-06-01: per-uid 송신도 _should_send 필터를 거쳐야 한다 — 안 그러면 cycle_complete
        등 사이클 이벤트가 모바일 알림설정(예: '사이클 완료' OFF)을 우회해 계속 푸시됐다(웹은 전부 수신)."""
        dead=[]
        for c in self.conns:
            m = self.meta.get(c) or {}
            if m.get("view_uid", m.get("uid")) == uid:
                try:
                    if _should_send(m, msg):
                        await c.send_json(msg)
                except: dead.append(c)
        for d in dead: self.disconnect(d)
    async def send_to_admins(self, msg):
        """ADMIN 연결에만 송신 (새 피드백 도착 알림). 일반 유저에겐 노출 안 함."""
        dead=[]
        for c in self.conns:
            u=(self.meta.get(c) or {}).get("uid")
            if u is None: continue
            try:
                if auth_store.is_admin(u): await c.send_json(msg)
            except: dead.append(c)
        for d in dead: self.disconnect(d)
ws_mgr = WS()
from main_swarm import set_broadcast_callback


async def _route(msg, uid=None):
    """main_swarm._broadcast 콜백 라우터 (Phase 2 멀티테넌트).
    uid 가 주어지면(오케스트레이터 사이클 이벤트) 그 유저 연결에만, None 이면(시스템 알림)
    전체 연결에 송신한다 → 다른 유저 대시보드로 이벤트가 새지 않는다."""
    if uid is not None:
        await ws_mgr.send_to_uid(uid, msg)
    else:
        await ws_mgr.broadcast(msg)

set_broadcast_callback(_route)

class Req(BaseModel):
    directive: Optional[str] = None
class CeoReq(BaseModel):
    message: str
class FundamentalResearchReq(BaseModel):
    code: str
    name: str = ""
    source: str = "manual_ai_berkshire"
    memo: str = ""
    verdict: str = ""
    business_quality_score: Optional[float] = None
    valuation_margin_score: Optional[float] = None
    moat_score: Optional[float] = None
    management_score: Optional[float] = None
    thesis_invalidators: list = []
    financial_checks: list = []



_rl_login = SlidingWindowLimiter(max_hits=int(os.getenv("ARQUANT_RL_LOGIN_MAX", "8")),
                                 window_sec=float(os.getenv("ARQUANT_RL_WIN", "900")))
# register:{ip} / recid:{ip} / recpw:{ip} 는 prefix 가 달라 IP당 독립 5회 윈도우(상호 잠식 없음).
_rl_recover = SlidingWindowLimiter(max_hits=int(os.getenv("ARQUANT_RL_RECOVER_MAX", "5")),
                                   window_sec=float(os.getenv("ARQUANT_RL_WIN", "900")))

def _client_ip(request: Request) -> str:
    # Cloudflare Tunnel 은 CF-Connecting-IP 에 실제 클라이언트 IP 를 넣는다(가장 신뢰 가능).
    # 그다음 X-Forwarded-For 첫 홉, 마지막으로 소켓 peer. (포트 직결 우회 시 헤더 위조
    # 가능 — 인프라에서 uvicorn --proxy-headers --forwarded-allow-ips 로 보강 권장.)
    cf = (request.headers.get("cf-connecting-ip") or "").strip()
    if cf:
        return cf
    xff = (request.headers.get("x-forwarded-for") or "").split(",")[0].strip()
    return xff or (request.client.host if request.client else "unknown")

def _throttle(lim: SlidingWindowLimiter, key: str) -> None:
    retry = lim.hit(key)
    if retry is not None:
        raise HTTPException(429, f"요청이 너무 많습니다. {int(retry)+1}초 후 다시 시도하세요.")

# ─── 인증 엔드포인트 (HYFE app.py:160-282 패턴) ───────────────────────────────
@app.get("/api/auth_status")
async def auth_status(request: Request):
    return {"authenticated": True, "provider": "cloudflare_access"}

@app.get("/api/me")
async def me(request: Request):
    uid = request.state.user_id
    c = auth_store.get_user_credentials(uid) or {}
    # 통합 계정(사장 지시 2026-07-20): 자격증명 표시는 활성(토글) 프로필 기준,
    # 정체성(username/is_admin)은 로그인 행 기준.
    profiles = auth_store.list_profiles(uid)
    eff = auth_store.resolve_profile_uid(uid)
    ec = auth_store.get_user_credentials(eff) or c
    active_kind = next((p["kind"] for p in profiles if p.get("is_active")), None)
    return {"user_id": uid, "username": c.get("username"), "label": c.get("label"),
            "kis_app_key_masked": auth_store._mask(ec.get("kis_app_key", "")),
            "kis_account_no_masked": auth_store._mask(ec.get("kis_account_no", ""), 4),
            "kis_base_url": ec.get("kis_base_url", ""),  # 프로필 '정보 변경'의 실전/모의 선택자 기본값용 (비밀 아님)
            "has_dart": bool(c.get("dart_key")),
            "account_mode": c.get("account_mode", auth_store.TRADING_MODE),
            "is_viewer": c.get("account_mode") == auth_store.VIEWER_MODE,
            "profiles": profiles, "active_profile_uid": eff,
            "paper_only": bool(__import__("config").PAPER_ONLY),
            "active_profile_kind": active_kind,
            "is_admin": bool(c.get("is_admin"))}  # 사장 피드백 2026-05-18: 코드변경 전체반영 권한 표시


@app.get("/api/profiles")
async def profiles_list(request: Request):
    """통합 계정의 매매 프로필 목록 + 활성 프로필 (헤더 토글용)."""
    uid = _uid_or_403(request)
    return {"profiles": auth_store.list_profiles(uid),
            "active_uid": auth_store.resolve_profile_uid(uid)}


class ProfileActivateReq(BaseModel):
    uid: int


@app.post("/api/profiles/activate")
async def profiles_activate(req: ProfileActivateReq, request: Request):
    """토글 전환 — 본인 소유 프로필로만 전환 가능. 각 프로필 스웜은 독립이므로
    전환은 보기/제어 대상만 바꾸며 다른 프로필의 매매 루프에 영향을 주지 않는다."""
    uid = _uid_or_403(request)
    if not auth_store.set_active_profile(uid, req.uid):
        raise HTTPException(400, "전환할 수 없는 프로필입니다.")
    return {"ok": True, "active_uid": auth_store.resolve_profile_uid(uid),
            "profiles": auth_store.list_profiles(uid)}


class CredsReq(BaseModel):
    kis_app_key: Optional[str] = None
    kis_app_secret: Optional[str] = None
    kis_account_no: Optional[str] = None
    kis_base_url: Optional[str] = None

class ProfileUpsertReq(BaseModel):
    kind: Literal["kis_real", "kis_paper"]
    kis_app_key: Optional[str] = None
    kis_app_secret: Optional[str] = None
    kis_account_no: Optional[str] = None

class DirectiveReq(BaseModel):
    text: str



@app.post("/api/profile/credentials")
async def profile_credentials(req: CredsReq, request: Request):
    # 통합 계정: '정보 변경'은 활성(토글) 프로필의 KIS 자격증명을 갱신한다.
    # 관전 계정은 프로필이 없어 자기 행이 그대로 대상(업그레이드 경로 유지).
    login_uid = _uid_or_403(request)
    uid = auth_store.resolve_profile_uid(login_uid)
    ip = _client_ip(request)
    creds_cr = auth_store.get_user_credentials(uid)
    uname_cr = (creds_cr or {}).get("username", "")
    cur = creds_cr or {}
    upgrading_viewer = cur.get("account_mode") == auth_store.VIEWER_MODE
    # Fix 2 — strip whitespace on provided fields (mirrors register handler)
    ak = req.kis_app_key.strip() if req.kis_app_key is not None else None
    as_ = req.kis_app_secret.strip() if req.kis_app_secret is not None else None
    an = req.kis_account_no.strip() if req.kis_account_no is not None else None
    bu = req.kis_base_url.strip().rstrip("/") if req.kis_base_url is not None else None
    # Resolve effective values for KIS validation (fall back to stored values when not provided)
    eff_ak = ak if ak is not None else cur.get("kis_app_key")
    eff_as = as_ if as_ is not None else cur.get("kis_app_secret")
    eff_bu = bu if bu is not None else cur.get("kis_base_url")
    if upgrading_viewer:
        if not all((ak, as_, an, bu)):
            raise HTTPException(
                400, "관전 모드 업그레이드는 KIS App Key/Secret, 계좌번호, 거래 환경을 모두 입력해야 합니다.")
    if ak is not None or as_ is not None or bu is not None:
        ok, msg = await _validate_kis(eff_ak, eff_as, eff_bu)
        if not ok:
            auth_store.audit("profile_credentials", username=uname_cr, ip=ip,
                             outcome="fail", detail="validate")
            raise HTTPException(400, msg)
    auth_store.update_credentials(
        uid, kis_app_key=ak,
        kis_app_secret=as_, kis_account_no=an,
        kis_base_url=bu)
    if upgrading_viewer:
        auth_store.set_account_mode(uid, auth_store.TRADING_MODE)
    auth_store.audit("profile_credentials", username=uname_cr, ip=ip, outcome="ok", detail="")
    # Phase 2: 이 유저의 컨텍스트(브로커/스왐)가 이미 살아있으면 새 자격증명으로 재생성되도록 리셋.
    # 단, 매매 루프가 도는 중이면 새 creds 채택을 위해 먼저 루프를 안전하게 멈춘다
    # (잘못된 계좌로 주문 방지 — 사용자가 다시 ▶실행을 눌러야 한다).
    ctx = REGISTRY.get(uid)
    if ctx is not None:
        if ctx.task and not ctx.task.done():
            await _stop_uid(uid)
        # 갱신된 자격증명을 컨텍스트가 다시 읽도록 새 creds 를 주입하고 broker/swarm 폐기.
        fresh = auth_store.get_user_credentials(uid)
        if fresh:
            ctx.creds = fresh
        ctx.reset()
    return {"ok": True, "upgraded": upgrading_viewer,
            "account_mode": auth_store.TRADING_MODE}


async def _reset_profile_ctx(uid: int):
    """자격증명 변경 후 이 프로필 컨텍스트를 새 creds로 재생성(진행 중 루프는 안전 정지)."""
    ctx = REGISTRY.get(uid)
    if ctx is not None:
        if ctx.task and not ctx.task.done():
            await _stop_uid(uid)
        fresh = auth_store.get_user_credentials(uid)
        if fresh:
            ctx.creds = fresh
        ctx.reset()


@app.post("/api/profile/upsert")
async def profile_upsert(req: ProfileUpsertReq, request: Request):
    """실전·모의 KIS 프로필을 추가하거나 갱신한다. 저장만으로 매매를 시작하지 않는다."""
    login_uid = _uid_or_403(request)
    master = auth_store.login_uid_of(login_uid)   # 서브 프로필로 로그인했어도 마스터 기준
    kind = (req.kind or "").strip()
    ip = _client_ip(request)
    if kind not in (auth_store.PROFILE_KIS_REAL, auth_store.PROFILE_KIS_PAPER):
        raise HTTPException(400, "알 수 없는 프로필 종류입니다.")
    existing = next((p for p in auth_store.list_profiles(master) if p["kind"] == kind), None)

    if kind in (auth_store.PROFILE_KIS_REAL, auth_store.PROFILE_KIS_PAPER):
        ak = (req.kis_app_key or "").strip()
        as_ = (req.kis_app_secret or "").strip()
        an = (req.kis_account_no or "").strip()
        base = DEFAULT_KIS_BASE_URL if kind == auth_store.PROFILE_KIS_REAL else MOCK_KIS_BASE_URL
        if existing:
            uid = int(existing["uid"])
            cur = auth_store.get_user_credentials(uid) or {}
            eff_ak = ak or cur.get("kis_app_key")
            eff_as = as_ or cur.get("kis_app_secret")
            if ak or as_:
                ok, msg = await _validate_kis(eff_ak, eff_as, base)
                if not ok:
                    raise HTTPException(400, msg)
            auth_store.update_credentials(
                uid, kis_app_key=(ak or None), kis_app_secret=(as_ or None),
                kis_account_no=(an or None), kis_base_url=base)
            await _reset_profile_ctx(uid)
        else:
            if not all((ak, as_, an)):
                raise HTTPException(400, "App Key/Secret, 계좌번호를 모두 입력하세요.")
            ok, msg = await _validate_kis(ak, as_, base)
            if not ok:
                raise HTTPException(400, msg)
            uid = auth_store.create_subprofile(
                master, kind, kis_app_key=ak, kis_app_secret=as_,
                kis_account_no=an, kis_base_url=base)
    # 관전(viewer) 마스터가 매매 프로필을 연결하면 매매 계정으로 승격.
    if auth_store.is_viewer(master):
        auth_store.set_account_mode(master, auth_store.TRADING_MODE)
    auth_store.set_active_profile(master, uid)   # 저장 즉시 이 프로필을 활성화(상단 배지 켜짐)
    auth_store.audit("profile_upsert", username=(auth_store.get_user_credentials(master) or {}).get("username", ""),
                     ip=ip, outcome="ok", detail=f"kind={kind} uid={uid}")
    return {"ok": True, "kind": kind, "uid": uid,
            "profiles": auth_store.list_profiles(master)}


@app.get("/api/profile/directives")
async def profile_directives_list(request: Request):
    uid = _require_trading(request)
    from infra import standing_directives as sd
    return {"directives": sd.load(uid)}

@app.post("/api/profile/directives")
async def profile_directives_add(req: DirectiveReq, request: Request):
    uid = _require_trading(request)
    from infra import standing_directives as sd
    added = sd.append_directive(uid, req.text)
    return {"ok": True, "added": added, "directives": sd.load(uid)}

@app.delete("/api/profile/directives/{did}")
async def profile_directives_del(did: str, request: Request):
    uid = _require_trading(request)
    from infra import standing_directives as sd
    sd.remove_directive(uid, did)
    return {"ok": True, "directives": sd.load(uid)}

# ── 사용자 전략 (사장 지시 2026-07-03) — 채팅에서 STRATEGY 로 인식돼 저장된 전략 관리 ──
@app.get("/api/profile/user_strategy")
async def profile_user_strategy_get(request: Request):
    uid = _require_trading(request)
    from infra import user_strategy
    return {"strategy": user_strategy.get_strategy(uid)}

@app.delete("/api/profile/user_strategy")
async def profile_user_strategy_del(request: Request):
    uid = _require_trading(request)
    import runtime
    from infra import user_strategy
    removed = user_strategy.clear_strategy(uid)
    if removed:  # 전략 해제 = 운용지원실장 자동 튜닝 복원
        runtime.set_ops_feedback(True, uid=uid, by="user_strategy_clear")
    return {"ok": True, "removed": removed}





@app.get("/api/accounts")
async def accounts():
    # Phase 2 멀티테넌트: 전역 '활성 계정' 개념 폐지 → 등록 계정 목록만 반환한다.
    # 각 세션은 자신의 request.state.user_id 로 식별되며 /api/me 가 그 정보를 준다.
    return {"accounts": auth_store.list_accounts()}

@app.get("/health")
async def health(): return {"status":"ok","service":"QuantInSight","timestamp":datetime.now().isoformat()}

@app.get("/api/status")
async def status(request: Request):
    # Phase 2 멀티테넌트: 요청 유저의 활성(토글) 프로필 스왐 상태를 반환한다.
    auth_uid = _uid_or_403(request)
    uid = _read_uid(request)
    viewer = auth_store.is_viewer(auth_uid)
    ctx = REGISTRY.get_or_create(uid)
    s = ctx.swarm.get_status()
    # Expose whether THIS user's background task is actively running so the frontend can sync buttons.
    s["is_running"] = bool(ctx.task and not ctx.task.done())
    # (사장 지시 2026-07-21: LLM 비용 표시 로직 제거 — 로컬 서버라 호출당 외부 비용 없음.)
    # 운용지원실장 피드백 on/off 토글 상태 (프로필별) — 요청 유저 본인 계정 기준.
    try:
        import runtime
        s["ops_feedback_enabled"] = False if viewer else runtime.ops_feedback_enabled(uid)
    except Exception:
        s["ops_feedback_enabled"] = True
    s["is_viewer"] = viewer
    return s

@app.post("/api/start")
async def start(req: Req, request: Request):
    uid0 = _uid_or_403(request)
    eff = auth_store.resolve_profile_uid(uid0)
    uid = _require_trading(request)
    await _start_uid(uid, req.directive)
    return {"message":"🟢 감시 시작"}

@app.post("/api/stop")
async def stop(request: Request):
    # 사장 지시 2026-05-22: 즉시 중지 — stop_event 만으로는 진행 중 사이클이 끝까지 돌므로,
    # 실행 중인 asyncio 태스크를 취소해 LLM 호출·분석을 그 자리에서 중단한다.
    # Phase 2: 요청 유저의 루프만 멈춘다(다른 유저 무영향).
    uid0 = _uid_or_403(request)
    eff = auth_store.resolve_profile_uid(uid0)
    uid = _require_trading(request)
    await _stop_uid(uid)
    return {"message": "🔴 즉시 중지됨"}

@app.post("/api/ceo")
async def ceo_command(req: CeoReq, request: Request):
    # 사장 지시 2026-05-21: 저장 여부는 체크박스 대신 ceo_directive 안에서 에이전트가
    # 지시 내용·결과로 자동 판단해 standing_directive 로 저장한다(자동 판단 경로).
    # Phase 2: 요청 유저의 스왐에 지시를 전달한다.
    uid0 = _uid_or_403(request)
    eff = auth_store.resolve_profile_uid(uid0)
    uid = _require_trading(request)
    resp = await REGISTRY.get_or_create(uid).swarm.ceo_directive(req.message)
    return {"response": resp}

@app.get("/api/history")
async def history(request: Request):
    uid = _read_uid(request)
    return {"cycles": REGISTRY.get_or_create(uid).swarm.get_history()}

@app.get("/api/events")
async def events(request: Request, limit: int = 500):
    """Persisted display log — the dashboard fetches this on load so a page refresh
    restores the trade/agent log. Accumulates until /api/events/clear.
    Phase 2 멀티테넌트: 요청 유저(request.state.user_id)의 로그만 반환한다."""
    uid = _read_uid(request)
    from main_swarm import get_recent_events
    return {"events": get_recent_events(limit, uid=uid)}

@app.post("/api/events/clear")
async def events_clear(request: Request):
    uid = _require_trading(request)
    from main_swarm import clear_event_log; clear_event_log(uid=uid)
    return {"message": "🧹 로그 초기화됨"}

@app.get("/api/alerts")
async def alerts(limit: int = 100, level: str = ""):
    """운영자 실패 알림 — 조용히 삼켜졌던 주문/체결/equity/루프 실패를 표면화.
    (auth 미들웨어로 로그인 필요 — 운영자 전용)"""
    from infra import notifier
    try:
        n = max(1, min(int(limit), 500))
    except (TypeError, ValueError):
        n = 100
    return {"alerts": notifier.recent(n, level or None)}

@app.get("/api/metrics")
async def metrics_snapshot(window_sec: int = 86400):
    """사이클 소요시간·주문 성공/실패·오류 카운트 최근 집계."""
    from infra import metrics
    try:
        w = max(60, min(int(window_sec), 7 * 86400))
    except (TypeError, ValueError):
        w = 86400
    return {"window_sec": w, "metrics": metrics.snapshot(w)}

@app.get("/api/news")
async def news(count: int = 20):
    # 사장 피드백 2026-05-18: 누적 뉴스를 깊게 받아 스크롤로 계속 열람 (1~500건).
    from tools.news_monitor import get_monitor; m = get_monitor()
    try:
        n = max(1, min(int(count), 500))
    except (TypeError, ValueError):
        n = 20
    return {"articles": m.get_recent_articles(n), "status": m.get_status()}

@app.post("/api/news/clear")
async def news_clear(request: Request):
    # 사장 피드백 2026-05-18: 뉴스 로그 비우기 — 단, 최근 20건은 유지.
    _require_trading(request)
    from tools.news_monitor import get_monitor; m = get_monitor()
    kept = m.clear_history(keep=20)
    return {"message": f"🧹 뉴스 로그 초기화됨 (최근 {kept}건 유지)", "kept": kept}

@app.get("/api/agents")
async def agents(request: Request):
    from config import MODEL_ASSIGNMENTS
    # 설명은 일관된 한국어 (사장 지시 2026-05-20). 산하 팀장(투자/경영/재무관리팀장)은 폐지되어 명단에 없음.
    roster = [
        {"name":"주식운용실장","role":"총괄 전략·종목 선정 (2패스)","model":MODEL_ASSIGNMENTS["chief_orchestrator"]},
        {"name":"글로벌리서치팀장","role":"거시·매크로 리서치","model":MODEL_ASSIGNMENTS["macro_analyst"]},
        {"name":"계량분석팀장","role":"후보 종목 정량 평가","model":MODEL_ASSIGNMENTS["quant_analyst"]},
        {"name":"마켓센티먼트팀장","role":"뉴스 감성·이벤트 분석","model":MODEL_ASSIGNMENTS["news_analyst"]},
        {"name":"프롭트레이딩팀장","role":"주문 실행·결과 보고","model":MODEL_ASSIGNMENTS["trader"]},
        {"name":"리스크관리실장","role":"리스크 게이트·DART 재심","model":f"룰 엔진(Python) + {MODEL_ASSIGNMENTS['risk_guard']}"},
        {"name":"사후관리실장","role":"보유 종목 매도 판단·슬리브 종합","model":MODEL_ASSIGNMENTS["post_manager"]},
        {"name":"포트폴리오기획팀장","role":"진입 thesis·보유계획 상기","model":MODEL_ASSIGNMENTS["fund_planner"]},
        {"name":"채권운용실장","role":"채권 ETF 매매·금리 전략","model":MODEL_ASSIGNMENTS["bond_manager"]},
        {"name":"원자재운용실장","role":"원자재 ETF 매매·실물자산","model":MODEL_ASSIGNMENTS["commodity_manager"]},
        {"name":"운용지원실장","role":"진단·프로필 전략 조정","model":MODEL_ASSIGNMENTS["ops_support"]},
        # 사장 지시 2026-07-22: 구 수탁자책임실 반려 기능을 컴플라이언스실장으로 부활.
        # 사이클 판정은 결정론 코드(agents/compliance.screen), LLM 은 @멘션 설명 전용.
        {"name":"컴플라이언스실장","role":"수탁자책임·ESG 반려·리서치 정직성",
         "model":f"룰 엔진(Python) + {MODEL_ASSIGNMENTS['compliance']}"},
    ]
    # 사장 지시 2026-05-20: 운용지원실장은 ADMIN·일반 유저 모두 사용 가능(프로필 한정 파라미터 조정).
    return {"agents": roster}

# 사장 피드백 2026-05-18: 운용지원실장 피드백 on/off 토글 (ADMIN 전용).
@app.get("/api/ops_feedback")
async def ops_feedback_get(request: Request):
    # 사장 지시 2026-05-20: 운용지원 토글은 프로필별 — 요청 유저 본인 계정 상태를 반환.
    import runtime
    login_uid = getattr(request.state, "user_id", None)
    uid = _read_uid(request)  # 통합 계정: 활성 프로필 기준 (set 경로와 대칭)
    _admin = False
    try:
        _admin = auth_store.is_admin(login_uid)
    except Exception:
        _admin = False
    st = runtime.ops_feedback_state(uid)
    return {"enabled": bool(st.get("enabled", True)), "since": st.get("since"),
            "by": st.get("by"), "is_admin": _admin, "uid": st.get("uid")}

@app.post("/api/ops_feedback")
async def ops_feedback_set(request: Request, req: dict):
    # 사장 지시 2026-05-20: 운용지원 토글은 프로필별 — 각 유저가 본인 계정 것만 켜고 끈다
    # (코드 자가수정 폐지로 더 이상 ADMIN 전용일 필요 없음).
    import runtime
    uid = _require_strategy_uid(request)
    enabled = bool((req or {}).get("enabled"))
    st = runtime.set_ops_feedback(enabled, uid=uid, by="dashboard")
    try:
        # Phase 2: 프로필별 토글이므로 요청 유저 연결에만 통지한다(다른 유저 무영향).
        await ws_mgr.send_to_uid(uid, {"type": "status", "state": "IDLE",
                          "message": f"🛠 운용지원실장 피드백 {'켜짐(ON)' if enabled else '꺼짐(OFF)'} (이 계정)"})
    except Exception:
        pass
    return {"enabled": bool(st.get("enabled", True)), "since": st.get("since"), "uid": st.get("uid")}

# 사장 지시 2026-05-21: 모바일 알림 설정 (프로필별) — 4종 푸시 on/off.
@app.get("/api/notif_settings")
async def notif_settings_get(request: Request):
    import runtime
    uid = _read_uid(request)  # 통합 계정: 활성 프로필 기준 (set 경로와 대칭)
    return {"settings": runtime.notif_settings(uid)}

@app.post("/api/notif_settings")
async def notif_settings_set(request: Request, req: dict):
    import runtime
    uid = _require_trading(request)
    st = runtime.set_notif_settings(req or {}, uid=uid)
    return {"ok": True, "settings": st}


# ── Coresight 수신함 — ADMIN 전용 (Implementation.md §3.3) ───────────
# 비관리자 → 403. 이 경로는 _PUBLIC_PATHS 에 포함되지 않는다 (인증 미들웨어 통과 필요).
# GET /api/coresight/pending   — 미처리 Coresight 제안 목록
# POST /api/coresight/approve  — 명시 승인 → standing_directive 추가 (자동 체결 금지)
# POST /api/coresight/reject   — 거부 → 큐 영구 제거

def _uid_or_403(request: Request) -> int:
    """request.state.user_id 반환. 인증 미들웨어가 먼저 설정한다."""
    uid = getattr(request.state, "user_id", None)
    if uid is None:
        raise HTTPException(401, "로그인이 필요합니다.")
    return uid


def _eff_uid(request: Request) -> int:
    """세션(로그인) uid → 현재 토글된 매매 프로필 uid (통합 계정, 사장 지시 2026-07-20).
    프로필이 없으면 자기 자신."""
    return auth_store.resolve_profile_uid(_uid_or_403(request))


def _read_uid(request: Request) -> int:
    """조회 대상 uid. 관전 계정은 ADMIN 의 KIS 모의 프로필, 그 외엔 활성(토글) 프로필."""
    uid = _uid_or_403(request)
    if auth_store.is_viewer(uid):
        target = auth_store.admin_view_uid()
        if target is None:
            raise HTTPException(503, "관전할 ADMIN 계정을 찾을 수 없습니다.")
        return target
    return auth_store.resolve_profile_uid(uid)


def _require_trading(request: Request) -> int:
    """거래 가능한 KIS 프로필을 반환한다."""
    uid = _uid_or_403(request)
    if auth_store.is_viewer(uid) and not auth_store.is_admin(uid):
        raise HTTPException(403, "관전 모드에서는 조회만 가능합니다.")
    eff = auth_store.resolve_profile_uid(uid)
    creds = auth_store.get_user_credentials(eff) or {}
    if auth_store.profile_kind_of(creds) not in auth_store.PROFILE_KIND_ORDER:
        raise HTTPException(403, "KIS 실전 또는 모의 프로필을 연결하세요.")
    return eff


def _require_strategy_uid(request: Request) -> int:
    return _require_trading(request)


def _admin_uid_or_403(request: Request) -> int:
    """uid 를 꺼내고 ADMIN 확인. 아니면 403 raise."""
    uid = _uid_or_403(request)
    try:
        _is_adm = auth_store.is_admin(uid)
    except Exception:
        _is_adm = False
    if not _is_adm:
        raise HTTPException(403, "Coresight 수신함은 ADMIN 전용입니다.")
    return uid


def _require_admin(request: Request) -> int:
    uid = _uid_or_403(request)
    if not auth_store.is_admin(uid):
        raise HTTPException(403, "ADMIN 전용 기능입니다.")
    return uid


# ─── ADMIN 탭 — 전역 설정(전체 계정 적용) + 회원관리 (사장 지시 2026-05-22) ───
class _AdminConfigReq(BaseModel):
    model_overrides: Optional[Dict[str, str]] = None
    news_crawl_interval_sec: Optional[int] = None


class _FeedbackReq(BaseModel):
    type: str = "etc"               # bug | feature | etc
    title: str = ""
    body: str = ""

class _FeedbackReplyReq(BaseModel):
    id: str
    reply: str = ""


@app.get("/api/admin/config")
async def admin_config_get(request: Request):
    _require_admin(request)
    from infra import admin_config as _ac
    from config import MODEL_ASSIGNMENTS
    try:
        from main_swarm import NEWS_CHECK_INTERVAL as _crawl_def
    except Exception:
        _crawl_def = 900
    cfg = _ac.get_all()
    # 사장 지시 2026-05-24: 모델 목록을 영문 키 대신 한글 직책으로 식별하게 라벨 제공
    # (이전엔 'post_manager' 같은 raw key만 떠서 어느 에이전트인지 헷갈렸다 → 사후관리실장 모델이
    #  의도와 다르게 설정되는 혼란의 원인).
    _labels = {
        "chief_orchestrator": "주식운용실장 (총괄·종목선정)",
        "macro_analyst": "글로벌리서치팀장 (매크로)",
        "quant_analyst": "계량분석팀장 (정량평가)",
        "news_analyst": "마켓센티먼트팀장 (감성·이벤트)",
        "insight_analyst": "기업리서치팀장 (기업 분석자료)",
        "news_curator": "뉴스 크롤러 (헤드라인 선별)",
        "macro_researcher": "매크로 리서치 (웹 리서치)",
        "trader": "프롭트레이딩팀장 (주문·보고)",
        "risk_guard": "리스크관리실장 (DART 재심)",
        "post_manager": "사후관리실장 (매도 판단)",
        "fund_planner": "포트폴리오기획팀장 (진입 thesis·강력 권고)",
        "bond_manager": "채권운용실장 (채권 매매)",
        "commodity_manager": "원자재운용실장 (원자재 매매)",
        "ops_support": "운용지원실장 (진단·조정)",
    }
    return {"model_overrides": cfg["model_overrides"],
            "news_crawl_interval_sec": cfg["news_crawl_interval_sec"],
            "news_crawl_interval_default": int(_crawl_def),
            "model_defaults": MODEL_ASSIGNMENTS,
            "model_labels": _labels,
            "model_keys": list(MODEL_ASSIGNMENTS.keys()),
            "model_choices": _ac.model_choices()}


@app.post("/api/admin/config")
async def admin_config_set(request: Request, req: _AdminConfigReq):
    _require_admin(request)
    from infra import admin_config as _ac
    cfg = _ac.set_config(model_overrides=req.model_overrides,
                         news_crawl_interval_sec=req.news_crawl_interval_sec)
    return {"ok": True, "config": cfg,
            "note": "모델 변경은 다음 재시작에 반영, 크롤 주기는 즉시 반영됩니다."}


# ─── 피드백/버그 제보 (사장 지시 2026-05-24) ───────────────────────────────
# 유저: 프로필 관리 창에서 제출·자기 항목 조회. ADMIN: 전체 조회·답글. 답글은 해당 유저에게 알림.
@app.get("/api/feedback")
async def feedback_list(request: Request):
    uid = _uid_or_403(request)
    from infra import feedback_store as fb
    return {"items": fb.list_for_user(uid), "unseen": fb.count_unseen_replies(uid)}

@app.post("/api/feedback")
async def feedback_submit(req: _FeedbackReq, request: Request):
    uid = _uid_or_403(request)
    from infra import feedback_store as fb
    c = auth_store.get_user_credentials(uid) or {}
    try:
        e = fb.submit(uid, c.get("username"), req.type, req.title, req.body)
    except ValueError as ve:
        raise HTTPException(400, str(ve))
    try:
        await ws_mgr.send_to_admins({"type": "feedback_new", "username": e["username"], "title": e["title"]})
    except Exception:
        pass
    return {"ok": True, "item": e}

@app.post("/api/feedback/seen")
async def feedback_seen(request: Request):
    uid = _uid_or_403(request)
    from infra import feedback_store as fb
    fb.mark_replies_seen(uid)
    return {"ok": True}

@app.get("/api/admin/feedback")
async def admin_feedback_list(request: Request):
    _require_admin(request)
    from infra import feedback_store as fb
    return {"items": fb.list_all(), "open": fb.count_open()}

@app.post("/api/admin/feedback/reply")
async def admin_feedback_reply(req: _FeedbackReplyReq, request: Request):
    _require_admin(request)
    from infra import feedback_store as fb
    e = fb.reply(req.id, req.reply)
    if not e:
        raise HTTPException(404, "해당 피드백을 찾을 수 없습니다.")
    try:
        await ws_mgr.send_to_uid(e["uid"], {"type": "feedback_reply", "title": e["title"]})
    except Exception:
        pass
    return {"ok": True, "item": e}




@app.get("/api/coresight/pending")
async def coresight_pending(request: Request):
    """미처리 Coresight 투자 신호 제안 목록 — ADMIN 전용."""
    uid = _admin_uid_or_403(request)
    from infra.coresight_inbox import list_pending
    return {"pending": list_pending(uid)}


class _CoresightItemReq(BaseModel):
    item_id: str


@app.post("/api/coresight/approve")
async def coresight_approve(request: Request, req: _CoresightItemReq):
    """Coresight 제안 명시 승인 → standing_directive 추가 — ADMIN 전용.
    자동 체결 금지: standing_directive 는 LLM 프롬프트 주입 경로일 뿐이며
    파이썬 리스크·guardrail 게이트를 우회하지 않는다."""
    uid = _admin_uid_or_403(request)
    from infra.coresight_inbox import approve
    ok = approve(uid, req.item_id)
    if not ok:
        raise HTTPException(404, f"item_id={req.item_id} 없음 또는 이미 처리됨")
    return {"ok": True, "item_id": req.item_id}


@app.post("/api/coresight/reject")
async def coresight_reject(request: Request, req: _CoresightItemReq):
    """Coresight 제안 거부 → 큐 영구 제거 — ADMIN 전용."""
    uid = _admin_uid_or_403(request)
    from infra.coresight_inbox import reject
    ok = reject(uid, req.item_id)
    if not ok:
        raise HTTPException(404, f"item_id={req.item_id} 없음")
    return {"ok": True, "item_id": req.item_id}


# ─── 정책 변경 승인 (토요일 ops 제안) — 로그인 유저 본인 계정 ───
# 거버넌스 2026-06-05: 평일 ops 는 정책 플래그 차단, 토요일 점검만 승인 대기로 회부.
# 사장(또는 계정 소유자)이 명시 승인해야만 set_overrides 로 적용된다.
class _PolicyKeyReq(BaseModel):
    key: str


@app.get("/api/policy_changes/pending")
async def policy_changes_pending(request: Request):
    uid = _require_trading(request)
    from infra.policy_approval_inbox import list_pending
    return {"pending": list_pending(uid)}


@app.post("/api/policy_changes/approve")
async def policy_changes_approve(request: Request, req: _PolicyKeyReq):
    uid = _require_trading(request)
    from infra.policy_approval_inbox import approve
    ok = approve(uid, req.key)
    if not ok:
        raise HTTPException(404, f"key={req.key} 없음 또는 이미 처리됨")
    return {"ok": True, "key": req.key}


@app.post("/api/policy_changes/reject")
async def policy_changes_reject(request: Request, req: _PolicyKeyReq):
    uid = _require_trading(request)
    from infra.policy_approval_inbox import reject
    ok = reject(uid, req.key)
    if not ok:
        raise HTTPException(404, f"key={req.key} 없음")
    return {"ok": True, "key": req.key}


@app.get("/api/dart")
async def dart_search(corp_name: str = "", days: int = 7):
    from tools.dart_disclosure import search_disclosures
    from datetime import timedelta
    bgn = (datetime.now() - timedelta(days=days)).strftime("%Y%m%d")
    result = await search_disclosures(corp_name=corp_name, bgn_de=bgn)
    return {"result": result}

@app.get("/api/price/kr/{code}")
async def kr_price(code: str, request: Request):
    broker = REGISTRY.get_or_create(_read_uid(request)).broker
    return {"result": await broker.kr_price_str(code)}

@app.get("/api/price/us/{ticker}")
async def us_price(ticker: str, request: Request):
    broker = REGISTRY.get_or_create(_read_uid(request)).broker
    return {"result": await broker.us_price(ticker)}

@app.get("/api/rank/volume")
async def volume_rank(request: Request):
    broker = REGISTRY.get_or_create(_read_uid(request)).broker
    return {"result": await broker.kr_volume_rank()}

@app.get("/api/balance")
async def balance(request: Request):
    """Account snapshot for the dashboard 'holdings' panel (single cached KIS read).
    Phase 2 멀티테넌트: 요청 유저의 브로커로 조회하고, 그 유저 equity_curve 에만 기록한다."""
    from main_swarm import record_equity, is_market_session_now
    auth_uid = _uid_or_403(request)
    uid = _read_uid(request)
    ctx = REGISTRY.get_or_create(uid)
    try:
        snap = await ctx.broker.portfolio_holdings()
        # 사장 지시 2026-05-24: 실제 정규장 세션(시간대+요일+휴장)일 때만 평가금액 포인트를 기록한다.
        # (모바일 위젯·열린 탭이 /api/balance 를 장외에도 폴링하면 equity_curve 에 장외 포인트가
        #  쌓여 수익률 KPI 가 장외에도 갱신됐다. 주말 밤 US 시간대 오인 방지 위해 요일·휴장까지 본다.)
        if not auth_store.is_viewer(auth_uid) and is_market_session_now():
            try:
                # 사장 지시 2026-06-11: 실거래 원장 평가도 같이 기록 (시드 전이면 None — 폴러가 시드).
                _led_val = None
                try:
                    from infra import trade_ledger
                    from tools.market_data import get_usdkrw
                    _led_val = trade_ledger.value_from_snap(uid, snap, fx=get_usdkrw(1510.0))
                except Exception as _le:
                    logger.warning(f"[balance] 원장 평가 실패(uid={uid}): {_le}")
                record_equity(ctx.swarm.equity_path, snap["buying_power"], "poll",
                              holdings=snap.get("holdings") or [], ledger_eval=_led_val,
                              is_mock=bool(getattr(ctx.broker, "is_mock", False)))
            except Exception as e:
                logger.warning(f"[balance] equity 기록 실패(uid={uid}): {e}")
        # 사장 지시 2026-06-01: 모의계정은 KIS 모의서버가 해외평가를 미지원해 잔고(총평가)가 부정확하므로,
        # 대시보드에서 '잔고'는 숨기고 '수익(실현손익)'만 표시한다 → 프론트가 is_mock 플래그로 분기.
        return {"buying_power": snap["buying_power"], "holdings": snap["holdings"],
                "holdings_stale": snap.get("holdings_stale", False),
                "is_mock": bool(getattr(ctx.broker, "is_mock", False))}
    except Exception as e:
        # KIS 잔고 글리치/조회실패를 조용히 0으로 표시하면 자산곡선이 튄다(알려진 함정) — 로깅으로 표면화.
        logger.warning(f"[balance] 조회 실패(uid={uid}) — buying_power 0 폴백: {e}")
        return {"buying_power": {"cash":0.0,"total_eval":0.0,"pnl_ratio":0.0,"ok":False}, "holdings": [],
                "is_mock": bool(getattr(ctx.broker, "is_mock", False))}

@app.get("/api/equity")
async def equity(request: Request, limit: int = 500, view: str = "realtime"):
    """Equity-curve points for the 수익률 그래프.
    view ∈ {'realtime', 'daily', 'monthly'} — see get_equity_series().
    Phase 2: 요청 유저의 equity_curve 를 읽는다."""
    from main_swarm import get_equity_series
    v = view if view in ("realtime", "daily", "monthly") else "realtime"
    uid = _read_uid(request)
    ep = REGISTRY.get_or_create(uid).swarm.equity_path
    return {"series": get_equity_series(ep, limit, v), "view": v}

async def _attach_current_fallback(base: dict, broker) -> dict:
    """사장 지시 2026-05-28: equity_curve 가 비어도 '실계좌 총평가' 는 KIS 직조회로 폴백 표시.
    base 가 이미 current 를 가지고 있으면(곡선 있음) 건드리지 않는다.
    broker 호출 실패·0/음수는 fail-soft (current 박지 않음)."""
    if base.get("current") is not None:
        return base
    try:
        snap = await broker.portfolio_holdings()
        bp = (snap or {}).get("buying_power") or {}
        cur = float(bp.get("total_eval") or 0.0)
        if cur > 0:
            base["current"] = cur
    except Exception:
        pass    # fail-soft
    return base


@app.get("/api/performance")
async def performance(request: Request):
    """수익률 탭 KPI 카드 — 누적·오늘·주·월 수익(원/%), MDD, 승률, 평균 보유일.
    Phase 2: 요청 유저의 equity_curve 기준.
    사장 지시 2026-05-28: 곡선 비어도 현재 총평가는 broker 폴백으로 항상 표시."""
    from main_swarm import performance_kpis
    uid = _read_uid(request)
    ctx = REGISTRY.get_or_create(uid)
    base = performance_kpis(ctx.swarm.equity_path, uid=uid)
    out = await _attach_current_fallback(base, ctx.broker)
    # 사장 지시 2026-06-01: 모의계정은 '현재 평가액(잔고)' 숨기고 '누적 수익'만 표시 → 프론트 분기용 플래그.
    out["is_mock"] = bool(getattr(ctx.broker, "is_mock", False))
    return out

def _scorecard_for_uid(uid: int):
    """에이전트 성과 스코어카드 (사장 지시 2026-06-04 ④) — 예측신호(scorecard_store)·체결(trade_log)·
    자산곡선·후속가격을 조인해 에이전트별 지표(IC·슬리피지·알파/베타)를 계산한다. 결손은 None/n 으로 표기."""
    import runtime as _rt
    from infra import scorecard_store, user_paths
    from tools.agent_scorecard import compute_scorecard
    from tools.market_data import forward_return_after
    from main_swarm import get_equity_series
    window = int(_rt.get("SCORECARD_WINDOW_DAYS", uid=uid) or 30)
    signals = scorecard_store.list_signals(uid=uid, limit=5000)
    try:
        ep = REGISTRY.get_or_create(uid).swarm.equity_path
        eq = get_equity_series(ep, limit=600, view="realtime")
    except Exception:
        eq = []
    try:
        import json as _json
        tp = user_paths.trade_log_path(uid)
        trades = _json.loads(tp.read_text(encoding="utf-8")) if tp.exists() else []
    except Exception:
        trades = []

    def _price_lookup(code, ts):
        try:
            return forward_return_after(code, ts, window_days=window)
        except Exception:
            return None
    return compute_scorecard(signals, trades, eq, bench=[], price_lookup=_price_lookup, window_days=window)


@app.get("/api/scorecard")
async def api_scorecard(request: Request):
    """에이전트 성과 귀인 카드 — 요청 유저 기준(인증 필요)."""
    return _scorecard_for_uid(_read_uid(request))


@app.get("/api/fundamental_research")
async def api_fundamental_research(request: Request, code: str = "", limit: int = 100):
    """ai-berkshire식 장기 펀더멘털 리서치 스냅샷 조회(advisory)."""
    uid = _read_uid(request)
    from infra import fundamental_research_store
    if code:
        row = fundamental_research_store.latest(uid, code)
        return {"item": row}
    return {"items": fundamental_research_store.list_snapshots(uid=uid, limit=max(1, min(int(limit or 100), 500)))}


@app.post("/api/fundamental_research")
async def api_record_fundamental_research(req: FundamentalResearchReq, request: Request):
    """수동 ai-berkshire/Codex 리서치 결과를 advisory 스냅샷으로 저장."""
    uid = _read_uid(request)
    from infra import fundamental_research_store
    from tools.fundamental_rigor import assess_fundamental_research
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    base = assess_fundamental_research(req.code, req.name or req.code, dart_text=req.memo, financial_text=req.memo, source=req.source)
    for k in ("verdict", "business_quality_score", "valuation_margin_score", "moat_score", "management_score"):
        v = getattr(req, k, None)
        if v not in (None, ""):
            base[k] = v
    if req.thesis_invalidators:
        base["thesis_invalidators"] = req.thesis_invalidators
    if req.financial_checks:
        base["financial_checks"] = req.financial_checks
    if req.memo:
        base["memo"] = req.memo[:2000]
    base.update({"uid": uid, "ts": ts, "cycle_started_at": ts, "source": req.source or "manual_ai_berkshire"})
    rid = fundamental_research_store.record_snapshot(base)
    return {"ok": bool(rid), "id": rid, "item": base}


@app.get("/api/benchmark")
async def benchmark(request: Request, view: str = "daily"):
    """벤치마크 곡선 — KOSPI·NASDAQ 를 자산곡선 시작 평가액에 리베이스해 겹쳐 보여준다.
    사장 지시 2026-05-21: 지수값은 5분 폴링(_equity_poller)이 잔고와 함께 equity 포인트에 같이 저장한
    kospi/nasdaq(QQQ 현재가 프록시)를 그대로 쓴다 → equity 와 지수가 동일 타임스탬프에 정렬되어
    일중 움직임이 자연히 표시된다(미검증 분봉 API 불필요). 지수값이 쌓이지 않은 포인트는 건너뛴다.
    Phase 2: 요청 유저의 equity_curve 기준."""
    from main_swarm import get_equity_series
    v = view if view in ("realtime", "daily", "monthly") else "daily"
    ep = REGISTRY.get_or_create(_read_uid(request)).swarm.equity_path
    series = get_equity_series(ep, limit=600, view=v)
    if len(series) < 2:
        return {"benchmarks": []}

    def valof(p):
        x = p.get("adj_total_eval")
        return x if x is not None else p.get("total_eval")

    base_val = valof(series[0])
    if not base_val:
        return {"benchmarks": []}

    out = []
    # 사장 지시 2026-06-16: nasdaq 값은 실제 지수가 아니라 QQQ 현재가 프록시이므로 라벨을 명확히 한다.
    for name, key in (("KOSPI", "kospi"), ("NASDAQ(QQQ)", "nasdaq")):
        # 지수값이 있는 첫 포인트에 리베이스 (오해를 주는 +0.00 평탄선 회피 — 값 없으면 생략)
        base_idx = next((p[key] for p in series if p.get(key)), None)
        if not base_idx:
            continue
        line = [{"label": p.get("label"), "value": base_val * (p[key] / base_idx)}
                for p in series if p.get(key)]
        if len(line) >= 2:
            out.append({"label": name, "series": line})
    return {"benchmarks": out}

@app.get("/api/trades")
async def trades(request: Request, limit: int = 500):
    """All real trade events (executed/failed), newest first — 전체 거래 내역.
    Phase 2 멀티테넌트: 요청 유저의 거래만 반환한다."""
    uid = _read_uid(request)
    from main_swarm import get_trade_history
    return {"trades": get_trade_history(limit, uid=uid)}

@app.post("/api/ledger/reseed")
async def ledger_reseed(request: Request):
    """실거래 원장 재시드 (사장 지시 2026-06-11) — 입출금·수동거래로 원장-KIS 괴리가 생겼을 때
    현재 KIS 보유/예수금 기준으로 원장을 새로 굽는다. (자산곡선의 과거 ledger 포인트는 보존.)"""
    from infra import trade_ledger
    uid = _require_trading(request)
    ctx = REGISTRY.get_or_create(uid)
    trade_ledger.reset(uid)
    try:
        snap = await ctx.broker.portfolio_holdings()
        led = await trade_ledger.seed(uid, ctx.broker, snap)
    except Exception as e:
        return {"ok": False, "message": f"재시드 실패: {e}"}
    if led is None:
        return {"ok": False, "message": "재시드 실패 — KIS 잔고 조회가 정상일 때 다시 시도하세요"}
    # 사장 지시 2026-06-16: 결제 과도기(US/해외 매수 D+2 미결제) 중 재시드하면 예수금(미차감)과
    # 보유(추가)가 이중계상되어 평가액이 일시 과대해질 수 있다(uid1 9.66M 사례). 추측으로 차감하면
    # 정상 결제 시 과소로 틀어지므로, 주의 안내만 덧붙이고 정확한 정정은 결제(D+2) 후 재시드로 유도한다.
    _settle_caution = (" · ⚠ 최근 해외(US) 매수가 미결제(D+2) 상태면 평가액이 일시 과대할 수 있습니다 — "
                       "결제 완료 후 한 번 더 재시드하면 정확해집니다")
    return {"ok": True, "message": f"원장 재시드 완료 — KRW {led['cash_krw']:,.0f} / USD {led['cash_usd']:,.2f} / 종목 {len(led['positions'])}개{_settle_caution}",
            "ledger": {k: led[k] for k in ("seeded_at", "seed_source", "cash_krw", "cash_usd")},
            "positions": led["positions"]}

@app.post("/api/trades/clear")
async def trades_clear(request: Request):
    """Wipe only the trade history (keeps system/agent logs intact). Used by the 수익률 탭 '비우기' button."""
    uid = _require_trading(request)
    from main_swarm import clear_trade_log
    removed = clear_trade_log(uid=uid)
    return {"message": f"🗑️ 거래 내역 {removed}건 초기화됨", "removed": removed}

@app.get("/api/strategy")
async def strategy_get(request: Request):
    """전략 탭에 필요한 모든 데이터 — active/presets/history + 한국어 메타데이터 (사장 지시 2026-05-14).
    사장 지시 2026-05-21: active.ops_since = 운용지원실장이 이 프로필 파라미터를 마지막으로 갱신한 시각
    (active.params 는 이미 프로필 오버라이드가 반영된 '효과적' 값이므로 설명도 전체 상세설정을 보여줄 수 있다)."""
    import runtime
    from infra import profile_overrides
    from config import STRATEGY_KEY_META, STRATEGY_TUNABLE_KEYS, STRATEGY_DEFAULTS
    uid = _read_uid(request)
    active = runtime.active(uid=uid)
    active["ops_since"] = profile_overrides.last_updated(uid)
    locked = []
    return {"active": active,
            "history": runtime.history(),
            "key_meta": STRATEGY_KEY_META, "key_order": STRATEGY_TUNABLE_KEYS,
            "defaults": STRATEGY_DEFAULTS, "locked_keys": locked}

@app.post("/api/strategy")
async def strategy_set(req: dict, request: Request):
    """현재 프로필의 전략 파라미터를 갱신한다."""
    import runtime
    from main_swarm import _broadcast
    uid = _require_strategy_uid(request)
    custom = (req or {}).get("params")
    if not custom:
        raise HTTPException(400, "params 필요")
    # 사장 지시 2026-07-21: 대시보드 '변경값 적용/기본값 적용'은 사용자의 최종 결정이므로
    # 운용지원(profile_overrides)보다 우선해야 한다. runtime.get 우선순위가
    # profile_overrides > set_strategy(_states) 이므로, 사용자가 바꾼 값도 profile_overrides 에
    # 함께 기록해야 실제로 반영된다(안 그러면 ops가 건드린 키에서 변경이 조용히 무시됨).
    try:
        from infra import profile_overrides as _po
        from config import STRATEGY_TUNABLE_KEYS as _TK
        _known = set(_TK)
        _po.set_overrides(int(uid), {k: v for k, v in custom.items() if k in _known})
    except Exception as _pe:
        logger.warning("전략 변경 profile_overrides 반영 실패 uid=%s: %s", uid, _pe)
    active = runtime.set_strategy(custom=custom, by="dashboard", uid=uid)
    try:
        await _broadcast({"type": "status", "state": "IDLE",
                          "message": f"⚙️ 전략 설정 변경 → {active['label']}"})
    except Exception: pass
    return {"active": active}

@app.get("/api/cycles")
async def cycles(request: Request, limit: int = 50, offset: int = 0, summary: int = 0):
    """Persisted analysis cycles (newest first). 사장 지시 2026-05-14 — 백테스트/장기 분석용.
    QuantInSight 이식(2026-07-18): summary=1 이면 사이클 탭 히스토리용 경량 목록(리포트 본문 제외)."""
    from infra import cycle_store
    if summary:
        return {"cycles": cycle_store.list_cycles_summary(limit, offset, uid=_read_uid(request))}
    return {"cycles": cycle_store.list_cycles(limit, offset, uid=_read_uid(request))}

@app.get("/api/cycles/{cycle_id}")
async def cycle_detail(cycle_id: int, request: Request):
    from infra import cycle_store
    row = cycle_store.get_cycle(cycle_id)
    if not row or row.get("uid") != _read_uid(request):
        raise HTTPException(404, f"cycle {cycle_id} 없음")
    return row

@app.get("/api/ops_history")
async def ops_history_endpoint(request: Request, limit: int = 100):
    """운용지원실장 자동 수정 이력 (사장 지시 2026-05-14). Newest-first 응답."""
    from infra import ops_history
    h = ops_history.load_history()
    from infra.strategy_research import list_proposals
    from tools.stock_policy import status as policy_status
    return {"history": list(reversed(h))[:max(1, int(limit))], "stats": ops_history.stats(),
            "research": list_proposals(_read_uid(request)), "automatic_tuning": False,
            "systematic": policy_status(_read_uid(request))}

@app.websocket("/ws")
async def ws_ep(ws: WebSocket):
    if ws.headers.get("origin") not in (None, "https://quantinsight.ai-ve.uk"):
        await ws.close(code=4403)
        return
    try:
        uid, expires = await asyncio.to_thread(cloudflare_access.verify,
            ws.headers.get("Cf-Access-Jwt-Assertion", ""))
    except (PermissionError, cloudflare_access.AccessNotConfigured):
        await ws.close(code=4401)
        return
    # 사장 지시 2026-05-21: 모바일 네이티브 클라이언트(?client=mobile)는 프로필 알림설정으로
    # 4종 푸시를 게이트한다. 웹(기본)은 전부 수신.
    client = (ws.query_params.get("client") or "web").strip().lower()
    view_uid = (auth_store.admin_view_uid() if auth_store.is_viewer(uid)
                else auth_store.resolve_profile_uid(uid))
    if view_uid is None:
        await ws.close(code=1013)
        return
    await ws_mgr.connect(ws, uid=uid, view_uid=view_uid,
                         client=("mobile" if client == "mobile" else "web"))
    try:
        import time
        while True:
            remaining = expires - time.time()
            if remaining <= 0:
                await ws.close(code=4401)
                break
            await asyncio.wait_for(ws.receive_text(), timeout=remaining)
    except asyncio.TimeoutError:
        await ws.close(code=4401)
    except WebSocketDisconnect:
        pass
    finally:
        ws_mgr.disconnect(ws)

SD = Path(__file__).parent / "static"; SD.mkdir(exist_ok=True)

@app.get("/", response_class=HTMLResponse)
async def dash():
    # 사장 지시 2026-05-21: 모바일 WebView가 옛 index.html을 캐시해 UI 수정이 안 보이던 문제 —
    # 대시보드 HTML은 항상 최신을 받도록 캐시 금지(파일은 작아 매 요청 디스크 읽기 부담 없음).
    p = SD / "index.html"
    content = p.read_text(encoding="utf-8") if p.exists() else "<h1>ArQuant v1.0</h1>"
    return HTMLResponse(content=content, headers={
        "Cache-Control": "no-cache, no-store, must-revalidate", "Pragma": "no-cache", "Expires": "0"})


@app.get("/corp", response_class=HTMLResponse)
async def corp():
    """운용위원회 사무실 시각화(사장 지시 2026-09-25) — 대시보드 헤더 '시각화' 버튼.
    말풍선은 /api/events·/ws 의 실제 agent_msg 다. 인증은 위 app_auth 미들웨어가 그대로 건다."""
    p = SD / "corp.html"
    return HTMLResponse(content=p.read_text(encoding="utf-8"), headers={
        "Cache-Control": "no-cache, no-store, must-revalidate", "Pragma": "no-cache", "Expires": "0"})


if SD.exists(): app.mount("/static", StaticFiles(directory=str(SD)), name="static")


# ─── 부팅 시 인증 부트스트랩 + 유저별 매매 루프 자동 재개 (Phase 2 멀티테넌트) ──────
# 1) auth_store 초기화·마이그레이션, 데이터 마이그레이션(전역→유저별 백업), .env 시드.
# 2) 직전 부팅 시 매매 중이던 유저(data/<uid>/.running 마커 보유)들의 루프를 각각 재개한다.
#    전역 단일 활성 계정/단일 _task 개념은 폐지 — 유저별로 독립 재개한다.
@app.on_event("startup")
async def _auth_bootstrap():
    # Credentials and trading ledgers remain encrypted in the private store.
    # No password migration, registration or environment account seeding.
    auth_store.init()


# ─── 분봉 크롤러 — KOSPI200+KOSDAQ150 분봉 수집 (market_bars/, 구 Lag_Trading 크롤러 이관) ──
# lead-lag 신호의 데이터원(data/bars.db). 별도 systemd 유닛 없이
# 트레이딩 서버 프로세스 안에서 백그라운드 데몬 스레드로 돈다(사장 지시 2026-07-21).
@app.on_event("startup")
async def _start_bar_crawler():
    try:
        from market_bars import start_background
        start_background()
    except Exception as e:
        logging.getLogger("quantinsight.bars").warning("분봉 크롤러 시작 실패(계속): %s", e)


# ─── 부팅 자동재개 — data/<uid>/.running 마커가 있는 유저별로 매매 루프를 다시 켠다 ───
# 운용지원실장 코드 갱신 후 재시작·서버 재시작 시, 직전에 감시 중이던 유저들의 루프를
# 끊김 없이 재개한다(유저별 독립). _start_uid 가 마커를 다시 쓰므로 멱등하다.
@app.on_event("startup")
async def _auto_resume_running_uids():
    async def _starter():
        await asyncio.sleep(2)  # 로깅·WS 준비 후 스왐이 broadcast 하도록 약간 지연
        from infra import user_paths
        data_dir = user_paths._DATA_DIR
        try:
            uid_dirs = [p for p in data_dir.iterdir() if p.is_dir() and p.name.isdigit()]
        except Exception as e:
            logging.getLogger("AUTH").warning("자동재개 디렉토리 스캔 실패: %s", e)
            return
        for d in uid_dirs:
            uid = int(d.name)
            if not (d / ".running").exists():
                continue
            try:
                await _start_uid(uid)
                from main_swarm import log_response_event
                log_response_event({"source": "system_event", "type": "status",
                                    "state": "MONITORING",
                                    "message": "🟢 자동 재개"},
                                   uid=uid)
            except Exception as e:
                logging.getLogger("AUTH").warning("uid=%s 자동 재개 실패: %s", uid, e)
    asyncio.create_task(_starter())

if __name__ == "__main__":
    import uvicorn; from config import APP_HOST, APP_PORT
    uvicorn.run(app, host=APP_HOST, port=APP_PORT)
