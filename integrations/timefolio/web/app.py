"""ArcTrade 매수/매도 신호 대시보드.

FastAPI, 127.0.0.1:8620 (Cloudflare 터널 뒤 루프백만 바인딩 — HYFE_IQC 패턴).
데이터원은 전부 ArcTrade 런타임 산출물(읽기 전용):
  data/leadlag_map.json  엔진의 선행-후행 지도
  data/signals.jsonl     전략 신호 피드 (BUY/SELL/SHORT_SIGNAL)
  data/positions.json    보유 포지션
  data/trades.csv        체결 내역
  data/bars.db           분봉 (스파크라인·실시간 WATCH 연산)

실행: python3 -m web.app   (또는 arctrade.service)
"""
import asyncio
import csv
import hashlib
import json
import logging
import sqlite3
import sys
import time
from contextlib import asynccontextmanager, suppress
from datetime import timedelta
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

# 루트 로거가 비어 있으면 autofolio 사이클/주문 로그(INFO)가 통째로 버려진다 — journald 로 흘린다.
# (엔진과 같은 lag_trading.log 파일을 두 프로세스가 rotate 하면 경합하므로 파일 핸들러는 쓰지 않는다.)
_root = logging.getLogger()
if not _root.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%m-%d %H:%M:%S"))
    _root.addHandler(_handler)
_root.setLevel(logging.INFO)
for _noisy in ("playwright", "urllib3", "asyncio", "httpx"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config                                  # noqa: E402
from utils import market_time as mt            # noqa: E402
from web import autofolio as af                # noqa: E402
from web import backtest                       # noqa: E402
from web import nlbacktest as nlbt             # noqa: E402
from web import signals_feed as sf             # noqa: E402
from web import experiment_store as experiments  # noqa: E402
from web import krximg                         # noqa: E402
from web import cnn_research                   # noqa: E402
from web import llm                            # noqa: E402
from web import strategy_runner                # noqa: E402

@asynccontextmanager
async def _lifespan(_app: FastAPI):
    cycle_task = asyncio.create_task(af.auto_loop())
    strategy_task = asyncio.create_task(strategy_runner.auto_loop())
    cnn_task = asyncio.create_task(cnn_research.auto_loop())
    try:
        yield
    finally:
        cycle_task.cancel()
        strategy_task.cancel()
        cnn_task.cancel()
        with suppress(asyncio.CancelledError):
            await cycle_task
        with suppress(asyncio.CancelledError):
            await strategy_task
        with suppress(asyncio.CancelledError):
            await cnn_task
        await asyncio.get_running_loop().run_in_executor(af._CYCLE_EXEC, af._drop_session)


app = FastAPI(title="ArcTrade", docs_url=None, redoc_url=None, lifespan=_lifespan)
from web.markets import router as markets_router
from web.impact_lab import router as impact_router
app.include_router(markets_router)
app.include_router(cnn_research.router)
app.include_router(impact_router)
STATIC = Path(__file__).resolve().parent / "static"

_names_cache = {}


def names():
    """{code: 종목명} — universe.csv 캐시."""
    global _names_cache
    if not _names_cache and config.UNIVERSE_CSV.exists():
        with open(config.UNIVERSE_CSV, newline="", encoding="utf-8") as f:
            _names_cache = {r["code"]: r["name"] for r in csv.DictReader(f)}
    return _names_cache


def db():
    conn = sqlite3.connect(f"file:{config.DB_PATH}?mode=ro", uri=True, timeout=5)
    return conn


def read_json(path, default):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return default


def latest_two_bars(conn, codes):
    """{code: [(ts, close) 최신2개]}"""
    if not codes:
        return {}
    rows = conn.execute(
        "SELECT code, ts, close FROM bars WHERE code IN (%s) ORDER BY code, ts DESC"
        % ",".join("?" * len(codes)), list(codes)).fetchall()
    out = {}
    for code, ts, close in rows:
        if len(out.setdefault(code, [])) < 2:
            out[code].append((ts, close))
    return out


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/summary")
def summary():
    m = read_json(config.LEADLAG_MAP_JSON, {})
    positions = read_json(config.POSITIONS_JSON, [])
    sig_count = 0
    if config.SIGNALS_JSONL.exists():
        with open(config.SIGNALS_JSONL, encoding="utf-8") as f:
            today = mt.now_kst().strftime("%Y-%m-%d")
            sig_count = sum(1 for ln in f if ln.startswith('{"ts": "' + today))
    return {
        "now": mt.now_kst().isoformat(timespec="seconds"),
        "market_open": mt.in_crawl_session(),
        "map_updated": m.get("updated"),
        "n_codes": m.get("n_codes", 0),
        "n_eligible": m.get("n_eligible", 0),      # 평균회귀 유의 종목수
        "n_candidates": m.get("n_candidates", 0),  # 현재 매수 후보수
        "n_positions": len(positions),
        "signals_today": sig_count,
        "live_trading": config.LIVE_TRADING,
        "regime": m.get("regime", {}),             # 시장 레짐(진입 허용 여부)
    }


def load_closes(codes):
    """bars.db → {code: {ts: close}}. 없거나 못 읽으면 그 종목은 빈 dict."""
    out = {c: {} for c in codes}
    if not codes:
        return out
    try:
        conn = db()
    except sqlite3.Error:
        return out
    try:
        qs = ",".join("?" * len(codes))
        for code, ts, close in conn.execute(
                f"SELECT code, ts, close FROM bars WHERE code IN ({qs})", tuple(codes)):
            out[code][ts] = close
    except sqlite3.Error:
        pass
    finally:
        conn.close()
    return out


# 지도 승률은 bars.db 를 훑는다 — 15초 폴링마다 다시 계산하면 낭비라 짧게 캐시한다.
_MAP_TOP_N = 40
_map_cache = {"key": None, "pairs": None, "at": 0.0}
_MAP_TTL = 60.0


@app.get("/api/map")
def score_map():
    """전 종목 s-score 스냅샷 — 과매도(음수) 순. stat-arb 재구성으로 리드-랙 페어표를 대체."""
    m = read_json(config.LEADLAG_MAP_JSON, {})
    nm = names()
    rows = []
    for code, sc in (m.get("scores") or {}).items():
        hl = sc.get("half_life")
        if hl is None or hl > config.STATARB_MAX_DISPLAY_HALFLIFE:
            continue                       # 반감기 과대(비-회귀·낙하칼류)는 표시하지 않는다
        rows.append({
            "code": code, "name": nm.get(code, code),
            "s": sc.get("s"), "half_life": hl,
            "exp_ret": sc.get("exp_ret"), "adf_t": sc.get("adf_t"),
            "sig_ok": sc.get("sig_ok"), "price": sc.get("price"),
            "dart_risky": sc.get("dart_risky")})
    rows.sort(key=lambda r: (r["s"] if r["s"] is not None else 0.0))   # 가장 과매도 먼저
    return {"updated": m.get("updated"), "n_eligible": m.get("n_eligible", 0), "scores": rows}


@app.get("/api/watch")
def watch():
    """현재 매수 후보: s<-진입문턱, 평균회귀 유의, 기대반전이 왕복마찰 초과 — 과매도 순.
    stat-arb 재구성으로 리드-랙 WATCH 를 대체."""
    m = read_json(config.LEADLAG_MAP_JSON, {})
    nm = names()
    out = []
    for c in (m.get("candidates") or []):
        out.append({
            "ts": m.get("updated"),
            "direction": "UP",                     # 롱 후보(반등 기대)
            "follower": c["code"], "follower_name": nm.get(c["code"], c["code"]),
            "s": c["s"], "half_life": c["half_life"],
            "exp_ret": round(c["exp_ret"], 5), "price": c["price"],
            "expect": f"반감기 {c['half_life']:.0f}분 내 반등 기대 (s={c['s']:+.2f}, "
                      f"기대반전 {c['exp_ret']*100:.2f}%)"})
    return {"watch": out, "updated": m.get("updated")}


@app.get("/api/signals")
def signals(limit: int = 60):
    """신호 로그 + 각 신호의 실현손익(ret/krw). 롱·숏을 한 응답에 담아 화면이 한 번만 그려진다.

    예전엔 프론트가 숏 청산가를 `/api/bars` 로 따로 받아 두 번 렌더했고, 그래서 롱숏 화면이
    15초마다 "롱만" → "롱+숏" 으로 깜빡였다.
    """
    out = []
    if config.SIGNALS_JSONL.exists():
        with open(config.SIGNALS_JSONL, encoding="utf-8") as f:
            lines = f.readlines()[-max(1, min(limit, 500)):]
        nm = names()
        for ln in lines:
            try:
                rec = json.loads(ln)
            except ValueError:
                continue
            rec["follower_name"] = nm.get(rec.get("follower", ""), rec.get("follower", ""))
            rec["leader_name"] = nm.get(rec.get("leader", ""), rec.get("leader", ""))
            # 엣지 게이트로 대회 계정 주문이 안 나간 BUY 는 화면에서 구분되게 표시한다.
            rec["gate_blocked"] = sf.gate_reason(rec)
            out.append(rec)

    shorts = {r["follower"] for r in out if r.get("kind") == "SHORT_SIGNAL" and r.get("follower")}
    res = backtest.resolve_signals(out, load_closes(shorts))   # out 은 시간 오름차순
    # 왕복 마찰비용(수수료 0.1%×2 + 거래세 0.2%)을 화면이 반드시 차감하게 함께 내려보낸다.
    # 이 전략의 평균 왕복 gross 는 ±0.1% 수준이라, 비용을 빼먹은 누적수익률은 부호 자체가 뒤집힌다
    # (실측 2026-07-14: gross -2.7% ↔ net -27.9%). 비용 없는 수익률은 의사결정을 오도한다.
    return {"signals": list(reversed(res["signals"])), "unresolved": res["unresolved"],
            "round_trip_cost_pct": float(config.AUTOFOLIO_ROUND_TRIP_COST_PCT or 0.0)}


@app.get("/api/positions")
def positions():
    nm = names()
    poss = read_json(config.POSITIONS_JSON, [])
    for p in poss:
        p["name"] = nm.get(p.get("code", ""), p.get("code", ""))
    return {"positions": poss}


@app.get("/api/trades")
def trades(limit: int = 50):
    out = []
    if config.TRADES_CSV.exists():
        with open(config.TRADES_CSV, newline="", encoding="utf-8") as f:
            out = list(csv.DictReader(f))[-max(1, min(limit, 300)):]
        nm = names()
        for t in out:
            t["name"] = nm.get(t.get("code", ""), t.get("code", ""))
    return {"trades": list(reversed(out))}


@app.get("/api/bars/{code}")
def bars(code: str, minutes: int = 120):
    if not code.isdigit() or len(code) != 6:
        raise HTTPException(400, "6자리 종목코드가 아님")
    try:
        conn = db()
    except sqlite3.Error:
        return JSONResponse({"code": code, "bars": []})
    try:
        rows = conn.execute(
            "SELECT ts, close FROM bars WHERE code=? ORDER BY ts DESC LIMIT ?",
            (code, max(10, min(minutes, 1200)))).fetchall()
    finally:
        conn.close()
    rows.reverse()
    return {"code": code, "name": names().get(code, code),
            "bars": [{"ts": ts, "close": c} for ts, c in rows]}


@app.get("/api/stream")
async def stream():
    """SSE — 실시간 예측(WATCH)·요약을 서버 푸시. 폴링·새로고침 없이 즉시 반영."""
    async def gen():
        last_sig = None
        last_emit = 0.0
        while True:
            try:
                payload = {"watch": (await asyncio.to_thread(watch))["watch"],
                           "summary": await asyncio.to_thread(summary)}
                body = json.dumps(payload, ensure_ascii=False)
                sig = hashlib.md5(body.encode()).hexdigest()
                now = time.monotonic()
                # 변경 시 즉시 + 15초 하트비트(클라우드플레어 연결 유지)
                if sig != last_sig or now - last_emit > 15:
                    last_sig, last_emit = sig, now
                    yield f"data: {body}\n\n"
            except Exception:
                yield ": keepalive\n\n"
            await asyncio.sleep(2)
    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/api/autofolio/summary")
def autofolio_summary():
    return af.summary()


@app.get("/api/autofolio/trades")
def autofolio_trades(limit: int = 50):
    return {"trades": af.trades(limit)}


@app.get("/api/autofolio/orders")
def autofolio_site_orders(limit: int = 50):
    return af.site_orders(limit)


@app.post("/api/autofolio/cycle")
async def autofolio_cycle():
    return await af.run_cycle("manual")


# ── 자연어 원샷 백테스트 (구 KRX 퀀트 시뮬레이터 통합) ──────────────────
@app.post("/api/nlbacktest")
def nlbacktest_start(body: dict, x_arctrade_profile: str | None = Header(default=None)):
    body = body or {}
    text = str(body.get("text") or "").strip()
    strategy = body.get("strategy")
    if not text and not isinstance(strategy, dict):
        raise HTTPException(400, "전략 아이디어나 생성된 전략이 필요합니다")
    if not nlbt.start(text, strategy=strategy, provider=body.get("provider"),
                      profile_id=x_arctrade_profile):
        raise HTTPException(409, "이미 백테스트가 진행 중입니다")
    return {"started": True}


@app.post("/api/strategy/generate")
def strategy_generate(body: dict):
    body = body or {}
    try:
        strategy = nlbt.generate(str(body.get("text") or ""), provider=body.get("provider"))
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(400, str(exc)) from exc
    return {"strategy": strategy}


def _autofolio_nav() -> float:
    try:
        return float((((af.summary().get("account") or {}).get("portfolio") or {}).get("total_eval")) or 0)
    except Exception:
        return 0.0


@app.get("/api/config/status")
def config_status():
    return {"providers": llm.provider_status(),
            "timefolio_configured": bool(config.AUTOFOLIO_SITE_USERNAME and config.AUTOFOLIO_SITE_PASSWORD),
            "live_orders": bool(config.AUTOFOLIO_LIVE_ORDERS)}


@app.get("/api/experiments")
def experiment_history(x_arctrade_profile: str | None = Header(default=None)):
    return experiments.get(x_arctrade_profile, nav=_autofolio_nav(),
                           signal_count=experiments.signal_count())


@app.post("/api/strategy/activate")
def strategy_activate(body: dict, x_arctrade_profile: str | None = Header(default=None)):
    body = body or {}
    strategy = body.get("strategy")
    if not isinstance(strategy, dict):
        raise HTTPException(400, "생성된 전략이 필요합니다")
    result = nlbt.get_result()
    if not result or result.get("error") or result.get("parsed") != strategy:
        raise HTTPException(409, "이 전략의 백테스트를 먼저 완료하세요")
    before = experiments.get(x_arctrade_profile, nav=_autofolio_nav(),
                             signal_count=experiments.signal_count())
    activated = experiments.activate(
        x_arctrade_profile, strategy, nav=_autofolio_nav(),
        signal_count=experiments.signal_count(),
        confirm_reset=bool(body.get("confirm_reset")),
    )
    if activated.get("requires_reset"):
        return JSONResponse(activated, status_code=409)
    if activated.get("changed"):
        old_id = ((before.get("active") or {}).get("strategy_id") or "initial")
        archive = experiments.reset_strategy_runtime(old_id)
        emitted = strategy_runner.evaluate_active(force=True, prepared_result=result)
        activated["runtime_reset"] = True
        activated["archive"] = archive.name
        activated["signal_evaluation"] = emitted
    return activated


@app.get("/api/nlbacktest/progress")
def nlbacktest_progress():
    return nlbt.get_progress()


@app.get("/api/nlbacktest/result")
def nlbacktest_result():
    r = nlbt.get_result()
    if r is None:
        raise HTTPException(404, "결과 없음")
    return r


@app.get("/api/candles/{code}")
def candles(code: str, tf: int = 5, n: int = 160):
    """분봉 종가(bars.db 는 close·vol_cum 뿐)를 tf 분 봉으로 묶은 캔들 — o/h/l/c 는 분 종가의 첫/최고/최저/끝, v 는 누적거래량 증분."""
    if not code.isdigit() or len(code) != 6 or tf not in (1, 5, 30, 1440):
        raise HTTPException(400, "6자리 종목코드와 tf 1/5/30/1440")
    try:
        conn = db()
    except sqlite3.Error:
        return {"code": code, "tf": tf, "candles": []}
    try:
        need = max(30, min(n, 400)) * (tf if tf < 1440 else 390)
        rows = conn.execute("SELECT ts, close, vol_cum FROM bars WHERE code=? ORDER BY ts DESC LIMIT ?", (code, need + 400)).fetchall()
    finally:
        conn.close()
    rows.reverse(); out = []; cur = None; prev_vol = None
    for ts, close, vol in rows:
        key = ts[:8] if tf == 1440 else ts[:8] + f"{(int(ts[8:10]) * 60 + int(ts[10:12])) // tf * tf:04d}"
        v = 0.0 if prev_vol is None or vol is None or vol < prev_vol else vol - prev_vol
        prev_vol = vol if vol is not None else prev_vol
        if cur is None or cur["k"] != key:
            cur = {"k": key, "t": ts, "o": close, "h": close, "l": close, "c": close, "v": v}; out.append(cur)
        else:
            cur["h"] = max(cur["h"], close); cur["l"] = min(cur["l"], close); cur["c"] = close; cur["v"] += v
    for c in out:
        c.pop("k"); c["t"] = f"{c['t'][:4]}-{c['t'][4:6]}-{c['t'][6:8]} {c['t'][8:10]}:{c['t'][10:12]}"
    return {"code": code, "name": names().get(code, code), "tf": tf, "candles": out[-max(30, min(n, 400)):]}


# ── 이미지 기반 전략(heatf CNN)의 KRX 백테스트 ─────────────────────────
@app.get("/api/krximg/results")
def krximg_results(cost: float = 0.002, q: float = 0.1, long_only: bool = True):
    return krximg.results(cost, q, long_only)


@app.post("/api/krximg/backtest")
def krximg_backtest(body: dict):
    body = body or {}
    cost = float(body.get("cost", 0.002)); q = float(body.get("q", 0.1))
    if not (0 <= cost <= 0.02 and 0.02 <= q <= 0.5):
        raise HTTPException(400, "cost 는 0~2%, q 는 2~50% 사이여야 합니다")
    if not krximg.start(cost, q, bool(body.get("long_only", True)), bool(body.get("shuffle", True))):
        raise HTTPException(409, "이미 백테스트가 진행 중입니다")
    return {"started": True}


@app.get("/api/krximg/progress")
def krximg_progress():
    return krximg.progress()


if STATIC.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")


if __name__ == "__main__":
    import uvicorn
    # access_log=False: 대시보드가 15초 주기로 폴링해 journald 가 GET 로그로 도배되고,
    # 정작 autofolio 오류/경고가 그 사이에 묻힌다(2026-07-13). 앱 로그(INFO+)만 남긴다.
    uvicorn.run(app, host=config.WEB_HOST, port=config.WEB_PORT, log_level="info", access_log=False)
