"""자연어 전략 생성과 백테스트.

한 줄로 전략을 말하면 파싱(로컬 LLM) → 재무 필터 → 유니버스 선정 → 일별 백테스트를
한 번에 돌린다. 종목선정·주가수집·백테스트를 나눠 클릭하던 흐름을 하나로 합친 것.

가격은 공유 저장소(QIS/data/daily_·investor_)를 먼저 읽고, 없을 때만 arcmarket/네이버 폴백
(quant.engine 이 이미 그렇게 재배선됨). krxsimul 서비스는 이 통합 후 삭제된다.
"""
import json
import os
import re
import threading

from quant import engine as qe
from web import llm

# 유니버스 상한 — 원샷을 응답성 있게(수 초~1분) 유지. 시총 상위부터 채운다.
MAX_UNIVERSE = int(os.environ.get("NLBT_MAX_UNIVERSE", "60"))
MIN_UNIVERSE = 8
_VALID_PORT = {"equal_weight", "market_cap", "momentum_weight", "risk_parity",
               "inverse_volatility", "kelly_criterion", "min_variance", "max_sharpe",
               "dynamic_asset", "all_in"}


def _llm(prompt, *, provider=None, json_mode=False, max_tokens=8000):
    return llm.complete(prompt, provider=provider, json_mode=json_mode, max_tokens=max_tokens)


# ── 진행률 (단일 실행 가정; thread-safe) ────────────────────────────────
_lock = threading.Lock()
_progress = {"status": "idle", "pct": 0, "msg": "", "error": None}


def _set(pct, msg, error=None):
    with _lock:
        if error is not None:
            _progress.update(status="error", error=error, msg=msg)
        else:
            _progress.update(status="running", pct=pct, msg=msg, error=None)


def get_progress():
    with _lock:
        return dict(_progress)


# ── 파싱: 자연어 → {financial_logic, buy_logic, sell_logic, period_months, portfolio_strategy} ──
def _logic_rules():
    path = os.path.join(os.path.dirname(qe.__file__), "logic.csv")
    try:
        with open(path, encoding="utf-8-sig") as f:
            return f.read()
    except Exception:
        return ""


# evaluator 어휘로 흔한 동의어 정규화(LLM 출력 편차 흡수). engine.TECH_INDICATORS_MAP 기준.
def _normalize(expr):
    if not expr:
        return expr
    expr = re.sub(r"(\d+)\s*일\s*(?:이동평균선|이평선|이평|평균선|이동평균)", r"\1일선", expr)
    expr = re.sub(r"스토[캐카]스틱\s*[Kk]", "STOCH_K", expr)
    expr = re.sub(r"스토[캐카]스틱\s*[Dd]", "STOCH_D", expr)
    expr = expr.replace("골든 크로스", "골든크로스").replace("데드 크로스", "데드크로스")
    expr = re.sub(r"RSI\s*지수", "RSI", expr)
    return expr


def _safe_logic(value: str, field: str) -> str:
    value = _normalize((value or "").strip())
    if len(value) > 600:
        raise ValueError(f"{field} 조건이 너무 깁니다")
    if value and ("__" in value or not re.fullmatch(r"[가-힣A-Za-z0-9_./%<>=!&|()+*\-\s]+", value)):
        raise ValueError(f"{field} 조건에 허용되지 않은 문자가 있습니다")
    return value


def parse_nl(text, provider=None):
    sys = f"""당신은 한국 주식 자동매매 전략을 설계하는 퀀트 연구자입니다.
아래 규칙표의 '영문코드/입력 예시' 어휘만 사용해 JSON 하나만 출력하세요. 설명·마크다운 금지.

[규칙표]
{_logic_rules()}

[강제 규칙]
1. 부등호는 '>=' 대신 '>', '<=' 대신 '<'.
2. 재무 지표(PER,PBR,ROE,부채비율,매출액증가율 등)는 financial_logic 에만.
3. 기술 지표는 이동평균선을 '5일선','20일선','60일선'(공백 없이) 으로 쓰고, RSI/MACD/골든크로스/스토캐스틱은 STOCH_K,STOCH_D 로.
4. 수급은 '기관순매수','외국인순매수','개인순매수' (순매수 거래량, >0 이면 순매수).
5. financial_logic 기본값 "전체", sell_logic 기본값 "" (빈 값 = 안 팔고 보유).
6. portfolio_strategy 는 {sorted(_VALID_PORT)} 중 하나(기본 equal_weight).
7. 사용자의 문장을 두 단계로 축약하지 마세요. AND·OR·괄호로 여러 재무·기술·수급·캔들·공시 조건을 함께 쓸 수 있습니다.
8. name은 30자 이하, summary와 risk_notes는 각각 한 문장으로 씁니다.
9. use_tax_fee는 실제 비용을 반영하려면 true가 기본입니다.

[예시]
Q: PER 10 미만이고 ROE 15 넘는 종목을 5일선이 20일선 위일 때 사서 3년 보유
A: {{"name":"저평가 추세","summary":"가치와 추세를 함께 확인한다.","financial_logic":"PER < 10 AND ROE > 15","buy_logic":"5일선 > 20일선","sell_logic":"데드크로스 == TRUE","period_months":36,"portfolio_strategy":"equal_weight","use_tax_fee":true,"risk_notes":"과거 성과는 미래 수익을 보장하지 않는다."}}
Q: 부채비율 100 미만 가치주를 외국인이 순매수하고 RSI 30 밑일 때 매수, 수익 20%면 매도
A: {{"name":"수급 반전","summary":"재무 안정 종목의 수급과 과매도를 함께 본다.","financial_logic":"부채비율 < 100","buy_logic":"외국인순매수 > 0 AND RSI < 30","sell_logic":"RSI > 65 OR 데드크로스 == TRUE","period_months":24,"portfolio_strategy":"inverse_volatility","use_tax_fee":true,"risk_notes":"급락장에서는 과매도 신호가 오래 지속될 수 있다."}}

반드시 JSON 만:
{{"name":"...","summary":"...","financial_logic":"...","buy_logic":"...","sell_logic":"...","period_months":24,"portfolio_strategy":"equal_weight","use_tax_fee":true,"risk_notes":"..."}}
사용자 입력: {text}"""
    raw, meta = _llm(sys, provider=provider, json_mode=True)
    raw = raw.strip()
    try:
        p = json.loads(raw)
    except Exception:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if not m:
            raise ValueError(f"파싱 실패: {raw[:200]}")
        p = json.loads(m.group(0))
    # 정규화 + 검증
    fin = _safe_logic(p.get("financial_logic") or "전체", "재무") or "전체"
    buy = _safe_logic(p.get("buy_logic") or "", "매수")
    sell = _safe_logic(p.get("sell_logic") or "", "매도")
    if sell.lower() in ("buy&hold", "buy&amp;hold", "매수후보유", "보유"):
        sell = ""
    try:
        months = int(p.get("period_months") or 24)
    except Exception:
        months = 24
    months = max(3, min(months, 120))
    port = p.get("portfolio_strategy") or "equal_weight"
    if port not in _VALID_PORT:
        port = "equal_weight"
    if not buy:
        if fin in ("", "전체"):
            raise ValueError("전략을 이해하지 못했습니다. 재무·기술 조건을 넣어 다시 말씀해 주세요.")
        buy = "1 > 0"                                # 재무 필터 종목을 매수 후 보유(순수 펀더멘털)
    return {"name": str(p.get("name") or "AI 생성 전략")[:30],
            "summary": str(p.get("summary") or "")[:240],
            "financial_logic": fin, "buy_logic": buy, "sell_logic": sell,
            "period_months": months, "portfolio_strategy": port,
            "use_tax_fee": bool(p.get("use_tax_fee", True)),
            "risk_notes": str(p.get("risk_notes") or "")[:300],
            "provider": meta["provider"], "model": meta["model"]}


def generate(text, provider=None):
    if not (text or "").strip():
        raise ValueError("전략 아이디어를 입력하세요")
    return parse_nl(text.strip(), provider=provider)


# ── 엔진 싱글턴 (CSV 로드 1회) ──────────────────────────────────────────
_engine = None
_engine_lock = threading.Lock()
_QIS_DIR = qe.QIS_DATA_DIR


def _dart_key():
    """DART 공시 이벤트(유상증자·배당·분할 등) 조건용 키. 없으면 "" (이벤트 없는 전략은 무영향)."""
    try:
        from quant.refresh_financials import _dart_key as k
        return k()
    except Exception:
        return ""


def _get_engine():
    global _engine
    with _engine_lock:
        if _engine is None:
            _engine = qe.QuantLogic()
        return _engine


def _in_store(code):
    return os.path.exists(os.path.join(_QIS_DIR, f"daily_{str(code).zfill(6)}.csv"))


def _pick_universe(m, financial_logic):
    """재무 필터(또는 전체=시총상위) → 유니버스. 공유 저장소 보유 종목 우선, 상한 적용."""
    if financial_logic in ("", "전체", "all", "ALL"):
        comps = list(m.financial_data.get("D-1y_data", []))
    else:
        comps = m.run_financial_filter(financial_logic)
    comps = [c for c in comps if float(c.get("시가총액") or 0) > 0]
    comps.sort(key=lambda c: -float(c.get("시가총액") or 0))
    total = len(comps)
    in_store = [c for c in comps if _in_store(c["종목코드"])]
    picked = in_store[:MAX_UNIVERSE]
    used_naver = 0
    if len(picked) < MIN_UNIVERSE:                       # 저장소 커버가 적으면 네이버로 소량 보충
        rest = [c for c in comps if c not in in_store][:MIN_UNIVERSE - len(picked)]
        picked += rest
        used_naver = len(rest)
    return picked, {"filtered": total, "used": len(picked),
                    "store_covered": len(picked) - used_naver, "via_naver": used_naver}


def run_strategy(parsed, *, include_report=True, track_progress=True):
    """검증된 전략 JSON을 백테스트한다."""
    try:
        if track_progress:
            _set(12, f"재무 필터 적용: {parsed['financial_logic']}")
        m = _get_engine()
        universe, meta = _pick_universe(m, parsed["financial_logic"])
        if not universe:
            if track_progress:
                _set(0, "조건에 맞는 종목이 없습니다", error="유니버스 0종목")
            return {"error": "조건에 맞는 종목이 없습니다.", "parsed": parsed}
        if track_progress:
            _set(20, f"유니버스 {meta['used']}종목(총 {meta['filtered']}) 백테스트 시작...")

        def cb(pct, msg, err=None):                       # 엔진 진행률 20~90 구간에 매핑
            if not track_progress:
                return
            if err is not None:
                _set(pct or 0, msg, error=err)
            elif pct is not None:
                _set(20 + int(pct * 0.7), msg)

        res = m.run_backtest_logic(
            universe, parsed["buy_logic"], parsed["sell_logic"],
            parsed["period_months"], "안함",
            portfolio_strategy=parsed["portfolio_strategy"],
            use_tax_fee=bool(parsed.get("use_tax_fee", True)), dart_key=_dart_key(),
            progress_callback=cb if track_progress else None)
        if "error" in res:
            if track_progress:
                _set(0, res["error"], error=res["error"])
            return {"error": res["error"], "parsed": parsed}

        if include_report:
            if track_progress:
                _set(92, "AI 해설 생성 중...")
            res["ai_report"] = _ai_report(res, parsed)
        res["parsed"] = parsed
        res["universe_meta"] = meta
        if track_progress:
            _set(100, "완료")
            with _lock:
                _progress["status"] = "done"
        return res
    except Exception as e:
        if track_progress:
            _set(0, str(e), error=str(e))
        return {"error": str(e)}


def run(text=None, *, strategy=None, provider=None, profile_id=None):
    """자연어 또는 전략 JSON을 백테스트한다."""
    try:
        parsed = strategy
        if parsed is None:
            _set(3, "AI가 전략을 설계하는 중...")
            parsed = generate(text or "", provider=provider)
        res = run_strategy(parsed)
        if not res.get("error"):
            from web import experiment_store
            experiment_store.record_backtest(profile_id, parsed, res.get("detailed_stats") or {})
        return res
    except Exception as exc:
        _set(0, str(exc), error=str(exc))
        return {"error": str(exc)}


def _ai_report(res, parsed):
    s = res.get("detailed_stats", {})
    try:
        prompt = (f"전문 퀀트 분석가로서 아래 백테스트를 코스피·Buy&Hold와 비교해 3~4문장 한국어로 평가하세요. "
                  f"전략 수익률이 코스피보다 크게 낮으면 '무의미한 전략'임을 분명히 하세요. 마크다운 금지.\n"
                  f"전략: 재무[{parsed['financial_logic']}] 매수[{parsed['buy_logic']}] 매도[{parsed['sell_logic'] or 'Buy&Hold'}]\n"
                  f"전략수익률 {s.get('Total Return','-')} · B&H {s.get('BnH Return','-')} · KOSPI {s.get('KOSPI Return','-')} · "
                  f"CAGR {s.get('CAGR','-')} · MDD {s.get('MDD','-')} · 샤프 {s.get('Sharpe','-')} · 승률 {s.get('Win Rate','-')}")
        raw, _ = _llm(prompt, provider=parsed.get("provider"))
        return re.sub(r"\*\*", "", raw).strip()
    except Exception:
        return "AI 해설을 생성하지 못했습니다."


# ── 백그라운드 실행 (CF 524 회피: POST 는 바로 반환, 클라이언트가 진행률 폴링) ──
_result = {"data": None}


def start(text=None, *, strategy=None, provider=None, profile_id=None):
    """이미 실행 중이면 거부, 아니면 백그라운드 스레드로 run() 시작."""
    with _lock:
        if _progress.get("status") == "running":
            return False
        _progress.update(status="running", pct=0, msg="시작...", error=None)
    _result["data"] = None

    def _work():
        _result["data"] = run(text, strategy=strategy, provider=provider, profile_id=profile_id)

    threading.Thread(target=_work, daemon=True).start()
    return True


def get_result():
    return _result["data"]


if __name__ == "__main__":  # 셀프체크: 파싱 정규화가 evaluator 어휘를 내는지
    assert _normalize("5일 이동평균선 > 20일 이평선") == "5일선 > 20일선"
    assert _normalize("스토캐스틱 K > 스토캐스틱 D") == "STOCH_K > STOCH_D"
    print("nlbacktest self-check OK")
