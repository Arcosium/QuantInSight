"""
Arquant v1.0 — runtime strategy state.
A "strategy" = the set of tunable trading parameters (sizing, TP/SL, risk gates, news threshold...).
사장 지시 2026-06-09: 프리셋 폐지 → config.STRATEGY_DEFAULTS 단일 기본값 위에 사장이
대시보드 '전략' 탭에서 직접 편집한 값(custom)을 프로필별로 영속·라이브 반영한다.

  runtime.get("PER_ORDER_BUDGET_RATIO")   # override-or-config-default
  runtime.active()                         # {name, label('사용자 설정'), params, since}
  runtime.set_strategy(custom={...})       # apply ad-hoc params, persist, append to history
  runtime.history()
"""
import json, os
from datetime import datetime
from pathlib import Path
import config

# import 시점에 data/ 를 생성·시드하므로, 테스트 격리를 위해 env 오버라이드를 허용한다
# (QUANTINSIGHT_DATA_DIR 미설정이면 기본 <project>/data). 라이브는 미설정 → 기존과 동일.
_DIR = Path(os.environ["QUANTINSIGHT_DATA_DIR"]) if os.environ.get("QUANTINSIGHT_DATA_DIR") else (Path(__file__).parent / "data")
_DIR.mkdir(parents=True, exist_ok=True)
_STATE = _DIR / "strategy_state.json"
_HIST = _DIR / "strategy_history.json"
_OPS_FLAG = _DIR / "ops_feedback.json"      # 사장 피드백 2026-05-18: 운용지원실장 피드백 on/off 토글

# ── 전략 상태 — 프로필(계정)별 (Task 10: 멀티테넌트 마무리) ──────────────────────
# "전략" = 튜닝 트레이딩 파라미터 묶음(sizing·TP/SL·리스크게이트·뉴스임계...).
# 이전엔 _state 가 프로세스 전역 단일 dict 라 한 계정의 전략/파라미터 변경이 모든
# 계정에 적용됐다(멀티테넌트 격리 깨짐). 이제 _notif_settings/_cost_mode 와 동일한
# {"_default":..., "<uid>":...} 키드 패턴으로 uid 별로 분리한다.
# 저장 형식: {"_default": {name,params,since}, "<uid>": {name,params,since}}.
# 구버전 flat 포맷({name,params,since})은 로드 시 _default 로 자동 이관(하위호환).
def _default_state() -> dict:
    return {"name": "default",
            "params": dict(config.STRATEGY_DEFAULTS),
            "since": datetime.now().isoformat()}


_states: dict = {"_default": _default_state()}


def _strat_key(uid=None) -> str:
    return str(int(uid)) if uid is not None else "_default"


def _get_state(uid=None) -> dict:
    """그 uid 의 전략 상태. 미설정 시 _default 로 폴백(없으면 새 기본 생성)."""
    return _states.get(_strat_key(uid)) or _states.get("_default") or _default_state()


# 사장 피드백 2026-05-18: 프로필 한정 오버라이드 레이어 (최상위 우선순위).
# Task 10: 전역 인메모리 레이어를 폐지하고 get(key, uid) 가 디스크
# (infra.profile_overrides.load(uid) = data/profiles/<uid>/overrides.json)를 직접 참조한다.
# uid=None 이면 프로필 오버라이드 미적용(시스템 기본).


# ── 운용지원실장 피드백 on/off — 프로필(계정)별 (사장 지시 2026-05-20) ───────────
# 계정마다 독립 토글. 기본 ON(기존 동작 보존). 끄면 그 프로필이 활성일 때 워커를
# 띄우지 않는다(재시작 불필요). 저장 형식:
#   {"_default": {enabled,since,by}, "<uid>": {enabled,since,by}, ...}
# 구버전 flat 포맷({"enabled":...})은 로드 시 _default 로 자동 이관(하위호환).
_ops_feedback: dict = {"_default": {"enabled": True, "since": datetime.now().isoformat(), "by": "default"}}


def _ops_key(uid=None) -> str:
    return str(int(uid)) if uid is not None else "_default"


def _load_ops_feedback():
    global _ops_feedback
    try:
        if _OPS_FLAG.exists():
            d = json.loads(_OPS_FLAG.read_text(encoding="utf-8"))
            if isinstance(d, dict) and "enabled" in d:
                # 구버전 flat 포맷 → _default 로 이관
                _ops_feedback = {"_default": {"enabled": bool(d["enabled"]),
                                              "since": d.get("since", datetime.now().isoformat()),
                                              "by": d.get("by", "load")}}
            elif isinstance(d, dict):
                _ops_feedback = {k: v for k, v in d.items() if isinstance(v, dict) and "enabled" in v}
                _ops_feedback.setdefault("_default", {"enabled": True,
                                                      "since": datetime.now().isoformat(), "by": "default"})
    except Exception:
        pass


def ops_feedback_enabled(uid=None) -> bool:
    """프로필(uid)별 토글. 해당 프로필 설정 없으면 _default(기본 ON)로 폴백."""
    st = _ops_feedback.get(_ops_key(uid)) or _ops_feedback.get("_default") or {}
    return bool(st.get("enabled", True))


def ops_feedback_state(uid=None) -> dict:
    key = _ops_key(uid)
    st = _ops_feedback.get(key) or _ops_feedback.get("_default") or {"enabled": True}
    return {**dict(st), "uid": (None if key == "_default" else int(key))}


def set_ops_feedback(enabled: bool, uid=None, by: str = "dashboard") -> dict:
    """운용지원실장 피드백 토글 — 프로필(uid)별. uid=None 이면 기본값. 재시작 불필요."""
    global _ops_feedback
    key = _ops_key(uid)
    _ops_feedback[key] = {"enabled": bool(enabled),
                          "since": datetime.now().isoformat(), "by": by}
    try:
        _OPS_FLAG.write_text(json.dumps(_ops_feedback, ensure_ascii=False, indent=2),
                             encoding="utf-8")
    except Exception:
        pass
    _append_history({"ops_feedback": bool(enabled),
                     "uid": (None if key == "_default" else int(key)), "by": f"{by}:ops_feedback"})
    return ops_feedback_state(uid)


# (사장 지시 2026-07-21: API 비용 표시 모드 로직 제거 — 로컬 서버라 비용 표시 폐지.)


# ── 모바일 알림 설정 — 프로필(계정)별 (사장 지시 2026-05-21) ──────────────────────
# 모바일 앱(네이티브 WsManager)이 띄울 4종 푸시 알림을 프로필별로 on/off:
#   order_submitted(체결 신청) · trade(체결 완료) · cycle(사이클 완료) · market_close(장 마감)
# 저장 형식: {"_default": {order_submitted,trade,cycle,market_close}, "<uid>": {...}} (전부 기본 ON).
# 웹 대시보드 표시는 이 설정과 무관(항상 전체) — 이 토글은 '모바일 푸시'만 게이트한다.
_NOTIF_FILE = _DIR / "notif_settings.json"
_NOTIF_KEYS = ("order_submitted", "trade", "cycle", "market_close")
_NOTIF_DEFAULT = {k: True for k in _NOTIF_KEYS}
_notif_settings: dict = {"_default": dict(_NOTIF_DEFAULT)}


def _notif_key(uid=None) -> str:
    return str(int(uid)) if uid is not None else "_default"


def _clean_notif(d: dict) -> dict:
    return {k: bool((d or {}).get(k, True)) for k in _NOTIF_KEYS}


def _load_notif_settings():
    global _notif_settings
    try:
        if _NOTIF_FILE.exists():
            d = json.loads(_NOTIF_FILE.read_text(encoding="utf-8"))
            if isinstance(d, dict):
                _notif_settings = {k: _clean_notif(v) for k, v in d.items() if isinstance(v, dict)}
                _notif_settings.setdefault("_default", dict(_NOTIF_DEFAULT))
    except Exception:
        pass


def notif_settings(uid=None) -> dict:
    """프로필(uid)별 모바일 알림 설정. 미설정 시 _default(전부 ON)로 폴백."""
    st = _notif_settings.get(_notif_key(uid)) or _notif_settings.get("_default") or _NOTIF_DEFAULT
    return _clean_notif(st)


def set_notif_settings(settings: dict, uid=None) -> dict:
    """모바일 알림 설정 변경 — 프로필(uid)별. 부분 갱신 허용. 재시작 불필요."""
    global _notif_settings
    cur = notif_settings(uid)
    cur.update({k: bool(v) for k, v in (settings or {}).items() if k in _NOTIF_KEYS})
    _notif_settings[_notif_key(uid)] = cur
    try:
        _NOTIF_FILE.write_text(json.dumps(_notif_settings, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass
    return cur


def _persist():
    try:
        _STATE.write_text(json.dumps(_states, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass


def _append_history(entry: dict):
    try:
        h = []
        if _HIST.exists():
            try: h = json.loads(_HIST.read_text(encoding="utf-8"))
            except Exception: h = []
        if not isinstance(h, list): h = []
        h.append({"ts": datetime.now().isoformat(), **entry})
        _HIST.write_text(json.dumps(h[-200:], ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass


def _normalize_state(d: dict) -> dict:
    return {"name": d.get("name", "custom"), "params": dict(d.get("params") or {}),
            "since": d.get("since", datetime.now().isoformat())}


def _load():
    global _states
    try:
        if _STATE.exists():
            d = json.loads(_STATE.read_text(encoding="utf-8"))
            if not isinstance(d, dict):
                return
            if d.get("params"):
                # 구버전 flat 포맷({name,params,since}) → _default 로 이관(하위호환)
                _states = {"_default": _normalize_state(d)}
            else:
                # 키드 포맷({"_default":..., "<uid>":...})
                loaded = {k: _normalize_state(v) for k, v in d.items()
                          if isinstance(v, dict) and v.get("params")}
                if loaded:
                    _states = loaded
                _states.setdefault("_default", _default_state())
    except Exception:
        pass


_load()
_load_ops_feedback()
_load_notif_settings()
if not _HIST.exists():
    _dflt = _get_state(None)
    _append_history({"name": _dflt["name"], "params": _dflt["params"], "by": "init"})


def get(key, default=None, uid=None):
    """Live tunable value — 프로필(uid)별. 우선순위 (Task 10):
      1) 이 uid 의 프로필 오버라이드 (비관리자 운용지원실장 튜닝 — 디스크에서 직접 로드,
         data/profiles/<uid>/overrides.json). uid=None 이면 미적용.
      2) 이 uid 의 전략 프리셋/커스텀 params (대시보드 '전략' 탭). uid=None → _default.
      3) config 모듈 상수 → 호출자 default
    """
    if uid is not None:
        try:
            from infra import profile_overrides as _po
            ov = _po.load(uid)
            if key in ov:
                return ov[key]
        except Exception:
            pass
    p = _get_state(uid).get("params") or {}
    if key in p:
        return p[key]
    return getattr(config, key, default)


# 지표 신호명 → QIW_* 튜닝키. main_swarm 2곳(루브릭·점수산정) + backtest/quant_ic 이 같은 표를
# 각자 들고 있어 축을 추가할 때마다 갱신을 빠뜨렸다(2026-08-02 통합).
QIW_KEYS = (("rsi", "QIW_RSI"), ("macd", "QIW_MACD"), ("adx", "QIW_ADX"), ("vwap", "QIW_VWAP"),
            ("vol", "QIW_VOL"), ("mom", "QIW_MOM"), ("cmf", "QIW_CMF"), ("flow", "QIW_FLOW"),
            ("high52", "QIW_HIGH52"), ("leadlag", "QIW_LEADLAG"), ("vol_surge", "QIW_VOLUME_SURGE"))


def quant_weights(uid=None) -> dict:
    """지표 가중치 {신호명: QIW 값} — tools.quant_score.compute_indicator_score 입력."""
    return {sig: get(key, uid=uid) for sig, key in QIW_KEYS}


def dim_weights(uid=None) -> dict:
    """차원 가중치 {QUANT/NEWS/MACRO} — tools.quant_score.combine_dimensions 입력."""
    return {"QUANT": get("DW_QUANT", uid=uid), "NEWS": get("DW_NEWS", uid=uid),
            "MACRO": get("DW_MACRO", uid=uid)}


def active(uid=None) -> dict:
    """그 uid 의 활성 전략(라벨·since + 효과적 params). 프리셋 폐지 → 항상 '사용자 설정'."""
    st = _get_state(uid)
    keys = config.STRATEGY_TUNABLE_KEYS
    return {"name": st.get("name", "default"), "label": "사용자 설정",
            "since": st.get("since"), "params": {k: get(k, uid=uid) for k in keys}}


def history() -> list:
    try:
        if _HIST.exists():
            h = json.loads(_HIST.read_text(encoding="utf-8"))
            return h if isinstance(h, list) else []
    except Exception:
        pass
    return []


def set_strategy(custom: dict = None, by: str = "user", uid=None) -> dict:
    """현재 적용 전략 파라미터를 갱신 — 프로필(uid)별. uid=None → _default.
    프리셋 폐지(2026-06-09): STRATEGY_DEFAULTS 베이스 위에 custom(알려진 키만)만 얇는다.
    """
    keys = config.STRATEGY_TUNABLE_KEYS
    base = dict(config.STRATEGY_DEFAULTS)
    if custom:
        known = set(keys)
        for k, v in custom.items():
            if k in known:
                base[k] = v
    new_state = {"name": "custom", "params": {k: base.get(k) for k in keys},
                 "since": datetime.now().isoformat()}
    _states[_strat_key(uid)] = new_state
    _persist()
    _append_history({"name": "custom", "label": "사용자 설정",
                     "params": new_state["params"], "by": by,
                     "uid": (None if uid is None else int(uid))})
    return active(uid)
