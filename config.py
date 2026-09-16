"""
NPS Swarm v1.0 - Central Configuration
All environment variables, model mappings, and system constants.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).parent
# 공용 환경(KRX_ID/KRX_PW 등)을 먼저 읽고, 전용 비밀값은 저장소의 git-ignored .env로 덮어쓴다.
# 경로는 BASE_DIR 기준 — 구 Oracle 절대경로(/home/opc)는 심링크에 기대던 지뢰였다.
load_dotenv(BASE_DIR.parent / ".env")
load_dotenv(BASE_DIR / ".env", override=True)

# ─── API Keys ───────────────────────────────────────────────────────────────
# LLM은 로컬 OpenAI 호환 서버만 사용한다. API 키를 읽거나 저장하지 않는다.
KIS_APP_KEY = os.getenv("KIS_APP_KEY", "")
KIS_APP_SECRET = os.getenv("KIS_APP_SECRET", "")
KIS_BASE_URL = os.getenv("KIS_BASE_URL", "https://openapi.koreainvestment.com:9443")
KIS_ACCOUNT_NO = os.getenv("KIS_ACCOUNT_NO", "")
OPENDART_API_KEY = os.getenv("OPENDART_API_KEY", "")
# 사장 피드백 2026-05-15 (8차): Tavily 제거 — alibaba/tongyi-deepresearch가 검색 + 분류 + 리서치 통합 담당

# ─── Coresight RAG Paths ────────────────────────────────────────────────────
_ARCAIVE = BASE_DIR.parent / "ArcAI.ve"
CORESIGHT_PATH = os.getenv("CORESIGHT_PATH", str(_ARCAIVE / "coresight_logs"))

# ─── Local LLM model assignments (by agent role) ────────────────────────────
# URL은 설치 시에만 지정한다. 예: http://127.0.0.1:8080/v1
# llama.cpp 서버는 보통 /v1 을 노출하므로 기본값도 그 형식으로 둔다.
LOCAL_LLM_BASE_URL = os.getenv("LOCAL_LLM_BASE_URL", "http://127.0.0.1:8080/v1").rstrip("/")
LOCAL_ADAPTER_MODEL = "qwen3.6-35b-a3b-uncensored-genesis-lora:latest"
_LOCAL_BASE_MODEL_NAMES = {
    "Qwen3.6-35B-A3B-Uncensored-Claude-Genesis-Q8_0.gguf",
    "qwen3.6-35b-a3b-uncensored",
    "qwen3.6-35b-a3b-uncensored:latest",
}


def _local_llm_model():
    model = (os.getenv("LOCAL_LLM_MODEL") or LOCAL_ADAPTER_MODEL).strip()
    return LOCAL_ADAPTER_MODEL if model in _LOCAL_BASE_MODEL_NAMES else model


LOCAL_LLM_MODEL = _local_llm_model()

# '+thinking'은 내부 가상 접미사다. 전송 직전에 제거하고 OpenAI 호환 요청의
# chat_template_kwargs.enable_thinking / reasoning.enabled 로 변환한다.
LOCAL_LLM_MODEL_THINKING = LOCAL_LLM_MODEL + "+thinking"

_PRO = LOCAL_LLM_MODEL_THINKING  # 결정 에이전트 — thinking ON
_FLASH = LOCAL_LLM_MODEL          # 분석/리서치/도구 — thinking OFF

MODEL_ASSIGNMENTS = {
    # ── 결정(pro 티어, reasoning ON) ──
    # 사장 원칙 2026-06-10: '주관이 들어가 결정을 내리는' 에이전트는 추론 켬.
    "chief_orchestrator": _PRO,   # 총괄·종목선정(2패스 결정)
    "post_manager": _PRO,         # 사후관리실장 매도 결정
    "ops_support": _PRO,          # 운용지원실장 파라미터 튜닝 결정
    "bond_manager": _PRO,         # 채권운용실장 슬리브 ETF 매매 결정
    "commodity_manager": _PRO,    # 원자재운용실장 슬리브 ETF 매매 결정
    # ── 분석/리서치/도구(flash 티어, reasoning OFF) ──
    # 사장 지시 2026-07-22: macro_analyst 를 _PRO → _FLASH 로 내린다.
    # 2026-06-10 원칙("결정 내리는 에이전트는 추론 켬")에 따라 '자산배분 권고=결정'으로 분류했으나,
    # 실제 프롬프트를 보면 리서치는 tools.global_search.deep_research 가 **미리** 수행하고,
    # 이 에이전트는 검증된 지수표·리서치 결과를 받아 고정 3블록(요약·비중·리스크)으로 옮겨 적는
    # **작문** 작업이다. 결정 성분은 합 100%인 네 숫자뿐. 2026-07-22 관측: 500토큰짜리 보고서를
    # 쓰겠다고 CoT 를 34,671토큰 태우고 타임아웃에 잘렸다(사이클 30분 정체).
    "macro_analyst": _FLASH,
    "quant_analyst": _FLASH,
    "news_analyst": _FLASH,
    "insight_analyst": _FLASH,    # 기업리서치팀장 — 기업 분석자료 리서치·요약(사장 지시 2026-07-21)
    "news_curator": _FLASH,
    # macro_researcher 는 Hermes web_search/web_extract(tool-calling)를 써야 한다 — reasoning
    # OFF 평문 슬러그 유지(tool-calling 안정성). (Qwen3.6-A3B 는 로컬 LLM 에서 도구호출 지원.)
    "macro_researcher": _FLASH,
    "trader": _FLASH,
    "risk_guard": _FLASH,
    "fund_planner": _FLASH,       # thesis 자문(veto 폐지) — 자문성이라 flash
    "compliance": _FLASH,         # 컴플라이언스실장 — 반려는 결정론 코드, LLM은 @멘션 설명만(사장 지시 2026-07-22)
}

# ─── Risk Management Constants ──────────────────────────────────────────────
# ⚠️ 2026-08-02 삭제: MAX_SINGLE_STOCK_RATIO(단일종목 20%)·MAX_DRAWDOWN_LIMIT(손실 10%)·
# MAX_ORDER_RETRY·MAX_VALIDATION_LOOP 는 **어디서도 읽지 않는 장식**이었다(선언만 존재).
# 실제로 동작하는 한도는 STRATEGY_DEFAULTS 의 튜닝 파라미터와 아래 CONSERVATIVE_MDD 다.
# 회로차단기(사장 지시 2026-06-17): 연속 US 매수실패(주문가능금액 초과)가 이 횟수 이상이면
# 해당 세션 US 매수를 보류한다(매도로 USD 확보·세션 전환 시 자동 해제). 안전망이라 ops 튜닝 비대상.
US_BUY_FAIL_STREAK_LIMIT = 2

# ─── LLM Cost-Reduction Knobs ───────────────────────────────────────────────
# Per-agent output token caps (replaces uniform 4096). Risk/Policy emit short
# structured JSON; analysts need more room.
# 2026-07-21 상향(사장 지시): thinking ON 에이전트(_PRO 6종)만 64000. 로컬 모델의 추론
# 길이는 1.1k~10.7k 로 분포가 넓어 예산이 작으면 추론이 예산을 다 쓰고 content 가 빈 채
# 끝난다(실측: 같은 프롬프트가 64000 이면 6/6 정상, 12000 은 2/6, 8000 은 0/3). llama-server
# 는 -c 524288/-np 4 라 슬롯당 131072 이고 KV 는 선할당이라 상한을 올려도 추가 메모리는 없다.
# _FLASH(추론 OFF) 에이전트는 그대로 둔다 — 추론을 안 하므로 큰 예산이 무의미하다.
#
# 2026-07-22 하향(사장 지시) 64000 → 40000: **예산이 타임아웃보다 길어 완주 불가**였다.
#   llama-server 실측 디코드 53 t/s → 64000토큰 = 1,208초인데 LOCAL_LLM_TIMEOUT_SEC 은 900초.
#   예산을 다 쓰는 생성은 15분에 잘리고 재시도는 토큰 0부터 다시 시작 → GRACE 소진 후 최종 실패.
#   실제로 24시간 동안 타임아웃 재시도 13회·최종 실패 4회(원자재운용실장·주식운용실장 등 전부 pro).
#   40000 = 관측 추론 최대(10.7k)의 3.7배 여유이면서 53 t/s 로 755초 → 새 타임아웃(1800초) 안에
#   2.4배 여유로 완주한다. 12000(2/6 실패)·8000(0/3) 과는 자릿수가 다르므로 빈 content 회귀 위험 없음.
#   타임아웃/그레이스는 infra/local_llm_client.py 에서 함께 상향했다 — 둘은 같이 움직여야 한다.
AGENT_MAX_TOKENS = {
    "chief_orchestrator": 40000,  # Qwen3.6-35B-A3B-Uncensored-Claude-Genesis-Q8_0.gguf+thinking로 교체(2026-05-19) — 비-reasoning이라 자연 완료 시 즉시 반환,
                                  # 12000은 안전한 상한(2패스 결정 출력 충분히 수용, 잠식 없음).
    "macro_analyst":       8000,   # thinking OFF 전환(2026-07-22)으로 CoT 소모가 없어졌다 — 순수 출력만.
                                  # 계량분석팀장(8000)과 동일. 4000은 리포트가 중간에 끊겼던 이력(2026-05-19).
    "quant_analyst":      8000,   # 5500으로도 5개 섹션 + 점수/진입가 줄이 잘리는 사례 발생 → 8000으로 재상향
    "news_analyst":       5000,   # 2026-06-11 2600→5000: 뉴스 다발 사이클(40건 등)에서 ②시장분위기·③매크로 시사점 섹션이 중간에 잘림(cycle294=1750자 컷 관측). flash라 max_tokens=출력 전량.
    "insight_analyst":    5000,   # 기업리서치팀장 — 기업 분석자료(펀더멘털·공시·리서치) 종목별 요약
    "news_curator":        600,   # 큐레이터는 번호 목록만 → 작게
    "macro_researcher":   8000,   # 매크로 리서치 — 시황·심리·정책 합성 응답
    "trader":             1500,   # 사장 피드백 (3차) — 자연어 보고용. 체결 결과 요약 + 매매 이유 정리
    "risk_guard":         2200,   # DART 공시 읽고 종목별 재심 + 사유
    # policy_filter 폐지(2026-05-18) → 반려 기능은 2026-07-22 컴플라이언스실장으로 부활(결정론)
    "compliance":         1200,   # @멘션 정책 설명용 — 사이클 판정은 LLM 미사용
    "post_manager":       40000,   # Qwen3.6-35B-A3B-Uncensored-Claude-Genesis-Q8_0.gguf+thinking(2026-05-24 교체) — chief_orchestrator와 동일 모델·동일 토큰 한도
    "ops_support":        40000,   # 코드 변경 JSON + 근거 설명 (사장 지시 2026-05-14 — 토큰 한도 상향)
    "fund_planner":       1200,   # 사장 지시 2026-05-28 — 4줄(목표가/손절가/계획 보유/사유) 정형 출력 + 한 단락 보강
    "bond_manager":       40000,   # 채권운용실장. 2026-06-10 pro 전환 후 2000→5000: pro=reasoning은 max_tokens에 CoT 포함이라 2000이면 본문이 잘림(391자 관측). CoT+결정문 여유 확보.
    "commodity_manager":  40000,   # 원자재운용실장. 동일 사유(pro reasoning CoT 토큰 소모) → 5000.
}
AGENT_HISTORY_TURNS = 3               # trailing history messages to resend (was 10)

# Macro/DART recompute throttling (slow-moving inputs — don't regenerate every cycle)
MACRO_CACHE_TTL_SEC = 30 * 60         # 30 min
DART_CACHE_TTL_SEC  = 24 * 60 * 60    # 1 day
# 뉴스 감성 리포트도 매크로와 같은 30분 공유 캐시(사장 지시 2026-07-21). 종전엔 캐시가 없어
# 과거 연속 사이클에서 마켓센티먼트팀장 LLM 을 새로 호출했다(모의 프로필 하룻밤 575회). 매크로와
# 동일 정책 — 개장 사이클(market_open)에선 캐시를 무시하고 반드시 새로 분석한다.
NEWS_CACHE_TTL_SEC  = 30 * 60         # 30 min

# 전체 리서치·선정·재량 매매 판단은 모든 프로필이 벽시계 정각에 시간당 1회만 수행한다.
# 그 사이에는 main_swarm 의 고정 60초 안전감시가 손절가·하드 손절만 확인한다.
PERIODIC_CYCLE_SEC      = 1 * 60 * 60
HEADLINE_DEDUP_RATIO    = 0.85        # difflib ratio above which two headlines are "the same"
NEWS_PREFILTER_TRIGGER  = 40          # 누적 헤드라인이 이 수를 넘으면 큐레이터로 사전 선별
NEWS_PREFILTER_LIMIT    = 40          # 사전 선별 후 마켓센티먼트팀장에게 넘길 최대 헤드라인 수
MATERIAL_NEWS_KEYWORDS = [
    "실적", "어닝", "영업이익", "순이익", "적자", "흑자전환", "가이던스",
    "M&A", "인수", "합병", "분할", "지분", "최대주주", "경영권",
    "유상증자", "무상증자", "감자", "배당", "자사주", "자기주식",
    "상장", "상장폐지", "관리종목", "거래정지", "공개매수",
    "규제", "제재", "과징금", "리콜", "소송", "특허", "파산", "회생",
    "금리", "인하", "인상", "FOMC", "환율", "관세",
    "수주", "계약", "공급계약", "임상", "승인", "허가",
    # 완화된 기준: 시장 동향, 업종, 산업 키워드 추가
    "급등", "급락", "상한가", "하한가", "신고가", "신저가",
    "반도체", "AI", "2차전지", "바이오", "로봇", "자율주행",
    "매출", "성장", "하락", "상승", "전망", "분석",
    "투자", "펀드", "채권", "ETF", "공모", "사모",
    "수출", "수입", "무역", "물가", "고용", "GDP",
    "트럼프", "중국", "미국", "일본", "유럽",
]

# ─── Live Trading (KIS) ─────────────────────────────────────────────────────
LIVE_TRADING         = True           # submit orders through the selected broker; PAPER_ONLY blocks live execution
PAPER_ONLY           = os.getenv("QIS_PAPER_ONLY", "1") != "0"  # current default; live support retained
SYSTEMATIC_POLICY_ENABLED = os.getenv("QIS_SYSTEMATIC_POLICY_ENABLED", "1") != "0"
MAX_TRADES_PER_CYCLE = 2              # at most this many real orders executed per analysis cycle (sells prioritized)

# ─── 유휴 USD → KRW 자동 역환전 (사장 보고 2026-06-26) ────────────────────────────
# 배경(KR/US 비대칭): US 매수의 KRW→USD 환전은 KIS '통합증거금'이 결제 시 자동 처리한다(명시 환전
# API 호출이 없다). 그러나 역방향(USD→KRW)은 KIS OpenAPI 에 공개 '환전' TR 이 존재하지 않아
# 자동 경로가 애초에 배선된 적이 없다 — 그래서 US 매도 후 남는 USD 예수금이 놀고, 사장이 매번
# MTS 에서 수동 환전해 왔다(2026-06-26 라이브 로그: 유휴 USD ≈₩1,424,204).
# 본 기능은 유휴 USD 를 주기적으로 감지해 (1) 자동 실행이 켜져 있고 실환전 경로가 있으면 실행,
# (2) 아니면(기본) 운영자에게 '환전 필요' 알림(중복억제)을 띄워 수동 환전을 자동 환기한다.
# KIS 가 환전 TR 을 공개/계약 제공하면 KIS_FX_EXCHANGE_TR 에 TR 을 넣고 broker.us_to_krw_exchange
# 한 곳만 실행 경로로 배선하면 된다(엔드포인트 발명 금지).
AUTO_USD_TO_KRW_RECONVERT = False     # True 라도 KIS_FX_EXCHANGE_TR 미설정이면 실주문 대신 알림만(안전 기본)
USD_RECONVERT_MIN_USD     = 100.0     # 이 미만(잔돈)은 무시 — 환전 알림/시도 안 함
KIS_FX_EXCHANGE_TR        = ""         # KIS 공개 환전 TR 없음 → 빈 값 = 수동 환전 알림 모드(미검증 엔드포인트 호출 금지)

# Sell / rebalance rules (BUY-only was too restrictive — now we also take profit / cut loss / trim).
ENABLE_SELL_REBALANCE = True
TAKE_PROFIT_PCT       = 12.0          # holding P&L ≥ +12%  → sell the position
STOP_LOSS_PCT         = 7.0           # holding P&L ≤ -7%   → sell the position
# 트레일링 익절(2026-06-18 고회전 수익화·비대칭 청산) — 보유 고점 대비 이 %만큼 되밀리면 매도해
# 승자를 길게 가져간다(고정 익절보다 추세 끝까지). 0=off(기본, 라이브 안전). 운용지원실장 튜닝.
TRAILING_TAKE_PROFIT_PCT = 0.0        # 0=off. >0 면 (고점 −현재)/고점 ≥ 이 % 일 때 매도
TRIM_OVER_RATIO       = True          # if a holding's notional exceeds CONSERVATIVE_STOCK_RATIO of total → trim down
# 사장 피드백 2026-05-15(#24): 데이트레이딩 0.5일 미만 회피 원칙 폐기. 기본 True(허용)로 변경.
# False로 두면 사후관리실장에게 "0.5일 미만 보유 종목은 데이트레이딩 회피" 가이드가 다시 들어간다.
ALLOW_DAY_TRADING     = True
# 0.5일 미만 회피가 켜졌을 때(ALLOW_DAY_TRADING=False) 적용되는 최소 보유일.
MIN_HOLDING_DAYS_FOR_SELL = 0.5
# 포트폴리오기획팀장은 계획기간 안의 첫 재량 매도에 1회 보류권을 행사한다.
# 손절·목표 도달·thesis invalidator는 즉시 통과하고, 다음 시간 정기 사이클에서도 매도 판단이
# 반복되면 보류를 해제한다. 설정 토글 없이 전 프로필 공통 정책으로 적용한다.

# ─── 넥스트레이드(NXT) 시간외 매매 (사장 지시 2026-06-08) ─────────────────────
# 프리마켓(08:00–08:50)·애프터마켓(15:50–20:00)을 NXT 거래소 경유로 매매. 정규장은 KRX 유지.
ENABLE_NXT_EXTENDED_HOURS    = True   # 마스터 스위치 (끄면 시간외 세션을 OFF_HOURS처럼 취급)
ENABLE_NXT_PRE_MARKET        = True   # 프리마켓 08:00–08:50 on/off
ENABLE_NXT_AFTER_MARKET      = True   # 애프터마켓 15:50–20:00 on/off
EXT_HOURS_LIMIT_SLIPPAGE_PCT = 0.5   # 시간외 지정가 밴드(%) — 매수=현재가×(1+x%), 매도=×(1−x%)
EXT_HOURS_MAX_PREMIUM_PCT    = 1.5   # 시간외 지정가 프리미엄 캡(%) — 정규 종가 대비 매수 상한·매도 하한(얇은 NXT 추종 방지)

# ─── 자산슬리브: 채권 ETF + 원자재 ETF 자동매매 ───────────────────────────────
# 매크로 자산배분 권고('채권 X%' / '원자재 W%')를 채권운용실장·원자재운용실장이 ETF 매수/
# 매도로 실현(주식 퀀트 파이프라인 우회 독립 트랙). 엔진은 infra/asset_sleeves.py 가 SleeveSpec
# 하나로 일반화한다. 풀 태그 = (code, name, duration, kind, fx):
#   duration ∈ {short, mid, long, na}  | kind ∈ {govt, rate, credit, tips, gold, oil, agri, broad}
#   fx ∈ {krw(원화자산), hedged(환헤지 (H)), exposed(환노출 USD)}
# 풀은 화이트리스트(LLM 티커 환각 방지) — 코드 오류=실주문 실패이므로 검증된 코드만 등재
# (KR 코드는 2026-06-09 웹 검증: 한국거래소 상장 ETF).
#
# ── 채권 슬리브 (사장 지시 2026-06-08, 2026-06-09 자산군 확장) ──
ENABLE_BOND_ETF         = True    # 마스터 스위치 (사장 지시 2026-06-09: 기본 ON)
BOND_TARGET_MAX_PCT     = 0.40    # 채권 비중 절대 상한(매크로 권고가 넘어도 이 값으로 클램프)
BOND_REBALANCE_BAND_PCT = 0.03    # 목표 대비 ±이 폭 이내면 매매 안 함(채권 churn 방지 데드존)
BOND_PER_CYCLE_RATIO    = 0.15    # 채권 전용 사이클 매수 예산비율(총평가 대비). 주식 MAX_CYCLE_BUDGET_RATIO 와 분리.
# 허용 채권 ETF 풀 — 국고채(govt)·CD금리(rate)·종합채권/회사채(credit)·환헤지 미국채(hedged).
BOND_ETF_POOL_KR = [
    ("153130", "KODEX 단기채권",                  "short", "govt",   "krw"),
    ("114260", "KODEX 국고채3년",                 "mid",   "govt",   "krw"),
    ("148070", "KOSEF 국고채10년",                "long",  "govt",   "krw"),
    ("357870", "TIGER CD금리투자KIS(합성)",       "short", "rate",   "krw"),  # CD 91일 금리(파킹)
    ("459580", "KODEX CD금리액티브(합성)",        "short", "rate",   "krw"),  # 보수 0.02%
    ("273130", "KODEX 종합채권(AA-이상)액티브",   "mid",   "credit", "krw"),  # 국채+회사채 4천종
    ("451540", "TIGER 종합채권(AA-이상)액티브",   "mid",   "credit", "krw"),
    ("458250", "TIGER 미국채30년스트립액티브(합성H)", "long", "govt", "hedged"),  # 환헤지 장기 미국채
]
BOND_ETF_POOL_US = [
    ("SHY", "iShares 1-3Y Treasury",  "short", "govt",   "exposed"),
    ("IEF", "iShares 7-10Y Treasury", "mid",   "govt",   "exposed"),
    ("TLT", "iShares 20+Y Treasury",  "long",  "govt",   "exposed"),
    ("LQD", "iShares IG Corp Bond",   "mid",   "credit", "exposed"),  # 투자등급 회사채
    ("HYG", "iShares High Yield Corp","mid",   "credit", "exposed"),  # 하이일드 회사채
    ("TIP", "iShares TIPS",           "mid",   "tips",   "exposed"),  # 물가연동
]

# ── 원자재 슬리브 (사장 지시 2026-06-09: 신설, 기본 ON) ──
# 실물자산(원유·금·농산물) ETF. 원자재는 듀레이션 개념 없음→na, kind 로 종류 구분.
ENABLE_COMMODITY_ETF         = True   # 마스터 스위치 (사장 지시 2026-06-09: 기본 ON)
COMMODITY_TARGET_MAX_PCT     = 0.20   # 원자재 비중 절대 상한
COMMODITY_REBALANCE_BAND_PCT = 0.03   # 목표 대비 데드존
COMMODITY_PER_CYCLE_RATIO    = 0.10   # 원자재 전용 사이클 매수 예산비율(총평가 대비)
COMMODITY_ETF_POOL_KR = [
    ("132030", "KODEX 골드선물(H)",          "na", "gold", "hedged"),
    ("261220", "KODEX WTI원유선물(H)",       "na", "oil",  "hedged"),
    ("137610", "TIGER 농산물선물Enhanced(H)","na", "agri", "hedged"),
]
COMMODITY_ETF_POOL_US = [
    ("GLD", "SPDR Gold Shares",          "na", "gold",  "exposed"),
    ("USO", "US Oil Fund",               "na", "oil",   "exposed"),
    ("DBA", "Invesco Agriculture Fund",  "na", "agri",  "exposed"),
    ("DBC", "Invesco Commodity Index",   "na", "broad", "exposed"),
]

# ─── 실행 프로필 간 공통 분석 공유 ──────────────────────────────────────────
# 실행 중인 프로필 하나가 뉴스·매크로 분석을 게시하고 다른 실행 프로필이 공유한다.
# 실전·모의 여부와 관리자 권한은 분석 역할에 영향을 주지 않는다.
SHARE_MARKET_INTELLIGENCE = True   # 마스터 토글. False면 전 계정이 현행대로 각자 계산
SHARE_PRODUCER_WAIT_SEC   = 420    # 소비자가 실행 중인 대표 프로필의 게시를 기다리는 단계별 최대 초(초과 시 자체계산 폴백)
                                   # 2026-07-22: 120초는 생산자 매크로 게시보다 짧아 매번 타임아웃 → 전 단계 자체계산(사이클 21분)
SHARE_STALE_OK_SEC        = 30*60  # 직전 게시분을 이 시간 안이면 재사용(매크로 캐시 TTL과 동일)
QUANT_CONCURRENCY         = 2      # 종목별 계량분석 동시 실행 수(사장 지시 2026-07-22: 사이클 최대 병목)
                                   # 태스크마다 전용 에이전트를 쓰므로 히스토리 누수 없음. 1이면 종전 순차.

# When no target is affordable with available cash, look for a cheaper liquid name in whatever
# market is tradeable right now (KR session → KRX volume rank; US session → the shortlist below).
ENABLE_CHEAP_FALLBACK = False        # 대체 후보(엉뚱한 저가 종목 매수) 비활성 — 사장 지시(2026-05). 살 게 없으면 그냥 매수 생략.
# 1주 매수 예산 허용 초과율 — 1주 가격이 1종목 예산을 이 배율 이내로만 넘으면 1주 매수 허용 (사장 지시: +20%)
PER_ORDER_BUDGET_OVERSHOOT = 1.20
# (대체 후보가 켜졌을 때만 사용) 거래량 상위에서 제외할 위험·부적격 키워드 + 최소 가격
CHEAP_FALLBACK_EXCLUDE_KEYWORDS = ["레버리지", "인버스", "곱버스", "2X", "3X", "선물", "ETN", "TR", "커버드콜"]
CHEAP_FALLBACK_MIN_PRICE = 2000      # 이 가격(원) 미만 종목은 대체 후보에서 제외 (잡주/저가 ETN 회피)
# Liquid, generally low-priced US names used as the US-session cheap fallback universe (대체 후보).
CHEAP_FALLBACK_US_TICKERS = ["F", "BAC", "T", "PFE", "KO", "CSCO", "INTC", "SOFI", "NU", "WBD", "SIRI", "KVUE", "VALE"]

# Which instruments the autonomous engine may BUY. KR equities & ETFs are 6-digit codes (same path).
# US needs USD cash; derivatives (선물/옵션) need margin/contract handling — off by default (high risk).
# 사장 지시 2026-06-08: 미국장 기능은 기본 비활성 — 대시보드'전략'탭에서 체크할 때만 활성(체크=per-uid override).
ALLOW_US_STOCKS       = False
ALLOW_DERIVATIVES     = False         # KR/해외 선물·옵션 자동매매 (마진·만기·증거금 처리 필요 — 기본 비활성)
# Position sizing — orders are sized to the *actual* account, not a fixed share count.
MAX_ORDER_QTY        = 0              # tunable share-count ceiling per order; 0 = fall back to HARD_MAX_ORDER_QTY
# 사장 지시 2026-06-04: 저가주에 예산이 몰리면 budget/단가 = 큰 수량이 산출돼 1회 한도에 걸려
# 매수가 '전량 반려'되던 버그(052900 10,227주) 방지용 절대 상한. 초과분은 '반려'가 아니라 이 값으로
# clamp 해서 주문이 살아 나가게 한다(매수 한정 — 매도는 위험회피라 미적용). MAX_ORDER_QTY(>0) 우선.
HARD_MAX_ORDER_QTY   = 1000           # absolute per-order share ceiling (safety backstop; buys clamped, not rejected)
PER_ORDER_BUDGET_RATIO = 0.10         # one order may spend at most this fraction of available cash (예수금)
MAX_CYCLE_BUDGET_RATIO = 0.25         # all orders in one cycle may spend at most this fraction of cash
MIN_CASH_BUFFER       = 1.10          # require cash ≥ notional × this (slippage/fee headroom) — else reject
# 매크로 목표 향한 예산 플로어 (사장 지시 2026-06-15): 매크로 주식 목표 > 현재 주식비중(여력)일 때
# ops 예산 컷이 배치를 과도하게 묶지 않도록 per-order·per-cycle 비율에 최소 플로어를 적용.
MACRO_DEPLOY_FLOOR_ENABLED   = True
PER_ORDER_BUDGET_FLOOR_RATIO = 0.10   # 여력 있을 때 1주문 예산 비율 하한
MAX_CYCLE_BUDGET_FLOOR_RATIO = 0.30   # 여력 있을 때 사이클 예산 비율 하한
# 2026-06-15 ROI 기능 토글 — 섀도우 우선(기본 OFF), 사장이 섀도우 로그 검토 후 점등.
ENABLE_DILUTION_GATE = False          # 매수 전 DART 희석(CB/유증) 게이트(#4) — ON이면 high 심각도 매수 보류
ENABLE_IC_SIZING     = False          # 스코어카드 IC 확신도를 실제 사이징에 반영(#2) — OFF면 섀도우 로그만
# Conservative risk gates (리스크관리실장 — tighter than the legacy 20%/-10% limits)
CONSERVATIVE_MDD       = 0.05         # block ALL new buys if account evaluation P&L ≤ -5%
CONSERVATIVE_STOCK_RATIO = 0.15       # a single position's notional may not exceed 15% of total eval (was 20%)

# ─── 전략 파라미터 확장 (사장 지시 2026-06-04) ───────────────────────────────
# 운용지원실장이 '급락장 대비'·'추세추종'·'역추세'·'모멘텀' 등 퀀트 전략 지시를 파라미터 조정만으로
# AI 매매에 반영할 수 있도록 추가한 '살아있는 노브'들. 전부 실제 매매 경로에 배선된다.
# spec: docs/superpowers/specs/2026-06-04-strategy-param-expansion-design.md
# (A) 종목 필터 — 매수 자격
MIN_QUANT_SCORE        = 6            # 결정론 게이트: 퀀트점수 < 이 값인 최종 매수대상은 제거(0~10)
# 완성 일봉의 비중복 5거래일 순수익을 검증한다. 변동성은 기대수익이 아니다.
ENABLE_COST_EDGE_GATE  = False        # retired; experimental edge is shadow-only
MIN_NET_EDGE_PCT       = 0.8          # conservative historical net estimate (%)
MAX_BUY_VOLATILITY_PCT = 0.0          # 프롬프트: 연환산 변동성(%)이 이 값 초과면 매수부적합 (0=off)
RSI_OVERBOUGHT_SKIP    = 0            # 프롬프트: RSI 이 값 초과(과매수)면 신규매수 회피 (0=off)
MIN_ADX_FOR_BUY        = 0            # 프롬프트: ADX 이 값 미만(추세약)이면 매수부적합 (0=off, 추세추종용)
REQUIRE_FOREIGN_NET_BUY = False       # 프롬프트: 외국인 순매수(+) 종목만 매수 적격
MAX_PRICE_EXTENSION_PCT = 0.0         # 프롬프트: VWAP/이평 대비 이격(%)이 이 값 초과면 추격매수 회피 (0=off)
# (B) 퀀트 채점 가중치(QW_*) — 2026-06-04 결정론 점수 엔진 도입으로 **완전 폐기**. 더 세밀한 QIW_*(지표별)로 대체.
# (C) 레짐 대응
MACRO_STOCK_GATE_ENABLED = True       # 결정론: 매크로 권고 주식비중 ≤ 현재 주식비중이면 신규 매수평가 스킵

# 운용지원실장 사이클 자동튜닝 최소 간격(초) — 사장 지시 2026-06-05: 매 사이클 spawn(낭비) → 시간당 1회.
OPS_THROTTLE_SEC = 3600
# anti-oscillation 윈도우(초) — 사장 지시 2026-06-18(버그 B): 운용지원실장이 같은 키를 이 시간 내
# 반대 방향으로 되감으면(예: 예산비율 0.3→1.0→0.3) 진동으로 보고 보류. 목표지향 같은방향 조정은 허용.
OPS_OSCILLATION_WINDOW_SEC = 7200

# 원장 허수(KIS<원장) 자동 정정 — 사장 지시 2026-06-17: 047810 2주가 6일간 ledger_eval 을
# 31만원 부풀려 리시드 시 가짜 -31만원 곡선단차를 만든 재발 방지. KIS 가 권위적으로 원장보다
# 적게 보유한 KR 포지션이 '연속 N회'(30분 간격 대조 → 약 1.5h) 확인되면 KIS 기준 하향 정정.
# 잔고 글리치(결제 과도기 일시 0)는 1~2틱이라 이 임계로 방어된다. KR 전용·하향 전용.
LEDGER_PHANTOM_PRUNE_CONFIRMATIONS = 3
# 원장 누락(KIS>원장) 자동 채택(adopt) 확인 임계 — 사장 지시 2026-06-19(defense-in-depth):
# prune_phantoms(하향)의 대칭. 매도 이중계상 등으로 원장이 KIS 아래로 떨어져 고착(161890 'KIS 65
# vs 원장 0')되면, KIS>원장 괴리가 '연속 N회' 확인될 때 KIS 기준 상향 채택 → 원장 qty 가 어떤
# 원인의 괴리든 KIS 로 자동 수렴. KR 전용·상향 전용·연속확인(글리치 방어). 주문으로 설명되는
# 부분체결 갭은 repair 가 1~2 사이클 내 먼저 해소하므로 채택은 '지속 갭'만 잡는다.
LEDGER_ORPHAN_ADOPT_CONFIRMATIONS = 3
# 원장 누락매수 자동보정(repair) 확인 임계 — 사장 지시 2026-06-18: KIS 잔고 글리치-高 읽기
# (일시적으로 보유가 부풀려 읽힘)를 원장에 그대로 baked 하던 버그(160980: 글리치 255 → 84주
# 주입 후 다음 사이클 KIS 171과 괴리) 방지. KIS>원장 괴리가 '연속 N회' 확인돼야 상향 보정한다.
LEDGER_REPAIR_CONFIRMATIONS = 2
# 매도 잠김(매도가능 0·펜딩없음) 에스컬레이션 임계 — 사장 지시 2026-06-18(버그 C): 손절/익절이
# 결제/제도 잠금으로 N사이클 연속 집행 불가면 강제 시장가 재청산 시도 + 경고로 표면화(무한 보류 차단).
LOCKED_SELL_ESCALATE_AFTER = 3

# ─── 결정론 점수 엔진 (사장 지시 2026-06-04: LLM 일관성 문제 → 점수는 무조건 파이썬) ───────
# spec: docs/superpowers/specs/2026-06-04-deterministic-score-engine-design.md
# 퀀트 점수 = 지표별 부호 가중치(QIW_*)로 결정론 산정. 뉴스·매크로와 함께 차원 가중치(DW_*)로 합성.
# 전부 signed(음수 허용) — 음수면 그 축을 반대로(예: QIW_VWAP 음수 = 과이격 추격 허용).
# 퀀트 지표 가중치 (신호명: rsi/macd/adx/vwap/vol/mom/cmf/flow/high52)
QIW_RSI   = 5     # 과매도→+ 과매수→− (평균회귀)
QIW_MACD  = 10    # 추세전환 모멘텀
QIW_ADX   = 8     # 추세 강도·방향
QIW_VWAP  = 8     # 과이격(추격) 페널티
QIW_VOL   = 8     # 고변동 페널티
QIW_MOM   = 12    # 1M·3M 모멘텀
QIW_CMF   = 8     # 매집/분산
QIW_FLOW  = 12    # 외인·기관 수급
QIW_HIGH52 = 8    # 신고가 근접(모멘텀)
# 차원 가중치 — 퀀트(기술) / 뉴스 / 매크로 레짐
DW_QUANT  = 60
DW_NEWS   = 25
DW_MACRO  = 15
# 점수 산정 주체 토글: True=파이썬 결정론(권장), False=구 LLM 채점으로 즉시 롤백
DETERMINISTIC_SCORING = True

# ─── 제도권 파이프라인 4기능 (사장 지시 2026-06-04) ───────────────────────────
# spec: docs/superpowers/specs/2026-06-04-institutional-pipeline-mimicry-design.md
# ① 퀀트 점수가 선정에 영향 — 한 사이클 최대 매수 종목 수(랭크 상위부터 자금배정)
MAX_BUY_NAMES          = 8            # 통과 후보가 이보다 많으면 퀀트점수 상위 N개만 매수(큰 값=무변화)
# ② 리스크기반 포지션 사이징
POSITION_SIZING_MODE   = "risk_weighted"  # 'equal'=균등(기존) / 'risk_weighted'=점수·역변동성 가중
SIZING_TILT_STRENGTH   = 0.5          # 0=완전균등 … 1=완전기울임 (균등분배와 raw가중 사이 보간)
SIZING_MAX_TILT        = 2.0          # 균등 대비 한 종목 최대/최소 배수(과집중 방지)
# ③ 유니버스 스크리닝 결정론화 (후보 풀 사전 배제 — 무음 금지, 로그 남김)
UNIVERSE_MIN_PRICE     = 0.0          # 현재가 < 이 값(원, US는 USD 별도 임계 미적용) 후보 배제 (0=off)
UNIVERSE_MIN_TURNOVER  = 0.0          # 일거래대금 < 이 값 후보 배제 (0=off)
UNIVERSE_EXCLUDE_LEVERAGED = True     # 레버리지/인버스/곱버스/ETN 후보 배제
# ⑤ 그랜저 선행-후행(lead-lag) 신호 — 30분 지평 (사장 지시 2026-07-21)
# 자체 분봉 크롤러(market_bars/, KOSPI200+KOSDAQ150)가 수집하는 bars.db 를 30분 버킷으로
# 리샘플해 선행주→후행주 관계를 재구성한다. 선행주가 최근 30분 이상 시장대비 오르면 후행주에
# 매수 우호 신호 → 계량 팩터(QIW_LEADLAG) + 매수후보 보강. (구 Lag_Trading 크롤러 이관·엔진 폐기)
ENABLE_LEADLAG_SIGNAL  = True         # 선행-후행 신호 사용(매수 후보 보강 + 계량 팩터). 파라미터 토글.
QIW_LEADLAG            = 8            # 계량 지표 가중치: 선행-후행(기본 편입). +면 선행주 상승 후행주 가점.
LEADLAG_LOOKBACK_MIN   = 30           # 선행주 '최근 이동' 판별 창(분). 사이클 20분+ 감안 최소 30.
LEADLAG_MIN_BUY_SIGNAL = 0.5          # 이 신호 이상인 후행주만 신규 매수 후보로 보강(0~1)
# 비튜닝 상수 (운용지원·웹 조정 대상 아님 — 구조/데이터 경로)
LEADLAG_BARS_DB        = os.getenv("LEADLAG_BARS_DB", str(BASE_DIR / "data" / "bars.db"))
LEADLAG_LOOKBACK_DAYS  = 10           # 구조적 맵 학습 영업일 수
LEADLAG_MIN_CONF       = 0.30         # 선행-후행 상관 임계(방향성 통과분만)
LEADLAG_TOP_LEADERS    = 3            # 후행주당 선행주 상위 N
LEADLAG_MAP_TTL_SEC    = 3600         # 구조적 맵 캐시 수명(초)
LEADLAG_MOVE_SCALE_PCT = 1.0          # 선행주 잔차수익률 이 %면 신호 만점(1.0=1%)
# ⑥ 추가 전략 파라미터 8종 (사장 지시 2026-07-21 후속 — 선행-후행 매도·상관분산·연속손절·
#    거래량급증·개장마감회피·갭업회피·레짐전환·뉴스감쇠)
ENABLE_LEADLAG_SELL    = True         # 선행주가 최근 30분+ 시장대비 급락하면 보유 후행주 부분매도(선행-후행 대칭)
MAX_PORTFOLIO_CORRELATION = 0.75      # 보유종목과 30분 상관이 이 값 초과인 신규 후보 매수 제한(0=off, 분산)
LOSS_STREAK_HALT       = 3            # 최근 실현매도가 N회 연속 손실이면 이번 사이클 신규매수 중단(0=off, 회로차단)
QIW_VOLUME_SURGE       = 8            # 계량 지표 가중치: 거래량 급증(평균比). +면 거래량 동반 돌파에 가점.
SKIP_OPEN_MIN          = 5            # 개장 후 이 분(分) 안에는 신규매수 회피(변동성 구간, 0=off)
SKIP_CLOSE_MIN         = 5            # 마감 전 이 분(分) 안에는 신규매수 회피(0=off)
MAX_GAP_UP_PCT         = 5.0          # 시가가 전일종가 대비 이 %+ 갭업한 종목 신규매수 제외(추격 회피, 0=off)
REGIME_ADAPTIVE        = True         # KOSPI 추세(20일선 대비)로 방어/공격 파라미터 자동 틸트
NEWS_DECAY_HOURS       = 24.0         # 이 시간(h) 초과된 뉴스는 감성 가중 감쇠(신선도, 0=off)
# 레짐 틸트 비튜닝 상수 (REGIME_ADAPTIVE 강도)
REGIME_TREND_LOOKBACK  = 20           # KOSPI 추세 판별 이동평균 일수
REGIME_DEFENSIVE_QS_BUMP = 1          # 방어 레짐 시 MIN_QUANT_SCORE 가산(엄선)
REGIME_DEFENSIVE_BUDGET_MULT = 0.7    # 방어 레짐 시 사이클 예산비율 배수
REGIME_AGGRESSIVE_BUDGET_MULT = 1.15  # 공격 레짐 시 사이클 예산비율 배수(캡 1.0)
# ④ 성과귀인 스코어카드 — 귀인 트레일링 윈도우(일)
SCORECARD_WINDOW_DAYS  = 30


def strategy_param_catalog_text():
    """STRATEGY_TUNABLE_KEYS + STRATEGY_KEY_META 로 전략 파라미터 카탈로그 텍스트를 만든다.
    운용지원실장 프롬프트에 런타임 주입 — 파라미터가 늘어도 카탈로그가 자동 최신화(stale 방지).
    각 줄: KEY (라벨) [type·범위·단위]: help. 효과: effect."""
    lines = []
    cur_group = None
    for k in STRATEGY_TUNABLE_KEYS:
        m = STRATEGY_KEY_META.get(k)
        if not m:
            continue
        g = m.get("group", "기타")
        if g != cur_group:
            lines.append(f"\n[{g}]")
            cur_group = g
        rng = ""
        if "min" in m and "max" in m:
            rng = f" {m['min']}~{m['max']}{m.get('unit','')}"
        eff = m.get("effect") or STRATEGY_KEY_EFFECT.get(k)
        eff_txt = f" 효과: {eff}" if eff else ""
        # 사장 지시 2026-06-09 #7: tier 표기 — 사이클 즉시 조정 가능 vs 토요일 백테스트 검증 후만.
        _tier = m.get("tier", "cycle")
        tier_txt = " 〔토요일 검증 후〕" if _tier == "weekly" else " 〔사이클 조정 가능〕"
        lines.append(f"- {k} ({m.get('label','')}) [{m.get('type','')}{rng}]{tier_txt}: {m.get('help','')}.{eff_txt}")
    return "\n".join(lines)


# ─── Strategy tunable parameters ─────────────────────────────────────────────
# A "strategy" is the set of tunable trading parameters. 사장 지시 2026-06-09: 프리셋
# (방어형~초공격형) 폐지 → 단일 STRATEGY_DEFAULTS(아래)가 기본값. 사장이 대시보드
# '전략' 탭에서 값을 직접 편집하면 runtime.py 가 프로필별로 영속·라이브 반영한다.
# Keys here MUST match module-level constant names above (runtime.get() falls back to those).
STRATEGY_TUNABLE_KEYS = [
    "PER_ORDER_BUDGET_RATIO", "PER_ORDER_BUDGET_OVERSHOOT", "MAX_CYCLE_BUDGET_RATIO", "MIN_CASH_BUFFER",
    "MACRO_DEPLOY_FLOOR_ENABLED", "PER_ORDER_BUDGET_FLOOR_RATIO", "MAX_CYCLE_BUDGET_FLOOR_RATIO",
    "ENABLE_DILUTION_GATE", "ENABLE_IC_SIZING",
    "CONSERVATIVE_MDD", "CONSERVATIVE_STOCK_RATIO",
    "MAX_TRADES_PER_CYCLE", "MAX_ORDER_QTY",
    # (A) 종목 필터 — 매수 자격 (사장 지시 2026-06-04)
    "MIN_QUANT_SCORE", "MAX_BUY_VOLATILITY_PCT", "RSI_OVERBOUGHT_SKIP", "MIN_ADX_FOR_BUY",
    "REQUIRE_FOREIGN_NET_BUY", "MAX_PRICE_EXTENSION_PCT",
    # 비용인지 진입 엣지 게이트 (고회전 수익화, 2026-06-18)
    "MIN_NET_EDGE_PCT",
    # (B) 결정론 점수 엔진 — 퀀트 지표 가중치 + 차원 가중치 + 토글 (사장 지시 2026-06-04, QW_* 대체)
    "QIW_RSI", "QIW_MACD", "QIW_ADX", "QIW_VWAP", "QIW_VOL", "QIW_MOM", "QIW_CMF", "QIW_FLOW", "QIW_HIGH52",
    "QIW_LEADLAG", "QIW_VOLUME_SURGE",
    "DW_QUANT", "DW_NEWS", "DW_MACRO", "DETERMINISTIC_SCORING",
    # 그랜저 선행-후행 신호 + 추가 전략 8종 (사장 지시 2026-07-21)
    "ENABLE_LEADLAG_SIGNAL", "LEADLAG_LOOKBACK_MIN", "LEADLAG_MIN_BUY_SIGNAL",
    "ENABLE_LEADLAG_SELL", "MAX_PORTFOLIO_CORRELATION", "LOSS_STREAK_HALT",
    "SKIP_OPEN_MIN", "SKIP_CLOSE_MIN", "MAX_GAP_UP_PCT", "REGIME_ADAPTIVE", "NEWS_DECAY_HOURS",
    # 제도권 파이프라인 4기능 (사장 지시 2026-06-04)
    "MAX_BUY_NAMES", "POSITION_SIZING_MODE", "SIZING_TILT_STRENGTH", "SIZING_MAX_TILT",
    "UNIVERSE_MIN_PRICE", "UNIVERSE_MIN_TURNOVER", "UNIVERSE_EXCLUDE_LEVERAGED",
    "SCORECARD_WINDOW_DAYS",
    # (C) 레짐 대응
    "MACRO_STOCK_GATE_ENABLED",
    "ENABLE_SELL_REBALANCE", "TAKE_PROFIT_PCT", "STOP_LOSS_PCT", "TRAILING_TAKE_PROFIT_PCT", "TRIM_OVER_RATIO",
    "ALLOW_DAY_TRADING", "MIN_HOLDING_DAYS_FOR_SELL",
    "ENABLE_CHEAP_FALLBACK", "ALLOW_US_STOCKS", "ALLOW_DERIVATIVES",
    # NXT 시간외 매매 (사장 지시 2026-06-08)
    "ENABLE_NXT_EXTENDED_HOURS", "ENABLE_NXT_PRE_MARKET", "ENABLE_NXT_AFTER_MARKET",
    "EXT_HOURS_LIMIT_SLIPPAGE_PCT", "EXT_HOURS_MAX_PREMIUM_PCT",
    # ADMIN 인텔리전스 공유 (사장 지시 2026-06-08)
    "SHARE_MARKET_INTELLIGENCE", "SHARE_PRODUCER_WAIT_SEC", "SHARE_STALE_OK_SEC",
    "QUANT_CONCURRENCY",
    # 채권 ETF 자동매매 (사장 지시 2026-06-08)
    "ENABLE_BOND_ETF", "BOND_TARGET_MAX_PCT", "BOND_REBALANCE_BAND_PCT",
    "BOND_PER_CYCLE_RATIO",
    # 원자재 ETF 자동매매 (사장 지시 2026-06-09)
    "ENABLE_COMMODITY_ETF", "COMMODITY_TARGET_MAX_PCT", "COMMODITY_REBALANCE_BAND_PCT",
    "COMMODITY_PER_CYCLE_RATIO",
]

# ─── 정책 플래그 봉인 (회사 운영 거버넌스, 사장 지시 2026-06-05) ────────────────
# 운용지원실장(ops)은 전술 파라미터는 적극 튜닝하되, 아래 '정책/구조 플래그'는
# 자율(cycle/weekly)로 변경할 수 없다 — 사장(대시보드 '전략' 탭 / 직접 지시)만 바꾼다.
#   • ALLOW_US_STOCKS / ALLOW_DERIVATIVES : 자산군 정책 (ops 가 미국 매매를 꺼 실거래
#     계정 매수가 막혔던 사고 2026-06-05 재발 방지)
#   • ENABLE_CHEAP_FALLBACK : 사장 영구 OFF 정책 (저가 대체매수 금지)
#   • DETERMINISTIC_SCORING : 채점 엔진 구조 토글 (LLM 롤백은 사장 결정 사항)
#   • ENABLE_NXT_* : NXT 시간외 매매 on/off 정책 (사장 지시 2026-06-11) — ops 가 자율로
#     꺼 profiles override 가 사장 전략탭 설정을 가려 '재시작마다 OFF 복귀'하던 사고 재발 방지.
# STRATEGY_TUNABLE_KEYS 에는 남겨 대시보드에서 사장이 직접 토글 가능하게 유지한다.
OPS_PROTECTED_KEYS = {
    "ALLOW_US_STOCKS", "ALLOW_DERIVATIVES",
    "ENABLE_CHEAP_FALLBACK", "DETERMINISTIC_SCORING",
    "ENABLE_NXT_EXTENDED_HOURS", "ENABLE_NXT_PRE_MARKET", "ENABLE_NXT_AFTER_MARKET",
    # 사장 지시 2026-06-12: 매크로 주식비중 매수게이트는 사장 방어 설정 — ops 자율 토글 금지.
    # ops 가 cyc305(00:05)에 OFF→cyc312(03:09) ON 으로 뒤집어, 매크로 '주식 0%' 권고에도
    # 1·2·3시 후보선정이 돌던 거버넌스 사고 재발 방지(2026-06-11 NXT 사건과 동형).
    "MACRO_STOCK_GATE_ENABLED",
}

# ─── Strategy parameter metadata (사장 지시 2026-05-14: UI에 한국어 라벨로 표시) ────
# type:
#   pct_ratio  → 저장값 0.10, UI 표시 10 (단위 %)
#   pct_raw    → 저장값 12.0, UI 표시 12.0 (단위 %)
#   multiplier → 저장값 1.20, UI 표시 1.20 (단위 ×)
#   int        → 정수 그대로
#   bool       → true/false 토글
STRATEGY_KEY_META = {
    "PER_ORDER_BUDGET_RATIO":     {"label": "1주문 예수금 사용 비율", "type": "pct_ratio", "unit": "%",
                                   "help": "한 번 매수 시 예수금의 최대 X%까지 사용 (예: 10 = 예수금의 10%)",
                                   "min": 1, "max": 100, "step": 1, "group": "사이징"},
    "PER_ORDER_BUDGET_OVERSHOOT": {"label": "1주 매수 예산 허용 초과율", "type": "multiplier", "unit": "×",
                                   "help": "1주 가격이 1주문 예산을 이 배율 이내로만 넘으면 매수 허용 (1.20 = +20%까지)",
                                   "min": 1.0, "max": 2.0, "step": 0.05, "group": "사이징"},
    "MAX_CYCLE_BUDGET_RATIO":     {"label": "사이클당 최대 예수금 사용", "type": "pct_ratio", "unit": "%",
                                   "help": "한 사이클의 모든 주문이 사용할 수 있는 예수금의 최대 비율",
                                   "min": 1, "max": 100, "step": 1, "group": "사이징"},
    "MIN_CASH_BUFFER":            {"label": "현금 안전 마진 (체결 슬리피지·수수료 여유)", "type": "multiplier", "unit": "×",
                                   "help": "주문 노티오날 × 이 배율보다 예수금이 적으면 거부 (1.10 = +10% 여유 요구)",
                                   "min": 1.0, "max": 1.5, "step": 0.01, "group": "사이징"},
    "MACRO_DEPLOY_FLOOR_ENABLED": {"label": "매크로 목표 향한 예산 플로어", "type": "bool",
                                   "help": "ON이면 매크로 주식목표 > 현재 주식비중(여력)일 때 예산컷이 배치를 과도하게 묶지 않도록 최소 플로어 적용",
                                   "group": "사이징"},
    "ENABLE_DILUTION_GATE":      {"label": "희석 공시 매수 게이트 (DART CB/유증)", "type": "bool",
                                   "help": "ON이면 매수 직전 DART 전환사채·유상증자 공시를 점검해 희석 위험 높은 종목 매수 보류(2026-06-15 ROI#4)",
                                   "group": "리스크"},
    "ENABLE_IC_SIZING":          {"label": "스코어카드 IC 확신도 사이징 반영", "type": "bool",
                                   "help": "ON이면 과거 예측력(IC) 기반 확신도를 실제 사이징에 반영. OFF면 섀도우 로그만(2026-06-15 ROI#2)",
                                   "group": "사이징"},
    "PER_ORDER_BUDGET_FLOOR_RATIO": {"label": "예산 플로어 — 1주문 하한", "type": "pct_ratio", "unit": "%",
                                   "help": "매크로 목표로 배치 여력이 있을 때 1주문 예산 비율의 하한",
                                   "min": 1, "max": 100, "step": 1, "group": "사이징"},
    "MAX_CYCLE_BUDGET_FLOOR_RATIO": {"label": "예산 플로어 — 사이클 하한", "type": "pct_ratio", "unit": "%",
                                   "help": "매크로 목표로 배치 여력이 있을 때 사이클 예산 비율의 하한",
                                   "min": 1, "max": 100, "step": 1, "group": "사이징"},
    "CONSERVATIVE_MDD":           {"label": "계좌 최대 손실 한도 (도달 시 신규 매수 차단)", "type": "pct_ratio", "unit": "%",
                                   "help": "계좌 평가손익이 -X% 이하면 신규 매수 전면 차단",
                                   "min": 1, "max": 30, "step": 1, "group": "리스크"},
    "CONSERVATIVE_STOCK_RATIO":   {"label": "단일 종목 최대 비중", "type": "pct_ratio", "unit": "%",
                                   "help": "한 종목 평가액이 총자산의 X%를 초과하면 매수 차단·초과분 매도",
                                   "min": 5, "max": 50, "step": 1, "group": "리스크"},
    "MAX_TRADES_PER_CYCLE":       {"label": "사이클당 최대 매매 건수", "type": "int", "unit": "건",
                                   "help": "한 사이클에서 실제 체결 시도할 주문의 최대 개수 (매도 우선)",
                                   "min": 0, "max": 10, "step": 1, "group": "리스크"},
    "MAX_ORDER_QTY":              {"label": "1주문 최대 수량 (0 = 제한 없음)", "type": "int", "unit": "주",
                                   "help": "한 주문이 살 수 있는 최대 주식 수. 0이면 사이징 룰에만 의존",
                                   "min": 0, "max": 10000, "step": 1, "group": "리스크"},
    # (A) 종목 필터 — 매수 자격 (사장 지시 2026-06-04)
    "MIN_QUANT_SCORE":            {"label": "최소 퀀트 점수 (미달 매수 제외)", "type": "int", "unit": "점",
                                   "help": "계량분석팀장 점수가 이 값 미만인 최종 매수대상을 결정론적으로 제거 (0~10)",
                                   "min": 0, "max": 10, "step": 1, "group": "종목 필터"},
    "MAX_BUY_VOLATILITY_PCT":     {"label": "매수 허용 최대 변동성 (연환산 %)", "type": "pct_raw", "unit": "%",
                                   "help": "연환산 변동성이 이 값을 초과하는 종목은 매수부적합 처리 (0 = 제한 없음)",
                                   "min": 0, "max": 200, "step": 5, "group": "종목 필터"},
    "RSI_OVERBOUGHT_SKIP":        {"label": "RSI 과매수 매수회피 기준", "type": "int", "unit": "",
                                   "help": "RSI(14)가 이 값을 초과하면 신규 매수 회피 (0 = 제한 없음, 예: 70~75)",
                                   "min": 0, "max": 100, "step": 1, "group": "종목 필터"},
    "MIN_ADX_FOR_BUY":            {"label": "매수 요구 최소 추세강도 (ADX)", "type": "int", "unit": "",
                                   "help": "ADX(14)가 이 값 미만이면(추세 약함) 매수부적합 (0 = 제한 없음, 추세추종용)",
                                   "min": 0, "max": 60, "step": 1, "group": "종목 필터"},
    "REQUIRE_FOREIGN_NET_BUY":    {"label": "외국인 순매수 종목만 매수", "type": "bool",
                                   "help": "ON이면 외국인 수급이 순매수(+)인 종목만 매수 적격으로 본다",
                                   "group": "종목 필터"},
    "MAX_PRICE_EXTENSION_PCT":    {"label": "매수 허용 최대 이격도 (VWAP/이평 대비 %)", "type": "pct_raw", "unit": "%",
                                   "help": "현재가가 VWAP/이동평균 대비 이 값 초과로 위에 있으면 추격매수 회피 (0 = 제한 없음)",
                                   "min": 0, "max": 50, "step": 1, "group": "종목 필터"},
    "ENABLE_COST_EDGE_GATE":      {"label": "비용인지 진입 엣지 게이트", "type": "bool",
                                   "help": "폐기된 설정입니다. 실험 진입 규칙은 기록만 하며 매매에 반영하지 않습니다.",
                                   "group": "종목 필터"},
    "MIN_NET_EDGE_PCT":           {"label": "실험 순엣지 기준 (관찰 전용, %)", "type": "pct_raw", "unit": "%",
                                   "help": "5거래일 평균수익을 보수적으로 축소하고 표준오차와 왕복비용(KR 0.35%, US 0.70% 가정)을 뺀 하한입니다. 관찰 기록에만 쓰며 매매 선정에는 반영하지 않습니다.",
                                   "min": 0, "max": 5, "step": 0.1, "group": "종목 필터"},
    # (B) 결정론 점수 엔진 — 퀀트 지표 가중치(signed, 음수 허용) (사장 지시 2026-06-04)
    "QIW_RSI":                    {"label": "지표 가중치: RSI(과매수/과매도)", "type": "int", "unit": "",
                                   "help": "RSI 신호 가중치. +면 과매도 우호·과매수 페널티(평균회귀). 음수면 반전(모멘텀)", "min": -50, "max": 50, "step": 1, "group": "퀀트 지표 가중치"},
    "QIW_MACD":                   {"label": "지표 가중치: MACD 모멘텀", "type": "int", "unit": "",
                                   "help": "MACD 히스토그램 신호 가중치. +면 상승 모멘텀 우호", "min": -50, "max": 50, "step": 1, "group": "퀀트 지표 가중치"},
    "QIW_ADX":                    {"label": "지표 가중치: ADX 추세강도", "type": "int", "unit": "",
                                   "help": "추세 강도·방향 가중치. +면 강한 상승추세 우호(추세추종)", "min": -50, "max": 50, "step": 1, "group": "퀀트 지표 가중치"},
    "QIW_VWAP":                   {"label": "지표 가중치: VWAP 이격", "type": "int", "unit": "",
                                   "help": "VWAP 이격 가중치. +면 과이격(추격) 페널티. 음수면 추격 허용", "min": -50, "max": 50, "step": 1, "group": "퀀트 지표 가중치"},
    "QIW_VOL":                    {"label": "지표 가중치: 변동성", "type": "int", "unit": "",
                                   "help": "σ20 가중치. +면 고변동 페널티(방어). 음수면 고변동 선호", "min": -50, "max": 50, "step": 1, "group": "퀀트 지표 가중치"},
    "QIW_MOM":                    {"label": "지표 가중치: 모멘텀(1M·3M)", "type": "int", "unit": "",
                                   "help": "수익률 모멘텀 가중치. +면 상승 추세 우호", "min": -50, "max": 50, "step": 1, "group": "퀀트 지표 가중치"},
    "QIW_CMF":                    {"label": "지표 가중치: CMF 자금흐름", "type": "int", "unit": "",
                                   "help": "Chaikin Money Flow 가중치. +면 매집 우호·분산 페널티", "min": -50, "max": 50, "step": 1, "group": "퀀트 지표 가중치"},
    "QIW_FLOW":                   {"label": "지표 가중치: 수급(외인·기관)", "type": "int", "unit": "",
                                   "help": "외인+기관 순매수 가중치. +면 순매수 우호", "min": -50, "max": 50, "step": 1, "group": "퀀트 지표 가중치"},
    "QIW_HIGH52":                 {"label": "지표 가중치: 52주 신고가 근접", "type": "int", "unit": "",
                                   "help": "신고가 근접 가중치. +면 신고가 모멘텀 우호", "min": -50, "max": 50, "step": 1, "group": "퀀트 지표 가중치"},
    "QIW_LEADLAG":                {"label": "지표 가중치: 선행-후행(그랜저)", "type": "int", "unit": "",
                                   "help": "선행주가 최근 30분(≥) 시장대비 오르면 후행주 가점(+). 자체 분봉(market_bars) 기반 30분 지평 신호", "min": -50, "max": 50, "step": 1, "group": "퀀트 지표 가중치"},
    "QIW_VOLUME_SURGE":           {"label": "지표 가중치: 거래량 급증", "type": "int", "unit": "",
                                   "help": "당일 거래량이 20일 평균 대비 급증하면 가점(+, 거래량 동반 돌파). 음수면 급증 회피", "min": -50, "max": 50, "step": 1, "group": "퀀트 지표 가중치"},
    # 차원 가중치 + 토글
    "DW_QUANT":                   {"label": "차원 가중치: 퀀트(기술)", "type": "int", "unit": "",
                                   "help": "최종 점수에서 퀀트(지표) 차원 비중(signed)", "min": -100, "max": 100, "step": 5, "group": "점수 차원"},
    "DW_NEWS":                    {"label": "차원 가중치: 뉴스", "type": "int", "unit": "",
                                   "help": "뉴스 감성 차원 비중(signed). 음수면 호재일수록 감점(역발상)", "min": -100, "max": 100, "step": 5, "group": "점수 차원"},
    "DW_MACRO":                   {"label": "차원 가중치: 매크로 레짐", "type": "int", "unit": "",
                                   "help": "매크로 권고 주식비중 차원(signed). +면 강세 매크로일수록 가점", "min": -100, "max": 100, "step": 5, "group": "점수 차원"},
    "DETERMINISTIC_SCORING":      {"label": "결정론 점수 사용(끄면 구 LLM 채점)", "type": "bool",
                                   "help": "ON=퀀트점수를 파이썬이 결정론적으로 산정(일관). OFF=구 LLM 정성 채점으로 롤백", "group": "점수 차원"},
    # 그랜저 선행-후행 신호 (사장 지시 2026-07-21)
    "ENABLE_LEADLAG_SIGNAL":      {"label": "선행-후행 신호 사용(그랜저 30분)", "type": "bool",
                                   "help": "ON이면 선행주가 최근 30분+ 시장대비 오른 후행주를 매수 후보로 보강 + 계량 팩터 반영. 자체 분봉(market_bars) 기반",
                                   "group": "선행-후행"},
    "LEADLAG_LOOKBACK_MIN":       {"label": "선행주 최근 이동 판별 창(분)", "type": "int", "unit": "분",
                                   "help": "선행주가 '최근에 올랐다'를 볼 창(분). 사이클이 20분+ 걸리므로 최소 30분",
                                   "min": 30, "max": 180, "step": 10, "group": "선행-후행"},
    "LEADLAG_MIN_BUY_SIGNAL":     {"label": "선행-후행 매수 보강 문턱(0~1)", "type": "pct_raw", "unit": "",
                                   "help": "후행주 신호가 이 값 이상일 때만 신규 매수 후보로 보강(올리면 강한 신호만·보수적)",
                                   "min": 0.0, "max": 1.0, "step": 0.05, "group": "선행-후행"},
    "ENABLE_LEADLAG_SELL":        {"label": "선행-후행 매도(선행주 급락→후행주)", "type": "bool",
                                   "help": "ON이면 선행주가 최근 30분+ 시장대비 급락한 보유 후행주를 절반 부분매도(선행-후행 대칭 방어)",
                                   "group": "선행-후행"},
    "MAX_PORTFOLIO_CORRELATION":  {"label": "포트폴리오 상관 상한(신규매수 제한)", "type": "pct_raw", "unit": "",
                                   "help": "신규 후보의 30분 수익률 상관이 보유종목 중 이 값을 초과하면 매수 제외(분산). 0=제한없음",
                                   "min": 0.0, "max": 1.0, "step": 0.05, "group": "리스크"},
    "LOSS_STREAK_HALT":           {"label": "연속 손절 매수중단 (회로차단)", "type": "int", "unit": "회",
                                   "help": "최근 실현매도가 이 횟수만큼 연속 손실이면 이번 사이클 신규매수 중단. 0=제한없음",
                                   "min": 0, "max": 10, "step": 1, "group": "리스크"},
    "SKIP_OPEN_MIN":              {"label": "개장 후 매수회피 시간(분)", "type": "int", "unit": "분",
                                   "help": "장 개장 후 이 분 안에는 신규매수 회피(개장 변동성 구간). 0=제한없음",
                                   "min": 0, "max": 60, "step": 5, "group": "매매 타이밍"},
    "SKIP_CLOSE_MIN":             {"label": "마감 전 매수회피 시간(분)", "type": "int", "unit": "분",
                                   "help": "장 마감 전 이 분 안에는 신규매수 회피. 0=제한없음",
                                   "min": 0, "max": 60, "step": 5, "group": "매매 타이밍"},
    "MAX_GAP_UP_PCT":             {"label": "매수 허용 최대 갭업(%)", "type": "pct_raw", "unit": "%",
                                   "help": "시가가 전일종가 대비 이 % 초과 갭업한 종목은 신규매수 제외(추격 회피). 0=제한없음",
                                   "min": 0.0, "max": 30.0, "step": 0.5, "group": "종목 필터"},
    "NEWS_DECAY_HOURS":           {"label": "뉴스 신선도 감쇠 시간(h)", "type": "pct_raw", "unit": "시간",
                                   "help": "이 시간을 초과한 뉴스는 감성 가중이 지수 감쇠(오래된 뉴스가 계속 가점되는 것 방지). 0=감쇠없음",
                                   "min": 0.0, "max": 168.0, "step": 6.0, "group": "종목 필터"},
    "REGIME_ADAPTIVE":            {"label": "레짐 자동전환(KOSPI 추세)", "type": "bool",
                                   "help": "ON이면 KOSPI가 20일선 아래(약세)면 방어(엄선·예산축소), 위(강세)면 공격(예산확대)으로 자동 틸트",
                                   "group": "레짐 대응"},
    # (C) 레짐 대응
    "MACRO_STOCK_GATE_ENABLED":   {"label": "매크로 주식비중 매수게이트", "type": "bool",
                                   "help": "ON이면 매크로 권고 주식비중 ≤ 현재 주식 평가비중일 때 신규 매수평가를 건너뜀(매도/관리만)",
                                   "group": "레짐 대응"},
    "ENABLE_SELL_REBALANCE":      {"label": "자동 익절/손절/편중축소 매도 활성화", "type": "bool",
                                   "help": "보유 종목이 임계 도달 시 자동 매도 (사후관리실장이 별도 언급 안 한 종목만)",
                                   "group": "매도 규칙"},
    "TAKE_PROFIT_PCT":            {"label": "익절 기준 (보유 수익률 ≥)", "type": "pct_raw", "unit": "%",
                                   "help": "보유 종목 평가손익이 +X%를 넘으면 자동 전량 매도",
                                   "min": 1, "max": 100, "step": 0.5, "group": "매도 규칙"},
    "STOP_LOSS_PCT":              {"label": "손절 기준 (보유 손실률 ≥)", "type": "pct_raw", "unit": "%",
                                   "help": "보유 종목 평가손익이 -X% 이하로 떨어지면 자동 전량 매도",
                                   "min": 1, "max": 50, "step": 0.5, "group": "매도 규칙"},
    "TRAILING_TAKE_PROFIT_PCT":   {"label": "트레일링 익절 (고점 대비 되밀림 %)", "type": "pct_raw", "unit": "%",
                                   "help": "보유 고점 대비 이 %만큼 되밀리면 매도해 승자를 길게 가져간다(추세추종). 0=off",
                                   "min": 0, "max": 30, "step": 0.5, "group": "매도 규칙"},
    "TRIM_OVER_RATIO":            {"label": "단일 종목 비중 초과 시 자동 축소", "type": "bool",
                                   "help": "단일 종목 비중이 한도를 넘으면 초과분만큼 부분 매도",
                                   "group": "매도 규칙"},
    "ALLOW_DAY_TRADING":          {"label": "데이트레이딩 허용 (단기 매도 OK)", "type": "bool",
                                   "help": "ON이면 보유기간 무관하게 매도 가능. OFF면 최소 보유일 미만은 데이트레이딩 회피",
                                   "group": "매도 규칙"},
    "MIN_HOLDING_DAYS_FOR_SELL":  {"label": "최소 보유일 (데이트레이딩 OFF일 때만)", "type": "pct_raw", "unit": "일",
                                   "help": "데이트레이딩 비허용 시 이 일수 미만 보유 종목은 매도 회피 (사후관리실장)",
                                   "min": 0, "max": 30, "step": 0.5, "group": "매도 규칙"},
    "ENABLE_CHEAP_FALLBACK":      {"label": "최종 종목 매수 불가 시 저가 대체 종목 매수", "type": "bool",
                                   "help": "OFF 권장 — 후보 모두 예산 초과 시 거래량 상위 저가주를 대신 매수",
                                   "group": "기타"},
    "ENABLE_NXT_EXTENDED_HOURS":  {"label": "넥스트레이드(NXT) 시간외 매매", "type": "bool",
                                   "help": "ON이면 프리마켓(08:00–08:50)·애프터마켓(15:50–20:00)에 NXT 경유 매매",
                                   "group": "시간외(NXT)"},
    "ENABLE_NXT_PRE_MARKET":      {"label": "프리마켓(08:00–08:50) 매매", "type": "bool",
                                   "help": "넥스트레이드 프리마켓 지정가 매매 (마스터 스위치 ON 전제)",
                                   "group": "시간외(NXT)"},
    "ENABLE_NXT_AFTER_MARKET":    {"label": "애프터마켓(15:50–20:00) 매매", "type": "bool",
                                   "help": "넥스트레이드 애프터마켓 지정가 매매 (마스터 스위치 ON 전제)",
                                   "group": "시간외(NXT)"},
    "EXT_HOURS_LIMIT_SLIPPAGE_PCT": {"label": "시간외 지정가 밴드", "type": "pct_raw", "unit": "%",
                                   "help": "시간외 지정가 = 현재가 ± 이 폭. 체결확률↑ vs 슬리피지 상한 트레이드오프",
                                   "min": 0, "max": 5, "step": 0.1, "group": "시간외(NXT)"},
    "EXT_HOURS_MAX_PREMIUM_PCT":   {"label": "시간외 프리미엄 캡", "type": "pct_raw", "unit": "%",
                                   "help": "정규 종가 대비 시간외 지정가 매수 상한·매도 하한. 얇은 NXT가 큰 프리미엄을 호가해도 추종 안 함(0=제한없음)",
                                   "min": 0, "max": 10, "step": 0.5, "group": "시간외(NXT)"},
    "SHARE_MARKET_INTELLIGENCE":  {"label": "시장 인텔리전스 공유(ADMIN 단일 생산)", "type": "bool",
                                   "help": "ON이면 관리자 계정이 매크로·뉴스 분석을 1회 산출, 다른 계정은 공유받아 LLM 중복 비용 절감",
                                   "group": "비용"},
    "SHARE_PRODUCER_WAIT_SEC":    {"label": "공유 대기 타임아웃", "type": "int", "unit": "초",
                                   "help": "공유받는 계정이 생산 계정의 게시를 기다리는 최대 초. 초과하면 자체 계산(생산자 부재 대응)",
                                   "min": 10, "max": 600, "step": 10, "group": "비용"},
    "SHARE_STALE_OK_SEC":         {"label": "공유 분석 재사용 허용 시간", "type": "int", "unit": "초",
                                   "help": "직전 시각에 게시된 매크로·뉴스 분석을 이 시간 안이면 재사용. 올리면 LLM 호출↓, 내리면 신선도↑",
                                   "min": 0, "max": 3600, "step": 300, "group": "비용"},
    "QUANT_CONCURRENCY":          {"label": "계량분석 동시 실행 수", "type": "int", "unit": "개",
                                   "help": "종목별 계량분석을 몇 개씩 동시에 돌릴지. 올리면 사이클이 짧아지고 로컬 LLM 슬롯을 더 씀(1=순차)",
                                   "min": 1, "max": 4, "step": 1, "group": "비용"},
    # 채권 ETF 자동매매 (사장 지시 2026-06-08)
    "ENABLE_BOND_ETF":         {"label": "채권 ETF 자동매매(채권운용실장)", "type": "bool",
                                "help": "켜면 매크로 채권 비중 권고를 채권 ETF 매수/매도로 실현. 끄면 채권 트랙 전체 스킵.",
                                "group": "매도 규칙"},
    "BOND_TARGET_MAX_PCT":     {"label": "채권 비중 상한", "type": "pct_raw", "unit": "%비율",
                                "help": "채권 평가비중 절대 상한. 매크로 권고가 이를 넘어도 이 값으로 클램프(0.40=40%)",
                                "min": 0.0, "max": 1.0, "step": 0.05, "group": "매도 규칙"},
    "BOND_REBALANCE_BAND_PCT": {"label": "채권 리밸런싱 데드존", "type": "pct_raw", "unit": "%비율",
                                "help": "목표 대비 ±이 폭 이내면 채권 매매 안 함(잦은 교체 방지). 0.03=±3%p",
                                "min": 0.0, "max": 0.2, "step": 0.01, "group": "매도 규칙"},
    "BOND_PER_CYCLE_RATIO":    {"label": "채권 사이클 예산비율", "type": "pct_raw", "unit": "%비율",
                                "help": "채권 전용 사이클 매수 한도(총평가 대비). 0.15=총평가의 15%까지 한 사이클에 채권 매수",
                                "min": 0.0, "max": 1.0, "step": 0.05, "group": "매도 규칙"},
    # 원자재 ETF 자동매매 (사장 지시 2026-06-09)
    "ENABLE_COMMODITY_ETF":    {"label": "원자재 ETF 자동매매(원자재운용실장)", "type": "bool",
                                "help": "켜면 매크로 원자재 비중 권고를 원유·금·농산물 ETF 매수/매도로 실현. 끄면 원자재 트랙 전체 스킵.",
                                "group": "매도 규칙"},
    "COMMODITY_TARGET_MAX_PCT":     {"label": "원자재 비중 상한", "type": "pct_raw", "unit": "%비율",
                                "help": "원자재 평가비중 절대 상한. 매크로 권고가 이를 넘어도 이 값으로 클램프(0.20=20%)",
                                "min": 0.0, "max": 1.0, "step": 0.05, "group": "매도 규칙"},
    "COMMODITY_REBALANCE_BAND_PCT": {"label": "원자재 리밸런싱 데드존", "type": "pct_raw", "unit": "%비율",
                                "help": "목표 대비 ±이 폭 이내면 원자재 매매 안 함(잦은 교체 방지). 0.03=±3%p",
                                "min": 0.0, "max": 0.2, "step": 0.01, "group": "매도 규칙"},
    "COMMODITY_PER_CYCLE_RATIO":    {"label": "원자재 사이클 예산비율", "type": "pct_raw", "unit": "%비율",
                                "help": "원자재 전용 사이클 매수 한도(총평가 대비). 0.10=총평가의 10%까지 한 사이클에 원자재 매수",
                                "min": 0.0, "max": 1.0, "step": 0.05, "group": "매도 규칙"},
    "ALLOW_US_STOCKS":            {"label": "미국 주식 매매 허용", "type": "bool",
                                   "help": "US 장 시간(KST 22:30~05:00)에 미국 상장 종목 거래",
                                   "group": "기타"},
    "ALLOW_DERIVATIVES":          {"label": "파생상품(선물/옵션) 매매 허용", "type": "bool",
                                   "help": "마진·증거금 처리 필요 — 기본 OFF",
                                   "group": "기타"},
    # 제도권 파이프라인 4기능 (사장 지시 2026-06-04)
    "MAX_BUY_NAMES":              {"label": "사이클당 최대 매수 종목 수", "type": "int", "unit": "개",
                                   "help": "퀀트점수 통과 후보가 이보다 많으면 점수 상위 N개만 매수(랭크 우선 자금배정)",
                                   "min": 1, "max": 20, "step": 1, "group": "종목 필터"},
    "POSITION_SIZING_MODE":       {"label": "포지션 사이징 방식", "type": "choice", "choices": ["equal", "risk_weighted"],
                                   "help": "equal=종목 균등분배(기존). risk_weighted=퀀트점수↑·변동성↓ 종목에 더 큰 비중",
                                   "group": "사이징"},
    "SIZING_TILT_STRENGTH":       {"label": "사이징 기울임 강도", "type": "pct_raw", "unit": "",
                                   "help": "0=완전 균등, 1=완전 기울임. risk_weighted일 때만 효과(0.5=절반 적용)",
                                   "min": 0.0, "max": 1.0, "step": 0.1, "group": "사이징"},
    "SIZING_MAX_TILT":            {"label": "사이징 최대 배수(과집중 방지)", "type": "multiplier", "unit": "×",
                                   "help": "한 종목이 균등분배 대비 받을 수 있는 최대/최소 배수(2.0=최대2배·최소0.5배)",
                                   "min": 1.0, "max": 4.0, "step": 0.5, "group": "사이징"},
    "UNIVERSE_MIN_PRICE":         {"label": "유니버스 최소 주가 (저가주 배제)", "type": "pct_raw", "unit": "원",
                                   "help": "현재가가 이 값 미만인 후보를 사전 배제(동전주 회피). 0=제한없음",
                                   "min": 0, "max": 10000, "step": 100, "group": "종목 필터"},
    "UNIVERSE_MIN_TURNOVER":      {"label": "유니버스 최소 일거래대금", "type": "pct_raw", "unit": "원",
                                   "help": "일거래대금이 이 값 미만인 후보를 사전 배제(유동성 확보). 0=제한없음",
                                   "min": 0, "max": 100000000000, "step": 100000000, "group": "종목 필터"},
    "UNIVERSE_EXCLUDE_LEVERAGED": {"label": "레버리지/인버스 ETF·ETN 배제", "type": "bool",
                                   "help": "ON이면 레버리지·인버스·곱버스·ETN 후보를 사전 배제",
                                   "group": "종목 필터"},
    "SCORECARD_WINDOW_DAYS":      {"label": "성과귀인 분석 기간(일)", "type": "int", "unit": "일",
                                   "help": "에이전트 스코어카드(IC·알파/베타)를 계산할 트레일링 윈도우",
                                   "min": 7, "max": 180, "step": 1, "group": "기타"},
}

# 사장 지시 2026-06-04: 각 파라미터를 '올리면/내리면(켜면/끄면) 어떤 방향'인지 — 운용지원실장이 전략
# 지시(급락장 대비·추세추종 등)를 파라미터로 충실히 번역하도록 카탈로그에 주입되는 effect 설명.
STRATEGY_KEY_EFFECT = {
    "PER_ORDER_BUDGET_RATIO": "올리면 한 종목에 더 크게 베팅(공격), 내리면 잘게 분산(방어).",
    "PER_ORDER_BUDGET_OVERSHOOT": "올리면 고가주도 1주 매수 허용, 내리면 예산 초과 매수 차단.",
    "MAX_CYCLE_BUDGET_RATIO": "올리면 한 사이클에 현금 더 투입(공격), 내리면 천천히(방어).",
    "MIN_CASH_BUFFER": "올리면 현금 여유 크게 요구→매수 보수적(급락장), 내리면 공격적.",
    "CONSERVATIVE_MDD": "내리면 작은 손실에도 신규매수 차단(방어), 올리면 손실 감내(공격).",
    "CONSERVATIVE_STOCK_RATIO": "내리면 종목당 비중 작게(분산·방어), 올리면 집중(공격).",
    "MAX_TRADES_PER_CYCLE": "내리면 매매 빈도↓(신중), 올리면 다종목 동시 매매(공격).",
    "MAX_ORDER_QTY": "0=제한없음. 줄이면 1회 주문 수량 상한(과대주문 방지).",
    "MIN_QUANT_SCORE": "올리면 더 엄선(고확신만 매수·방어/모멘텀), 내리면 폭넓게 매수.",
    "MAX_BUY_VOLATILITY_PCT": "내리면 고변동 종목 회피(급락장·방어), 0=제한없음(공격).",
    "RSI_OVERBOUGHT_SKIP": "내리면(예 70) 과매수 추격 회피(역추세·방어), 0=제한없음.",
    "MIN_ADX_FOR_BUY": "올리면 강한 추세 종목만 매수(추세추종), 0=제한없음.",
    "REQUIRE_FOREIGN_NET_BUY": "켜면 외국인 순매수 종목만(수급 방어), 끄면 무관.",
    "MAX_PRICE_EXTENSION_PCT": "내리면 이평/VWAP 멀리 뜬 종목 추격 회피(역추세), 0=제한없음.",
    "ENABLE_COST_EDGE_GATE": "폐기된 설정. 변동성을 기대수익으로 취급하지 않는다.",
    "MIN_NET_EDGE_PCT": "5거래일 추정치의 관찰 기준. 매매에 반영하지 않으며 별도의 검증이 필요하다.",
    "QIW_RSI": "+면 과매도 매수·과매수 회피(평균회귀). 음수면 반대(RSI 높을수록 가점=모멘텀).",
    "QIW_MACD": "+면 MACD 상승 모멘텀에 가점. 음수면 역추세.",
    "QIW_ADX": "+면 강한 상승추세에 가점(추세추종). 0이면 추세 무시.",
    "QIW_VWAP": "+면 평균(VWAP) 위로 많이 뜬 종목 감점(추격 회피). 음수면 추격 허용.",
    "QIW_VOL": "+면 고변동 종목 감점(방어). 음수면 고변동 선호(공격).",
    "QIW_MOM": "+면 1M·3M 상승 모멘텀에 가점. 음수면 역추세(낙폭과대).",
    "QIW_CMF": "+면 매집(자금유입) 가점·분산 감점. 음수면 반대.",
    "QIW_FLOW": "+면 외인·기관 순매수에 가점. 음수면 반대(역행).",
    "QIW_HIGH52": "+면 52주 신고가 근접에 가점(모멘텀). 음수면 고점 회피.",
    "QIW_LEADLAG": "+면 선행주(그랜저)가 최근 30분+ 오른 후행주에 가점(선행-후행 추종). 0이면 선행-후행 무시.",
    "ENABLE_LEADLAG_SIGNAL": "켜면 선행-후행 신호로 후행주 매수후보 보강 + 계량 팩터 반영(30분 지평), 끄면 미사용.",
    "LEADLAG_LOOKBACK_MIN": "올리면 더 오래된 선행주 이동까지 반영(느린 추종), 내리면 최신 이동만(단, 최소 30분).",
    "LEADLAG_MIN_BUY_SIGNAL": "올리면 강한 선행-후행 신호만 신규 매수(보수적), 내리면 폭넓게 보강.",
    "QIW_VOLUME_SURGE": "+면 거래량 급증(돌파)에 가점(추세추종). 음수면 급증 회피(과열 경계).",
    "ENABLE_LEADLAG_SELL": "켜면 선행주 급락 시 보유 후행주 부분매도(선행-후행 대칭 방어), 끄면 매도 신호 미사용.",
    "MAX_PORTFOLIO_CORRELATION": "내리면 보유와 비슷한 종목 매수 강하게 제한(분산·방어), 올리면 허용(집중). 0=제한없음.",
    "LOSS_STREAK_HALT": "내리면 연속손절 몇 번에 바로 매수중단(방어·급락장), 올리면 손실 감내. 0=제한없음.",
    "SKIP_OPEN_MIN": "올리면 개장 변동성 구간을 더 오래 피함(안정), 0=제한없음.",
    "SKIP_CLOSE_MIN": "올리면 마감 임박 매수를 더 오래 피함(종가 왜곡 회피), 0=제한없음.",
    "MAX_GAP_UP_PCT": "내리면 작은 갭업도 추격매수 회피(보수), 올리면 갭 추격 허용. 0=제한없음.",
    "NEWS_DECAY_HOURS": "내리면 최신 뉴스만 감성 반영(신선도↑), 올리면 오래된 뉴스도 유지. 0=감쇠없음.",
    "REGIME_ADAPTIVE": "켜면 KOSPI 약세장엔 방어(엄선·예산↓)·강세장엔 공격(예산↑)으로 자동 전환, 끄면 고정.",
    "DW_QUANT": "올리면 최종 점수에서 기술(지표) 비중↑.",
    "DW_NEWS": "올리면 뉴스 감성 비중↑. 음수면 호재일수록 감점(역발상).",
    "DW_MACRO": "올리면 매크로 레짐 비중↑(강세 매크로 가점·약세 감점). 음수면 역발상.",
    "DETERMINISTIC_SCORING": "켜면 파이썬 결정론 점수(일관). 끄면 구 LLM 정성 채점.",
    "MACRO_STOCK_GATE_ENABLED": "켜면 매크로가 주식 추가매수 불가일 때 신규매수 스킵(방어), 끄면 매크로 무시하고 매수평가.",
    "ENABLE_SELL_REBALANCE": "켜면 자동 익절/손절/편중축소, 끄면 사후관리실장 판단만.",
    "TAKE_PROFIT_PCT": "내리면 빨리 익절(보수), 올리면 길게 보유(추세추종·공격).",
    "STOP_LOSS_PCT": "내리면 타이트한 손절(급락장·방어), 올리면 느슨(공격).",
    "TRIM_OVER_RATIO": "켜면 비중 초과분 자동 부분매도(리밸런싱).",
    "TRAILING_TAKE_PROFIT_PCT": "올리면 승자를 더 길게(되밀림 크게 허용·추세추종), 내리면 빨리 차익실현. 0=off. 비대칭 청산(타이트 손절+트레일링 익절)으로 고회전 기대값↑.",
    "ALLOW_DAY_TRADING": "켜면 보유기간 무관 매도, 끄면 최소보유일 미만 단타 회피.",
    "MIN_HOLDING_DAYS_FOR_SELL": "올리면 더 오래 보유 강제(데이트레이딩 OFF 시).",
    "ENABLE_CHEAP_FALLBACK": "OFF 권장 — 켜면 후보 매수불가 시 저가주 대체매수(권장X).",
    "ALLOW_US_STOCKS": "켜면 미국 주식 매매 허용.",
    "ALLOW_DERIVATIVES": "켜면 파생(선물/옵션) — 고위험, 기본 OFF.",
    "MAX_BUY_NAMES": "내리면 고확신 소수 종목에 집중(분산↓), 올리면 폭넓게 분산 매수.",
    "POSITION_SIZING_MODE": "risk_weighted=점수높고 변동성낮은 종목에 큰 비중(샤프 개선). equal=균등(기존).",
    "SIZING_TILT_STRENGTH": "올리면 점수·변동성 차이를 비중에 강하게 반영(공격), 0이면 균등.",
    "SIZING_MAX_TILT": "올리면 한 종목 비중 편차 허용(집중), 내리면 균등에 가깝게(분산·방어).",
    "UNIVERSE_MIN_PRICE": "올리면 저가·동전주 배제(품질 방어), 0=제한없음.",
    "UNIVERSE_MIN_TURNOVER": "올리면 거래 활발한 종목만(유동성·급락장 방어), 0=제한없음.",
    "UNIVERSE_EXCLUDE_LEVERAGED": "켜면 레버리지/인버스/ETN 배제(변동성 방어), 끄면 허용.",
    "SCORECARD_WINDOW_DAYS": "올리면 더 긴 기간으로 에이전트 성과 평가(안정), 내리면 최근 위주(민감).",
    "ENABLE_NXT_EXTENDED_HOURS": "켜면 프리/애프터마켓(NXT) 시간외 매매 활성, 끄면 정규장(KRX)만(방어).",
    "ENABLE_NXT_PRE_MARKET": "켜면 프리마켓(08:00–08:50) NXT 매매, 끄면 해당 구간 매매 안 함.",
    "ENABLE_NXT_AFTER_MARKET": "켜면 애프터마켓(15:50–20:00) NXT 매매, 끄면 해당 구간 매매 안 함.",
    "EXT_HOURS_LIMIT_SLIPPAGE_PCT": "올리면 시간외 지정가를 현재가에서 더 멀리(체결확률↑·슬리피지↑), 내리면 가깝게.",
    "SHARE_MARKET_INTELLIGENCE": "켜면 ADMIN이 매크로·뉴스 분석을 1회만 하고 다른 계정이 공유(LLM 비용↓), 끄면 계정마다 각자 계산.",
    "QUANT_CONCURRENCY": "올리면 종목별 계량분석을 더 많이 동시 실행(사이클 단축), 내리면 LLM 슬롯 점유↓(1=순차).",
    "SHARE_STALE_OK_SEC": "직전 시각에 게시된 매크로·뉴스 분석을 이 시간 안이면 재사용(중복 LLM 호출 방지).",
    "SHARE_PRODUCER_WAIT_SEC": "올리면 대표 프로필의 분석을 더 오래 기다림(공유 적중↑), 내리면 빨리 자체계산으로 전환(지연↓).",
    "ENABLE_BOND_ETF": "켜면 매크로 채권 권고를 채권 ETF로 실현(자산배분 충실), 끄면 채권 매매 안 함.",
    "BOND_TARGET_MAX_PCT": "올리면 채권에 더 많이 배분 허용, 내리면 채권 상한 축소.",
    "BOND_REBALANCE_BAND_PCT": "올리면 채권 교체 둔감(churn↓), 내리면 목표 추종 민감.",
    "BOND_PER_CYCLE_RATIO": "올리면 채권을 더 빠르게 목표비중까지 매수, 내리면 천천히.",
    "ENABLE_COMMODITY_ETF": "켜면 매크로 원자재 권고를 원유·금·농산물 ETF로 실현(인플레 헤지), 끄면 원자재 매매 안 함.",
    "COMMODITY_TARGET_MAX_PCT": "올리면 원자재에 더 많이 배분 허용, 내리면 원자재 상한 축소.",
    "COMMODITY_REBALANCE_BAND_PCT": "올리면 원자재 교체 둔감(churn↓), 내리면 목표 추종 민감.",
    "COMMODITY_PER_CYCLE_RATIO": "올리면 원자재를 더 빠르게 목표비중까지 매수, 내리면 천천히.",
}

# ─── ops 파라미터 tier: 사이클 조정 가능 vs 토요일 검증 후 (사장 지시 2026-06-09 #7) ─────
# 운용지원실장이 (a) **매 사이클** 자유롭게 조정해도 안전한 '전술·반응형' 파라미터(cycle)와
# (b) 모델·구조를 바꿔 **토요일 백테스트+실데이터 검증 후에만** 조정해야 하는 파라미터(weekly)를
# 코드로 강제 구분한다(infra.ops_param_clamp.partition_by_tier 가 enforcement).
#   • weekly = 점수엔진 가중치(QIW_*·DW_*)·사이징 모델·유니버스 스크린·종목수·스코어카드 윈도우·
#     슬리브 구조값(상한/밴드/사이클예산)·슬리브 마스터스위치·결정론 채점 토글.
#   • cycle  = 그 외 전술 파라미터(사이징 비율·리스크 한도·매도룰·종목 필터·레짐 게이트·NXT·공유).
# (OPS_PROTECTED_KEYS 는 tier 와 무관하게 ops 자율변경 불가 — partition_protected 가 먼저 처리.)
STRATEGY_WEEKLY_TIER_KEYS = {
    "QIW_RSI", "QIW_MACD", "QIW_ADX", "QIW_VWAP", "QIW_VOL", "QIW_MOM", "QIW_CMF", "QIW_FLOW", "QIW_HIGH52",
    "QIW_LEADLAG", "QIW_VOLUME_SURGE",
    "DW_QUANT", "DW_NEWS", "DW_MACRO", "DETERMINISTIC_SCORING",
    "POSITION_SIZING_MODE", "SIZING_TILT_STRENGTH", "SIZING_MAX_TILT",
    "UNIVERSE_MIN_PRICE", "UNIVERSE_MIN_TURNOVER", "UNIVERSE_EXCLUDE_LEVERAGED",
    "MAX_BUY_NAMES", "SCORECARD_WINDOW_DAYS",
    "ENABLE_BOND_ETF", "BOND_TARGET_MAX_PCT", "BOND_REBALANCE_BAND_PCT", "BOND_PER_CYCLE_RATIO",
    "ENABLE_COMMODITY_ETF", "COMMODITY_TARGET_MAX_PCT", "COMMODITY_REBALANCE_BAND_PCT", "COMMODITY_PER_CYCLE_RATIO",
    "ENABLE_DILUTION_GATE", "ENABLE_IC_SIZING",   # 2026-06-15 ROI 안전 토글 — weekly tier(ops 매사이클 토글 금지)
}
for _mk, _mm in STRATEGY_KEY_META.items():
    _mm["tier"] = "weekly" if _mk in STRATEGY_WEEKLY_TIER_KEYS else "cycle"
del _mk, _mm


# ─── 단일 기본 전략값 (사장 지시 2026-06-09: 프리셋 폐지) ──────────────────────
# 과거 STRATEGY_PRESETS["balanced"] 가 모든 프로필의 기본 베이스였다. 프리셋을 없애고
# 그 균형형 값을 유일한 기본값으로 승격한다. 균형형에 없던 후발 추가 키(NXT/인텔리전스/
# 채권)는 모듈 상수에서 채워 전 튜닝키를 빠짐없이 커버한다(기존 폴백 동작 보존).
STRATEGY_DEFAULTS = {
    "PER_ORDER_BUDGET_RATIO": 0.10, "PER_ORDER_BUDGET_OVERSHOOT": 1.20,
    "MAX_CYCLE_BUDGET_RATIO": 0.25, "MIN_CASH_BUFFER": 1.10,
    "CONSERVATIVE_MDD": 0.05, "CONSERVATIVE_STOCK_RATIO": 0.15,
    "MAX_TRADES_PER_CYCLE": 2, "MAX_ORDER_QTY": 0,
    "MIN_QUANT_SCORE": 6, "MAX_BUY_VOLATILITY_PCT": 0, "RSI_OVERBOUGHT_SKIP": 0, "MIN_ADX_FOR_BUY": 0,
    "REQUIRE_FOREIGN_NET_BUY": False, "MAX_PRICE_EXTENSION_PCT": 0,
    "ENABLE_COST_EDGE_GATE": False, "MIN_NET_EDGE_PCT": 0.8,
    "QIW_RSI": 5, "QIW_MACD": 10, "QIW_ADX": 8, "QIW_VWAP": 8, "QIW_VOL": 8,
    "QIW_MOM": 12, "QIW_CMF": 8, "QIW_FLOW": 12, "QIW_HIGH52": 8, "QIW_LEADLAG": 8, "QIW_VOLUME_SURGE": 8,
    "ENABLE_LEADLAG_SIGNAL": True, "LEADLAG_LOOKBACK_MIN": 30, "LEADLAG_MIN_BUY_SIGNAL": 0.5,
    "ENABLE_LEADLAG_SELL": True, "MAX_PORTFOLIO_CORRELATION": 0.75, "LOSS_STREAK_HALT": 3,
    "SKIP_OPEN_MIN": 5, "SKIP_CLOSE_MIN": 5, "MAX_GAP_UP_PCT": 5.0,
    "REGIME_ADAPTIVE": True, "NEWS_DECAY_HOURS": 24.0,
    "DW_QUANT": 60, "DW_NEWS": 25, "DW_MACRO": 15,
    "DETERMINISTIC_SCORING": True, "MACRO_STOCK_GATE_ENABLED": True,
    "MAX_BUY_NAMES": 8, "POSITION_SIZING_MODE": "risk_weighted",
    "SIZING_TILT_STRENGTH": 0.5, "SIZING_MAX_TILT": 2.0,
    "UNIVERSE_MIN_PRICE": 0, "UNIVERSE_MIN_TURNOVER": 0,
    "UNIVERSE_EXCLUDE_LEVERAGED": True, "SCORECARD_WINDOW_DAYS": 30,
    "ENABLE_SELL_REBALANCE": True, "TAKE_PROFIT_PCT": 12.0, "STOP_LOSS_PCT": 5.0,
    "TRAILING_TAKE_PROFIT_PCT": 0.0, "TRIM_OVER_RATIO": True,
    "ALLOW_DAY_TRADING": True, "MIN_HOLDING_DAYS_FOR_SELL": 0.5,
    "ENABLE_CHEAP_FALLBACK": False, "ALLOW_US_STOCKS": True, "ALLOW_DERIVATIVES": False,
}
# 균형형에 없던 후발 키(NXT/인텔리전스/채권 등)는 모듈 상수에서 보충 → 단일 기본값 완성.
for _k in STRATEGY_TUNABLE_KEYS:
    STRATEGY_DEFAULTS.setdefault(_k, globals().get(_k))
del _k

# ─── Server ──────────────────────────────────────────────────────────────────
APP_HOST = os.getenv("APP_HOST", "0.0.0.0")
APP_PORT = int(os.getenv("APP_PORT", "8500"))
