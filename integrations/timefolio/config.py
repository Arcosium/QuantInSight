"""ArcTrade 전역 설정.

실시간 정보 전이 네트워크(선행-후행) 트레이딩 시스템의 파라미터.
환경변수로 오버라이드 가능한 값은 _env() 로 읽는다.
"""
import os
from pathlib import Path

# BLAS 스레드 상한 — 그랜저 엔진은 프로세스 단위로 병렬화하므로 워커 내부
# BLAS 까지 멀티스레드면 (워커수 × 코어수) 스레드가 생겨 머신 전체가 thrash 한다
# (실측: 16워커×20스레드 = 스캔 35s/사이클 → 1스레드 8워커 = 5s).
# numpy 첫 import 전에 걸려야 하므로 config 최상단에 둔다 (main.py 가 config 를 먼저 import).
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_k, "1")

BASE_DIR = Path(__file__).resolve().parent
_xdg_data = Path(os.environ.get("XDG_DATA_HOME") or (Path.home() / ".local" / "share"))
PRIVATE_DATA_DIR = Path(os.environ.get("ARCTRADE_PRIVATE_DATA_DIR") or
                        (Path.home() / "vault" / "ArcTrade"))
PRIVATE_DATA_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR = Path(os.environ.get("ARCTRADE_DATA_DIR") or (PRIVATE_DATA_DIR / "runtime"))
DATA_DIR.mkdir(parents=True, exist_ok=True)

DB_PATH = DATA_DIR / "bars.db"
UNIVERSE_CSV = DATA_DIR / "universe.csv"
LEADLAG_MAP_JSON = DATA_DIR / "leadlag_map.json"      # 매 사이클 선행 확률 지도 덤프
TRADES_CSV = DATA_DIR / "trades.csv"
POSITIONS_JSON = DATA_DIR / "positions.json"
SIGNALS_JSONL = DATA_DIR / "signals.jsonl"            # 웹 대시보드(ArcTrade) 신호 피드


def _env(key, default, cast=str):
    v = os.environ.get(key)
    if v is None or v == "":
        return default
    try:
        return cast(v)
    except (TypeError, ValueError):
        return default


# ── 유니버스 / 데이터 ────────────────────────────────────────────
# 사장 지시 2026-07-03: KOSPI 200 + KOSDAQ 150(KOSDAQ150 지수급 우량주 — 시총 상위 근사) = 350.
UNIVERSE_KOSPI = _env("LAG_UNIVERSE_KOSPI", 200, int)   # KOSPI 시총 상위 N (우선주 제외)
UNIVERSE_KOSDAQ = _env("LAG_UNIVERSE_KOSDAQ", 150, int) # KOSDAQ 시총 상위 N (스팩·우선주 제외)
UNIVERSE_SIZE = _env("LAG_UNIVERSE_SIZE", UNIVERSE_KOSPI + UNIVERSE_KOSDAQ, int)
# 학습창(그랜저 추정에 쓰는 영업일 수). 예전엔 3일이었는데, 그건 "매분 재스캔"이 1분 예산에
# 들어가는 한계치였을 뿐 통계적 근거가 아니었다. 이제 지도는 하루 1회만 재구축하므로(학습창이
# 오늘을 제외하니 하루 종일 같은 지도다) 창을 길게 가져갈 수 있다. DB에 쌓인 만큼만 쓴다 —
# 지금 3일뿐이어도 동작하고, 매 영업일 1일씩 늘어 RETAIN_DAYS 까지 자동 누적된다.
BUFFER_DAYS = _env("LAG_BUFFER_DAYS", 20, int)        # 학습창: 최근 N 영업일(오늘·홀드아웃 제외)
# **분봉 아카이브는 영구 보존한다** (사장 지시 2026-07-14): 지나간 분봉은 다시 살 수 없고,
# 쌓일수록 그 자체가 자산이다. 0 = 무제한(삭제 안 함). 예전엔 purge_old 가 BUFFER_DAYS(3일)로
# DB를 지워버려서 표본이 영영 3일을 못 넘었다 — 학습창을 늘리려 해도 데이터가 이미 없었다.
# 학습창(BUFFER_DAYS)은 연산량 때문에 제한하지만, 저장은 제한하지 않는다(둘은 완전히 별개다).
# 용량 감각: 349종목 × 1분봉 ≈ 14MB/영업일 → 연 3.5GB.
RETAIN_DAYS = _env("LAG_RETAIN_DAYS", 0, int)         # 0 = 영구 보존. >0 이면 그 일수만 유지.
CRAWL_WORKERS = _env("LAG_CRAWL_WORKERS", 8, int)     # 분당 수집 스레드 수
CRAWL_SEC_MIN, CRAWL_SEC_MAX = 2, 5                   # 매분 02~05초 사이 분산 호출
CRAWL_FAIL_LIMIT = 3                                  # 연속 실패 N회 → 알림 후 다음 분 재시도
REQUEST_TIMEOUT = 7                                   # HTTP 타임아웃(초)
# QuantInSight 의 market_bars 크롤러가 이미 같은 bars.db(심링크)에 매분 분봉을 수집한다.
# 0 이면 Lag 자체 크롤러(data 프로세스)를 띄우지 않아 네이버 이중수집·SQLite 경합을 피한다.
# (bars.db·universe.csv 는 QIS data/ 로 심링크 — main.py run_live 가 이 값으로 data 프로세스를 건너뛴다.)
RUN_CRAWLER = bool(_env("LAG_RUN_CRAWLER", 1, int))

# ── 연산 엔진 ────────────────────────────────────────────────────
# 2026-07-30 정리: 그랜저 전수스캔(GRANGER_LAGS·FDR_Q·OOS_*·MAP_*·TE_*·CONF_THRESHOLD·
# STABLE_CYCLES·ENGINE_PROCS·ENGINE_BUDGET_SEC) 파라미터를 전부 제거했다. 7/28 stat-arb
# 전환으로 스캔 자체가 사라져 아무도 안 읽는 값들이었다(선행 확신 게이트는 LEADLAG_* 가 담당).
ENGINE_INTERVAL_SEC = 60                              # 연산 주기
ENGINE_NICE = _env("LAG_ENGINE_NICE", 5, int)         # 엔진 nice — 공유 호스트 배려
MIN_OBS = 120                                         # 잔차 적합 최소 관측치

# ── 시장중립화 팩터 / 유동성 (2026-07-15) ────────────────────────
# 상관/그랜저 산정 전에 공통 시장·시간대·(선택)섹터 성분을 제거해 '쭉 얻고 쭉 잃는'
# 스트리킹(가짜 동반이동)을 줄인다. 효과는 지도 재구축 후 leadlag_map.json 의
# n_oos_pass / 표본밖 |corr| 상승으로 확인.
NORMALIZE_TOD_VOL = _env("LAG_NORMALIZE_TOD_VOL", 1, int)     # 잔차를 분(HHMM)별 변동성으로 등분산화(개장·마감 U자 완화)
# 0=off. 학습창 평균 일거래대금(원) 하한 미달 종목은 페어 후보 제외. 기본 30억/일 =
# 유니버스 하위 ~9%(저유동 꼬리)만 제거(317/349 통과, 2026-07-15 학습창 실측). 저유동 후행주는
# 체결·상관이 불안정해 가짜 페어를 만든다. 대형주뿐이라 대부분 통과 — 값은 env 로 조정.
MIN_DAILY_TURNOVER_KRW = _env("LAG_MIN_TURNOVER", 3_000_000_000.0, float)
USE_SECTOR_FACTOR = _env("LAG_USE_SECTOR", 1, int)           # 1이면 시장 2팩터에 더해 섹터 동시점평균도 중립화(2026-07-29)
# ── 크립토 로직 이식 (2026-08-07 사장 지시) ──────────────────────
# crypto/ 에서 검증한 것을 KRX 에도 그대로 적용한다. 자세한 근거는 crypto/README.md.
# 1) 팩터를 KOSPI·KOSDAQ 등가중평균 2개 대신 **고유포트폴리오** 상위 k개로.
#    지수 등가중평균은 대형주에 끌려다니고, 크립토처럼 지수가 없는 시장엔 아예 못 쓴다.
#    끄면(0) 기존 2팩터로 돌아간다 — 라이브가 이걸로 돌고 있었으니 되돌릴 길을 남긴다.
USE_EIGEN_FACTORS = _env("LAG_USE_EIGEN", 1, int)
EIGEN_FACTORS = _env("LAG_EIGEN_K", 2, int)                  # 제거할 고유포트폴리오 수
# 2) 섹터 디민의 라벨 출처. "cluster"=가격 상관으로 발견(core/clusters.py) · "csv"=거래소 업종.
#    거래소 업종은 사업 분류이지 '같이 움직이는가'의 분류가 아니다.
SECTOR_SOURCE = _env("LAG_SECTOR_SOURCE", "cluster")
CLUSTER_K = _env("LAG_CLUSTER_K", 40, int)                   # KRX 349종목 → 군당 중앙 ~6종목
CLUSTER_FACTORS = _env("LAG_CLUSTER_FACTORS", 2, int)
CLUSTER_REFRESH_HOURS = _env("LAG_CLUSTER_REFRESH_H", 20.0, float)

# ── 전략: 통계적 차익거래 (Avellaneda-Lee 잔차 평균회귀, 롱온리) ──
# 2026-07-28 리드-랙 → stat-arb 전면 재구성. 각 종목의 시장중립 잔차수익률 ε_i(t)(to_residuals,
# 2팩터 KOSPI/KOSDAQ 제거)를 롤링창에서 누적 X_i=Σε → OU(평균회귀) AR(1) 적합 →
# s = (X - m)/σ_eq (Avellaneda & Lee 2010). s가 크게 음수 = 팩터대비 저평가(과매도) →
# 상승 반전 기대 → 매수(현물 롱온리라 롱 다리만 취함). 참고: [[arctrade-revived-from-filehistory]]
STATARB_WINDOW_MIN = _env("SA_WINDOW_MIN", 180, int)   # 잔차 누적·OU 추정 롤링창(분)
STATARB_S_ENTRY = _env("SA_S_ENTRY", 1.25, float)      # s < -이값 이면 매수 (논문 기본 s̄_bo)
STATARB_S_EXIT = _env("SA_S_EXIT", 0.5, float)         # s > -이값 이면 청산 (논문 s̄_sc, 평균회귀 완료)
STATARB_S_STOP = _env("SA_S_STOP", 3.0, float)         # s < -이값 이면 손절 (낙하칼 방어 — 논문에 없는 롱온리 필수 추가)
STATARB_MAX_HALFLIFE_FRAC = _env("SA_MAX_HL_FRAC", 0.5, float)  # 반감기 < 창×이값 이어야 거래(회귀가 창의 절반보다 빨라야)
STATARB_MIN_HALFLIFE_MIN = _env("SA_MIN_HL_MIN", 3.0, float)    # 반감기 하한(분) — 초단타 마이크로구조 노이즈 제외
STATARB_ADF_T = _env("SA_ADF_T", -2.0, float)          # (b-1) DF t검정 상한: 이보다 작아야 단위근 기각(평균회귀 유의)
STATARB_MIN_OBS = _env("SA_MIN_OBS", 90, int)          # OU 적합 최소 관측치(분)
# 최소 홀딩(분) — 잔차 s가 (다른 종목 움직임으로) 되돌아가도 이 종목 가격이 안 움직였으면
# REVERT/SSTOP 소프트청산을 미룬다. 이걸 안 두면 진입가≈청산가로 즉시 청산해 왕복 슬리피지만
# 남는 −0.1% churn 이 대량 발생한다(2026-07-29). SL/TP/TIMEOUT 리스크 청산은 항상 유효.
STATARB_MIN_HOLD_MIN = _env("SA_MIN_HOLD_MIN", 10.0, float)
# ── 시장 레짐 / 낙하칼 가드 (2026-07-29, 손실 대응) ──
# 역추세(평균회귀)는 시장이 강한 추세일 때 실패한다(추세 지속=역추세 손실 — 실측 개장 첫날 승률 0%).
# 시장팩터 효율비(Kaufman |Σr|/Σ|r|, 1=완전추세·0=완전횡보)가 이 값 초과면 '추세 레짐'→신규 진입 중단.
STATARB_MAX_TREND_ER = _env("SA_MAX_TREND_ER", 0.35, float)
# 창 내 시장 누적수익률이 이보다 더 하락(급락장)이면 롱온리 베타손실 커 진입 중단.
STATARB_MARKET_FALL = _env("SA_MARKET_FALL", 0.015, float)
# 종목 자체가 창 내 이보다 더 급락(절대)했으면 낙하칼(뉴스·부실 추정) → 잔차 저평가여도 매수 제외.
STATARB_MAX_ABS_DROP = _env("SA_MAX_ABS_DROP", 0.04, float)
# 반감기 이 값(분) 초과 종목은 대시보드 s-score 표에 표시하지 않는다(비-회귀·낙하칼류 잡음).
STATARB_MAX_DISPLAY_HALFLIFE = _env("SA_MAX_DISP_HL", 100.0, float)
# ── DART 재무위험 게이트 (2026-07-29, OpenDART 일일 캐시) ──
# 재무부실(자본잠식·극단고부채·영업손실)은 잔차 저평가가 '반등'이 아니라 지속 하락(낙하칼)이라
# 매수 후보에서 뺀다. LLM 은 느려 못 쓰지만 OpenDART 재무제표는 분기단위라 하루 1회 캐시로 충분.
DART_RISK_ENABLED = bool(_env("DART_RISK_ENABLED", 1, int))
DART_MAX_DEBT_RATIO = _env("DART_MAX_DEBT_RATIO", 5.0, float)   # 부채총계/자본총계 이 값 초과=고부채 부실(500%)
DART_REFRESH_HOURS = _env("DART_REFRESH_HOURS", 20.0, float)    # 캐시 이 시간 지나면 백그라운드 재계산
# ── lead-lag 결합 (2026-07-29, 원 ArcTrade 그랜저 부활 → stat-arb 확신 게이트) ──
# 잔차 시차상관으로 종목 간 선행-후행 맵을 만들어, 과매도 후보 B 의 '선행주가 방금 올랐나'로
# 진입 확신을 더한다(교집합). 다른 정보축(종목 간)이라 거래당 엣지를 키워 0.4% 비용을 넘길 여지.
LEADLAG_ENABLED = bool(_env("LL_ENABLED", 1, int))
LEADLAG_TRAIN_DAYS = _env("LL_TRAIN_DAYS", 5, int)     # lead-lag 맵 학습 영업일(+1일 홀드아웃 재현검증)
LEADLAG_LAGS = (1, 2, 3, 5)
LEADLAG_MAX_LEADERS = _env("LL_MAX_LEADERS", 3, int)
LEADLAG_MIN_CORR = _env("LL_MIN_CORR", 0.10, float)   # 학습·홀드아웃 모두 이 |시차상관| 이상이어야 페어 채택
LEADLAG_CONFIRM = _env("LL_CONFIRM", 0.0, float)      # 예측 잔차이동(Σ corr×선행최근) 이 값 초과여야 확정
STATARB_REQUIRE_LEADLAG = bool(_env("SA_REQUIRE_LL", 1, int))  # 1이면 lead-lag 확신 있는 후보만 매수(교집합)
LEADLAG_MAP_CACHE = DATA_DIR / "leadlag_pairs.json"   # (s-score 캐시 LEADLAG_MAP_JSON 과 별개)
# 리드-랙 잔재(대시보드 watch·backtest 가 아직 참조 — 신 전략은 안 씀)
LEADER_JUMP = 0.02
FOLLOWER_FLAT = 0.002
TAKE_PROFIT = _env("SA_TP_PCT", 0.015, float)          # 가격 기반 이익실현 상한(s청산 전 안전판)
STOP_LOSS = _env("SA_SL_PCT", -0.008, float)           # 가격 기반 손절선
ORDER_NOTIONAL = _env("LAG_ORDER_NOTIONAL", 1_000_000, int)  # 1회 주문 금액(원)
MAX_POSITIONS = _env("LAG_MAX_POSITIONS", 5, int)
STRATEGY_TICK_SEC = 5                                 # 청산 감시 주기(초)

# ── 실행 / 브로커 ────────────────────────────────────────────────
LIVE_TRADING = bool(_env("LAG_LIVE_TRADING", 0, int)) # True 면 실주문 브로커 필요 (기본 모의)
PAPER_SLIPPAGE_BPS = 5                                # 페이퍼 체결 슬리피지(bp)

# ── 웹 (ArcTrade) ────────────────────────────────────────────────
WEB_HOST = os.environ.get("ARCTRADE_HOST", "127.0.0.1")  # 터널 뒤 루프백만 (HYFE_IQC 패턴)
WEB_PORT = _env("ARCTRADE_PORT", 8620, int)

# ── Auto_folio (타임폴리오 규칙 모의거래 — ArcTrade 통합) ─────────
# 장부는 저장소 밖의 PRIVATE_DATA_DIR/autofolio_state.json에 두고 시세는 네이버에서 읽는다.
AUTOFOLIO_ENABLED = bool(_env("AUTOFOLIO_ENABLED", 1, int))
AUTOFOLIO_CYCLE_MIN = _env("AUTOFOLIO_CYCLE_MIN", 1, int)           # 신호가 없을 때의 보유 감시(TP/SL) 주기(분)
AUTOFOLIO_POLL_SEC = _env("AUTOFOLIO_POLL_SEC", 1.5, float)         # signals.jsonl 감시 간격(초) — BUY/SELL 뜨면 즉시 사이클
AUTOFOLIO_SESSION_MAX_MIN = _env("AUTOFOLIO_SESSION_MAX_MIN", 30, int)  # 로그인 세션 정기 교체(분) — 신호 사이클은 교체하지 않음
# 대회 왕복 마찰비용(대시보드 누적수익률에서 차감). 대회는 수수료·거래세를 실제로 뗀다
# (사이트 체결내역에 수수료·거래세 라인이 찍힘 — 2026-07-15 사장 확인). 기본 0.4% =
# 수수료 0.1%×2 + 거래세 0.2%. 참고: 거래세는 2025년부터 0.15%(KOSPI 농특세 / KOSDAQ)라
# 실제 왕복은 ~0.35%일 수 있음 → 체결내역의 실제 수수료·거래세로 확정 예정.
# 사이트 원장에는 체결가가 없어 거래내역 가격은 당시 시장가 스냅샷 추정치다.
# 진입 게이트는 이 값이 아니라 AUTOFOLIO_MIN_LEADER_RET_PCT(선행 급등폭)가 담당한다.
AUTOFOLIO_ROUND_TRIP_COST_PCT = _env("AUTOFOLIO_ROUND_TRIP_COST_PCT", 0.4, float)
# 문턱을 엔진 진입선(LEADER_JUMP=2%)과 맞춘다 — 사장 지시 2026-07-09 저녁.
# 4.0 이었을 때 당일 BUY 신호 3건(082740 +2.23% · 039030 +2.01% · 050890 +3.16%)이 전부
# 진입 차단돼 대회 계정에 주문이 한 건도 안 나갔다. 2.0 이면 엔진이 내는 신호는 모두 통과한다.
AUTOFOLIO_MIN_LEADER_RET_PCT = _env("AUTOFOLIO_MIN_LEADER_RET_PCT", 2.0, float)
AUTOFOLIO_INITIAL_CASH = _env("AUTOFOLIO_INITIAL_CASH", 1_000_000_000, int)
AUTOFOLIO_MAX_BUYS = _env("AUTOFOLIO_MAX_BUYS", 1, int)             # 사이클당 최대 신규 매수
AUTOFOLIO_TAKE_PROFIT_PCT = _env("AUTOFOLIO_TP_PCT", 8.0, float)
AUTOFOLIO_STOP_LOSS_PCT = _env("AUTOFOLIO_SL_PCT", 3.0, float)
# ── 2026-07-13 매도 불능 사건 후 추가된 안전장치 ──
# 사이트에 N분 넘게 살아 있는 미체결(작동) 주문은 취소하고 더 깊은 틱으로 재주문한다.
AUTOFOLIO_STALE_ORDER_MIN = _env("AUTOFOLIO_STALE_ORDER_MIN", 3.0, float)
# 열린 BUY 신호가 없는 보유가 이 시간(분)을 넘으면 강제 청산 — 신호창(10분)을 놓친 뒤
# TP/SL 전까지 팔 로직이 없어 단타가 스윙으로 변질되는 것을 막는다. 0 이면 비활성.
AUTOFOLIO_MAX_HOLD_MIN = _env("AUTOFOLIO_MAX_HOLD_MIN", 15.0, float)
# 이 비중(%) 미만의 극소 매수는 하지 않는다 — 폼 정밀도 밑 잔여물(못 파는 7주짜리) 방지.
AUTOFOLIO_MIN_ORDER_WEIGHT_PCT = _env("AUTOFOLIO_MIN_ORDER_WEIGHT_PCT", 0.3, float)
# 1이면 ArcTrade Auto_folio 사이클이 타임폴리오 대회 사이트에 주문을 제출한다.
# 타임폴리오 자체가 모의투자이므로 기본값은 실주문 모드다.
# 자격증명이 없으면 페이퍼로 대체하지 않고 사이클을 실패 처리한다.
AUTOFOLIO_LIVE_ORDERS = bool(_env("AUTOFOLIO_LIVE_ORDERS", 0, int))
AUTOFOLIO_TIMEFOLIO_HEADLESS = bool(_env("AUTOFOLIO_TIMEFOLIO_HEADLESS", 1, int))
AUTOFOLIO_SITE_USERNAME = (os.environ.get("AUTOFOLIO_SITE_USERNAME") or os.environ.get("TIMEFOLIO_USERNAME") or "").strip()
AUTOFOLIO_SITE_PASSWORD = os.environ.get("AUTOFOLIO_SITE_PASSWORD") or os.environ.get("TIMEFOLIO_PASSWORD") or ""

# ── 알림 ─────────────────────────────────────────────────────────
TELEGRAM_BOT_TOKEN = os.environ.get("LAG_TG_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("LAG_TG_CHAT_ID", "")

# ── AI 전략 생성(BYOK) ──────────────────────────────────────────
# 키는 환경변수로만 받는다. 브라우저나 저장소에 평문 키를 저장하지 않는다.
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "auto").strip().lower()
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-5-mini").strip()
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.7-flash").strip()
GEMINI_BASE_URL = os.environ.get(
    "GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai"
).rstrip("/")
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "").strip()
DEEPSEEK_MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-flash").strip()
DEEPSEEK_BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/")
LOCAL_LLM_BASE_URL = os.environ.get("LOCAL_LLM_BASE_URL", "http://127.0.0.1:11434/v1").rstrip("/")
LOCAL_LLM_MODEL = os.environ.get("LOCAL_LLM_MODEL", "qwen3.6-35b-a3b-uncensored").strip()
AI_STRATEGY_REFRESH_SEC = _env("AI_STRATEGY_REFRESH_SEC", 300, int)
