# 🏛️ QuantInSight — 근거를 읽고 판단하는 AI 멀티에셋 퀀트 트레이딩 시스템

## 평가자용 요약

QuantInSight는 LLM의 의견을 그대로 주문으로 보내지 않는다. 10개 에이전트가 리서치와 운용 판단을 나누고, 사이징·예수금·종목 비중·공시 위험은 결정론적 게이트가 통제한다. 결과는 KIS 집계값에 의존하지 않고 자체 체결 원장으로 다시 계산한다.

| 평가 관점 | 확인할 구성 |
|---|---|
| 금융 문제 이해 | 한국·미국 주식, NXT 시간외, 채권·원자재 ETF 슬리브 |
| AI 설계 | 10개 역할 에이전트, 시장 인텔리전스 공유, 주간 피드백 루프 |
| 안전성 | 결정론적 리스크 게이트, 계정별 격리, 암호화 자격증명, 감사 기록 |
| 검증 가능성 | 자체 실거래 원장, walk-forward 백테스트, 에이전트 성과귀인 |
| 제품 완성도 | 웹 대시보드, Android 앱·위젯, 소유자별 거래환경 분리, 재부팅 자동 복구 |

실행 중인 서비스: **https://quantinsight.ai-ve.uk**

> 구 프로젝트명 **ArQuant** — 2026-07-20 QuantInSight 로 개명 (공개 주소·레포지토리 동반 이전)

> **https://quantinsight.ai-ve.uk** — Cloudflare Access 소유자 전용 · KIS 모의매매 기본 · 실전 프로필 보존

**LLM 멀티에이전트 "스웜"이 한국·미국 주식과 채권·원자재 ETF를 자동 매매하는 시스템.**
10명의 AI 에이전트(로컬 LLM 구동)가 매시 정각 시장 감시 사이클을 돌며 뉴스·매크로·퀀트·리스크를
분담 분석하고, 한국투자증권(KIS) OpenAPI 모의환경으로 주문을 낸다. 계정(uid)별로 독립된 매매 루프·전략·
데이터를 가지며 실전투자와 모의투자를 모두 지원한다.

- 웹 대시보드: FastAPI + WebSocket (port **8500**, cloudflared 터널)
- 모바일: Android 네이티브 셸(WebView + 푸시 알림 + 위젯) `arquant_mobile/`
- AI: **로컬 LLM** — OpenAI 호환 서버(llama.cpp 등)에서 `Qwen3.6-35B-A3B` GGUF 구동, 외부 API 키 불필요
- 증권: **KIS OpenAPI** — KRX 정규장 + **NXT 시간외(프리/애프터마켓)** + 미국주식
- 성과 평가: KIS 집계 TR에 의존하지 않는 **자체 실거래 원장(trade ledger)** 기반

---

## 📁 프로젝트 구조

```
QuantInSight/
├── main_swarm.py            # 핵심 오케스트레이터 — 세션별 무한 감시 루프 + 사이클 실행
├── config.py                # 전략 기본값(STRATEGY_DEFAULTS 53키) · 모델 배정 · 세션 스케줄
├── runtime.py               # 라이브 파라미터 오버라이드 — 재시작 없이 반영 (전략 탭)
├── agents/
│   ├── specialists.py       # 에이전트 페르소나 10종 (시스템 프롬프트 + 대화 흐름 규칙)
│   ├── base_agent.py        # LLM 호출 공통 + 토큰/비용 추적 (로컬 모델 단가 0)
│   └── guardrails.py        # 결정론 리스크 게이트 (주문 초안 검증)
├── infra/
│   ├── kis_broker.py        # KIS API — 국내(kr_*)·해외(us_*/_overseas_*) 별도 경로, NXT 주문
│   ├── trade_ledger.py      # 실거래 원장 — KIS 집계 비의존 자산평가 (시드→체결진화→M2M)
│   ├── nxt_blacklist.py     # NXT 거래불가 종목 전 계정 공유 블랙리스트 (자동 학습)
│   ├── asset_sleeves.py     # 멀티에셋 슬리브 엔진 (채권/원자재 ETF 공통 스펙)
│   ├── auth_store.py        # 소유자 프로필 · Fernet 암호화 자격증명 · 감사로그
│   ├── cloudflare_access.py # Access JWT 검증
│   ├── user_context.py      # per-uid 컨텍스트 레지스트리 (브로커·스웜 격리)
│   ├── user_paths.py        # per-uid 데이터 경로 (data/<uid>/...)
│   ├── market_intel.py      # ADMIN 인텔리전스 공유 (매크로/뉴스 1회 생산 → 전 계정 공유)
│   ├── cycle_store.py       # 사이클 영속 (data/cycles.db)
│   └── weekly_review.py     # 토요일 주간 피드백 루프 (운용지원실장 + 현재 설정 백테스트)
├── tools/
│   ├── market_data.py       # 일봉/수급/분봉/지수/환율(5분 크롤) 수집
│   ├── quant_score.py       # 결정론 퀀트 점수 엔진 (지표 가중치 QIW_* · 차원 가중치 DW_*)
│   ├── universe_screen.py   # 유니버스 스크리닝 (레버리지/저가/거래대금 배제)
│   ├── global_search.py     # Hermes 웹 검색 리서치 (deep research)
│   ├── agent_scorecard.py   # 에이전트 성과귀인 (IC·알파/베타)
│   └── gen_manual.js        # 사용설명서.docx 생성기 (node)
├── server/
│   ├── app.py               # FastAPI — 68 라우트 + WebSocket + 부팅 자동재개(.running)
│   └── static/index.html    # 웹 대시보드 (단일 파일 SPA)
├── backtest/                # 백테스트 하네스 (현재 설정 파라미터 평가)
├── arquant_mobile/          # Android 앱 (Kotlin — WebView 셸 + WsRelayService 알림 + 위젯)
├── tests/                   # pytest 168 파일 / 850+ 케이스 — 반드시 python3.11 로 실행
└── data/                    # 런타임 데이터 (대부분 gitignore) — data/<uid>/ 가 계정별 영역
```

---

## 🏛️ AI 에이전트 조직 (10 페르소나 — 사이드바 3구역)

**FRONT (리서치·운용전략)** / **MIDDLE (리스크 통제·사후관리)** / **BACK (실행·인프라 지원)**

| 페르소나 | 역할(코드) | 티어 | 담당 |
|---|---|---|---|
| **주식운용실장** | chief_orchestrator | pro | 총괄 — 매크로/퀀트/뉴스 종합, 2패스 종목 선정(후보→최종), 사장 지시 처리 |
| **글로벌리서치팀장** | macro_analyst | pro | 거시경제 분석 → **4분할 자산배분 권고** (주식/채권/원자재/현금, 합 100%) |
| **계량분석팀장** | quant_analyst | flash | 후보 종목 정량 해설 (점수 자체는 결정론 엔진이 산출) |
| **마켓센티먼트팀장** | news_analyst | flash | 뉴스 감성(-1.0~+1.0)·이벤트 분석, KR/US/공통 분류 |
| **채권운용실장** | bond_manager | pro | 매크로 채권 비중 권고를 채권 ETF 매매로 실현 (듀레이션·종목 재량) |
| **원자재운용실장** | commodity_manager | pro | 원자재 비중 권고를 원유/금/농산물 ETF 매매로 실현 |
| **리스크관리실장** | (결정론+DART 재심) | — | 리스크 게이트(파이썬) + 매수 2차 재심(DART 공시·재무) |
| **사후관리실장** | post_manager | pro | 보유 종목 매도/유지/익절·손절 **최종 판단** (주식 + 슬리브 매도 종합) |
| **포트폴리오기획팀장** | fund_planner | flash | 매수 직후 thesis(목표가/손절가/계획 보유기간) 기록 → 매도 직전 강력 권고(거부권 없음) |
| **프롭트레이딩팀장** | trader | flash | 체결 결과 보고 (결정론 템플릿 — 사실만) |
| **운용지원실장** | ops_support | pro | 사이클 진단 + **전략 변경안 검증 대기** (WHAT-not-HOW, 코드 수정 불가) |

- 두 **티어는 같은 로컬 모델**을 추론(thinking) ON/OFF로 굴린다: '결정' 에이전트는
  **thinking ON = pro**, 분석·리서치·도구 에이전트는 **thinking OFF = flash**.
- 매크로 **웹 검색 단계**(macro_researcher)는 Hermes 도구 호출(tool-calling) 안정성을 위해
  **thinking OFF 유지**.
- 한글 페르소나 이름이 곧 @멘션 라우팅 키다 (`@사후관리실장 보유 점검`).
- 대화 흐름 규칙: 팀 회의처럼 누적 — 직전 발언 재서술 금지, 자기 관점만 추가.
- 모델 오버라이드: `data/admin_config.json`의 `model_overrides` (ADMIN 탭, 재시작 반영).

---

## ⚙️ 의사결정 사이클

**트리거**: ① 매시 **정각(:00) 벽시계 앵커** (재시작 불변) ② 장 개장 순간 ③ ▶ 실행 직후 1회.
직전 사이클 5분 이내·휴장일(거래량 검증)·예수금 5,000원 미만이면 건너뛴다.

```
뉴스 수집(10분 크롤·큐레이터 선별) ─┐
글로벌 지수·보유종목·DART 공시      ├→ 마켓센티먼트(감성) → 글로벌리서치(매크로·4분할 배분)
                                   │
   ┌── 주식 트랙: 주식운용실장 PASS1(후보 선정) → 유니버스 스크리닝 → 퀀트 데이터 수집
   │     → 결정론 퀀트점수(MIN_QUANT_SCORE 게이트) → PASS2(최종 매수 결정) → 리스크 사이징
   ├── 채권 슬리브: 채권운용실장 — 금리 전망 + 목표비중 추종 → ETF 매수/매도 결정
   ├── 원자재 슬리브: 원자재운용실장 — 인플레·달러·지정학 → ETF 매수/매도 결정
   └── 매도 트랙: 슬리브 매니저 '매도 제안' → 사후관리실장이 thesis 권고와 함께 종합·확정
                                   │
       리스크 게이트(결정론) + DART 공시·재무 재심(매수만) → 주문 실행(KIS) →
       체결 확인(즉시/5분 폴링) → 실거래 원장 반영 → 프롭트레이딩 보고 → 사이클 리포트
```

**ADMIN 인텔리전스 공유** (`SHARE_MARKET_INTELLIGENCE`): 시장 전역 분석(매크로·뉴스 분류)은
ADMIN 계정이 1회 생산하고 비관리자 계정은 공유받는다(producer-first wait + 타임아웃 폴백) — LLM 비용 절감.

**리스크 게이트 (결정론, LLM 미사용)**: 단일 종목 비중 한도 · 사이클 누적 예산 ·
예수금 안전마진 · 계좌 손익 차단선(CONSERVATIVE_MDD) · 비정상 수량 차단 · KR 매도는
**매도가능수량(ord_psbl_qty)** 기준 사이징(미체결 잠금 물량 거부 예방). 슬리브 ETF
리밸런싱 매수는 주식 예산 게이트 면제(`*_PER_CYCLE_RATIO` 별도 예산). 매도는 리스크 축소
행위라 통과. 매수는 추가로 DART 공시·직전연도 재무 재심(관리종목·거래정지·횡령·연속적자
적신호 반려)을 거친다.

---

## 🧺 멀티에셋 슬리브 (채권·원자재 ETF — 기본 ON)

공통 엔진 `infra/asset_sleeves.py`(`SleeveSpec`)가 두 슬리브를 동일 구조로 굴린다:
글로벌리서치팀장의 "채권 X% / 원자재 W%" 권고 → 목표비중 추종(±3% 데드존) →
매니저가 종목·페이스 재량 결정 → 매수는 즉시, **매도는 제안만**(사후관리실장이 확정).

| 슬리브 | KR ETF 풀 | US ETF 풀 | 비중 상한 | 사이클 예산 |
|---|---|---|---|---|
| 채권 (bond) | 153130 단기채 · 114260 국고3y · 148070 국고10y · 357870/459580 CD금리 · 273130/451540 종합채 · 458250 미국채30y스트립(H) | SHY · IEF · TLT · LQD · HYG · TIP | 40% | 15% |
| 원자재 (commodity) | 132030 금(H) · 261220 WTI원유(H) · 137610 농산물(H) | GLD · USO · DBA · DBC | 20% | 10% |

---

## ⏰ 세션 & NXT 시간외 (KST)

| 세션 | 시간 | 거래소 | 동작 |
|---|---|---|---|
| KR_PRE_MARKET | 08:00–08:50 | **NXT** | 시간외 프리마켓 — 지정가(슬리피지 밴드 0.5%) |
| KR_TRADING | 09:00–15:30 | KRX | 국내 정규장 |
| KR_CLOSE_REVIEW | 15:30–16:30 | — | 장 마감 리뷰 (매매 없음) |
| KR_AFTER_MARKET | 15:50–20:00 | **NXT** | 시간외 애프터마켓 — 지정가 |
| US_TRADING | 22:30–05:00 | 미국 | 해외주식 (야간) |
| OFF_HOURS | 그 외 | — | 뉴스 수집만 |

- NXT 주문은 신 TR(TTTC0012U/0011U + `EXCG_ID_DVSN_CD`) 사용, 정규장은 KRX 보존.
  모의투자는 NXT 미지원이라 하드 차단.
- **NXT 블랙리스트** (`data/nxt_untradable.json`, 전 계정 공유): "NXT 상장종목인지
  확인하세요" 거부가 나온 종목을 자동 학습 → 이후 시간외 세션에서 매수 후보 제외 +
  주문 시도 자체 보류(정규장 재시도).
- 휴장일은 하드코딩 목록 없이 개장 후 **거래량 검증**으로 판정.

---

## 📒 실거래 원장 — KIS 집계 비의존 자산평가 (`infra/trade_ledger.py`)

KIS 통합총자산 TR 3종은 서로 불일치하고(자기모순), USD 결제 과도기(T+2)엔 매도대금이
어느 예수금 필드에도 안 잡혀 자산곡선·수익률이 환각을 일으켰다. QuantInSight는 KIS를
**체결 사실·종목 평단·시세**에만 신뢰하고 평가는 자체 원장으로 한다:

1. **시드(1회)** — KIS 보유(qty/평단) + KRW 예수금(D+2) + USD 외화예수금 +
   **미결제 USD 매도대금**(해외 체결내역 TTTS3035R 기반 — 순매수일은 통합증거금으로 KRW에서
   차감되므로 일별 양수 net만 가산)으로 초기 원장 구성.
2. **체결 진화** — 이후엔 우리 체결만으로 현금·포지션이 움직인다 (US 수수료 매수·매도 각 0.3%).
3. **M2M 평가** — 자체 시세(시세 결손 시 직전가 carry-forward) × 5분 크롤 환율 →
   equity 포인트의 `ledger_eval`. **원장 포인트가 있으면 곡선·KPI가 원장 시리즈만 사용.**
4. **대조·재시드** — 30분마다 KIS 보유수량과 대조해 괴리 경고. 입출금·수동매매 후엔
   `POST /api/ledger/reseed`.

수익률 KPI 8종: 상단 4칸(전체/오늘/주/월)은 원장 **평가금액 변동**(미실현 포함),
**누적수익**은 체결 실현손익(비용 반영 — US 0.3%/leg, KR 0%, KIS 실평단 권위) 별도 집계.
나머지는 MDD·승률·현재 평가액(모의계정은 잔고 숨김).

---

## 🎛️ 전략 파라미터 (53키 — 프리셋 폐지, 단일 설정값)

2026-06-09부터 빌트인 프리셋(방어형~초공격형)을 폐지하고 **단일 STRATEGY_DEFAULTS**
(균형형 기반) + 대시보드 '전략' 탭의 그룹별 편집 패널로 일원화했다. 변경은 `runtime.py`
라이브 오버라이드로 **재시작 없이** 계정별 반영된다.

| 그룹 | 대표 키 |
|---|---|
| 사이징 | PER_ORDER_BUDGET_RATIO(10%) · MAX_CYCLE_BUDGET_RATIO(25%) · MIN_CASH_BUFFER(1.10) |
| 리스크 | CONSERVATIVE_MDD(−5% 차단) · CONSERVATIVE_STOCK_RATIO(15%) · MAX_TRADES_PER_CYCLE(2) |
| 종목 필터 | MIN_QUANT_SCORE(6) · RSI_OVERBOUGHT_SKIP · MIN_ADX_FOR_BUY · REQUIRE_FOREIGN_NET_BUY |
| 퀀트 지표 가중치 | QIW_RSI/MACD/ADX/VWAP/VOL/MOM/CMF/FLOW/HIGH52 (음수 허용) |
| 점수 차원 | DW_QUANT(60)/DW_NEWS(25)/DW_MACRO(15) · DETERMINISTIC_SCORING(ON) |
| 레짐 대응 | TAKE_PROFIT_PCT(12%) · STOP_LOSS_PCT(5%) · ALLOW_DAY_TRADING(ON) · MACRO_STOCK_GATE |
| 제도권 파이프라인 | MAX_BUY_NAMES(8) · POSITION_SIZING_MODE(risk_weighted) · UNIVERSE_* · SCORECARD_WINDOW |
| NXT 시간외 | ENABLE_NXT_EXTENDED_HOURS/PRE/AFTER(ON) · EXT_HOURS_LIMIT_SLIPPAGE_PCT(0.5%) |
| 채권/원자재 ETF | ENABLE_BOND_ETF(ON)·BOND_*  /  ENABLE_COMMODITY_ETF(ON)·COMMODITY_* |
| 비용 | SHARE_MARKET_INTELLIGENCE(ON) · SHARE_PRODUCER_WAIT_SEC |

- 정책 봉인 키(운용지원실장 자율 조정 금지): `ALLOW_US_STOCKS` `ALLOW_DERIVATIVES`
  `ENABLE_CHEAP_FALLBACK`(영구 OFF) `DETERMINISTIC_SCORING` — 사장만 변경.
- 토요일 **주간 피드백 루프**: 운용지원실장이 지난 7일을 진단하고 현재 설정으로
  백테스트를 돌려 보고. 정책 변경 제안은 '전략 탭 승인 대기 박스'에서 명시 승인/거부.

---

## 접근 제어와 거래환경

Cloudflare Access가 소유자 이메일 한 개만 허용한다. 서버도 Access JWT의 서명·발급자·대상·만료·이메일을 검증한다. 자체 로그인·가입·복구·회원 관리 API는 제거했다. 단순 이메일 헤더나 과거 세션으로는 접근할 수 없다. 로그아웃은 `/cdn-cgi/access/logout`을 사용한다.

Access 설정은 `/home/arcosium/vault/QuantInSight/cloudflare_access.json`에 둔다. `team_domain`, `audience`, `owner_email`, `owner_uid`가 필요하다. 값이 없으면 접근을 거부한다. 원본 서버는 `127.0.0.1:8500`에서만 받는다.

`QIS_PAPER_ONLY=1`이 기본이다. 실전 프로필을 시작할 수 없고 브로커도 실계좌 주문·정정·취소를 거부한다. 실전 시세 조회는 NXT 수집에 사용한다. 모의투자는 KRX 정규장과 지원되는 미국 모의 주문을 사용하며 NXT 주문은 보류한다. 나중에 실전 운용을 시작할 때는 서버 환경에서 `QIS_PAPER_ONLY=0`으로 바꾸고 실전 프로필을 명시적으로 시작한다.

소유자의 KIS 실전·모의 프로필만 유지한다. 프로필 전환은 조회·제어 대상을 바꾸는 동작이며 자동으로 매매를 시작하지 않는다. 자격증명은 vault의 DB에 암호화해 보관한다.

### 의사결정과 위험 예산

실전과 모의 계정은 같은 정량 운용안과 위원회 심의, 주문 수량 계산, 매도 규칙을 사용한다. `QIS_SYSTEMATIC_POLICY_ENABLED=1`이 기본이며 계정의 실전·모의 여부로 전략을 바꾸지 않는다. 주문 전송과 체결 확인은 선택한 KIS 계정에서 수행한다. 실전 실행 잠금은 이 전략 설정과 별개다.

리서치, 운용위원회, 리스크관리, 집행, 사후 평가의 역할은 유지한다. 계량 점수가 없는 종목은 매수에서 제외하고 최종 심의 종목을 살 수 없으면 현금을 유지한다. 저가 대체 종목을 자동으로 사지 않는다.

주문 직전에는 기존 보유분과 같은 사이클의 주문을 합산한다. 주식 한 종목은 총평가의 10%, 손절 거리와 비용으로 추정한 위험은 종목당 0.5%·전체 3% 이내로 제한한다. 변동성이 크면 수량을 줄이며 현금은 총평가의 10%를 남긴다. 슬리브 ETF는 기존 자산배분 한도와 공통 현금 한도를 적용한다. 이 값은 손실 보장선이 아니며 갭과 체결 지연으로 실제 손실이 더 클 수 있다.

운용지원의 자동 제안은 당시 설정과 함께 `data/<uid>/research/`에 고정 저장한다. 토요일 리뷰도 검증 없이 설정을 자동 변경하지 않는다. 사용자 직접 설정은 계속 가능하다. 5거래일 경험 수익 진입 규칙은 과거 비교에서 채택 근거를 확보하지 못해 관찰 기록만 남긴다. 기관 전체 전략의 수익 개선이 입증된 상태는 아니다.

검증 명령: `python -m tools.evaluate_entry_policy --output /home/arcosium/vault/QuantInSight/entry-policy-replay.json`. 현재 유니버스·캐시 가격의 자료 한계를 포함하는 진입 규칙 단독 진단이다.

---

## 🖥️ 웹 대시보드 & 📱 모바일

단일 SPA(`server/static/index.html`) — 본문 6탭 + 사이드바 에이전트 조직도(3구역):

| 탭 | 내용 |
|---|---|
| 📊 대시보드 | 상태 카드(세션·뉴스·다음 사이클·체결 등) + 에이전트 통신 로그 + @멘션 입력 |
| 💰 수익률 | KPI 8종(평가변동 4칸 + MDD·승률·누적 실현수익·현재 평가액) + 원장 곡선·벤치마크 |
| 💼 보유종목 | 잔고·보유 목록(KR/US 배지) + 전체 거래 내역(상세·FIFO·실현손익) |
| 📰 뉴스 | KR/US/공통 분류 피드 |
| ⚙️ 전략 | 53키 그룹 편집 + 운용지원 ON/OFF + (토요일) 정책 변경 승인 박스 |
| 🛡️ ADMIN | 전역 설정·피드백 답글 (관리자 전용) |

모바일(`arquant_mobile/`): WebView 셸이라 **서버 화면 수정은 재설치 없이 반영**.
네이티브 = Cloudflare Access 인증 WebView + Access 쿠키를 사용하는 **WsRelayService**(포그라운드
서비스로 백그라운드 푸시 4종: 체결 신청/체결 완료/사이클 완료/장 마감 — 프로필별 필터) +
홈 위젯(보유·총평가·수익률). 빌드: `./build_apk.sh` (QEMU docker, ~20분 — 네이티브 변경 시에만).

---

## 🔌 API 엔드포인트

| 그룹 | 라우트 |
|---|---|
| 인증·계정 | `/api/auth_status` `/api/me` |
| 프로필 | `/api/profile/credentials` `/api/profile/directives`(GET/POST/DELETE) `/api/notif_settings` `/api/cost_mode` `/api/feedback`(+`/seen`) |
| 운용 | `/api/start` `/api/stop` `/api/status` `/api/ceo`(@멘션) `/api/agents` `/api/events`(+`/clear`) `/api/history` `/api/cycles`(+`/{id}`) `/api/alerts` `/api/metrics` |
| 성과·데이터 | `/api/balance` `/api/equity` `/api/performance` `/api/trades`(+`/clear`) **`/api/ledger/reseed`** `/api/scorecard` `/api/benchmark` `/api/news`(+`/clear`) `/api/dart` `/api/price/kr/{code}` `/api/price/us/{ticker}` `/api/rank/volume` |
| 전략·운용지원 | `/api/strategy`(GET/POST) `/api/ops_feedback` `/api/ops_history` `/api/policy_changes/*` |
| ADMIN | `/api/admin/config` `/api/admin/feedback`(+`/reply`) `/api/coresight/*` |
| 실시간 | `WS /ws` (대시보드 피드 + 모바일 알림 — per-connection 필터) |

---

## 🚀 실행 · 배포 · 테스트

```bash
# 배포 = 서비스 재시작 (코드 변경 반영)
sudo systemctl restart arquant.service
sudo systemctl status arquant.service      # 헬스 확인 (port 8500)

# 수동/로컬
./start_server.sh        # uvicorn + cloudflared 터널 (포트 정리·헬스 확인 포함)
./supervise.sh           # 20초 주기 watchdog (서버·터널 자동 재기동)

# 테스트 — 반드시 python3.11 (기본 python은 argon2 import에서 실패)
python3.11 -m pytest                 # 전체 (tests/test_*.py, 850+ 케이스)
python3.11 -m pytest tests/test_x.py

# Android APK (네이티브 변경 시에만)
./build_apk.sh
```

- `config.py`의 `LIVE_TRADING=True`면 실주문이 나간다. 처음엔 모의투자 권장.
- 외부 도구가 주기적으로 `git add -A` + `Backup:` 커밋/푸시를 수행한다.
- **pytest 안전 가드**: 운영 호스트에서 테스트가 라이브 데이터(`data/<uid>/`)를
  삭제·오염하지 않도록 운영코드(`_rmtree_pytest_guarded`, trade_ledger/nxt_blacklist
  쓰기 가드)와 테스트 격리(tmp monkeypatch)가 이중으로 걸려 있다.

---

## 💵 모델 · 비용

LLM은 **로컬 OpenAI 호환 서버**(llama.cpp 등)에서 GGUF 모델을 구동한다 — 외부 클라우드 API·키
없이 동작하므로 **호출당 외부 API 비용은 0**이다. 모든 LLM 호출은 `LOCAL_LLM_BASE_URL`로 간다.

| 티어 | 모델 (기본값) | 추론(thinking) | 사용처 |
|---|---|---|---|
| flash | `Qwen3.6-35B-A3B` (GGUF Q8_0) | OFF | 뉴스·퀀트 해설·트레이더·thesis·매크로 웹검색(도구 호출) |
| pro | 〃 (`+thinking` 변종) | ON | 오케스트레이터·매크로 결정·사후관리·슬리브 매니저·운용지원 |

- 서버 주소·모델은 env로 지정: `LOCAL_LLM_BASE_URL`(기본 `http://127.0.0.1:8080/v1`) ·
  `LOCAL_LLM_MODEL`(기본 `Qwen3.6-35B-A3B-...-Q8_0.gguf`). thinking ON/OFF는 `+thinking` 가상
  접미사로 표현하고 전송 직전 `chat_template_kwargs.enable_thinking` / `reasoning.enabled` 로
  변환한다(`infra/local_llm_client.py` — 파일명은 유지, 내용은 로컬 클라이언트).
- 비용 추적 인프라(토큰 롤업 `data/api_cost_rollup.json` · 대시보드 💵 배지)는 유지되나 로컬
  모델 단가가 0이라 표시는 항상 $0.00 이다. ADMIN 탭에서 역할별 모델·추론 토글 오버라이드 가능(재시작 반영).
- DART 키는 서버 소유 단일 env(전 계정 공통, 사용자 입력 불필요).

---

## ⚠️ 알려진 함정 (Gotchas)

- **KR/US 비대칭**: KIS 국내(`kr_*`)와 해외(`us_*`/`_overseas_*`)는 완전 별개 경로.
  "KR엔 되는데 US엔 안 됨"이면 입력→평가→주문→실행 전 단계에서 `_overseas_*` 누락을 점검.
- **KIS 결제 과도기 글리치**: 잔고 필드가 순간 빈값/0으로 읽힐 수 있다(빈 보유·cash 0·
  D1/D2 0). D0(dnca) 예수금은 부풀려진 값이라 평가에 절대 사용 금지 — D+2 기준.
- **T+2 미결제 USD**: 결제 전 매도대금은 KIS 예수금 TR에 안 잡힌다 — 원장이 체결내역
  기반으로 보정(위 '실거래 원장' 참고).
- **모의서버 한계**: 해외 TR(체결내역·미체결조회·NXT) 다수 미지원, 해외 평가·환율 garbage.
- **실주문 누락 금지**: fail-closed보다 다중 폴백 전송 우선. 매도 보류 시엔 반드시 사유 발화.
- `data/`는 대부분 gitignore — git에 있다고 가정하지 말 것.

---

## 정량 모의 운용

모의 계정에서는 리서치 부서가 60·20일 상대강도와 변동성으로 후보·목표 비중을 정합니다.
최근 5거래일을 제외한 상대강도를 사용하고 유동성, 추세, 급등 추격 여부를 확인합니다.
후보는 5거래일 유지하며 뉴스가 없다는 이유만으로 분석을 생략하지 않습니다.
위원회가 공시·뉴스 위험을 심의하고 리스크 부서가 기존 보유를 포함해 주문량을 제한합니다.

신규 주식 진입은 총평가액 40% 한도에서 시험합니다. 시장 추세가 약하면 목표 비중을 더 낮춥니다.
운용안에서 빠진 종목과 초과 비중은 사후관리 단계에 매도·축소 제안으로 전달합니다.
고정 수익률에 도달했다는 이유만으로 전량 익절하는 규칙은 이 모의 실험에서 끄고 손절·기존 위험 관리는 유지합니다.
이 정책이 실전 계정을 자동으로 활성화하지는 않습니다.

국내 일봉은 공용 보관소 `KRX/daily_cnn`, 미국 일봉은 `USA/daily_policy`를 사용합니다.
현재 미국 후보 풀은 40개 종목으로 고정했습니다. 자료가 오래됐거나 90% 이상을 확인하지 못하면 신규 진입을 보류합니다.
운용안은 `data/<uid>/stock_policy/`에 기록하며 원본 파일은 vault에 있습니다.
정량 규칙의 과거 시험은 전체 위원회·주문 체결의 성과와 구분해 화면에 표시합니다.

새 국내 체결부터 원장에 매수 0.05%, 매도 0.30%의 비용 가정을 반영합니다.
부분 매도 손익에도 해당 수량의 매수 비용을 배분합니다. 이는 정확한 증권사 요율이 아니라
순수익 평가를 위한 추정치이며 과거 원장은 다시 쓰지 않습니다.

## 📌 운용 범위

현재 배포 설정은 모의매매입니다. 향후 실전 연결 기능은 남겨 두되 지금은 실제 주문을 차단합니다.
과거 시험의 양수 수익률이 미래 수익을 보장하지 않습니다. AI 분석과 시세에는 오류·지연·중단이 발생할 수 있습니다.

*최종 업데이트: 2026-06-23 (branch `local-llm-no-api-keys` — 로컬 LLM 전환)*
