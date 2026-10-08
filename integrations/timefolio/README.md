# ArcTrade

ArcTrade는 KRX·NXT·미국 주식·크립토를 조회하고 백테스트하는 연구 도구입니다. 기존 자연어 한국 주식 전략은 Auto-folio에 연결합니다. OpenAI, Gemini, DeepSeek 또는 로컬 OpenAI 호환 모델 가운데 하나를 골라 쓸 수 있습니다. 재무 지표와 수급, 이동평균, RSI, MACD, 캔들, 공시 조건을 한 전략 안에서 함께 다룹니다.

> 이 프로젝트는 투자 자문이 아닙니다. 공개 배포본은 `AUTOFOLIO_LIVE_ORDERS=0`이 기본이며, 전략을 활성화해도 먼저 로컬 모의 장부로 실행됩니다.

## 주요 기능

- 자연어 아이디어를 실행 가능한 전략 JSON으로 변환
- AI 전략 생성, 백테스트, 전략 활성화를 각각 확인할 수 있는 3개 버튼
- AND, OR, 괄호를 이용한 복합 진입·청산 조건과 여러 자산 배분 방식
- 거래비용을 반영한 일별 백테스트와 KOSPI·Buy & Hold 비교
- 활성 전략의 운용 시간, 수익률, 신호 수를 실시간 기록
- 전략 교체 전 경고창과 기존 실험 이력 자동 보관
- 전략을 바꾸면 전략 수익률·신호 로그·모의 포지션만 초기화
- Auto-folio의 타임폴리오 가격, 보유 내역, 주문 원장은 그대로 보존
- 타임폴리오 계정과 모든 API 키를 환경변수로 주입

과매도 s-score 화면은 제거했습니다. 해당 연구 코드는 재현을 위해 남겨 두었지만 새 대시보드와 AI 전략 운용 흐름에서는 쓰지 않습니다.

## 네 시장 조회와 검증

대시보드의 **마켓** 탭에서 시장과 종목을 고른다. KRX·NXT·크립토는 공통 분봉 폴더를 읽고 미국 주식은 수정 일봉을 조회한다. 미국 티커는 직접 입력할 수도 있다. 마켓 탭은 가격 조회와 이동평균 전략 백테스트를 제공하며 기존 AI 전략·타임폴리오 주문은 국내 주식 경로를 유지한다.

백테스트는 완성된 이전 봉으로 신호를 만들고 다음 봉에서 체결한다. KRX 과거 자료는 종가만 있으므로 다음 종가를 사용한다. 왕복비용을 차감한 전략 수익, 단순 보유 수익, 최대 낙폭과 후반 30% 수익을 표시한다. 비용은 사용자가 정하는 가정값이며 매개변수를 반복해 고르면 후반 구간도 독립 검증이 아니다.

공통 경로는 `minute_data -> /home/arcosium/vault/CryptoBars/data`다. `ARCTRADE_MINUTE_DATA_DIR`로 별도 경로를 지정할 수 있다. KRX·NXT 저장소와 기존 크립토 Parquet를 같은 루트에서 관리한다.

## 빠른 시작

Python 3.11 이상을 권장합니다.

```bash
git clone https://github.com/Arcosium/ArcTrade.git
cd ArcTrade
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
playwright install chromium
mkdir -p /home/arcosium/vault/ArcTrade
cp -n .env.example /home/arcosium/vault/ArcTrade/.env
ln -s /home/arcosium/vault/ArcTrade/.env .env
```

`.env`에 사용할 AI 제공사의 키 하나만 입력합니다. 여러 키를 넣고 `LLM_PROVIDER=auto`로 두면 OpenAI, Gemini, DeepSeek 순서로 사용 가능한 제공사를 고릅니다. 키 없이 로컬 모델을 쓰려면 `LLM_PROVIDER=local`과 `LOCAL_LLM_BASE_URL`을 설정합니다.

환경변수를 적용한 뒤 서버를 실행합니다.

```bash
set -a
source .env
set +a
python -m web.app
```

브라우저에서 `http://127.0.0.1:8620`을 엽니다.

## AI 제공사 설정

키 값은 브라우저 입력창이나 저장소에 저장하지 않습니다. 서버 프로세스의 환경변수에서만 읽습니다.

```dotenv
OPENAI_API_KEY=""
GEMINI_API_KEY=""
DEEPSEEK_API_KEY=""
```

각 제공사의 모델과 기본 URL도 바꿀 수 있습니다. Gemini와 DeepSeek는 공식 OpenAI 호환 Chat Completions 엔드포인트를 사용합니다. 사내 게이트웨이나 프록시가 있다면 해당 `*_BASE_URL`을 지정하면 됩니다.

## 타임폴리오 연결

타임폴리오 계정은 소스 코드에 넣지 않습니다.

```dotenv
TIMEFOLIO_USERNAME=""
TIMEFOLIO_PASSWORD=""
AUTOFOLIO_LIVE_ORDERS=0
```

로그인이 확인되고 주문 위험을 이해한 뒤에만 `AUTOFOLIO_LIVE_ORDERS=1`로 바꾸십시오. 이 값이 0이면 Auto-folio는 로컬 모의 장부를 사용합니다. 계정 정보는 런타임 저장소에서 암호화되며 Git 추적 대상이 아닙니다.

## 전략 연구 흐름

전략 연구실에서 아이디어를 적고 `AI로 전략 생성`을 누릅니다. 생성된 재무 필터, 매수·매도 조건, 검증 기간, 배분 방식을 확인한 다음 `백테스트`를 실행합니다. 백테스트를 마친 전략만 `이 전략으로 매매하기`로 활성화할 수 있습니다.

이미 다른 전략이 운용 중이면 ArcTrade가 수익률과 신호 로그를 초기화할지 묻습니다. 확인하면 기존 기록을 `strategy_archives`에 보관하고 새 전략을 0%부터 추적합니다. Auto-folio의 실제 계정 장부는 이 과정에 포함되지 않습니다.

브라우저마다 무작위 프로필 ID를 만들기 때문에 실험 기록이 서로 섞이지 않습니다. 프로필 ID에는 이름이나 이메일을 넣지 않습니다. 기록 파일은 기본적으로 `~/.local/share/arctrade/profiles`에 저장됩니다.

## 데이터와 비밀 관리

기본 런타임 경로는 저장소 밖인 `~/.local/share/arctrade`입니다. 다음 파일은 커밋하지 않습니다.

- `.env`와 API 키, 타임폴리오 아이디·비밀번호
- 사용자별 실험 기록과 Auto-folio 상태
- SQLite DB, 신호 로그, 거래 로그, 브라우저 세션
- 시세 캐시와 Parquet 파일

경로를 바꾸려면 `ARCTRADE_PRIVATE_DATA_DIR`, `ARCTRADE_DATA_DIR`, `ARCTRADE_MARKET_DATA_DIR`, `ARCTRADE_QUANT_CACHE_DIR`을 지정합니다.

## 지원하는 전략 표현

재무 필터에는 PER, PBR, ROE, 부채비율, 성장률 등을 쓸 수 있습니다. 매수·매도 조건에는 이동평균선, RSI, MACD, 스토캐스틱, 거래대금, 외국인·기관·개인 순매수, 캔들 패턴, DART 공시 이벤트를 조합할 수 있습니다.

포트폴리오 방식은 동일비중, 시가총액, 모멘텀, 위험균형, 역변동성, Kelly 근사, 최소분산 근사, 최대 Sharpe 근사, 동적 배분, 단일 종목 집중을 지원합니다. 일부 방식은 휴리스틱 근사이므로 결과를 실제 운용 전에 별도로 검증해야 합니다.

## KRX CNN 모의 연구

`KRX CNN` 화면에서 60일 OHLCV 차트로 학습한 소형 CNN을 확인합니다. 학습·검증 경계에서
미래 수익률 라벨이 겹치는 관측치를 제외합니다. 시험 구간의 수익률로 모델이나 시드를 고르지 않습니다.
CNN, 상대강도·저변동, 유동성 상위 동일비중을 같은 계좌 계산으로 비교합니다.

다음 거래일 시가, 정수 주식 수, 현금 잔액, 편도 0.2% 비용을 반영합니다. 가격 제한 부근의 주문과
거래정지일에는 체결을 보수적으로 제한합니다. 공표된 종가가 없는 보유분은 성과 검증 실패로 표시합니다.
현재 유니버스의 생존편향과 배당·기업행동 검증 한계가 있어 과거 성과만으로 실전 전환하지 않습니다.

서버는 별도 모의 계좌 세 개를 갱신합니다. 결정 저장 이후에 열린 장에서만 가상 체결하며,
기존 타임폴리오 주문과 연결하지 않습니다. 모델이 바뀌면 기존 모의 원장을 이어 붙이지 않습니다.
공용 코드 `arcmarket.systematic`과 PyTorch가 필요합니다. GPU나 외부 유료 학습 서버는 사용하지 않습니다.

```bash
python3 -m quant.krx_cnn_data       # 공용 KRX 일봉 갱신
python3 -m quant.krx_cnn train      # 고정 실험 학습·시험, 새 실행 폴더 생성
python3 -m quant.krx_cnn_worker     # 데이터·현재 후보·모의 계좌 갱신
```

일봉은 공용 보관소의 `KRX/daily_cnn`, 미국 정량 운용 자료는 `USA/daily_policy`에 둡니다.
모델·예측·보고서·모의 계좌는 `~/vault/ArcTrade/krx_cnn`에 보관합니다. API는 `/api/krx-cnn/status`입니다.
차트 표현은 [Jiang·Kelly·Xiu의 연구](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3756587)를 참고했으며 논문을 그대로 재현한 모델은 아닙니다.

## 시장충격 실험 (HYFE 9.28)

상단의 **시장충격 실험** 링크에서 관측한 Bybit 현물 호가창에 가상 주문을 넣습니다.
같은 총량의 일괄·분할 주문에 따른 평균 체결가, 시작 mid price 대비 비용, 남은 잔량을 비교합니다.
표시된 호가 밖 수량은 미체결로 남기며, 두 방식이 모두 전량 체결됐을 때만 비용 차이를 제시합니다.
실제 주문·계좌·Auto-folio 실행 경로와 연결하지 않습니다.

`ARCTRADE_IMPACT_DIR`의 `study.json`을 읽습니다. 기본 경로는 `~/vault/ArcTrade/market_impact`입니다.
자료에는 실험 4·5·6의 관측 곡선, 표본 수, 신뢰구간, 검열된 회복 시간과 대표 호가창이 들어갑니다.
회복 표본이 부족하면 해당 상태의 실측 보정값을 사용하지 않습니다. 직접 입력한 값은 가정으로 표시합니다.
가상 시장은 원래 가격대의 잔량만 회복하므로 미래 가격 추세나 다른 참여자의 반응을 예측하지 않습니다.

```bash
python3 -m quant.impact_collect --symbol ETHUSDT --seconds 1800 --output /path/to/new-capture
python3 -m quant.impact_research --input /path/to/new-capture --output /path/to/new-study
ARCTRADE_IMPACT_DIR=/path/to/new-study python3 -m uvicorn web.impact_lab:preview_app --host 127.0.0.1 --port 18628
pytest -q tests/test_market_impact.py
```

출력 경로는 매번 새 디렉터리를 지정합니다. 원자료는 gzip JSONL, 분석 산출물은 `study.json`,
`events.csv`와 그래프입니다. 원자료의 SHA-256과 고정한 사건 선정 기준을 `study.json`에 기록합니다.
별도 미리보기 서버는 운용 작업을 시작하지 않습니다. 기존 서버에 새 API를 반영하려면 재시작이 필요합니다.

### 동일 가격 refill 보완 분석과 본인 주문 관측

시장충격 화면의 **동일 가격 refill 보완 분석 · 본인 주문 관측기** 링크에서 보완 결과를 엽니다.
개별 가격의 양의 잔량 변화/최초 소모량, Depth 하위·상위 20%, 스프레드·변동성·활동량,
동일 방향·수량 차이 10% 이내 비교, 반대 방향 흐름과 사건 종료 후 반응을 함께 확인합니다.
호가의 양의 변화는 총 신규 공급의 하한 대용치이며 정확한 주문별 신규 공급량으로 단정하지 않습니다.

```bash
python3 -m quant.impact_extension --input /path/to/capture --output /path/to/new-study --tick-size 0.01
node --test tests/impact_pilot.test.mjs
node quant/impact_pilot_cli.mjs /path/to/new-study/study.json /private/path/orders.json /private/path/new-observation.json
```

분석할 종목의 tick size를 확인해 지정합니다. `study.json`에는 기존 실험을 유지하면서
`extension`과 공개 호가의 `pilot_market`을 추가합니다. `ARCTRADE_IMPACT_DIR`을 새 결과 폴더로 연결하면
기존 API가 이를 읽습니다. 현재 서버는 요청마다 파일을 읽으므로 데이터·정적 화면 갱신에는 재시작이 필요 없습니다.

본인 주문 관측기는 브라우저에서 JSON 기록을 읽으며 파일을 서버로 보내지 않습니다.
`schema_version: 1`, `orders` 배열 안에 `venue: "Bybit spot"`, `symbol`, `environment: "live"`,
`side: "Buy" | "Sell"`, `order_type: "Market" | "Limit"`, 기초자산 `quantity`,
거래소 기준 `created_ms`, `end_ms`, `status: "Filled" | "Cancelled" | "PartiallyFilledCancelled"`가 필요합니다.
지정가는 `limit_price`, 체결은 `fills: [{exec_id, ts_ms, quantity, price}]`로 기록합니다.
실제 주문과 같은 종목·시간대의 공개 호가가 필요하며 demo·testnet 기록은 본시장 호가와 섞지 않습니다.
주문 식별자는 결과에 싣지 않습니다. 기록 진위의 거래소 확인이나 인과효과 추정은 하지 않습니다.
관측기를 구현·검사한 상태와 실제 본인 주문 파일럿을 수행한 상태는 구분하며, 주문 기록이 없으면 미수행으로 표시합니다.

### 과거 호가 백필과 300초 관찰

**90일 백필 · 5분 가격 반응 확대 분석**에서 기간별 refill, 수량을 맞춘 상태 비교,
300초 가격 반응과 회복률을 확인합니다. 기존 가상 주문 모형과 본인 주문 관측기의 자료 범위는 따로 표시합니다.

`quant.impact_backfill.download_day(root, day)`로 분석할 모든 UTC 날짜의 공식 현물 ZIP/GZIP을 먼저 받습니다.
기존 원본은 덮어쓰지 않으며 CRC와 SHA-256을 검증합니다. `prepare_day`는 200단계 호가를 복원하고
상위 50단계를 날짜별 메모리 매핑 파일에 보관합니다. 큰 호가 파일을 여러 개 처리하므로 여유 디스크가 필요합니다.

```bash
python3 -m quant.impact_backfill --root /private/path/archives --day 2026-06-30 --prepare --max-gap-ms 0
python3 -m quant.impact_longitudinal --root /private/path/archives --output /private/path/new-study --start 2026-06-30 --end 2026-09-27 --calibration-end 2026-08-28 --workers 2
pytest -q tests/test_impact_backfill.py tests/test_impact_longitudinal.py
```

큰 체결 문턱과 상태 경계는 기준 설정 기간에서만 정합니다. 10초 사건의 간격은 11초 이상,
300초 사건은 301초 이상이며, 날짜 단위 bootstrap으로 점별 95% 구간을 계산합니다.
연속 update ID 사이의 호가 상태를 유지하되, 1초보다 오래된 호가를 제외한 민감도도 따로 제공합니다.
100ms 백필의 첫 refill 속도는 기존 20ms 실시간 자료와 직접 비교하지 않습니다.

확대 분석의 `study.json`은 `schema_version: 2`입니다. 웹에서는 기존 `schema_version: 1` 자료를
보존한 새 묶음의 `longitudinal` 필드로 포함합니다. 기존 API에 버전 2 단독 파일을 연결하면 읽을 수 없습니다.
확대 결과는 공개 시장의 관측이며 수익성 검증이나 실제 본인 주문 관측을 대신하지 않습니다.

### 완료된 실시간 수집의 재분석

`quant.impact_live_study`는 수집 manifest의 완료 표시와 예정 시간을 먼저 확인한 뒤 raw를 엽니다.
gzip CRC, 메시지 개수, SHA-256, 갱신 순서와 호가 복원을 검사하며, 알려진 오류 구간을 지나는
사건은 제외합니다. Update ID가 1보다 크게 증가한 경우는 별도 집계하고 누락으로 단정하지 않습니다.

```bash
python3 -m quant.impact_live_study --input /private/path/completed-capture --reference /private/path/historical-study/study.json --output /private/path/new-live-study
pytest -q tests/test_impact_live_study.py
```

큰 체결과 시장 상태 경계는 과거 기준 설정 기간의 값을 고정합니다. 10초 주표본은 호가가 1초 이내인
사건이며, 300초 관찰에서는 전체 구간의 신선도 조건을 적용한 민감도도 제공합니다. 신뢰구간과
짝맞춤은 같은 10분 구간을 단위로 합니다. 첫·다음 1시간의 결과는 시간대별 기술통계입니다.

새 `study.json`은 기존 schema 1을 유지하고 `live_study`에 주분석을 담습니다. 새 호가와 관측 가능한
회복 반감기로 기존 `/api/impact/simulate`를 호출하면 같은 5회 균등 분할 전략을 재실행할 수 있습니다.
회복값을 추정하지 못한 상태는 `refill: "none"`으로 실행하고 과거 값을 대신 넣지 않습니다.
`calibration_events.csv`에는 반감기를 검산할 100ms 잔량 경로를 저장합니다.

`/static/impact-live.html`에서 실시간 결과와 `paper_execution` 기록을 함께 봅니다. 90일 비교 자료는
`longitudinal`, 이전 페이퍼 기록은 `paper_execution_archive`에 별도로 보존합니다. 공개 API용 묶음에서는
로컬 보관 경로가 포함된 `provenance`를 제외하고, 전체 출처 기록은 개인 분석 폴더에 보관합니다.
이 페이퍼는 표시 호가를 가상으로 소모하는 집행비용 실험이며 실제 본시장 주문을 보내지 않습니다.

### 호가 회복 신호의 오프라인 알파 백테스트

`quant.impact_alpha_backtest`는 보관된 현물 호가·체결만 읽습니다. 주문 API나 계좌를 연결하지 않습니다.
강한 매수 체결 뒤 낮은 ask 보충·얇은 호가를 따르는 매수 전략과, 강한 매도 체결을 높은 bid 보충·두꺼운
호가가 흡수할 때 매수하는 반등 전략을 비교합니다. 매수 흐름만 쓰는 기준선과 보충 조건만 더한 기준선도 제공합니다.

보충을 1초 관찰한 뒤 100ms 지연을 두고 ask VWAP로 진입하며, 정해진 시간이 지나면 bid VWAP로 청산합니다.
5·10·30초 중 보유시간은 기준 설정 기간의 순수익으로만 고릅니다. 기준선은 돌파 전략과 같은 보유시간을 씁니다.
수량은 0.1 ETH, 전략별 가상 시작자금은 100,000 USDT, 기본 수수료는 편도 10bp입니다. 포지션은 겹치지 않습니다.
수수료 0·1·5·10bp, 추가 슬리피지 편도 1bp, 진입 지연 500ms도 별도로 비교합니다.

```bash
python3 -m quant.impact_alpha_backtest --cache-root /private/path/archives/cache --reference /private/path/historical-study/study.json --live-input /private/path/completed-capture --output /private/path/new-alpha-backtest --workers 2
pytest -q tests/test_impact_alpha_backtest.py tests/test_impact_alpha_stats.py
```

`protocol.json`에는 결과를 보기 전에 고정한 규칙을, `results.json`에는 학습·시간 분리 평가·실시간 구간의 성과를
저장합니다. 신호는 과거 1초까지만 사용해 다시 추출합니다. 향후 10초·300초의 관측 가능성을 조건으로 골랐던
설명용 사건표를 매매 신호로 재사용하지 않습니다. 청산 자료가 없으면 열린 포지션을 조용히 버리지 않고 실행을 실패시킵니다.

평가 날짜가 기존 기술통계 연구에서 이미 검토됐을 수 있으므로 새 미관측 표본으로 부르지 않습니다. 날짜 군집 신뢰구간은
탐색용이며, 표시된 최대낙폭은 청산 손익 기준으로 보유 중 손실을 포함하지 않습니다. 이 경로는 과거 호가 재생이므로
자기 주문이 이후 시장에 미치는 영향과 실제 계좌별 수수료는 별도 검증이 필요합니다.

### 매일 갱신하는 롤링 알파 연구

`quant.impact_alpha_rolling`은 21일 학습, 7일 진입 기준 선택, 다음 1일 평가를 하루씩 이동합니다.
가격·체결·시간 정보 14개를 쓰는 기준 모델과, 보충 속도·호가 두께·초기 충격 회복 등을 더한
33개 변수 모델을 비교합니다. 30초·5분·30분 보유와 학습 점수 상위 10·3·1% 기준 가운데
직전 7일 성과로만 진입 기준을 고릅니다. 각 학습·선택 구간의 청산 결과는 다음 구간 시작 전에 확정돼야 합니다.

편도 10bp를 기본 비용으로 두며, 1bp는 가상의 저비용 환경에 대한 민감도 분석입니다.
예상 수익이 왕복 수수료에 1bp를 더한 값 이상이고, 직전 7일에 30회 이상·4일 이상 거래했으며,
날짜 단위 재표본추출로 계산한 순수익 신뢰구간 하한이 양수인 후보만 다음 날 거래할 수 있습니다.
조건을 통과한 후보가 없으면 현금을 유지합니다. 별도 `rank97` 결과는 비용 전 신호를 살피는
연구용 비교이며 이 수익성 조건을 통과한 전략이 아닙니다. 현금 유지 자체를 알파로 해석하지 않습니다.

```bash
python3 -m quant.impact_alpha_rolling --cache-root /private/path/archives/cache --live-input /private/path/completed-capture --output /private/path/new-rolling-backtest --workers 2
pytest -q tests/test_impact_alpha_features.py tests/test_impact_alpha_rolling.py
```

`dataset/`에는 시점별 변수와 체결 가능 여부, `folds/`에는 날짜별 학습 범위·선택 후보·예측·거래를 저장합니다.
`results.json`과 전략별 거래 CSV로 전체 평가 결과를 확인합니다. 변수는 결정 시점까지의 자료만 사용하고,
미래 체결 자료가 없는 행도 삭제하지 않고 표시합니다. 선택된 거래의 청산을 계산할 수 없으면 실행을 중단합니다.
매수·매도 양쪽의 실제 표시 잔량으로 0.1 ETH VWAP를 계산하며, 같은 전략의 포지션은 겹치지 않습니다.

평가 신뢰구간에는 7일 블록 재표본추출 결과도 제공합니다. 기존에 살펴본 기간을 다시 사용하므로
성과는 탐색 결과입니다. 비용·대기열·자기 주문에 따른 후속 시장 변화까지 검증한 실거래 수익으로 해석하지 않습니다.
이 명령은 로컬 파일만 읽으며 주문·계좌·서비스 설정을 변경하지 않습니다.

## 테스트

```bash
pytest -q
python -m web.nlbacktest
```

웹 화면은 1440×900과 390×844에서 확인합니다. 공개 전에는 아래 명령으로 추적 대상에 비밀이 없는지 다시 검사하는 편이 안전합니다.

```bash
git grep -nEi '(api[_-]?key|password|secret|token|@gmail\.com)'
```

변수명과 빈 예시는 검색되더라도 실제 값은 없어야 합니다.

## 구조

```text
ArcTrade/
├── web/               FastAPI 서버, AI 제공사 연결, 실험 기록, 대시보드
├── quant/             자연어 조건 백테스트 엔진과 재무 데이터 갱신
├── core/              분봉 연구 엔진과 통계 모듈
├── Auto_folio/        타임폴리오 모의·사이트 주문 어댑터
├── crypto/            별도 암호자산 연구 코드
├── tests/             핵심 전략·주문 로직 테스트
├── .env.example       비밀이 없는 설정 예시
└── requirements.txt   Python 의존성
```

MIT License로 공개합니다.
