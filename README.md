# QuantInSight

전략 연구와 계좌 현황: https://quantinsight.ai-ve.uk

데이터셋 / 한국주식 / 미국주식 / 크립토 / 타임폴리오로 구분한다.
한국주식은 전략 연구·한투 실매매·한투 모의매매, 미국주식과 크립토는 전략 연구·모의매매를 제공한다.
타임폴리오는 13회 대회 계정을 직접 조회하며 이전 대회 장부와 합치지 않는다.
크립토 모의매매는 기존 AutoCrypto 화면과 논문 기반 고정 모델 장부를 같은 출처에서 읽는다.

## 로그인과 연결

관리자는 로컬에서 최초 생성한다. 회원가입은 일반 사용자만 생성하며 서버 자원 설정은 관리자 전용이다.
회원은 OpenAI, Anthropic, Gemini, DeepSeek, OpenRouter 중 하나와 모델 ID·API Key를 등록해 알파 후보를 만든다.
관리자는 로컬 AI를 사용한다. 모델 응답은 허용된 전략 파라미터로 검증하며 생성된 코드는 실행하지 않는다.
회원가입의 한국투자증권·타임폴리오 연결은 선택 항목이다. 가입 후 계좌 연결 화면에서 추가·변경할 수 있다.
계좌 정보·API Key는 회원별로 암호화해 vault에 두고 비밀번호는 scrypt 해시로 저장한다.
계좌 조회는 해당 회원의 연결만 사용한다. 기존 관리자 KIS 연결에 대한 호환 경로만 관리자에게 유지한다.
세션은 HttpOnly·SameSite 쿠키, 외부 HTTPS에서는 Secure 속성을 사용한다. 변경 API는 동일 출처를 검사한다.
Cloudflare는 이 도메인에 한해 앱 로그인 화면으로 통과시키며 모든 데이터 API는 앱 인증을 거친다.

## 데이터와 연구 상태

데이터셋에는 실제 보유 시작·종료일을 표시한다. 현재·과거 분봉은 자산별로 묶는다.
날짜는 DB·Parquet·패널 원본에서 읽으며 파일 수정일을 데이터 기간으로 대신하지 않는다.
보관소의 최소·최대일은 합집합 범위이며 모든 종목에 연속 자료가 있다는 의미가 아니다.

신규 평가 기간은 최근 36개 완료된 달로 고정한다. 주식은 시장 거래일 달력, 크립토는 전체 달력일과 일치해야 한다.
학술제의 동결된 검증 시세는 2026-02-27까지이며 그대로 보존한다. QuantInSight의 별도 확장 입력은
한국 NAVER 일봉 303종목과 미국 동일 제공사 수정 일봉 40종목이다. 평가 구간은 2023-10-01~2026-09-30,
한국 728거래일·미국 752거래일이다. 2026년 한국 추가 휴장일 6월 3일·7월 17일을 달력에 반영했다.
현재 수집 대상 종목을 사용하는 개발 결과이므로 생존·수집 선택 편향과 기업행사·체결 근사의 한계가 있다.

`autofolio.campaign`은 모멘텀·단기 반전·추세 조건·변동성 조정과 차트 표현의 기준 비교를 먼저 진행한다.
그다음 진입 조건, 보유 5/14/28거래일, 보유 종목 10/20/30개, Ridge 강도를 한 항목씩 바꾼 뒤 조합을 평가한다.
가격 전략의 무의미한 Ridge 중복은 제외한다. 후보는 자동 보충하고 같은 정의를 반복 실행하지 않는다.
유효한 조합을 모두 소진하면 종료하며 평가 월이 바뀌면 새 기간으로 다시 검증한다.
자료 대기 시장이 준비된 시장을 막지 않으며 실패 후보를 무한 재시도하지 않는다.

순차 학습은 해당 분기 전에 결과가 확정된 라벨만 사용한다. 가격 신호는 당일 종가까지의 정보로 만들고
다음 거래일 시가에 체결한다. 학술제의 롱온리 계좌 엔진으로 비용·보유 제한·유동성을 반영한다.
원본 학술제 실행 파일과 수집기는 변경하지 않는다. 입력은 기간별 단일 스냅샷을 공유하고 반복 학습의
모델 가중치·이미지 행렬은 저장하지 않는다. 워커 cgroup의 CPU·메모리·동시 실험 한도와 저장 용량 가드를 적용한다.
관리 설정은 `research_campaign_enabled`, `research_campaign_owner`; 실제 상태는 시장 연구 API와 진행 로그에 표시한다.
크립토 신규 36개월 순차 모델은 아직 준비 상태이며 기존 논문 모의매매 모델은 그대로 유지한다.

전략의 적용 메뉴는 시장에 맞는 계좌를 선택해 회원별로 저장한다. 현재 상태는 `awaiting_signal`이다.
최신 운용 신호와 주문 실행 연결은 아직 완료되지 않았으므로 선택 저장을 주문 활성화나 가상 체결로 표시하지 않는다.
미국 모의 계좌도 검증된 신호가 연결되기 전에는 잔고·수익률을 만들어 표시하지 않는다.

## 실행과 검사

```sh
python3 -m pip install -r requirements.txt
python3 -m uvicorn autofolio.app:app --host 127.0.0.1 --port 8997
python3 -m autofolio.worker
python3 -m pytest tests -q
```

운영 호스트의 시장 달력 의존성은 vault/QuantInSight/python-deps에 격리 설치했다. 공유 Python 패키지는 덮어쓰지 않는다.
`quantinsight.service`의 기존 `server.app:app`도 같은 앱이다. CPU·메모리 한도는 autofolio-worker cgroup에 실제 적용한다.
연구 결과와 기존 계좌 데이터는 vault 링크를 유지한다. 연구 모델·DB·수집 원본을 반복 복사하지 않는다.
배경 이미지·폰트는 바이너리 공개 검사 정책에 따라 GitHub 소스 사본에서 제외한다.

API 구현 참고: [OpenAI](https://developers.openai.com/api/reference/resources/chat),
[Anthropic](https://platform.claude.com/docs/en/api/messages/create),
[Gemini](https://ai.google.dev/api/generate-content),
[DeepSeek](https://api-docs.deepseek.com/api/create-chat-completion/),
[OpenRouter](https://openrouter.ai/docs/api/api-reference/chat/send-chat-completion-request).
