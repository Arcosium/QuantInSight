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
현재 한국·미국 학술제 검증 시세는 2022-05-02~2026-02-27이다. 2026-10 기준 평가 구간은 2023-10-01~2026-09-30이므로
검증 원본 갱신 전 후보는 자료 대기로 표시하며 과거의 짧은 폴드나 마지막 달 일부를 36개월 성과로 게시하지 않는다.
한국·미국 차트 이미지 실험은 학술제의 차트 변환·인과적 라벨·다음 거래일 체결 함수를 재사용하는 순차 학습 어댑터다.
원본 연구 파일과 실행 중 수집기를 변경하지 않는다. 크립토 신규 36개월 순차 모델의 학습·평가 연결은 아직 준비 상태이며,
기존 논문 모의매매 모델을 과거 전체 기간에 재사용해 미래 정보가 섞인 성과를 만들지 않는다.

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
