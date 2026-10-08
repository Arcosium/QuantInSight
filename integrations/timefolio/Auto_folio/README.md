# Auto_folio

QuantInSight 내부의 타임폴리오 투자대회 모의투자 모듈입니다. 외부 타임폴리오 사이트 계정을 KIS 주문에 우회 연결하지 않고, QuantInSight 안에서 별도 대회 계정과 포트폴리오를 만들고 대회 규정을 검사한 뒤 자체 모의 주문을 체결 반영합니다.

## 주요 기능

- 프로필 모달의 `타임폴리오 모의투자` 섹션에서 별도 대회 아이디/비밀번호 가입
- 독립 현금, 보유, 거래내역, 주간 회전율 원장 저장
- 종목 메타데이터 저장: 시장, 보통주 여부, GICS 섹터, 시장 섹터 비중, 시가총액, 5일 평균 거래대금, 상장 후 거래일 수, 지정 상태
- 주문 전 대회 규정 검사 및 거부 사유 기록
- 기존 QuantInSight/KIS 실주문 흐름과 분리된 `/api/autofolio/*` API
- KIS 없이 네이버금융 일봉/종목 페이지 크롤링으로 가격, 5일 평균 거래대금, 시가총액 일부 자동 보강
- `POST /api/autofolio/cycle`로 Auto_folio 전용 네이버 사이클 실행

## 적용 규정

- 투자 대상: KOSPI/KOSDAQ 보통주
- 매수 불가:
  - 5일 평균 거래대금 30억원 이하
  - 상장 후 6영업일 미도래 또는 최근 거래일 5일 미만
  - 투자주의/경고/위험, 투자주의환기, 관리, 거래정지 등 지정 상태
  - 시가총액 1,000억원 미만
- 편입 한도:
  - 기본 종목별 15% 이하
  - 삼성전자 `005930` 최대 40%
  - SK하이닉스 `000660` 최대 30% (2026-07-01 기준)
  - GICS 섹터 비중은 시장 비중의 2배 이하, 시장 비중 5% 이하 섹터는 10%까지
  - 시가총액 1조원 미만 종목 합산 30% 이하
- 주간 회전율:
  - 현재 주 회전율을 `(매수총액 + 매도총액) / 평균 운용금액 * 0.5 * 100%`로 추적
  - 주문 차단 조건이 아니라 계정 상태에 5% 충족/미달로 표시

## API

- `POST /api/autofolio/register`
- `GET /api/autofolio/account`
- `GET /api/autofolio/securities`
- `POST /api/autofolio/securities`
- `POST /api/autofolio/order`
- `POST /api/autofolio/cycle`

데이터는 `Auto_folio/data/contest_state.json`에 저장됩니다.
