# QuantInSight

흑연색 관제실 디자인의 전략 연구·계좌 현황 화면. 주소는 `https://quantinsight.ai-ve.uk`다.
Cloudflare Access의 기존 소유자 정책을 유지한다.

데이터셋, 타임폴리오 모의매매, 한국투자증권 실매매·모의매매를 한 화면에서 확인한다.
각 매매 환경에는 전략 연구와 매매 현황 서브탭이 있다. 연구 결과는 공통 KRX 계좌 실험이며,
매매 현황의 계좌·보유·체결·수익률 기록은 환경별로 분리한다.

평가는 최근 36개 달로 고정한다. 아직 수집하지 못한 마지막 거래일은 채우지 않으며 실제 평가 종료일을 표시한다.
현재 확정 입력은 2026-09-23까지다. 달이 바뀌면 이전 기간 결과를 새 기간의 성과로 표시하지 않는다.
기존 33개월 이하 전략과 255거래일 실험은 활성 목록에서 폐기했다.

손실 개월 수 축은 내림차순이고 파레토 경계는 오른쪽에 표시한다. 리더보드는 10개씩 표시한다.
새 결과를 5초마다 확인하고 신규 점에만 등장 효과를 적용한다. 모션 감소 설정을 존중한다.
진행 로그, 실제 CPU·메모리 한도 설정, 로컬 AI의 전략 유전자 생성 기능을 제공한다.
지원하지 않는 전략 조건은 임의로 바꾸지 않고 알려 준다. 로컬 모델만 호출한다.

## 실행과 검사

```sh
python3 -m uvicorn autofolio.app:app --host 127.0.0.1 --port 8997
python3 -m autofolio.runner --baseline
python3 -m autofolio.worker
python3 -m unittest discover -s tests -v
```

`quantinsight.service`의 기존 `server.app:app` 진입점도 같은 앱을 실행한다.
`autofolio-worker.service`가 새 연구를 실행하며 자원 한도는 해당 서비스의 cgroup에 적용한다.
타임폴리오 계좌 어댑터는 `integrations/timefolio/web/gateway.py`이며 이전 자동매매 전략 루프를 시작하지 않는다.
한국투자증권의 인증·시세·수집 모듈은 기존 원본 데이터 경로를 보존한다. 이 화면에서 주문을 제출하지 않는다.

`research_data/`와 `runs/`는 기존 vault/Autofolio 원본을, `data/`는 vault/QuantInSight 원본을 참조한다.
ArcTrade의 필요한 코드와 확정 입력 참조는 `integrations/timefolio/`로 이동했다.
기존 Autofolio·ArcTrade 경로는 외부 수집기·연구 참조를 위한 호환 링크다. 별도 프로젝트 복사·업로드 대상이 아니다.
폐기한 앱·실험 목록의 복구용 백업은 vault에 있다. 실행 중인 분봉 DB는 이동하지 않았다.

배경 이미지 `static/observatory.png`와 로컬 폰트는 바이너리 공개 검사 정책에 따라 보관용 사본에만 포함한다. 이미지가 없어도 화면의 기본 흑연색 배경과 기능은 유지된다.
