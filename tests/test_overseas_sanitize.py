"""모의서버 비정상 기준환율(frst_bltn_exrt) garbage 전파 차단 — 사장 지시 2026-06-16.

배경: 모의서버가 비정상 환율(exrt 223.85, 정상 USD/KRW≈1500)을 주면 해외 주식평가가
과대 계산된다. 기존 가드는 총평가 합산값(krw)만 0 처리하고 비중계산용 주식분(stock,
_overseas_stock_krw)은 오염값을 흘려보내, 매크로 주식비중이 100% 로 부풀어 매수가 영구
차단됐다(uid2). _sanitize_overseas 는 krw·stock 을 함께 0 처리한다.
"""
from infra.kis_broker import _sanitize_overseas


def test_abnormal_exrt_zeros_both_krw_and_stock():
    # uid2 실측: 외화평가 379M·주식분 80.8M·exrt 223.85(비정상) → 둘 다 0
    krw, stock = _sanitize_overseas(379_000_000, 80_818_068, 223.85)
    assert krw == 0.0
    assert stock == 0.0


def test_normal_exrt_passthrough():
    # 실거래 정상 환율(~1500) → 입력 그대로 (가드 미발동)
    krw, stock = _sanitize_overseas(379_000_000, 80_818_068, 1500.0)
    assert krw == 379_000_000
    assert stock == 80_818_068


def test_unknown_exrt_is_not_touched():
    # exrt 0/None(미상)은 함부로 0 처리하지 않는다(조회 실패와 비정상을 구분, 보수적)
    assert _sanitize_overseas(100.0, 50.0, 0.0) == (100.0, 50.0)
    assert _sanitize_overseas(100.0, 50.0, None) == (100.0, 50.0)
