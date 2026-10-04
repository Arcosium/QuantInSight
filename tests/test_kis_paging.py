"""KIS 잔고 신뢰성 — 인프라 기반(Group A): 모의 tr_id 변환범위, 전역 호출간격(rate-lock),
연속조회(tr_cont 페이징)+부분성공 보존. (2026-06-01 KIS 공식샘플 정독 반영)
모두 라이브 KIS 미사용 — 세션/응답을 스텁한다."""
import asyncio
import pytest

from infra.kis_broker import KISBroker


def _creds(mock=False):
    return {
        "kis_app_key": "APPKEY", "kis_app_secret": "SECRET",
        "kis_account_no": "12345678-01",
        "kis_base_url": ("https://openapivts.koreainvestment.com:29443" if mock
                         else "https://openapi.koreainvestment.com:9443"),
    }


# ── Task 2: 모의 tr_id 변환 T/J/C ──────────────────────────────────────────────
def test_mock_tr_converts_t_j_c_prefixes(tmp_path):
    b = KISBroker(_creds(mock=True), token_path=tmp_path / "tok.json")
    # 'T' 시작(현행 동작 유지)
    assert b._mock_tr("TTTC8908R") == "VTTC8908R"
    # 'C' 시작(현행 버그: 미변환) — CTRP6504R 류
    assert b._mock_tr("CTRP6504R") == "VTRP6504R"
    # 'J' 시작
    assert b._mock_tr("JTTT1004R") == "VTTT1004R"
    # 오버라이드맵 우선(해외매도 예외)
    assert b._mock_tr("TTTT1006U") == "VTTT1001U"
    # 시세성(FH...)·기타는 불변
    assert b._mock_tr("FHKST01010100") == "FHKST01010100"


def test_mock_tr_noop_on_real_account(tmp_path):
    b = KISBroker(_creds(mock=False), token_path=tmp_path / "tok.json")
    assert b._mock_tr("CTRP6504R") == "CTRP6504R"
    assert b._mock_tr("TTTC8908R") == "TTTC8908R"


# ── Task 1: 전역 호출간격 rate-lock ───────────────────────────────────────────
def test_pace_enforces_min_interval(tmp_path):
    b = KISBroker(_creds(mock=False), token_path=tmp_path / "tok.json")
    assert hasattr(b, "_min_interval") and b._min_interval > 0   # 실전 기본 간격 존재
    b._min_interval = 0.05

    async def run():
        loop = asyncio.get_event_loop()
        await b._pace()          # 첫 호출 — 즉시
        t0 = loop.time()
        await b._pace()          # ~0.05 대기
        await b._pace()          # ~0.05 대기
        return loop.time() - t0

    elapsed = asyncio.run(run())
    assert elapsed >= 0.09, f"두 번의 _pace 가 최소간격(2×0.05)만큼 직렬화돼야 함 (실측 {elapsed:.3f}s)"


def test_min_interval_larger_for_mock(tmp_path):
    real = KISBroker(_creds(mock=False), token_path=tmp_path / "r.json")
    mock = KISBroker(_creds(mock=True), token_path=tmp_path / "m.json")
    assert mock._min_interval > real._min_interval  # 모의서버는 더 보수적으로


# ── Task 3: 연속조회(tr_cont 페이징) + 부분성공 보존 ─────────────────────────────
class _FakeResp:
    def __init__(self, body, tr_cont=""):
        self._body = body
        self.headers = {"tr_cont": tr_cont}

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def json(self):
        return self._body


class _FakeSession:
    def __init__(self, pages):
        self._pages = list(pages)   # [(body, tr_cont), ...]
        self.calls = []

    def get(self, url, headers=None, params=None):
        self.calls.append({"url": url, "headers": dict(headers or {}), "params": dict(params or {})})
        body, trc = self._pages.pop(0)
        return _FakeResp(body, trc)


def _wire(b, fake, monkeypatch):
    async def _s():
        return fake

    async def _tok(force=False):
        return "TOK"

    monkeypatch.setattr(b, "_s", _s)
    monkeypatch.setattr(b, "token", _tok)
    b._min_interval = 0.0


def test_paged_get_accumulates_and_paginates(tmp_path, monkeypatch):
    b = KISBroker(_creds(mock=False), token_path=tmp_path / "tok.json")
    pages = [
        ({"rt_cd": "0", "msg1": "ok", "output1": [{"x": 1}],
          "ctx_area_fk200": "F2", "ctx_area_nk200": "N2"}, "M"),
        ({"rt_cd": "0", "msg1": "ok", "output1": [{"x": 2}]}, "D"),
    ]
    fake = _FakeSession(pages)
    _wire(b, fake, monkeypatch)
    res = asyncio.run(b._paged_get("/p", "TR", {"CANO": "c"},
                                   fk_key="CTX_AREA_FK200", nk_key="CTX_AREA_NK200",
                                   out_keys=("output1",)))
    assert [r["x"] for r in res["output1"]] == [1, 2], "두 페이지 종목이 모두 누적돼야 함"
    assert res["ok"] is True and res["partial"] is False
    # 2페이지 요청이 직전 ctx 키를 이어받고 tr_cont='N' 으로 보냈는가
    assert fake.calls[1]["params"].get("CTX_AREA_FK200") == "F2"
    assert fake.calls[1]["params"].get("CTX_AREA_NK200") == "N2"
    assert fake.calls[1]["headers"].get("tr_cont") == "N"


def test_paged_get_partial_success_on_late_failure(tmp_path, monkeypatch):
    b = KISBroker(_creds(mock=False), token_path=tmp_path / "tok.json")
    pages = [
        ({"rt_cd": "0", "output1": [{"x": 1}],
          "ctx_area_fk200": "F", "ctx_area_nk200": "N"}, "M"),
        ({"rt_cd": "1", "msg1": "일시 오류", "msg_cd": "EGW9999", "output1": []}, ""),
    ]
    fake = _FakeSession(pages)
    _wire(b, fake, monkeypatch)
    res = asyncio.run(b._paged_get("/p", "TR", {"CANO": "c"},
                                   fk_key="CTX_AREA_FK200", nk_key="CTX_AREA_NK200",
                                   out_keys=("output1",)))
    assert [r["x"] for r in res["output1"]] == [1], "후반 페이지 실패 시 앞 페이지는 보존돼야 함"
    assert res["ok"] is True and res["partial"] is True


def test_paged_get_first_page_failure_marks_not_ok(tmp_path, monkeypatch):
    b = KISBroker(_creds(mock=False), token_path=tmp_path / "tok.json")
    pages = [({"rt_cd": "1", "msg1": "조회 실패", "msg_cd": "X", "output1": []}, "")]
    fake = _FakeSession(pages)
    _wire(b, fake, monkeypatch)
    res = asyncio.run(b._paged_get("/p", "TR", {"CANO": "c"},
                                   fk_key="CTX_AREA_FK200", nk_key="CTX_AREA_NK200",
                                   out_keys=("output1",)))
    assert res["output1"] == [] and res["ok"] is False
    assert res["msg_cd"] == "X"


class _ExchSession:
    """거래소(OVRS_EXCG_CD)별로 자체 페이지 큐를 갖는 가짜 세션 — 거래소별 연속조회를 정확히 모델링."""
    def __init__(self, by_exch):
        self._by = {k: list(v) for k, v in by_exch.items()}
        self.calls = []

    def get(self, url, headers=None, params=None):
        self.calls.append({"headers": dict(headers or {}), "params": dict(params or {})})
        exch = (params or {}).get("OVRS_EXCG_CD", "")
        q = self._by.get(exch) or []
        body, trc = q.pop(0) if q else ({"rt_cd": "0", "output1": []}, "")
        return _FakeResp(body, trc)


# ── Task 10: _overseas_holdings 가 거래소별로 연속조회(페이징) ──────────────────
def test_overseas_holdings_paginates_each_exchange(tmp_path, monkeypatch):
    b = KISBroker(_creds(mock=False), token_path=tmp_path / "t.json")
    by_exch = {
        "NASD": [
            ({"rt_cd": "0", "output1": [{"ovrs_pdno": "AAA", "ovrs_cblc_qty": "2",
                                         "now_pric2": "10", "ovrs_item_name": "A"}],
              "ctx_area_fk200": "F", "ctx_area_nk200": "N"}, "M"),     # 1페이지(다음 있음)
            ({"rt_cd": "0", "output1": [{"ovrs_pdno": "BBB", "ovrs_cblc_qty": "3",
                                         "now_pric2": "20", "ovrs_item_name": "B"}]}, "D"),  # 2페이지(끝)
        ],
        "NYSE": [({"rt_cd": "0", "output1": []}, "")],
        "AMEX": [({"rt_cd": "0", "output1": []}, "")],
    }
    fake = _ExchSession(by_exch)
    _wire(b, fake, monkeypatch)
    res = asyncio.run(b._overseas_holdings())
    codes = sorted(h["code"] for h in res)
    assert codes == ["AAA", "BBB"], "한 거래소의 2페이지 종목이 모두 포함돼야 함(현행은 1페이지만 읽어 BBB 누락)"
    assert b._overseas_holdings_authoritative is True


def test_overseas_holdings_empty_is_not_authoritative_if_one_exchange_fails(tmp_path, monkeypatch):
    """빈 목록이어도 세 거래소 중 하나가 실패하면 전량매도 확정으로 쓰면 안 된다."""
    b = KISBroker(_creds(mock=False), token_path=tmp_path / "t.json")
    fake = _ExchSession({
        "NASD": [({"rt_cd": "0", "output1": []}, "")],
        "NYSE": [({"rt_cd": "1", "msg1": "temporary failure"}, "")],
        "AMEX": [({"rt_cd": "0", "output1": []}, "")],
    })
    _wire(b, fake, monkeypatch)
    assert asyncio.run(b._overseas_holdings()) == []
    assert b._overseas_holdings_authoritative is False
