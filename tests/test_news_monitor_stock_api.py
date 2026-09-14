"""2026-09-10 네이버가 finance.naver.com/news 를 stock.naver.com(클라이언트 렌더링)으로 옮기면서
HTML 파서가 오류 없이 0건만 뽑았다 — 9/10 19:45~9/14 무수집, daily_news 쇼츠 사흘 공백.

2026-09-14 대응(사장 지시):
- 1차 소스 = stock.naver.com 목록 JSON, 2차 = 언론사 RSS(늘 함께 읽는다)
- 소스별로 3시간 연속 링크 0건이면 WARNING(건강 시각은 파일에 영속 — 수집기는 매번 새 프로세스)
- 수집은 독립 타이머가 하고 QIS 는 pull_from_disk 로 읽기만 한다
"""
import json
import logging

from tools import news_monitor as nm

API_BODY = {"articles": [
    {"officeId": "215", "officeHname": "한국경제TV", "articleId": "0001265821",
     "title": "美 기준금리 인상은 이제 상수…10년물 5% 시대까지 대비해야",
     "datetime": "2026-09-14 07:42:14", "subcontent": "미 증시 살펴보며 한국 증시 투자아이디어까지"},
    {"officeId": "018", "officeHname": "이데일리", "articleId": "0006368762",
     "title": "아모레퍼시픽, 더마·헤어 앞세워 성장 본격화…스케일업 진입",
     "datetime": "2026-09-14 07:42:12", "subcontent": "하나증권은 14일 아모레퍼시픽에 대해"},
]}
RSS_BODY = """<?xml version="1.0" encoding="UTF-8"?><rss><channel><title>연합뉴스 마켓</title>
<item><title><![CDATA[코스피, 외국인 매수에 2% 급반등…반도체 강세]]></title>
<link>https://www.yna.co.kr/view/AKR20260914000100008</link>
<description><![CDATA[<p>14일 코스피가 외국인 순매수에 힘입어 급반등했다.</p>]]></description>
<pubDate>Mon, 14 Sep 2026 07:55:00 +0900</pubDate></item>
<item><title>깨진 & 문자가 섞인 제목도 읽는다 — 원화 강세 수혜주 점검</title>
<link>https://www.yna.co.kr/view/AKR20260914000200008</link>
<pubDate>Mon, 14 Sep 2026 07:50:00 +0900</pubDate></item>
</channel></rss>"""


class _Resp:
    def __init__(self, body=None, text=""):
        self._body, self.text, self.encoding = body, text, "utf-8"

    def raise_for_status(self):
        pass

    def json(self):
        return self._body


def _router(api=API_BODY, rss=RSS_BODY):
    def get(url, **kw):
        if "stock.naver.com/api" in url:
            return _Resp(api)
        if url.endswith(".xml") or "rss" in url:
            return _Resp(text=rss)
        return _Resp(text="<html><title>Npay 증권</title></html>")   # 기사 없는 SPA 껍데기
    return get


def _monitor(tmp_path, monkeypatch):
    monkeypatch.setattr(nm, "_NEWS_STATE_FILE", tmp_path / "news_history.json")
    m = nm.NaverFinanceMonitor()
    m._seen_links, m._seen_titles, m._article_history = set(), [], []
    monkeypatch.setattr(m, "_mirror_to_archive", lambda arts: None)
    return m


def test_api_and_rss_articles_are_merged(tmp_path, monkeypatch):
    m = _monitor(tmp_path, monkeypatch)
    monkeypatch.setattr(nm.requests, "get", _router())
    arts = m.crawl_once()
    links = [a["link"] for a in arts]
    assert "https://n.news.naver.com/article/215/0001265821" in links
    assert "https://www.yna.co.kr/view/AKR20260914000100008" in links     # RSS 원문 링크 그대로
    yna = next(a for a in arts if "yna.co.kr" in a["link"])
    assert yna["source"] == "연합뉴스" and yna["date"] == "2026-09-14 07:55"
    assert "외국인 순매수" in yna["summary"] and "<p>" not in yna["summary"]
    assert m.crawl_once() == []          # 같은 목록 재호출은 중복 제거


def test_rss_keeps_collecting_when_naver_breaks(tmp_path, monkeypatch):
    m = _monitor(tmp_path, monkeypatch)
    monkeypatch.setattr(nm.requests, "get", _router(api={"articles": []}))
    arts = m.crawl_once()
    assert arts and all("yna.co.kr" in a["link"] for a in arts)


def test_dead_source_warns_even_if_others_work(tmp_path, monkeypatch, caplog):
    m = _monitor(tmp_path, monkeypatch)
    monkeypatch.setattr(nm.requests, "get", _router(api={"articles": []}))
    m._last_links_at["네이버 증권"] = 0.0       # 아주 오래전에 마지막으로 성공
    with caplog.at_level(logging.WARNING, logger="NEWS_MONITOR"):
        m.crawl_once()
    msgs = [r.message for r in caplog.records]
    assert any("무수집" in x and "네이버 증권" in x for x in msgs)
    assert not any("연합뉴스" in x for x in msgs)   # 살아있는 소스는 경고 안 함


def test_reader_pulls_only_articles_added_after_start(tmp_path, monkeypatch):
    state = tmp_path / "news_history.json"
    monkeypatch.setattr(nm, "_NEWS_STATE_FILE", state)
    old = {"title": "기동 전부터 있던 기사입니다 코스피", "link": "https://x/old"}
    state.write_text(json.dumps({"article_history": [old]}), encoding="utf-8")
    reader = nm.NaverFinanceMonitor()                     # QIS 기동 — 기존 기사 복원
    assert reader.pull_from_disk() == []                  # 재시작 직후 옛 기사를 신규로 쏟지 않는다
    new = {"title": "수집기가 방금 넣은 기사 원화 강세", "link": "https://x/new"}
    state.write_text(json.dumps({"article_history": [old, new], "last_crawl_time": "2026-09-14 08:10:00"}),
                     encoding="utf-8")
    assert [a["link"] for a in reader.pull_from_disk()] == ["https://x/new"]
    assert reader.pull_from_disk() == []
    assert reader.last_crawl_time == "2026-09-14 08:10:00"
    assert [a["link"] for a in reader.get_recent_articles(5)][-1] == "https://x/new"


def test_history_is_time_ordered_across_sources(tmp_path, monkeypatch):
    m = _monitor(tmp_path, monkeypatch)
    monkeypatch.setattr(nm.requests, "get", _router())
    m.crawl_once()
    dates = [a["date"] for a in m._article_history]
    assert dates == sorted(dates)
    assert m._article_history[-1]["date"] == "2026-09-14 07:55"      # 가장 최근이 꼬리
