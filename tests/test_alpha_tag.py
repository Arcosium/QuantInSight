"""P2 — 알파 귀인 태그 (2026-08-02).

점수 breakdown 의 축 기여도로 '이 매수가 어느 알파에서 나왔나'를 결정론 태깅하고,
청산 후 계열별 성과로 집계한다(LLM 호출 없음).
"""
from tools.quant_score import alpha_tag


def test_tag_is_dominant_positive_axis():
    bd = {"S_quant": 7.0, "S_news": 5.0,
          "indicators": {"mom": 8.4, "rsi": -2.0, "cmf": 1.1}}
    assert alpha_tag(bd) == "모멘텀"
    assert alpha_tag({"S_quant": 6.0, "indicators": {"cmf": 4.0, "mom": 1.0}}) == "수급"


def test_strong_news_overrides_quant():
    bd = {"S_quant": 6.0, "S_news": 9.0, "indicators": {"mom": 3.0}}
    assert alpha_tag(bd) == "뉴스이벤트"
    # 뉴스가 강해도 퀀트가 더 높으면 퀀트 축이 이긴다
    assert alpha_tag({"S_quant": 9.5, "S_news": 7.6, "indicators": {"mom": 3.0}}) == "모멘텀"


def test_no_positive_axis_is_honest():
    assert alpha_tag({"S_quant": 3.0, "indicators": {"mom": -1.0, "rsi": -2.0}}) == "기타"
    assert alpha_tag(None) == "기타"


def test_tag_stats_filters_small_samples(tmp_path, monkeypatch):
    import infra.user_paths as up
    import infra.trade_reflections as tr
    monkeypatch.setattr(up, "user_dir", lambda uid: tmp_path)
    for e in [
        {"code": "A", "status": "resolved", "alpha_tag": "모멘텀", "raw_ret_pct": 3.0, "alpha_pct": 1.0},
        {"code": "B", "status": "resolved", "alpha_tag": "모멘텀", "raw_ret_pct": -1.0, "alpha_pct": -2.0},
        {"code": "C", "status": "resolved", "alpha_tag": "평균회귀", "raw_ret_pct": 9.0},   # 1건 → 제외
        {"code": "D", "status": "pending", "alpha_tag": "모멘텀"},                          # 미확정 → 제외
    ]:
        tr._save(1, tr._load(1) + [e])
    st = tr.tag_stats(1, min_n=2)
    assert set(st) == {"모멘텀"}
    assert st["모멘텀"] == {"n": 2, "wins": 1, "win_rate": 50.0,
                          "avg_ret_pct": 1.0, "avg_alpha_pct": -0.5}
    assert "모멘텀: 2건 · 승률 50%" in tr.tag_stats_block(1, min_n=2)
    assert tr.tag_stats_block(1, min_n=5) == ""
