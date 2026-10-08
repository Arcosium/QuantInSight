import json

from web import experiment_store as store


def _paths(monkeypatch, tmp_path):
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    monkeypatch.setattr(store, "_PROFILES", tmp_path / "profiles")
    monkeypatch.setattr(store, "_ACTIVE", tmp_path / "active_strategy.json")
    monkeypatch.setattr(store.config, "PRIVATE_DATA_DIR", tmp_path)
    monkeypatch.setattr(store.config, "SIGNALS_JSONL", runtime / "signals.jsonl")
    monkeypatch.setattr(store.config, "TRADES_CSV", runtime / "trades.csv")
    monkeypatch.setattr(store.config, "POSITIONS_JSON", runtime / "positions.json")
    return runtime


def _strategy(name):
    return {"name": name, "financial_logic": "전체", "buy_logic": "RSI < 30",
            "sell_logic": "RSI > 70", "period_months": 24,
            "portfolio_strategy": "equal_weight", "use_tax_fee": True}


def test_strategy_switch_requires_confirmation(monkeypatch, tmp_path):
    _paths(monkeypatch, tmp_path)
    first = store.activate("browser-a", _strategy("처음"), nav=100, signal_count=0)
    assert first["changed"] is True
    warning = store.activate("browser-a", _strategy("다음"), nav=110, signal_count=3)
    assert warning["requires_reset"] is True
    assert warning["current"]["return_pct"] == 10.0


def test_reset_archives_strategy_files_but_not_autofolio(monkeypatch, tmp_path):
    runtime = _paths(monkeypatch, tmp_path)
    (runtime / "signals.jsonl").write_text('{"kind":"BUY"}\n', encoding="utf-8")
    (runtime / "trades.csv").write_text("one trade\n", encoding="utf-8")
    (runtime / "positions.json").write_text('[{"code":"005930"}]', encoding="utf-8")
    autofolio = tmp_path / "autofolio_state.json"
    autofolio.write_text('{"cash":123}', encoding="utf-8")

    archive = store.reset_strategy_runtime("old-strategy")

    assert (archive / "signals.jsonl").exists()
    assert (runtime / "signals.jsonl").read_text(encoding="utf-8") == ""
    assert json.loads((runtime / "positions.json").read_text(encoding="utf-8")) == []
    assert autofolio.read_text(encoding="utf-8") == '{"cash":123}'
