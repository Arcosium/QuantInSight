import json
from infra import strategy_research as research


def test_proposal_preserves_baseline_and_never_applies(tmp_path,monkeypatch):
    import runtime
    import config
    monkeypatch.setattr(research.user_paths,"user_dir",lambda _:tmp_path)
    monkeypatch.setattr(config,"STRATEGY_TUNABLE_KEYS",["STOP_LOSS_PCT"])
    monkeypatch.setattr(runtime,"get",lambda key,uid=None:5.0)
    key=research.record_proposal(1,{"STOP_LOSS_PCT":4},rationale="test")
    first=(tmp_path/"research"/f"{key}.json").read_text()
    assert research.record_proposal(1,{"STOP_LOSS_PCT":4},rationale="changed") == key
    assert (tmp_path/"research"/f"{key}.json").read_text() == first
    record=json.loads(first)
    assert record["baseline"] == {"STOP_LOSS_PCT":5.0}
    assert record["overrides"] == {"STOP_LOSS_PCT":4.0}
    assert record["applied"] is False


def test_automatic_ops_cannot_apply_unvalidated_strategy(tmp_path,monkeypatch):
    from infra import ops_support_worker as worker, profile_overrides, ops_history
    import main_swarm
    called=[]
    monkeypatch.setattr(research.user_paths,"user_dir",lambda _:tmp_path)
    monkeypatch.setattr(research,"record_proposal",lambda *a,**kw:called.append(a) or "research-test")
    monkeypatch.setattr(profile_overrides,"set_overrides",lambda *a:(_ for _ in ()).throw(AssertionError("unvalidated apply")))
    monkeypatch.setattr(profile_overrides,"record_proposal",lambda *a:None)
    monkeypatch.setattr(ops_history,"append_run",lambda *a:None)
    monkeypatch.setattr(main_swarm,"log_response_event",lambda *a,**kw:None)
    for trigger in ("cycle","weekly"):
        worker._handle_param_tuning({"param_overrides":{"STOP_LOSS_PCT":2}},1,"ops_support",
                                   "2026-09-16 00:00:00",trigger,1)
    assert len(called) == 2
