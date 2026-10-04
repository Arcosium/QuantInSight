"""정책 변경 승인 인박스 — 토요일 ops 제안 → 사장 승인 시 오버라이드 적용."""
import infra.policy_approval_inbox as box
from infra import profile_overrides


def _isolate(tmp_path, monkeypatch):
    monkeypatch.setattr(box, "_PROFILES_DIR", tmp_path)
    return tmp_path


def test_enqueue_then_list(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    box.enqueue(7, "ALLOW_US_STOCKS", True, False, "미국 재개 권고")
    p = box.list_pending(7)
    assert len(p) == 1
    assert p[0]["key"] == "ALLOW_US_STOCKS"
    assert p[0]["proposed_value"] is True
    assert p[0]["current_value"] is False
    assert p[0]["status"] == "pending"


def test_enqueue_dedupes_by_key_updates_value(tmp_path, monkeypatch):
    # 같은 key 를 다시 제안하면 항목이 늘지 않고 최신 제안값·근거로 덮어쓴다.
    # (기대치 갱신 2026-07-22) 예전엔 2회차를 (proposed=False, current=False) 로 넣었는데,
    # 2026-07-21 사장 지시로 proposed == current 는 '실제 변경 없음'이라 적재 대상이 아니게 됐다
    # → 그 픽스처로는 dedupe 가 아니라 no-op prune 을 타서 pending 이 0 이 된다.
    # 검증하려는 행동(키당 1건 + 최신값 반영)은 그대로 두고, 2회차를 '진짜 변경'
    # (그 사이 플래그가 켜졌고 이제 끄자는 제안)으로 바꾼다. no-op prune 자체는 아래 별도 테스트.
    _isolate(tmp_path, monkeypatch)
    box.enqueue(7, "ALLOW_US_STOCKS", True, False, "v1")
    box.enqueue(7, "ALLOW_US_STOCKS", False, True, "v2")
    p = box.list_pending(7)
    assert len(p) == 1
    assert p[0]["proposed_value"] is False
    assert p[0]["rationale"] == "v2"


def test_enqueue_noop_is_not_queued_and_prunes_existing(tmp_path, monkeypatch):
    """사장 지시 2026-07-21: proposed == current 는 승인 요청 대상이 아니다.
    적재를 거부할 뿐 아니라 이미 쌓여 있던 같은 key 의 대기 항목도 정리한다."""
    _isolate(tmp_path, monkeypatch)
    assert box.enqueue(7, "ALLOW_DERIVATIVES", True, True, "무변경") is None
    assert box.list_pending(7) == []
    box.enqueue(7, "ALLOW_DERIVATIVES", True, False, "켜자")
    assert len(box.list_pending(7)) == 1
    assert box.enqueue(7, "ALLOW_DERIVATIVES", False, False, "역시 그대로") is None
    assert box.list_pending(7) == []


def test_approve_applies_override(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    called = {}
    monkeypatch.setattr(profile_overrides, "set_overrides",
                        lambda uid, params: called.setdefault("v", (uid, dict(params))))
    box.enqueue(7, "ALLOW_US_STOCKS", True, False, "x")
    assert box.approve(7, "ALLOW_US_STOCKS") is True
    assert called["v"] == (7, {"ALLOW_US_STOCKS": True})
    assert box.list_pending(7) == []   # 더 이상 pending 아님


def test_reject_removes_without_applying(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    monkeypatch.setattr(profile_overrides, "set_overrides",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("적용 금지")))
    box.enqueue(7, "ALLOW_DERIVATIVES", True, False, "x")
    assert box.reject(7, "ALLOW_DERIVATIVES") is True
    assert box.list_pending(7) == []


def test_approve_missing_returns_false(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    assert box.approve(7, "NOPE") is False


def test_list_pending_none_uid_safe(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    assert box.list_pending(None) == []
