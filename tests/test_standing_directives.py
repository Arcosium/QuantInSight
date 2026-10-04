"""ITEM6 회귀 테스트: 계정별 상시 지시사항 영속성·격리·주입.

- 지시사항이 uid별로 저장되고 uid가 다른 계정에는 보이지 않는다
- 주식운용실장 컨텍스트 블록에 지시사항이 포함된다
- 중복 추가는 멱등 처리된다
"""
import pytest
from datetime import datetime
from pathlib import Path


@pytest.fixture
def isolated_profiles(tmp_path, monkeypatch):
    """data/profiles 를 tmp_path로 격리."""
    import infra.standing_directives as sd
    monkeypatch.setattr(sd, "_PROFILES_DIR", tmp_path / "profiles")
    monkeypatch.setattr(sd, "_DATA_DIR", tmp_path)
    return sd


def test_append_and_load(isolated_profiles):
    sd = isolated_profiles
    uid = 42
    sd.append_directive(uid, "달러 자산 우선 편입")
    directives = sd.load(uid)
    assert len(directives) == 1
    assert "달러 자산 우선 편입" in directives[0]["text"]


def test_isolation_between_uids(isolated_profiles):
    """uid A의 지시가 uid B에는 보이지 않아야 한다."""
    sd = isolated_profiles
    uid_a, uid_b = 1, 2
    sd.append_directive(uid_a, "A 계정 전용 지시")
    sd.append_directive(uid_b, "B 계정 전용 지시")

    directives_a = sd.load(uid_a)
    directives_b = sd.load(uid_b)

    assert len(directives_a) == 1
    assert "A 계정" in directives_a[0]["text"]
    assert len(directives_b) == 1
    assert "B 계정" in directives_b[0]["text"]

    # 교차 오염 확인
    all_a_texts = [d["text"] for d in directives_a]
    all_b_texts = [d["text"] for d in directives_b]
    assert not any("B 계정" in t for t in all_a_texts)
    assert not any("A 계정" in t for t in all_b_texts)


def test_idempotent_append(isolated_profiles):
    """동일 내용 중복 추가는 1건만 저장."""
    sd = isolated_profiles
    uid = 99
    text = "매크로 붕괴 대비 달러 전환"
    added_first = sd.append_directive(uid, text)
    added_second = sd.append_directive(uid, text)

    assert added_first is True
    assert added_second is False  # 중복 → False
    assert len(sd.load(uid)) == 1


def test_clear_directives(isolated_profiles):
    sd = isolated_profiles
    uid = 7
    sd.append_directive(uid, "지시1")
    sd.append_directive(uid, "지시2")
    count = sd.clear_directives(uid)
    assert count == 2
    assert sd.load(uid) == []


def test_build_orchestrator_directive_block_present(isolated_profiles):
    """지시가 있으면 블록 텍스트에 지시 내용 포함."""
    sd = isolated_profiles
    uid = 10
    sd.append_directive(uid, "원화 자산 최소화")
    block = sd.build_orchestrator_directive_block(uid)
    # TASK B: 헤더/푸터 문구가 완화된 형태로 업데이트됨
    assert "사장님 상시 지침" in block
    assert "원화 자산 최소화" in block
    assert "균형 있게" in block
    assert "파이썬 리스크·guardrail 게이트" in block


def test_build_orchestrator_directive_block_empty_when_no_directives(isolated_profiles):
    """지시가 없으면 빈 문자열 반환 — 프롬프트에 불필요한 노이즈 없음."""
    sd = isolated_profiles
    block = sd.build_orchestrator_directive_block(uid=999)
    assert block == ""


def test_build_orchestrator_directive_block_none_uid(isolated_profiles):
    """uid=None 이면 빈 문자열."""
    sd = isolated_profiles
    assert sd.build_orchestrator_directive_block(None) == ""


def test_multiple_directives_all_present_in_block(isolated_profiles):
    """여러 지시가 모두 블록에 포함되어야 한다."""
    sd = isolated_profiles
    uid = 20
    sd.append_directive(uid, "달러 MMF 핵심 축")
    sd.append_directive(uid, "금·비트코인 배제")
    sd.append_directive(uid, "리밸런싱 트리거 수립")
    block = sd.build_orchestrator_directive_block(uid)
    assert "달러 MMF 핵심 축" in block
    assert "금·비트코인 배제" in block
    assert "리밸런싱 트리거 수립" in block


def test_day_only_directive_expires_after_created_day(isolated_profiles, monkeypatch):
    """'오늘' 지시를 상시로 잘못 저장해도 다음 날부터 프롬프트에는 주입하지 않는다."""
    sd = isolated_profiles
    uid = 21
    monkeypatch.setattr(sd, "_now_kst", lambda: "2026-07-22 09:52:43")
    sd.append_directive(uid, "오늘 신규 매수는 최소화하고 차익을 단계적으로 실현하라")
    monkeypatch.setattr(sd, "_now_dt_kst", lambda: datetime(2026, 7, 23, 9, 0, tzinfo=sd.KST))
    assert sd.load(uid), "감사 기록은 삭제하면 안 됨"
    assert sd.build_orchestrator_directive_block(uid) == ""


def test_today_onward_directive_does_not_expire(isolated_profiles, monkeypatch):
    sd = isolated_profiles
    uid = 22
    monkeypatch.setattr(sd, "_now_kst", lambda: "2026-07-22 09:52:43")
    sd.append_directive(uid, "오늘부터 계속 익절 기준을 지켜라")
    monkeypatch.setattr(sd, "_now_dt_kst", lambda: datetime(2026, 8, 10, 9, 0, tzinfo=sd.KST))
    assert "오늘부터 계속" in sd.build_orchestrator_directive_block(uid)


def test_persistence_across_calls(isolated_profiles, tmp_path):
    """append 후 모듈을 재-import 해도 데이터가 유지되는지 확인
    (파일 기반 영속성 검증 — 메모리 캐시가 아님)."""
    sd = isolated_profiles
    uid = 55
    sd.append_directive(uid, "영속 테스트 지시사항")

    # 파일이 실제로 존재하는지 확인
    p = sd._directives_path(uid)
    assert p.exists()

    # 파일 내용 직접 확인
    import json
    data = json.loads(p.read_text(encoding="utf-8"))
    assert isinstance(data, list)
    assert len(data) == 1
    assert "영속 테스트" in data[0]["text"]


def test_seed_admin_directive_safe_no_op_without_admin(isolated_profiles, monkeypatch):
    """admin 계정이 DB에 없어도 크래시 없이 종료 (safe no-op)."""
    # auth_store.find_user_by_username 을 None 반환으로 패치
    import infra.auth_store as _as
    monkeypatch.setattr(_as, "find_user_by_username", lambda *a, **kw: None)

    # 크래시 없이 실행되어야 함
    try:
        from infra.standing_directives import seed_admin_directive as _real_seed
        _real_seed()
    except Exception as e:
        pytest.fail(f"seed_admin_directive raised unexpectedly: {e}")


# ─── 삭제 tombstone / 부활 방지 테스트 (사장 지시 2026-05-20) ───────────────────

def _make_fake_user(uid: int):
    return {"id": uid, "username": "hh09080"}


def test_seed_adds_directive_first_time(isolated_profiles, monkeypatch):
    """최초 시드: 지시 1건 추가, 삭제 안 했으므로 tombstone 없음."""
    sd = isolated_profiles
    import infra.auth_store as _as
    fake_uid = 77
    monkeypatch.setattr(_as, "find_user_by_username", lambda *a, **kw: _make_fake_user(fake_uid))

    sd.seed_admin_directive()

    assert len(sd.load(fake_uid)) == 1
    macro_id = sd._directive_id(sd.MACRO_COLLAPSE_DIRECTIVE)
    assert not sd._is_tombstoned(fake_uid, macro_id), "삭제 안 했으면 tombstone 없어야 함"


def test_permanent_delete_via_clear(isolated_profiles, monkeypatch):
    """시드 → clear_directives → 재시드해도 부활하지 않는다 (명시 삭제 = 영구)."""
    sd = isolated_profiles
    import infra.auth_store as _as
    fake_uid = 88
    monkeypatch.setattr(_as, "find_user_by_username", lambda *a, **kw: _make_fake_user(fake_uid))

    sd.seed_admin_directive()
    assert len(sd.load(fake_uid)) == 1

    deleted = sd.clear_directives(fake_uid)
    assert deleted == 1
    assert sd.load(fake_uid) == []

    macro_id = sd._directive_id(sd.MACRO_COLLAPSE_DIRECTIVE)
    assert sd._is_tombstoned(fake_uid, macro_id), "명시 삭제 → tombstone 기록되어야 함"

    sd.seed_admin_directive()  # 재시작 시뮬
    assert sd.load(fake_uid) == [], "tombstone 있으면 재시드 금지"


def test_permanent_delete_via_remove(isolated_profiles, monkeypatch):
    """시드 → remove_directive → 재시드해도 부활하지 않는다."""
    sd = isolated_profiles
    import infra.auth_store as _as
    fake_uid = 89
    monkeypatch.setattr(_as, "find_user_by_username", lambda *a, **kw: _make_fake_user(fake_uid))

    sd.seed_admin_directive()
    did = sd.load(fake_uid)[0]["id"]

    assert sd.remove_directive(fake_uid, did) is True
    assert sd.load(fake_uid) == []

    sd.seed_admin_directive()  # 재시작 시뮬
    assert sd.load(fake_uid) == [], "remove 후 tombstone 이 재시드를 차단해야 한다"


def test_reseed_readds_when_not_deleted(isolated_profiles, monkeypatch):
    """삭제 안 한 상태에서 지시 파일만 소실되면 재부팅 시 다시 채워진다.
    (사장 요구사항 2: 삭제 버튼 안 누르고 재부팅 → 다시 추가)."""
    sd = isolated_profiles
    import infra.auth_store as _as
    fake_uid = 90
    monkeypatch.setattr(_as, "find_user_by_username", lambda *a, **kw: _make_fake_user(fake_uid))

    sd.seed_admin_directive()
    assert len(sd.load(fake_uid)) == 1

    # 명시 삭제가 아니라, 지시 파일만 소실 (프로필 디렉토리 리셋 시뮬)
    sd._directives_path(fake_uid).unlink()
    assert sd.load(fake_uid) == []

    sd.seed_admin_directive()  # 재부팅
    assert len(sd.load(fake_uid)) == 1, "삭제 안 했으면 재부팅 시 다시 채워져야 한다"


def test_tombstone_survives_profile_dir_wipe(isolated_profiles, monkeypatch):
    """핵심 회귀: 명시 삭제 후 프로필 디렉토리가 통째로 날아가도 부활하지 않는다.
    (tombstone 이 data/ 최상위에 있어 프로필 리셋에 생존 — 과거 부활 버그 근본 차단)."""
    import shutil
    sd = isolated_profiles
    import infra.auth_store as _as
    fake_uid = 93
    monkeypatch.setattr(_as, "find_user_by_username", lambda *a, **kw: _make_fake_user(fake_uid))

    sd.seed_admin_directive()
    sd.clear_directives(fake_uid)  # 명시 삭제 → tombstone

    # 프로필 디렉토리 통째로 삭제 (과거 부활 버그의 원인)
    shutil.rmtree(sd._profile_dir(fake_uid), ignore_errors=True)

    sd.seed_admin_directive()  # 재부팅
    assert sd.load(fake_uid) == [], "프로필 리셋돼도 명시 삭제는 영구 보존되어야 한다"


def test_manual_readd_clears_tombstone(isolated_profiles, monkeypatch):
    """삭제 후 같은 지시를 수동으로 다시 추가하면 tombstone 이 해제된다 (삭제 의사 철회)."""
    sd = isolated_profiles
    import infra.auth_store as _as
    fake_uid = 94
    monkeypatch.setattr(_as, "find_user_by_username", lambda *a, **kw: _make_fake_user(fake_uid))

    sd.seed_admin_directive()
    sd.clear_directives(fake_uid)
    macro_id = sd._directive_id(sd.MACRO_COLLAPSE_DIRECTIVE)
    assert sd._is_tombstoned(fake_uid, macro_id)

    # 사용자가 동일 지시를 다시 추가
    sd.append_directive(fake_uid, sd.MACRO_COLLAPSE_DIRECTIVE)
    assert not sd._is_tombstoned(fake_uid, macro_id), "재추가 시 tombstone 해제"
    assert len(sd.load(fake_uid)) == 1

    sd.seed_admin_directive()  # 재부팅 — 멱등 유지
    assert len(sd.load(fake_uid)) == 1


def test_multiple_restarts_idempotent(isolated_profiles, monkeypatch):
    """여러 번 재시작해도 지시는 1건만 (append 멱등)."""
    sd = isolated_profiles
    import infra.auth_store as _as
    fake_uid = 92
    monkeypatch.setattr(_as, "find_user_by_username", lambda *a, **kw: _make_fake_user(fake_uid))

    for _ in range(5):
        sd.seed_admin_directive()

    assert len(sd.load(fake_uid)) == 1, f"5번 재시드해도 지시는 1건이어야 함."
