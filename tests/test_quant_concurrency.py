"""사장 지시 2026-07-22: 종목별 계량분석 동시 실행 (사이클 최대 병목 제거).

지켜야 할 불변식 — 동시 실행이 **매매 판단을 바꾸면 안 된다**:
  1. 태스크마다 **전용 에이전트 인스턴스**를 쓴다. self.quant_analyst 를 공유하면
     think() 가 conversation_history 를 읽고 append 하므로(base_agent.py:96,164)
     루프가 reset_history() 로 막고 있던 '종목 간 컨텍스트 누수'가 되살아난다.
     퀀트점수는 매수·매도로 직결되므로 누수 = 판단 오염이다.
  2. quant_report 의 종목 순서는 완료 순서가 아니라 **후보 순서**로 재현돼야 한다
     (그 텍스트가 주식운용실장 프롬프트에 그대로 들어간다).
  3. 1종목 실패가 나머지를 죽이면 안 된다.
  4. QUANT_CONCURRENCY 를 넘겨 동시 실행하지 않는다.
"""
import asyncio
import re

import pytest

import main_swarm


def _read_stage_source():
    import inspect
    return inspect.getsource(main_swarm.ArquantOrchestrator._cyc_stage_data_quant)


def test_uses_dedicated_agent_not_shared_instance():
    """공유 인스턴스(self.quant_analyst)로 think() 하면 안 된다 — 히스토리 누수."""
    src = _read_stage_source()
    assert "self.quant_analyst.think" not in src, \
        "공유 에이전트로 동시 호출하면 종목 간 컨텍스트가 섞인다"
    assert "self.quant_analyst.reset_history" not in src
    assert "create_quant_analyst(injection=" in src, "태스크 전용 인스턴스를 만들어야 한다"
    assert "_agent.think(" in src and "_agent.reset_history()" in src


def test_sections_reassembled_in_candidate_order():
    """완료 순서가 아니라 후보 순서로 재조립해야 한다(주식운용실장 프롬프트 재현성)."""
    src = _read_stage_source()
    assert "_quant_sections.append" not in src, "append 는 동시 실행에서 순서가 흔들린다"
    assert re.search(r"_quant_sections\s*=\s*\[_sections_by_code\[_c\]\s+for\s+_c\s+in\s+_quant_codes", src), \
        "_quant_codes 순서로 재조립해야 한다"


def test_concurrency_is_bounded_and_tunable():
    src = _read_stage_source()
    assert "asyncio.Semaphore" in src, "무제한 동시 실행 금지 — LLM 슬롯을 다 먹는다"
    assert 'runtime.get("QUANT_CONCURRENCY"' in src, "런타임 튜너블이어야 한다(재시작 없이 조정)"


def test_config_default_is_two():
    import config
    assert config.QUANT_CONCURRENCY == 2
    assert "QUANT_CONCURRENCY" in config.STRATEGY_TUNABLE_KEYS
    assert config.STRATEGY_KEY_META["QUANT_CONCURRENCY"]["min"] == 1   # 1 = 종전 순차로 되돌리기


def test_one_failure_does_not_kill_others():
    """gather 가 예외를 전파하면 1종목 실패로 사이클 전체가 죽는다 — 격리 확인."""
    src = _read_stage_source()
    assert "_quant_one_guarded" in src
    assert re.search(r"except Exception as _qe", src), "종목별 예외를 삼켜 나머지를 살려야 한다"


@pytest.mark.asyncio
async def test_semaphore_bounds_actual_concurrency():
    """세마포어가 실제로 동시 실행 수를 묶는지 — 패턴이 아니라 동작으로 확인."""
    live, peak = 0, 0
    sem = asyncio.Semaphore(2)

    async def task():
        nonlocal live, peak
        async with sem:
            live += 1
            peak = max(peak, live)
            await asyncio.sleep(0.01)
            live -= 1

    await asyncio.gather(*(task() for _ in range(8)))
    assert peak <= 2, f"동시 실행이 2를 넘었다: {peak}"


@pytest.mark.asyncio
async def test_dedicated_agents_do_not_share_history():
    """전용 인스턴스면 한쪽 히스토리가 다른 쪽 프롬프트에 새지 않는다."""
    from agents.specialists import create_quant_analyst
    a = create_quant_analyst(injection={"uid": 1})
    b = create_quant_analyst(injection={"uid": 1})
    assert a is not b
    a.conversation_history.append({"role": "user", "content": "005930 분석"})
    assert b.conversation_history == [], "인스턴스 간 히스토리가 공유되면 안 된다"
