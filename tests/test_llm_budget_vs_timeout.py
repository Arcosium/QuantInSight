"""LLM 토큰 예산 ↔ 타임아웃 정합성 회귀 (사장 지시 2026-07-22).

2026-07-22 사고: `AGENT_MAX_TOKENS` 의 pro(thinking ON) 예산이 64000 인데
`LOCAL_LLM_TIMEOUT_SEC` 은 900초였다. 로컬 llama-server 실측 디코드가 약 53 t/s 이므로
64000 토큰은 1,208초가 필요해 **예산을 다 쓰는 생성은 구조적으로 완주 불가**였다.
타임아웃에 잘리면 재시도는 토큰 0부터 다시 시작하므로 그레이스만 태우다 최종 실패한다.
실제 24시간 로그에서 재시도 13회·최종 실패 4회(원자재운용실장·주식운용실장 등 전부 pro)가 관측됐다.

이 테스트는 그 조합이 다시 어긋나는 것을 막는다. 예산을 올릴 거면 타임아웃도 같이 올려야 한다.
"""
import config
from infra import local_llm_client as llm

# llama-server 실측 디코드 속도(2026-07-22, Qwen3.6-35B-A3B Q8_0, slot 3 print_timing).
# 보수적으로 잡는다 — 실측보다 낮게 잡아야 테스트가 안전한 쪽으로 실패한다.
OBSERVED_TPS = 53.0
# 프롬프트 길이 변동·GPU 경합을 흡수할 최소 배수.
MIN_SAFETY_FACTOR = 1.5
# 관측된 추론(CoT) 길이 상한 — config.py 주석의 1.1k~10.7k 분포 근거.
OBSERVED_MAX_REASONING_TOKENS = 10_700

# macro_analyst 는 2026-07-22 에 _FLASH 로 내려갔다(작문 작업 — CoT 불필요).
PRO_AGENTS = ("chief_orchestrator", "post_manager",
              "ops_support", "bond_manager", "commodity_manager")


def _seconds_for(tokens: int) -> float:
    return tokens / OBSERVED_TPS


def test_every_agent_budget_fits_in_timeout():
    """모든 에이전트의 예산이 타임아웃 안에 안전배수까지 포함해 완주 가능해야 한다."""
    timeout = llm._LOCAL_LLM_TIMEOUT_SEC
    offenders = []
    for agent, budget in config.AGENT_MAX_TOKENS.items():
        need = _seconds_for(budget)
        if need * MIN_SAFETY_FACTOR > timeout:
            offenders.append(
                f"{agent}: 예산 {budget}토큰 = {need:.0f}초 × 안전배수 {MIN_SAFETY_FACTOR}"
                f" = {need * MIN_SAFETY_FACTOR:.0f}초 > 타임아웃 {timeout}초")
    assert not offenders, (
        "예산이 타임아웃을 넘겨 완주 불가한 에이전트가 있습니다 "
        "(예산을 낮추거나 LOCAL_LLM_TIMEOUT_SEC 을 올리십시오):\n  " + "\n  ".join(offenders))


def test_grace_allows_at_least_one_retry():
    """그레이스는 타임아웃보다 커야 한다 — 같으면 재시도 여지가 0이라 모델 재기동을 못 견딘다."""
    assert llm._LOCAL_LLM_GRACE_SEC > llm._LOCAL_LLM_TIMEOUT_SEC, (
        f"GRACE({llm._LOCAL_LLM_GRACE_SEC}) 가 TIMEOUT({llm._LOCAL_LLM_TIMEOUT_SEC}) 이하라 "
        "타임아웃 시 재시도가 즉시 포기됩니다.")


def test_pro_budget_still_covers_observed_reasoning():
    """반대 방향 회귀도 막는다 — 예산을 너무 낮추면 추론이 예산을 다 써 content 가 빈다.
    (실측: 12000 은 2/6 정상, 8000 은 0/3. 관측 추론 상한의 최소 2배는 유지한다.)"""
    floor = OBSERVED_MAX_REASONING_TOKENS * 2
    for agent in PRO_AGENTS:
        budget = config.AGENT_MAX_TOKENS[agent]
        assert budget >= floor, (
            f"{agent} 예산 {budget} < {floor} — 추론이 예산을 소진해 빈 응답이 날 수 있습니다.")


def test_pro_agents_are_the_thinking_tier():
    """예산 정책이 붙는 대상이 실제 thinking ON 에이전트와 일치하는지 확인."""
    for agent in PRO_AGENTS:
        assert config.MODEL_ASSIGNMENTS[agent].endswith("+thinking"), (
            f"{agent} 가 thinking 티어가 아닙니다 — PRO_AGENTS 목록을 갱신하십시오.")
