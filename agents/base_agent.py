"""
NPS Swarm v1.0 - Base Agent
Local-LLM-powered agent base class with dynamic model allocation.
"""
import logging
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any, Optional

from infra.error_log import record_error  # 사장 지시 2026-05-19 — 상세 에러를 운용지원실장이 열람 가능하게

logger = logging.getLogger("AGENT")
KST = timezone(timedelta(hours=9))

# 사장 지시 2026-07-21: LLM 비용 추적·표시 로직 전면 제거(로컬 GGUF 서버라 호출당 외부 비용 없음).
# 우상단 비용 배지·/api/cost_mode·롤업 파일 배관을 모두 삭제했다.


class BaseAgent:
    """
    Base agent class that communicates via the local OpenAI-compatible LLM server.
    Each agent has a persona, system prompt, assigned LLM model, and tool set.
    """

    def __init__(
        self,
        name: str,
        role: str,
        system_prompt: str,
        model_key: str = "quant_analyst",
        injection: Optional[Dict[str, Any]] = None,
    ):
        from config import MODEL_ASSIGNMENTS, AGENT_MAX_TOKENS, AGENT_HISTORY_TURNS
        self.name = name
        self.role = role
        self.system_prompt = system_prompt
        inj = injection or {}
        self.uid = inj.get("uid")
        # Per-uid model override wins; else admin global override; else config default.
        _ov = (inj.get("model_overrides") or {}).get(model_key) or ""
        if not _ov:
            try:
                from infra import admin_config
                _ov = admin_config.get_model_override(model_key)
            except Exception:
                _ov = ""
        self.model = _ov or MODEL_ASSIGNMENTS.get(model_key, "")
        # Local server is unauthenticated. Keep the attribute for call-site compatibility.
        self.api_key = ""
        self.conversation_history: List[Dict[str, str]] = []
        # ── Cost-reduction knobs ──────────────────────────────────────────
        self.max_tokens = AGENT_MAX_TOKENS.get(model_key, AGENT_MAX_TOKENS.get(role, 2048))
        self.history_turns = AGENT_HISTORY_TURNS
        # Coresight RAG 주입 대상: 프롬프트에 query_coresight 툴 줄이 주입된 에이전트
        # (= admin 게이트를 이미 통과한 macro/news/post-manager). query_coresight 가
        # 내부에서 admin 재검증하므로 이중 방어. 비대상이면 None → 주입 안 함.
        self.coresight_uid = self.uid if "query_coresight" in (system_prompt or "") else None

    async def _coresight_context(self, message: str) -> str:
        """이 메시지에 관련된 Coresight 과거 지식을 검색해 주입용 블록으로 반환.
        admin 미대상·오류·빈결과면 '' (에이전트 루프 절대 안 죽인다, fail-soft)."""
        if not getattr(self, "coresight_uid", None):
            return ""
        try:
            from tools.coresight_rag import query_coresight
            out = await query_coresight((message or "")[:500], top_k=3, uid=self.coresight_uid)
            if out and not any(k in out for k in ("비활성", "찾지 못했", "설정 오류")):
                return "[Coresight 과거 지식 — 참고]\n" + out
        except Exception as e:  # noqa: BLE001
            logger.debug(f"[{self.name}] Coresight 주입 실패(무시): {e}")
        return ""

    async def think(
        self,
        message: str,
        context: Optional[str] = None,
        *,
        max_tokens: Optional[int] = None,
        timeout_sec: int = 300,
        thinking: Optional[bool] = None,
    ) -> str:
        """
        Process a message and return the agent's response.

        Args:
            message: Input message from another agent or orchestrator
            context: Optional additional context (e.g., market data)
            max_tokens/thinking/timeout_sec: 정형·경량 단계의 호출별 제한. 생략하면
                기존 에이전트 기본값과 모델의 thinking 설정을 그대로 사용한다.

        Returns:
            Agent's response text
        """
        messages = [{"role": "system", "content": self.system_prompt}]

        # Add trailing conversation history only (was: last 10 messages → now last N, default 3)
        for h in self.conversation_history[-self.history_turns:]:
            messages.append(h)

        # Coresight RAG (opt-in·admin·fail-soft): 이 메시지 관련 과거 지식을 context 에 병합
        cs_block = await self._coresight_context(message)
        if cs_block:
            context = f"{context}\n\n{cs_block}" if context else cs_block

        user_content = message
        if context:
            user_content = f"[컨텍스트]\n{context}\n\n[메시지]\n{message}"

        messages.append({"role": "user", "content": user_content})

        _max_tokens = int(max_tokens if max_tokens is not None else self.max_tokens)
        try:
            from infra.local_llm_client import chat_completion
            data = await chat_completion(
                api_key=self.api_key, model=self.model, messages=messages,
                max_tokens=_max_tokens, temperature=0.3, timeout_sec=timeout_sec,
                thinking=thinking,
            )

            # 공급자가 빈 JSON/None을 반환해도 명시적 오류로 변환한다.
            # 방어 코드로 비정상 응답을 빈 reply + 명시적 에러 메시지로 변환.
            if not data or not isinstance(data, dict):
                logger.error(f"[{self.name}] 비정상 응답 (빈 JSON 또는 dict 아님): {str(data)[:200]}")
                return f"[{self.name} 에러] API 응답 비어있음 (재시도 필요)"
            _choices = data.get("choices") or []
            if not _choices or not isinstance(_choices, list):
                logger.error(f"[{self.name}] choices 비어있음: {str(data)[:200]}")
                return f"[{self.name} 에러] API 응답에 choices 없음 (재시도 필요)"
            reply = (_choices[0] or {}).get("message", {}).get("content", "") or ""

            # 디버그+복원력 2026-05-19: HTTP 200인데 content가 비어있는 경우
            # (주로 finish_reason="length" — 큰 프롬프트 + 작은 max_tokens; 전략리서치팀장
            # 0-char 사례). finish_reason/usage를 남겨 원인을 진단하고, max_tokens를
            # 키워 1회 재시도한다. (캐시 오염 수정과 함께 — 빈 응답이 캐시되면 안 됨.)
            if not (reply or "").strip():
                _fr = (_choices[0] or {}).get("finish_reason")
                _u0 = data.get("usage") or {}
                logger.warning(
                    f"[{self.name}] 빈 응답 (status=200, finish_reason={_fr}, "
                    f"prompt_tok={_u0.get('prompt_tokens')}, completion_tok={_u0.get('completion_tokens')}, "
                    f"max_tokens={_max_tokens}) → max_tokens 상향 후 1회 재시도")
                # ceiling 2026-07-21 16000→64000(사장 지시): 추론 ON 에이전트 base 가
                # 64000 이라 16000 실링은 '상향'이 아니라 오히려 삭감이었다. 로컬 모델 추론이
                # 1.1k~10.7k 로 길어 빈 응답의 주원인이 예산 부족인 만큼, 재시도는 확실히
                # 더 큰 예산으로 간다(슬롯 131072 이내).
                _retry_max = min(max(_max_tokens * 2, 4000), 64000)
                try:
                    _d2 = await chat_completion(
                        api_key=self.api_key, model=self.model, messages=messages,
                        max_tokens=_retry_max, temperature=0.3, timeout_sec=timeout_sec,
                        thinking=thinking,
                    )
                    _c2 = _d2.get("choices") or []
                    if _c2:
                        _r2c = (_c2[0] or {}).get("message", {}).get("content", "") or ""
                        if _r2c.strip():
                            reply = _r2c
                            data = _d2
                            logger.info(f"[{self.name}] 빈응답 재시도 성공 "
                                        f"({len(reply)} chars, max_tokens={_retry_max})")
                except Exception as _re:
                    record_error(self.name, _re, context=(
                        f"빈응답 재시도 예외 | model={self.model} | max_tokens={_retry_max}"), uid=self.uid)

            # 사장 지시(재발 2026-07-29): 마크다운 굵게(`**`)는 채팅 UI에서 그대로 노출된다.
            # 프롬프트 금지만으로는 로컬 모델이 계속 쓰므로 여기서 결정론적으로 제거한다
            # (히스토리에 넣기 전에 지워 다음 턴이 `**` 를 흉내내지 않게 한다).
            from infra.local_llm_client import strip_markdown_emphasis
            reply = strip_markdown_emphasis(reply)

            # Update history (bounded — only the last few exchanges are ever resent anyway)
            self.conversation_history.append({"role": "user", "content": user_content})
            self.conversation_history.append({"role": "assistant", "content": reply})
            if len(self.conversation_history) > max(self.history_turns * 2, 6):
                self.conversation_history = self.conversation_history[-(self.history_turns * 2):]

            logger.info(f"[{self.name}] 응답 생성 완료 ({len(reply)} chars, model: {self.model})")
            return reply

        except Exception as e:
            # 사장 지시 2026-05-19: TimeoutError 등 str(e)='' 인 예외도 타입+repr+트레이스백을
            # 남겨 운용지원실장이 '어디서 왜' 났는지 진단할 수 있게 한다.
            record_error(self.name, e, context=f"think() | model={self.model}", uid=self.uid)
            return f"[{self.name} 에러] {type(e).__name__}: {str(e) or repr(e)}"

    def reset_history(self):
        self.conversation_history.clear()

    def __repr__(self):
        return f"<Agent:{self.name} role={self.role} model={self.model}>"
