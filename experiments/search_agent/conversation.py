"""Session state around fresh, independently grounded single-question runs."""
import asyncio
import copy
from dataclasses import asdict, dataclass, replace
import time

from .core import AgentLimits, AgentResult, run
from .resolution import prepare_resolution_context


@dataclass(frozen=True)
class ConversationLimits:
    max_rounds: int = 6
    history_tokens: int = 3000
    resolver_seconds: float = 10


@dataclass
class ConversationResult(AgentResult):
    session_id: str = ""
    question: str = ""
    effective_question: str = ""
    history_truncated: bool = False


class Conversation:
    def __init__(self, session_id, *, limits=None):
        if not isinstance(session_id, str) or not session_id.strip():
            raise ValueError("session_id_required")
        self.session_id = session_id
        self.limits = limits or ConversationLimits()
        if (type(self.limits.max_rounds) is not int or not 0 <= self.limits.max_rounds <= 6
                or type(self.limits.history_tokens) is not int or not 0 < self.limits.history_tokens <= 3000
                or not 0 < self.limits.resolver_seconds <= 10):
            raise ValueError("invalid_conversation_limits")
        self._turns = []
        self._pending = None
        self._history_truncated = False
        self._lock = asyncio.Lock()

    @property
    def turns(self):
        return copy.deepcopy(self._turns)

    @property
    def pending(self):
        return copy.deepcopy(self._pending)

    async def clear(self):
        async with self._lock:
            self._clear()

    def _clear(self):
        self._turns = []
        self._pending = None
        self._history_truncated = False

    def _clarify(self, question, clarification, reset):
        if reset:
            self._clear()
        pending = self.pending
        if pending is None:
            pending = {"question": question, "clarification": clarification, "messages": []}
        else:
            pending["clarification"] = clarification
            pending["messages"].append(question)
        self._pending = pending

    def _commit(self, question, result, context, reset):
        raw = question
        if not reset and self._pending:
            raw = "\n".join([self._pending["question"], *self._pending["messages"], question])
        retained = len(context["history"]) // 2
        turns = self._turns[-retained:] if retained and not reset else []
        turns = [*turns, {"user": raw, "assistant": result.answer, "effective_question": result.effective_question}]
        truncated = len(turns) > self.limits.max_rounds
        self._turns = turns[-self.limits.max_rounds:] if self.limits.max_rounds else []
        self._pending = None
        self._history_truncated = truncated or (result.history_truncated and not reset)

    async def ask(self, question, backend, limits=None):
        limits = limits or AgentLimits()
        if not 0 <= limits.max_actions <= 2 or not 1 <= limits.max_decisions <= 3 or not 0 < limits.search_seconds <= limits.total_seconds <= 60:
            raise ValueError("invalid_agent_limits")
        async with self._lock:
            return await self._ask(question, backend, limits)

    async def _ask(self, question, backend, limits):
        start = time.monotonic()
        result = ConversationResult(session_id=self.session_id, question=question)
        try:
            context = prepare_resolution_context(question, self._turns, self._pending, backend.count_tokens,
                                                 max_rounds=self.limits.max_rounds, budget=self.limits.history_tokens)
            context["history_truncated"] |= self._history_truncated
            result.history_truncated = context["history_truncated"]
            resolution = await asyncio.wait_for(backend.resolve(context),
                                                timeout=min(self.limits.resolver_seconds, limits.total_seconds - (time.monotonic() - start)))
            event = {"event": "resolution", "value": asdict(resolution)}
            result.trace.append(event)
            if resolution.kind == "clarification":
                result.status = "needs_clarification"
                result.stop_reason = "missing_user_information"
                result.answer = resolution.clarification
                self._clarify(question, result.answer, resolution.reset)
            elif resolution.kind == "question":
                result.effective_question = resolution.question
                if result.effective_question != question and not result.effective_question.endswith("\n用户本轮原话：" + question):
                    result.effective_question += "\n用户本轮原话：" + question
                elapsed = time.monotonic() - start
                if elapsed >= limits.search_seconds:
                    result.status, result.stop_reason = "timeout", "search_timeout"
                else:
                    remaining = replace(limits, total_seconds=limits.total_seconds - elapsed, search_seconds=limits.search_seconds - elapsed)
                    single = await run(result.effective_question, backend, remaining)
                    result = ConversationResult(**asdict(single), session_id=self.session_id, question=question,
                                                effective_question=result.effective_question, history_truncated=result.history_truncated)
                    result.trace.insert(0, event)
                    if result.status == "ok":
                        if result.stop_reason == "missing_user_information":
                            result.status = "needs_clarification"
                            self._clarify(question, result.answer, resolution.reset)
                        else:
                            self._commit(question, result, context, resolution.reset)
            else:
                raise ValueError("invalid_resolution_kind")
        except TimeoutError:
            result.status, result.stop_reason = "timeout", "resolution_timeout"
        except Exception as exc:
            result.status, result.stop_reason = "technical_error", "resolution_error"
            result.error = str(exc)[:200]
        result.latency_ms = round((time.monotonic() - start) * 1000)
        result.usage = copy.deepcopy(getattr(backend, "metrics", {}))
        return result
