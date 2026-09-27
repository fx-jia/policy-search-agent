"""One-pass RAG with a small literal context; no benchmark or controller access."""
import asyncio
from dataclasses import replace
import re
import time

from .core import AgentLimits, AgentResult
from .evidence import validate_views


def context_text(items):
    blocks = []
    for i, e in enumerate(items):
        title = re.sub(r"\n+", " ", e.title)
        blocks.append(f"\nID: {i}\n├── Title: {title}\n└── Content:\n{e.text}")
    return "\n------\n" + "\n\n------\n\n".join(blocks) if blocks else ""


def select_context(items, count_tokens, budget=2000):
    """Take at most the first three unique chunks, retaining whole paragraph prefixes.

    Retrieval order wins. An oversized paragraph ends that chunk's prefix;
    later paragraphs cannot lose their preceding applicability conditions.
    This does not guarantee that all required evidence fits the smaller budget.
    """
    if budget <= 0:
        raise ValueError("invalid_context_budget")
    unique = {}
    for e in items:
        previous = unique.setdefault(e.id, e)
        if (previous.document_id, previous.title, previous.text) != (e.document_id, e.title, e.text):
            raise ValueError("conflicting_chunk_identity")
    selected = []
    for e in list(unique.values())[:3]:
        if e.text.strip() and count_tokens(context_text(selected + [e])) <= budget:
            selected.append(replace(e, spans=((0, len(e.text)),), truncated=False))
            continue
        candidate = None
        # Indexed text contains soft wraps, including numeric values on their own
        # line. A line break alone cannot prove the preceding clause is complete.
        for match in re.finditer(r"[^\r\n]+", e.text):
            if not re.search(r"[。！？!?；;.]\s*[”’\"'）)\]]*\s*$", match.group()):
                continue
            end = match.end()
            view = replace(e, text=e.text[:end], spans=((0, end),), truncated=end < len(e.text))
            if count_tokens(context_text(selected + [view])) > budget:
                break
            candidate = view
        if candidate:
            selected.append(candidate)
    validate_views(selected, list(unique.values()))
    return selected


async def run(question, backend, limits=None):
    limits = limits or AgentLimits()
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question_required")
    if not 0 < limits.search_seconds <= limits.total_seconds <= 60:
        raise ValueError("invalid_time_limits")
    started = time.monotonic()
    result = AgentResult(stop_reason="one_pass")
    phase = "retrieval"
    timings = {}
    try:
        phase_start = time.monotonic()
        initial = await asyncio.wait_for(backend.search(question), limits.search_seconds)
        timings["retrieval_ms"] = round((time.monotonic() - phase_start) * 1000)
        phase = "selection"
        phase_start = time.monotonic()
        chosen = select_context(initial, backend.count_tokens)
        timings["selection_ms"] = round((time.monotonic() - phase_start) * 1000)
        timings["context_tokens"] = backend.count_tokens(context_text(chosen))
        result.trace.append({"event": "fast_context", "initial_ids": [e.id for e in initial],
                             "selected_ids": [e.id for e in chosen], "context_tokens": timings["context_tokens"]})
        result.selected = [{"chunk_id": e.id, "document_id": e.document_id, "spans": e.spans,
                            "text": e.text, "truncated": e.truncated} for e in chosen]
        if not chosen:
            result.status = "context_unavailable"
            result.stop_reason = "no_results" if not initial else "no_complete_paragraph_fits"
        else:
            phase = "generation"
            remaining = limits.total_seconds - (time.monotonic() - started)
            if remaining <= 0:
                raise TimeoutError
            phase_start = time.monotonic()
            answer = await asyncio.wait_for(backend.generate(question, chosen, native=False), remaining)
            timings["generation_ms"] = round((time.monotonic() - phase_start) * 1000)
            result.answer, result.reference = answer.get("answer", ""), answer.get("reference", {})
            if not result.answer.strip():
                result.status = result.stop_reason = "generation_error"
    except TimeoutError:
        result.status, result.stop_reason = "timeout", phase + "_timeout"
    except Exception as exc:
        result.status, result.stop_reason, result.error = "technical_error", phase + "_error", str(exc)[:200]
    result.latency_ms = round((time.monotonic() - started) * 1000)
    result.usage = {**getattr(backend, "metrics", {}), **timings}
    return result
