"""A bounded search controller; provider/runtime accounting is the backend's job."""
import asyncio
from dataclasses import dataclass, field
import re
import time

from .adapter import prepare_judge_views
from .evidence import DecisionError, evidence_segments, pack_evidence, read_chunk, validate_decision, validate_views
from .evidence import _anchor_cost, _queries, _score


@dataclass(frozen=True)
class AgentLimits:
    max_actions: int = 2
    max_decisions: int = 3
    total_seconds: float = 60
    search_seconds: float = 40
    controller_evidence_tokens: int = 4800
    final_evidence_tokens: int = 5600
    enhance: bool = True
    allow_search: bool = True
    select_context: bool = True
    fallback_on_invalid_decision: bool = True
    gap_contract: bool = True


@dataclass
class AgentResult:
    status: str = "ok"
    stop_reason: str = ""
    answer: str = ""
    reference: dict = field(default_factory=dict)
    trace: list = field(default_factory=list)
    selected: list = field(default_factory=list)
    latency_ms: int = 0
    error: str = ""
    usage: dict = field(default_factory=dict)
    fallback_used: bool = False
    controller_error: str = ""


def action_signature(action):
    if action["type"] == "search_policy":
        return ("search_policy", " ".join(action["query"].split()), tuple(sorted(set(action.get("document_ids") or []))))
    if action["type"] == "read_chunk":
        return ("read_chunk", action["chunk_id"], action["need_index"])
    return ("read_neighbors", action["chunk_id"], action["direction"])


def _gap_action(value, views, seen, backend, actions, shown_views):
    action = value["action"]
    if action["type"] == "finish" and value.get("stop_kind") == "missing_user_information":
        return action
    idx = action.get("need_index", next(i for i, n in enumerate(value["needs"]) if n["status"] != "supported"))
    need = value["needs"][idx]["text"]
    queries = _queries(need)
    ranked = sorted(views, key=lambda e: -_score(e.text, queries))
    # Explicit local reads remain model-directed, but nonexistent neighbors
    # fall through to another available action.
    if action["type"] == "read_chunk":
        return action
    if action["type"] == "read_neighbors":
        directions = backend.neighbor_directions(seen[action["chunk_id"]])
        if action["direction"] in directions or (action["direction"] == "both" and directions):
            return action
    for view in ranked:
        candidate = {"type": "read_chunk", "chunk_id": view.id, "need_index": idx}
        if not view.truncated:
            continue
        unseen = read_chunk(seen[view.id], need, shown_views, backend.count_tokens)
        if unseen is not None and _score(unseen.text, queries) > 0:
            return candidate
    for view in ranked:
        if _score(view.text, queries) <= 0:
            continue
        for direction in backend.neighbor_directions(seen[view.id]):
            candidate = {"type": "read_neighbors", "chunk_id": view.id, "direction": direction, "need_index": idx}
            if action_signature(candidate) not in actions:
                return candidate
    if action["type"] == "search_policy":
        return action
    # Original question is appended by the executor, preserving its numbers.
    query = re.sub(r"\d+(?:\.\d+)?", "", need).strip() or "相关政策规定"
    candidate = {"type": "search_policy", "query": query, "need_index": idx}
    return candidate if action_signature(candidate) not in actions else {"type": "finish"}


def _anchors(decision, read_anchors):
    result = {cid: list(values) for cid, values in read_anchors.items()}
    for need in (decision or {}).get("needs", []):
        for anchor in need["evidence"]:
            result.setdefault(anchor["chunk_id"], []).append(anchor)
    return result


async def run(question, backend, limits=None):
    limits = limits or AgentLimits()
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question_required")
    if not 0 <= limits.max_actions <= 2 or not 1 <= limits.max_decisions <= 3 or not 0 < limits.search_seconds <= limits.total_seconds <= 60:
        raise ValueError("invalid_agent_limits")
    result = AgentResult()
    start = time.monotonic()
    seen = {}
    last = None
    shown_views = []
    read_anchors = {}
    phase = "search"

    async def bounded(coro, deadline):
        return await asyncio.wait_for(coro, timeout=max(0.001, deadline - time.monotonic()))

    def control_event(event):
        result.trace.append(event)
        backend.record_control_event(event)

    def merge(items):
        added = []
        for e in items:
            if e.id in seen and seen[e.id] != e:
                raise ValueError("evidence_changed_during_request")
            if e.id not in seen:
                seen[e.id] = e
                added.append(e.id)
        return added

    try:
        initial = await bounded(backend.search(question), start + limits.search_seconds)
        merge(initial)
        result.trace.append({"event": "initial_search", "query": question, "chunk_ids": list(seen)})
        actions = set()
        used = 0
        result.stop_reason = "decision_limit"
        if not limits.enhance:
            result.stop_reason = "native_replay"
        elif not seen:
            result.stop_reason = "no_evidence"
        else:
            for _ in range(limits.max_decisions):
                if backend.remaining_generation_calls() < 2:
                    result.stop_reason = "final_call_reserved"
                    break
                if time.monotonic() >= start + limits.search_seconds:
                    result.stop_reason = "search_deadline"
                    break
                anchors = _anchors(last, read_anchors)
                phase = "decision"
                try:
                    views = prepare_judge_views(question, list(seen.values()), last, backend.count_tokens, limits.controller_evidence_tokens,
                                                anchors=anchors, neighbors={e.id: backend.neighbor_directions(e) for e in seen.values()},
                                                gap_contract=limits.gap_contract)
                    shown_views.extend(views)
                    value = await bounded(backend.decide(question, views, last), start + limits.search_seconds)
                    value = validate_decision(value, views, question)
                    if last and [n["text"] for n in last["needs"]] != [n["text"] for n in value["needs"]]:
                        raise DecisionError("requirements_changed_between_decisions")
                except DecisionError as exc:
                    result.controller_error = str(exc)[:200]
                    control_event({"event": "controller_error", "error": result.controller_error})
                    # No retry or extra Judge. Provider budget/usage is checked again
                    # by generate before sending; timeouts and source errors never land here.
                    if not limits.fallback_on_invalid_decision or not initial or start + limits.total_seconds - time.monotonic() < 5:
                        raise
                    result.fallback_used = True
                    result.stop_reason = "controller_fallback"
                    control_event({"event": "fallback", "mode": "native_initial"})
                    break
                last = value
                result.trace.append({"event": "decision", "value": value, "views": [{"chunk_id": e.id, "spans": e.spans, "truncated": e.truncated} for e in views]})
                if all(n["status"] == "supported" for n in value["needs"]):
                    result.stop_reason = "sufficient"
                    break
                action = value["action"]
                if limits.gap_contract and limits.allow_search and used < limits.max_actions:
                    action = _gap_action(value, views, seen, backend, actions, shown_views)
                    if action != value["action"]:
                        control_event({"event": "action_override", "proposed": value["action"], "executed": action,
                                       "reason": "address_known_evidence_gap"})
                if action["type"] == "finish":
                    result.stop_reason = value.get("stop_kind", "insufficient")
                    break
                if not limits.allow_search:
                    result.stop_reason = "search_disabled"
                    break
                if used >= limits.max_actions:
                    result.stop_reason = "action_limit"
                    break
                # Retaining the verbatim question mechanically preserves original constraints.
                signature = action_signature(action)
                if action["type"] != "read_chunk" and signature in actions:
                    result.stop_reason = "duplicate_action"
                    break
                actions.add(signature)
                used += 1
                phase = "search"
                if action["type"] == "read_chunk":
                    # This round's newly selected anchors must be reserved too.
                    anchors = _anchors(value, read_anchors)
                    reserved = _anchor_cost(list(seen.values()), anchors, backend.count_tokens)
                    available = max(0, limits.controller_evidence_tokens - reserved - 64 * bool(reserved))
                    item = read_chunk(seen[action["chunk_id"]], value["needs"][action["need_index"]]["text"],
                                      shown_views, backend.count_tokens, min(700, available))
                    if item is not None:
                        combined = {cid: list(values) for cid, values in anchors.items()}
                        combined.setdefault(item.id, []).extend(evidence_segments([item]).values())
                        try:
                            prepare_judge_views(question, list(seen.values()), value, backend.count_tokens, limits.controller_evidence_tokens,
                                                anchors=combined, neighbors={e.id: backend.neighbor_directions(e) for e in seen.values()},
                                                gap_contract=limits.gap_contract)
                        except DecisionError:
                            result.stop_reason = "context_budget"
                            break
                        for anchor in evidence_segments([item]).values():
                            read_anchors.setdefault(item.id, []).append(anchor)
                    result.trace.append({"event": "tool", "action": action, "new_chunk_ids": [],
                                         "new_spans": list(item.spans) if item is not None else []})
                    if item is None:
                        result.stop_reason = "no_progress"
                        break
                    continue
                if action["type"] == "search_policy":
                    query = question + "\n待补充检索：" + action["query"]
                    items = await bounded(backend.search(query, action.get("document_ids")), start + limits.search_seconds)
                else:
                    items = await bounded(backend.neighbors(seen[action["chunk_id"]], action["direction"]), start + limits.search_seconds)
                added = merge(items)
                result.trace.append({"event": "tool", "action": action, "new_chunk_ids": added})
                if not added:
                    result.stop_reason = "no_progress"
                    break
        native = result.fallback_used or not limits.enhance or not limits.select_context
        chosen = initial if native else pack_evidence(list(seen.values()), last, question, backend.count_tokens, limits.final_evidence_tokens)
        validate_views(chosen, list(seen.values()))
        result.selected = [{"chunk_id": e.id, "document_id": e.document_id, "spans": e.spans, "text": e.text, "truncated": e.truncated} for e in chosen]
        phase = "generation"
        answer = await bounded(backend.generate(question, chosen, native=native), start + limits.total_seconds)
        result.answer = answer.get("answer", "")
        result.reference = answer.get("reference", {})
        if not result.answer.strip():
            result.status = "generation_error"
    except TimeoutError:
        result.status = "timeout"
        result.stop_reason = phase + "_timeout"
    except Exception as exc:
        result.status = "invalid_decision" if phase == "decision" and isinstance(exc, (ValueError, KeyError, TypeError)) else "technical_error"
        result.stop_reason = result.status
        result.error = str(exc)[:200]
    result.latency_ms = round((time.monotonic() - start) * 1000)
    result.usage = dict(getattr(backend, "metrics", {}))
    return result
