"""Literal, source-preserving views. This module never reads benchmark data."""
from dataclasses import dataclass, field, replace
import copy
import re

SEPARATOR = "\n[原文中间部分未展示]\n"


@dataclass(frozen=True)
class Evidence:
    id: str
    document_id: str
    title: str
    text: str
    order: int
    native: dict = field(default_factory=dict, repr=False)
    spans: tuple = ()
    truncated: bool = False
    source: dict = field(default_factory=dict)


class DecisionError(ValueError):
    """Invalid model decision, distinct from source, transport and budget errors."""


def _source_for_prompt(source):
    """Business provenance only; the full immutable source remains on Evidence."""
    fields = ("title", "source_url", "source_kind", "document_number", "published_at", "signed_at",
              "effective_from", "effective_to", "repealed_at", "confirmation_status", "relations")
    result = {key: copy.deepcopy(source[key]) for key in fields if key in source}
    if "temporal_metadata" in source:
        result["temporal_metadata"] = {key: copy.deepcopy(value) for key, value in source["temporal_metadata"].items()
                                       if key not in {"evidence_archive_key", "evidence_archive_sha256"}}
    if str(source.get("asset_role", "")).split(":", 1)[0] == "attachment":
        result["asset_role"] = "attachment"
        if "asset_url" in source:
            result["asset_url"] = source["asset_url"]
    return result


def evidence_segments(visible):
    """Assign call-local IDs to literal clauses, preserving original offsets."""
    result = {}
    for item in visible:
        spans = item.spans or ((0, len(item.text)),)
        for part, (start, end) in zip(item.text.split(SEPARATOR), spans, strict=True):
            if len(part) != end - start:
                raise ValueError("invalid_visible_span")
            # Line breaks may occur inside a number or sentence after parsing.
            for match in re.finditer(r".+?(?:[。！？；]|$)", part, re.S):
                if match[0].strip():
                    result[f"s{len(result)}"] = {"chunk_id": item.id, "quote": match[0],
                                               "_source_span": [start + match.start(), start + match.end()]}
    return result


def _terms(text):
    result = set(re.findall(r"[A-Za-z0-9]+", text.lower()))
    for word in re.findall(r"[\u4e00-\u9fff]+", text):
        result.update(word[i:i + 2] for i in range(len(word) - 1))
    return result


def _fit(text, count, budget):
    lo, hi = 0, len(text)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if count(text[:mid]) <= budget:
            lo = mid
        else:
            hi = mid - 1
    return lo


def _queries(query, needs=()):
    # Source titles remain in retrieval and Judge input; only window ranking
    # removes the preamble so it cannot overwhelm the requested clauses.
    text = re.sub(r"《[^》]*》", "", query)
    text = re.sub(r"^(?:按|根据|依据)[^，,：:]{0,100}[，,：:]", "", text)
    return [_terms(q) for q in (list(needs) or re.split(r"[？?；;]", text)) if q.strip()] or [_terms(text)]


def _paragraphs(text):
    # Parsed line breaks can split amounts and eligibility conditions.
    return [(m.start(), m.end()) for m in re.finditer(r".+?(?:[。！？；]|$)", text, re.S)]


def _score(text, queries):
    terms = _terms(text)
    matches = [len(terms & q) / max(1, len(q)) for q in queries]
    return max(matches, default=0) + sum(matches) / max(1, len(matches))


def _render(item, spans):
    merged = []
    for a, b in sorted(set(spans)):
        if merged and a <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(b, merged[-1][1]))
        else:
            merged.append((a, b))
    return merged, SEPARATOR.join(item.text[a:b] for a, b in merged)


def _shared_views(items, query, count, budget, *, needs=(), anchors=None):
    anchors = anchors or {}
    if not items:
        return []
    # Business sources reserve the same complete decoration emitted by kb_prompt.
    # Literal text still owns the remaining budget; source fields are never cut.
    headers = []
    for index, item in enumerate(items):
        if not item.source:
            continue
        source = _source_for_prompt(item.source)
        fields = {"Title": source.get("title") or item.title, "URL": source.get("source_url", ""), **source}
        header = f"\nID: {index}"
        for key, value in fields.items():
            if value is not None and str(value):
                header += f"\n├── {key}: " + re.sub(r"\n+", " ", str(value))
        headers.append(header + "\n└── Content:\n")
    budget -= sum(count(header) for header in headers)
    if headers and budget <= 0:
        raise ValueError("evidence_source_metadata_budget")
    queries = _queries(query, needs)
    costs = [count(e.text) for e in items]
    allocation = [count(_render(e, [tuple(a["_source_span"]) for a in anchors.get(e.id, [])])[1]) for e in items]
    if sum(allocation) > budget:
        raise ValueError("selected_anchor_budget")
    remaining = budget - sum(allocation)
    floor = min(128, remaining // len(items))
    for i, cost in enumerate(costs):
        addition = min(max(0, floor - allocation[i]), cost - allocation[i])
        allocation[i] += addition
        remaining -= addition
    weights = [1 + 8 * max((_score(e.text[a:b], queries) for a, b in _paragraphs(e.text)), default=0) for e in items]
    while remaining > 0:
        active = [i for i in range(len(items)) if allocation[i] < costs[i]]
        if not active:
            break
        total = sum(weights[i] for i in active)
        available = remaining
        for i in sorted(active, key=lambda i: -weights[i]):
            addition = min(remaining, costs[i] - allocation[i], max(1, int(available * weights[i] / total)))
            allocation[i] += addition
            remaining -= addition
    return [_window(e, queries, count, size, anchors.get(e.id, [])) for e, size in zip(items, allocation) if size > 0]


def _window(item, query, count, budget, quotes=()):
    if count(item.text) <= budget:
        return replace(item, spans=((0, len(item.text)),), truncated=False)
    paragraphs = _paragraphs(item.text)
    queries = _queries(query) if isinstance(query, str) else query
    scored = sorted(paragraphs, key=lambda s: (-_score(item.text[s[0]:s[1]], queries), s[0]))
    anchors = []
    for anchor in quotes:
        quote = anchor["quote"]
        start, end = anchor["_source_span"]
        if item.text[start:end] != quote:
            raise ValueError("anchor_not_in_source")
        # Preserve the containing clause, including its eligibility qualifier.
        left = max([0] + [b for a, b in paragraphs if b <= start])
        right = min([len(item.text)] + [b for a, b in paragraphs if b >= end])
        anchors.append((left, right) if count(item.text[left:right]) <= budget else (start, end))
    selected = []

    def render(spans):
        return _render(item, spans)

    # Reserve every selected literal span before adding neighboring conditions.
    # Otherwise expanding the first anchor can evict another required clause.
    if quotes:
        selected, text = render([tuple(a["_source_span"]) for a in quotes])
        if count(text) > budget:
            raise ValueError("selected_anchor_budget")
    for span in anchors + scored:
        merged, text = render(selected + [span])
        if count(text) <= budget:
            selected = merged
    if not selected:
        start, end = (anchors + scored)[0] if anchors or scored else (0, len(item.text))
        end = start + _fit(item.text[start:end], count, budget)
        selected = [(start, end)] if end > start else []
    spans, text = render(selected)
    return replace(item, text=text, spans=tuple(spans), truncated=True)


def make_views(items, query, count, budget=4800, *, needs=(), anchors=None):
    return _shared_views(items, query, count, budget, needs=needs, anchors=anchors)


def read_chunk(item, need, previous_views, count, budget=700):
    if budget <= 0:
        return None
    shown = [span for view in previous_views if view.id == item.id for span in (view.spans or ((0, len(view.text)),))]
    candidates = [(a, b) for a, b in _paragraphs(item.text) if not any(x <= a and b <= y for x, y in shown)]
    queries = _queries(need)
    bounded = []
    for a, b in candidates:
        if count(item.text[a:b]) <= budget:
            bounded.append((a, b))
            continue
        terms = set().union(*queries)
        hits = sorted({a + m.start() for term in terms for m in re.finditer(re.escape(term), item.text[a:b])
                       if not any(x <= a + m.start() < y for x, y in shown)})
        cursor = a
        for x, y in sorted(shown):
            if x <= cursor < y:
                cursor = y
        for hit in (hits[:64] or [cursor]):
            start = max(a, hit - min(120, budget // 3))
            end = start + _fit(item.text[start:b], count, budget)
            if end <= hit:
                start = hit
                end = start + _fit(item.text[start:b], count, budget)
            if start < end and not any(x <= start and end <= y for x, y in shown):
                bounded.append((start, end))
    candidates = bounded
    candidates.sort(key=lambda s: (-_score(item.text[s[0]:s[1]], queries), s[0]))
    selected = []
    for span in candidates:
        merged, text = _render(item, selected + [span])
        if count(text) <= budget:
            selected = merged
    if not selected:
        return None
    spans, text = _render(item, selected)
    return replace(item, text=text, spans=tuple(spans), truncated=spans != [(0, len(item.text))])


def _anchor_cost(items, anchors, count):
    return sum(count(_render(item, [tuple(a["_source_span"]) for a in anchors.get(item.id, [])])[1]) for item in items)


def validate_decision(value, visible, question):
    if not isinstance(value, dict) or not isinstance(value.get("needs"), list) or not 1 <= len(value["needs"]) <= 6:
        raise DecisionError("invalid_needs")
    if set(value) - {"action", "constraints", "needs", "reason", "stop_kind"}:
        raise DecisionError("unexpected_decision_field")
    if value.get("stop_kind") not in (None, "sufficient", "missing_user_information", "no_more_evidence"):
        raise DecisionError("invalid_stop_kind")
    value = copy.deepcopy(value)
    if not isinstance(value.get("constraints"), list) or len(value["constraints"]) > 12:
        raise DecisionError("invalid_constraints")
    for constraint in value["constraints"]:
        if not isinstance(constraint, str) or not constraint or constraint not in question:
            raise DecisionError("constraint_not_from_question")
    by_id = {e.id: e for e in visible}
    segments = evidence_segments(visible)
    for need in value["needs"]:
        if not isinstance(need, dict) or not isinstance(need.get("text"), str) or not 1 <= len(need["text"]) <= 200:
            raise DecisionError("invalid_need_text")
        if set(need) - {"text", "status", "evidence_ids"}:
            raise DecisionError("unexpected_need_field")
        if need.get("status") not in ("supported", "missing", "uncertain") or not isinstance(need.get("evidence_ids"), list):
            raise DecisionError("invalid_need_status")
        ids = need["evidence_ids"]
        if len(ids) > 6 or (need["status"] == "supported" and not ids):
            raise DecisionError("supported_without_evidence")
        if any(not isinstance(sid, str) or sid not in segments for sid in ids):
            raise DecisionError("unknown_evidence_id")
        need["evidence"] = [copy.deepcopy(segments[sid]) for sid in dict.fromkeys(ids)]
    if len({a["chunk_id"] for n in value["needs"] for a in n["evidence"]}) > 8:
        raise DecisionError("evidence_source_limit")
    action = value.get("action")
    if not isinstance(action, dict) or action.get("type") not in ("finish", "search_policy", "read_neighbors", "read_chunk"):
        raise DecisionError("invalid_action")
    fields = {"finish": {"type"}, "search_policy": {"type", "query", "need_index", "document_ids"},
              "read_neighbors": {"type", "chunk_id", "direction", "need_index"},
              "read_chunk": {"type", "chunk_id", "need_index"}}
    if set(action) - fields[action["type"]]:
        raise DecisionError("unexpected_action_field")
    if not isinstance(value.get("reason"), str) or len(value["reason"]) > 400:
        raise DecisionError("invalid_reason")
    if action["type"] == "finish":
        return value
    idx = action.get("need_index")
    if type(idx) is not int or not 0 <= idx < len(value["needs"]) or value["needs"][idx]["status"] == "supported":
        raise DecisionError("action_must_target_missing_need")
    if action["type"] == "search_policy":
        query = action.get("query")
        if not isinstance(query, str) or not 1 <= len(query) <= 300:
            raise DecisionError("invalid_search_query")
        available = question + "\n" + "\n".join(e.text for e in visible)
        if not set(re.findall(r"\d+(?:\.\d+)?", query)) <= set(re.findall(r"\d+(?:\.\d+)?", available)):
            raise DecisionError("invented_query_number")
        scope = action.get("document_ids")
        if scope is not None and (not isinstance(scope, list) or not scope or any(not isinstance(d, str) or d not in {e.document_id for e in visible} for d in scope)):
            raise DecisionError("unseen_document_scope")
    else:
        cid = action.get("chunk_id")
        if not isinstance(cid, str) or cid not in by_id:
            raise DecisionError("invalid_read_seed")
        if action["type"] == "read_chunk" and not by_id[cid].truncated:
            raise DecisionError("chunk_already_visible")
        if action["type"] == "read_neighbors" and action.get("direction") not in ("previous", "next", "both"):
            raise DecisionError("invalid_neighbor_seed")
    return value


def pack_evidence(items, decision, query, count, budget=5600):
    by_id = {e.id: e for e in items}
    if len(by_id) != len(items):
        raise ValueError("duplicate_evidence_id")
    anchors = {}
    for need in (decision or {}).get("needs", []):
        for quote in need["evidence"]:
            anchors.setdefault(quote["chunk_id"], []).append(quote)
    order = list(anchors) + [e.id for e in items if e.id not in anchors]
    chosen = [by_id[cid] for cid in order[:8]]
    if not chosen:
        return []
    selected = _shared_views(chosen, query, count, budget, needs=[n["text"] for n in (decision or {}).get("needs", [])], anchors=anchors)
    return [e for e in selected if e.text]


def validate_views(selected, originals):
    by_id = {e.id: e for e in originals}
    for item in selected:
        original = by_id.get(item.id)
        if original is None or item.document_id != original.document_id or item.title != original.title:
            raise ValueError("unseen_selected_evidence")
        if item.source != original.source:
            raise ValueError("selected_source_metadata_changed")
        if item.order != original.order:
            raise ValueError("selected_source_order_changed")
        spans = item.spans or ((0, len(original.text)),)
        if any(type(a) is not int or type(b) is not int or not 0 <= a < b <= len(original.text) for a, b in spans):
            raise ValueError("invalid_excerpt_spans")
        if item.text != SEPARATOR.join(original.text[a:b] for a, b in spans):
            raise ValueError("excerpt_differs_from_approved_source")
