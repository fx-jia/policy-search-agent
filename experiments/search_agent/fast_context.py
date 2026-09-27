"""Question-aware literal context selection; no models, gold labels or retrieval."""
from dataclasses import replace
import math
import re
import unicodedata

from .evidence import _render, _terms, validate_views


def _normalized(text):
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", text)).lower()


def _windows(text):
    """Keep full numbered articles, or complete sentences with both neighbors.

    A global preamble and the article's actual chapter/section accompany each
    excerpt. Unstructured windows retain the first sentence. Soft newlines never
    delimit a sentence.
    This preserves local qualifiers, without asserting semantic completeness.
    """
    headers = list(re.finditer(r"(?m)^[ \t]*(?:#{1,6}[ \t]*)?第[一二三四五六七八九十百千万\d]+(?P<kind>[章节条])", text))
    if headers:
        prefix = [(0, headers[0].start())] if headers[0].start() else []
        chapter, section, windows = [], [], []
        has_articles = any(h["kind"] == "条" for h in headers)
        for i, h in enumerate(headers):
            span = [(h.start(), headers[i + 1].start() if i + 1 < len(headers) else len(text))]
            if h["kind"] == "章":
                chapter, section = span, []
            elif h["kind"] == "节":
                section = span
            else:
                windows.append(prefix + chapter + section + span)
                continue
            if not has_articles:
                windows.append(prefix + chapter + section)
        return windows
    sentences = [(m.start(), m.end()) for m in re.finditer(r".+?(?:[。！？!?]|$)", text, re.S) if m.group().strip()]
    return [[sentences[0], (sentences[max(0, i - 1)][0], sentences[min(len(sentences) - 1, i + 1)][1])]
            for i in range(len(sentences))]


def select_context(question, items, count_tokens, render_context, budget=2000):
    """Choose up to three of the same eight candidates within the full rendered budget."""
    if not isinstance(question, str) or budget <= 0 or len(items) > 8:
        raise ValueError("invalid_selection_input")
    unique = {}
    for e in items:
        previous = unique.setdefault(e.id, e)
        if (previous.document_id, previous.title, previous.text) != (e.document_id, e.title, e.text):
            raise ValueError("conflicting_chunk_identity")
    originals = [e for e in unique.values() if e.text.strip()]
    if not originals:
        return []
    intent = re.sub(r"《[^》]*》", " ", question)
    intent = re.sub(r"仅按库内|仅依据库内|按照|根据|请问|是否|能否|可否|分别|如何|多少|多久", "，", intent)
    needs = [_terms(_normalized(p)) for p in re.split(r"[，,；;？?]|以及|同时", intent)]
    needs = [p for p in needs if p]
    query = set().union(*needs) if needs else set()
    bodies = {e.id: _terms(_normalized(e.text)) for e in originals}
    weights = {t: 1 + math.log((len(originals) + 1) / (1 + sum(t in terms for terms in bodies.values()))) for t in query}

    def coverage(terms, target=query):
        return sum(weights[t] for t in terms & target) / (sum(weights[t] for t in target) or 1)

    options = {}
    for e in originals:
        options[e.id] = []
        for spans in _windows(e.text):
            merged, text = _render(e, spans)
            terms = _terms(_normalized(text))
            score = .5 * coverage(terms) + .5 * max((coverage(terms, n) for n in needs), default=0)
            options[e.id].append((score, merged, text))
        options[e.id].sort(key=lambda v: -v[0])
    feasible = [e for e in originals if any(
        count_tokens(render_context([replace(e, text=text)])) <= budget for _, _, text in options[e.id])]
    # Penalize repeated coverage across selected sources, while keeping relevance.
    planned, covered = [], set()
    while len(planned) < min(3, len(feasible)):
        pool = [e for e in feasible if e not in planned]

        def priority(e):
            relevance = max((v[0] for v in options[e.id]), default=0)
            novelty = coverage(bodies[e.id] - covered)
            return .6 * novelty + .4 * relevance

        best = max(pool, key=priority)
        planned.append(best)
        covered |= bodies[best.id]

    views = {}

    def as_view(e, spans):
        merged, text = _render(e, spans)
        return replace(e, text=text, spans=tuple(merged), truncated=merged != [(0, len(e.text))])

    def ordered(proposed):
        return [proposed[e.id] for e in originals if e.id in proposed]

    def fits(proposed):
        return count_tokens(render_context(ordered(proposed))) <= budget

    for i, e in enumerate(planned):
        remaining = budget - count_tokens(render_context(ordered(views)))
        share = remaining // (len(planned) - i)
        full = as_view(e, [(0, len(e.text))])
        if count_tokens(render_context([full])) <= share and fits({**views, e.id: full}):
            views[e.id] = full
            continue
        for _, spans, _ in options[e.id]:
            view = as_view(e, spans)
            if count_tokens(render_context([view])) <= share and fits({**views, e.id: view}):
                views[e.id] = view
                break
    # Spend the remaining budget on full original context before adding windows.
    # Sources unable to fit their initial share can still fit the final remainder.
    for e in planned:
        full = as_view(e, [(0, len(e.text))])
        if fits({**views, e.id: full}):
            views[e.id] = full
            continue
        for _, spans, _ in options[e.id]:
            prior = list(views[e.id].spans) if e.id in views else []
            view = as_view(e, prior + spans)
            if fits({**views, e.id: view}):
                views[e.id] = view
    # A planned source can lose its slot when its complete unit cannot fit the
    # residual budget. Backfill from the same initial candidates, without cutting.
    for e in feasible:
        if len(views) >= 3:
            break
        if e.id in views:
            continue
        for _, spans, _ in options[e.id]:
            view = as_view(e, spans)
            if fits({**views, e.id: view}):
                views[e.id] = view
                break
    chosen = ordered(views)
    validate_views(chosen, originals)
    return chosen
