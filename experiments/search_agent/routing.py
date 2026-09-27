"""Local pre-answer evidence signals. Lexical coverage is not semantic proof."""
import math
import re
import unicodedata

from .evidence import _terms, validate_views

FEATURE_NAMES = (
    "chosen_count", "candidate_doc_fraction", "chosen_doc_fraction", "context_fraction", "truncated_fraction",
    "all_coverage", "chosen_coverage", "selected_retention", "coverage_gap", "best_chosen_coverage",
    "omitted_best_gap", "need_min_retention", "need_max_gap", "need_unknown_fraction",
    "anchor_count", "anchor_loss_fraction", "anchor_absent_fraction", "title_chosen_fraction",
    "title_all_fraction", "omitted_new_doc_fraction",
)
_FILLER = re.compile(r"仅按库内|仅依据库内|文中规定|相关规定|该办法|本办法|该通知|请问|是否|能否|可否|有无|分别|如何|多少|多久|按照|根据|仅按|库内")
_ANCHOR = re.compile(r"第[一二三四五六七八九十百千万\d]+条|(?:19|20)\d{2}年?|江苏|南京|苏州|无锡|常州|南通|镇江|扬州|泰州|徐州|淮安|盐城|宿迁|连云港|职工|居民|生育")


def _normalize(text):
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", text)).lower()


def _intent(question):
    text = re.sub(r"《[^》]*》", " ", question)
    text = re.sub(r"[（(][^）)]*(?:文末|生效|施行|印发)[^）)]*[）)]", " ", text)
    return _FILLER.sub(" ", text)


def evidence_features(question, initial, chosen):
    """Use only the current question, initial chunks and literal chosen views.

    IDs validate source identity but never become model features. Weights are
    computed within this request's candidates, without benchmark/corpus fitting.
    Text removed from a partially visible chunk remains potentially missing.
    """
    if not isinstance(question, str):
        raise TypeError("question_text_only")
    if len(initial) > 8 or len(chosen) > 3:
        raise ValueError("unexpected_context_size")
    unique = {}
    for e in initial:
        prior = unique.setdefault(e.id, e)
        if (e.document_id, e.title, e.text) != (prior.document_id, prior.title, prior.text):
            raise ValueError("conflicting_chunk_identity")
    initial = list(unique.values())
    if len({e.id for e in chosen}) != len(chosen):
        raise ValueError("duplicate_chosen_chunk")
    validate_views(chosen, initial)
    body = {e.id: _normalize(e.text) for e in initial}
    terms = {cid: _terms(text) for cid, text in body.items()}
    visible = [_normalize(e.text) for e in chosen]
    visible_terms = [_terms(text) for text in visible]
    all_terms = set().union(*terms.values()) if terms else set()
    shown_terms = set().union(*visible_terms) if visible_terms else set()
    intent = _intent(question)
    query = _terms(_normalize(intent))
    weights = {t: 1 + math.log((len(initial) + 1) / (1 + sum(t in ts for ts in terms.values()))) for t in query}

    def coverage(haystack, needle=query):
        return sum(weights[t] for t in needle & haystack) / (sum(weights[t] for t in needle) or 1)

    all_coverage, chosen_coverage = coverage(all_terms), coverage(shown_terms)
    chosen_ids = {e.id for e in chosen}
    shortened = {e.id for e in chosen if e.text != unique[e.id].text}
    omitted = [terms[e.id] for e in initial if e.id not in chosen_ids or e.id in shortened]
    best = max((coverage(ts) for ts in visible_terms), default=0)
    # Separate clauses reveal a gap even when other demands dominate word overlap.
    needs = [_terms(_normalize(p)) & query for p in re.split(r"[，,；;？?]|以及|同时", intent)]
    needs = [ts for ts in needs if len(ts) >= 2] or ([query] if query else [])
    need_all = [coverage(all_terms, ts) for ts in needs]
    need_chosen = [coverage(shown_terms, ts) for ts in needs]
    anchors = set(_ANCHOR.findall(_normalize(question)))
    observed = {a for a in anchors if any(a in text for text in body.values())}
    kept = {a for a in anchors if any(a in text for text in visible)}
    titles = [_normalize(t) for t in re.findall(r"《([^》]*)》", question)]
    chosen_docs = {e.document_id for e in chosen}
    all_docs = {e.document_id for e in initial}
    values = (
        len(chosen), len(all_docs) / max(1, len(initial)), len(chosen_docs) / max(1, len(chosen)),
        sum(len(e.text) for e in chosen) / max(1, sum(len(e.text) for e in initial)),
        len(shortened) / max(1, len(chosen)), all_coverage, chosen_coverage,
        chosen_coverage / all_coverage if all_coverage else 0, all_coverage - chosen_coverage, best,
        max((coverage(ts) for ts in omitted), default=0) - best,
        min((b / a for a, b in zip(need_all, need_chosen) if a), default=0),
        max((a - b for a, b in zip(need_all, need_chosen)), default=0),
        sum(a == 0 for a in need_all) / max(1, len(needs)), len(anchors),
        len(observed - kept) / max(1, len(observed)), len(anchors - observed) / max(1, len(anchors)),
        sum(any(t in _normalize(e.title) for e in chosen) for t in titles) / max(1, len(titles)),
        sum(any(t in _normalize(e.title) for e in initial) for t in titles) / max(1, len(titles)),
        len(all_docs - chosen_docs) / max(1, len(all_docs)),
    )
    return dict(zip(FEATURE_NAMES, values, strict=True))


def evidence_rule(value):
    """A conservative diagnostic baseline, not a correctness guarantee."""
    sufficient = (value["chosen_count"] > 0 and value["all_coverage"] >= .35
                  and value["selected_retention"] >= .95 and value["need_max_gap"] <= .05
                  and value["omitted_best_gap"] <= .05 and value["anchor_loss_fraction"] == 0
                  and value["truncated_fraction"] == 0)
    return "fast" if sufficient else "ab"
