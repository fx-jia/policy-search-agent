"""Model-free context selection; this module has no benchmark or network access."""

from dataclasses import dataclass
import html
import re


@dataclass(frozen=True)
class Candidate:
    id: str
    document_id: str
    source_document_id: str
    rank: int
    order: int
    content: str
    tokens: int


def normalize(text):
    return re.sub(r"\s+", "", re.sub(r"<[^>]+>", "", html.unescape(text)))


def _unfinished_end(content):
    text = normalize(content).rstrip('”’"\'）)]】')
    return bool(text) and text[-1] not in "。！？.!?；;：:"


def select_context(candidates, policy, budget=2560, seed_top_k=3):
    """Pack whole chunks, skipping non-fitting chunks; expand only original seeds.

    All neighbors must already be in this query's saved candidate set. No
    recursive expansion, source ID preferences, questions or labels are used.
    Token counts include each passage's source/chunk header and separator.
    """
    if policy not in {"ranked", "neighbors", "boundary"} or budget < 0 or seed_top_k < 0:
        raise ValueError("Invalid selection policy or budget")
    if len({c.id for c in candidates}) != len(candidates) or len({c.rank for c in candidates}) != len(candidates):
        raise ValueError("Duplicate candidate ID or rank")
    by_order = {(c.document_id, c.order): c for c in candidates}
    if len(by_order) != len(candidates):
        raise ValueError("Ambiguous document chunk order")
    if any(c.tokens <= 0 or c.rank < 1 or c.order < 0 for c in candidates):
        raise ValueError("Invalid candidate metadata")
    selected, seen, used = [], set(), 0

    def add(c):
        nonlocal used
        if c is None or c.id in seen or used + c.tokens > budget:
            return False
        seen.add(c.id)
        selected.append(c)
        used += c.tokens
        return True

    for candidate in sorted(candidates, key=lambda c: c.rank):
        add(candidate)
        if candidate.id not in seen or policy == "ranked" or candidate.rank > seed_top_k:
            continue
        following = by_order.get((candidate.document_id, candidate.order + 1))
        preceding = by_order.get((candidate.document_id, candidate.order - 1))
        # The ordering of these two offers is fixed before evaluation.
        if policy == "neighbors" or _unfinished_end(candidate.content):
            add(following)
        if policy == "neighbors" or (preceding and _unfinished_end(preceding.content)):
            add(preceding)
    return selected


def evidence_coverage(evidence, selected):
    """Exact normalized span match within contiguous chunks from one source.

    This measures reference coverage, not semantic answer correctness. The
    original chunk boundaries remain barriers when a missing chunk intervenes.
    """
    runs = []
    previous = None
    for c in sorted(selected, key=lambda c: (c.document_id, c.order)):
        if previous and c.document_id == previous.document_id and c.source_document_id == previous.source_document_id and c.order == previous.order + 1:
            runs[-1][1] += normalize(c.content)
        else:
            runs.append([c.source_document_id, normalize(c.content)])
        previous = c
    return [bool(normalize(e["text"])) and any(sid == e["source_document_id"] and normalize(e["text"]) in content for sid, content in runs) for e in evidence]
