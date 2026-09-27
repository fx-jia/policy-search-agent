"""Fuse frozen rank lists without questions, document text or relevance labels."""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class Hit:
    id: str
    score: float


def _project(hits):
    if len({h.id for h in hits}) != len(hits) or any(not h.id or not math.isfinite(h.score) for h in hits):
        raise ValueError('Invalid or duplicate hit')
    if any(a.score < b.score for a, b in zip(hits, hits[1:])):
        raise ValueError('Input must preserve descending retrieval scores')
    if not hits:
        return {}
    low, high = min(h.score for h in hits), max(h.score for h in hits)
    return {h.id: {'rank': i, 'raw': h.score, 'normalized': (h.score - low) / (high - low) if high > low else 1.0} for i, h in enumerate(hits, 1)}


def fuse(bm25, dense, method, rrf_k=60, dense_weight=0.5, limit=64):
    if method not in {'weighted', 'rrf'} or rrf_k <= 0 or not math.isfinite(rrf_k) or not 0 <= dense_weight <= 1 or limit < 1:
        raise ValueError('Invalid fusion configuration')
    routes = {'bm25': _project(bm25), 'dense': _project(dense)}
    rows = []
    for cid in sorted(set(routes['bm25']) | set(routes['dense'])):
        row = {'id': cid, 'score': 0.0}
        for name, lookup in routes.items():
            hit = lookup.get(cid)
            contribution = 0.0
            if hit:
                contribution = 1 / (rrf_k + hit['rank']) if method == 'rrf' else hit['normalized'] * (dense_weight if name == 'dense' else 1 - dense_weight)
            row.update({f'{name}_rank': hit['rank'] if hit else None, f'{name}_raw_score': hit['raw'] if hit else None, f'{name}_normalized_score': hit['normalized'] if hit else None, f'{name}_contribution': contribution})
            row['score'] += contribution
        rows.append(row)
    return sorted(rows, key=lambda r: (-r['score'], r['id']))[:limit]
