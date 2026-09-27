"""Separate document ranking from evidence coverage in a chunk context."""

import math
import statistics


def retrieval_metrics(chunks, relevant, ks=(1, 5, 10)):
    relevant = set(relevant)
    if not relevant:
        return None  # Unanswerable/unlabelled samples are not retrieval recall tasks.
    documents = list(dict.fromkeys(chunk["source_document_id"] for chunk in chunks))
    result = {}
    for k in ks:
        hits = len(set(documents[:k]) & relevant)
        result[f"recall@{k}"] = hits / len(relevant)
        result[f"hit_rate@{k}"] = float(hits > 0)
        dcg = sum(1 / math.log2(rank + 1) for rank, doc in enumerate(documents[:k], 1) if doc in relevant)
        ideal = sum(1 / math.log2(rank + 1) for rank in range(1, min(k, len(relevant)) + 1))
        result[f"ndcg@{k}"] = dcg / ideal
        context_docs = {chunk["source_document_id"] for chunk in chunks[:k]}
        result[f"context_document_recall@{k}_chunks"] = len(context_docs & relevant) / len(relevant)
    result["mrr@10"] = next((1 / rank for rank, doc in enumerate(documents[:10], 1) if doc in relevant), 0.0)
    result["returned_window_recall"] = len(set(documents) & relevant) / len(relevant)
    return result


def percentile(values, q):
    if not values:
        return None
    values = sorted(values)
    position = (len(values) - 1) * q
    lower, upper = math.floor(position), math.ceil(position)
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def aggregate(rows):
    scored = [row["metrics"] for row in rows if row["metrics"] is not None]
    latency = [row["latency_ms"] for row in rows if row["status"] == "ok"]
    return {
        "attempted": len(rows),
        "successful": len(latency),
        "errors": len(rows) - len(latency),
        "scored": len(scored),
        "metrics": {key: statistics.mean(row[key] for row in scored) for key in scored[0]} if scored else {},
        "latency_ms": {"mean": statistics.mean(latency) if latency else None, "p50": percentile(latency, 0.5), "p95": percentile(latency, 0.95)},
        "generation": {
            "status": "Not Run",
            "answer_accuracy": None,
            "faithfulness": None,
            "citation_accuracy": None,
            "end_to_end_ms": None,
            "input_tokens": None,
            "output_tokens": None,
            "estimated_cost": None,
        },
    }
