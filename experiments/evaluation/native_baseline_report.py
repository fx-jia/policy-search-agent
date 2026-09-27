"""Mechanical metrics and local semantic-review packets for native responses."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import statistics

from experiments.evaluation.context_completion import Candidate, evidence_coverage
from experiments.evaluation.metrics import percentile

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "experiments/results/private/native-corpus1000-v2"
REVIEW = REPO / "experiments/results/private/native-corpus1000-v2-preflight"
CITATION_INSTRUCTION = "Based on the provided document or chat history, add citations to the input text"
CITATION_PATTERN = re.compile(r"\[(?:ID:)?([0-9\u0660-\u0669\u06f0-\u06f9]+)\]|##(\d+)\$\$")


def knowledge_prompt(system):
    # The installed runtime appends this constant after message_fit_in. Its
    # examples contain unrelated ID blocks and must not count as policy evidence.
    return system.rsplit(CITATION_INSTRUCTION, 1)[0]


def map_prompt_chunks(system, chunks):
    """Match native ID/title/content blocks, including a budget-truncated tail."""
    knowledge = knowledge_prompt(system)
    rows = []
    native_id = 0
    for chunk in chunks:
        cid = chunk.get("chunk_id", chunk.get("id"))
        content = chunk["content_with_weight"]
        row = {"chunk_id": cid, "native_id": None, "status": "empty_not_rendered", "sent_content": ""}
        if not content:
            rows.append(row)
            continue
        row["native_id"] = native_id
        title = re.sub(r"\n+", " ", chunk["docnm_kwd"])
        header = f"\nID: {native_id}"
        if title:
            header += f"\n├── Title: {title}"
        header += "\n└── Content:\n"
        native_id += 1
        positions = [match.start() for match in re.finditer(re.escape(header), knowledge)]
        if len(positions) != 1:
            row["status"] = "absent" if not positions else "indeterminate"
        else:
            actual = knowledge[positions[0] + len(header) :]
            if actual.startswith(content):
                row.update(status="full", sent_content=content)
            elif actual and content.startswith(actual):
                row.update(status="partial", sent_content=actual)
            else:
                row["status"] = "indeterminate"
        rows.append(row)
    return rows


def map_citations(answer, refs, retrieved, prompt_chunks):
    """Resolve actual native markers without assuming changed references align."""
    ref_ids = [c.get("id", c.get("chunk_id")) for c in refs]
    retrieved_ids = [c.get("chunk_id", c.get("id")) for c in retrieved]
    prompt_status = {c["chunk_id"]: c["status"] for c in prompt_chunks}
    rows = []
    for match in CITATION_PATTERN.finditer(answer):
        native_id = int(match.group(1) or match.group(2))
        row = {"native_id": native_id, "marker": match.group(), "answer_start": match.start()}
        if ref_ids == retrieved_ids and native_id < len(ref_ids):
            cid = ref_ids[native_id]
            row.update(status="resolved", chunk_id=cid, prompt_status=prompt_status.get(cid, "indeterminate"))
        elif native_id >= len(ref_ids) and native_id >= len(retrieved_ids):
            row["status"] = "invalid"
        else:
            # A different reference order/length could reflect pruning or
            # renumbering. Without an explicit map, neither convention is proven.
            row.update(status="indeterminate_reference_mapping", candidate_chunk_ids=list(dict.fromkeys(ids[native_id] for ids in (ref_ids, retrieved_ids) if native_id < len(ids))))
        rows.append(row)
    return rows


def jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def build(response_files):
    config = json.loads((OUT / "generation_frozen.json").read_text())
    planned = set(config["question_ids"])
    if len(planned) != len(config["question_ids"]):
        raise ValueError("Duplicate planned question")
    rubric = json.loads((REVIEW / "combined_answer_rubric_v2.json").read_text())
    rubric_map = {q["question_id"]: q for q in rubric["questions"]}
    benchmark = {}
    for split in ("dev", "test"):
        benchmark.update({q["question_id"]: q for q in jsonl(REPO / f"experiments/datasets/private/benchmark_split_v1/{split}.jsonl")})
    dev_qrels = json.loads((REPO / "experiments/datasets/private/relevance_review_v1/completed_batch_01_v1/qrels_v2.json").read_text())["relevant_document_ids"]
    original_chunks = {c["id"]: c for c in json.loads((REPO / "experiments/datasets/private/context_completion_v1/chunks.json").read_text())["chunks"]}
    records = []
    packets = []
    seen = set()
    for filename in response_files:
        for row in jsonl(filename):
            qid = row["question_id"]
            if qid in seen or qid not in planned:
                raise ValueError("Duplicate or unplanned question result")
            seen.add(qid)
            q = benchmark[qid]
            known = set(dev_qrels.get(qid, q["relevant_document_ids"]))
            chunks = row.get("retrieved_chunks", [])
            if len(chunks) > 8:
                raise ValueError("Native eight-chunk window exceeded")
            source_ids = [config["document_map"]["documents"][c["doc_id"]] for c in chunks]
            documents = list(dict.fromkeys(source_ids))
            data = (row.get("body") or {}).get("data") or {}
            if not isinstance(data, dict):
                data = {}
            refs = (data.get("reference") or {}).get("chunks", [])
            envelope = row.get("outbound_request") or {}
            system = envelope.get("system", "")
            prompt_chunks = map_prompt_chunks(system, row.get("outbound_chunks", []))
            included, candidates = [], []
            for c, mapping in zip(row.get("outbound_chunks", []), prompt_chunks):
                cid = c.get("chunk_id", c.get("id"))
                if mapping["status"] == "full":
                    included.append(cid)
                if mapping["status"] in {"full", "partial"}:
                    orig = original_chunks[cid]
                    candidates.append(Candidate(cid, c["doc_id"], config["document_map"]["documents"][c["doc_id"]], len(candidates) + 1, orig["order"], mapping["sent_content"], 1))
            coverage = evidence_coverage(q["evidence"], candidates)
            answer = data.get("answer") or ""
            citations = map_citations(answer, refs, chunks, prompt_chunks)
            cited = [c["native_id"] for c in citations]
            invalid_citations = [c["native_id"] for c in citations if c["status"] == "invalid"]
            record = {
                "question_id": qid,
                "split": q["split"],
                "status": row["status"],
                "known_positive_ids": sorted(known),
                "retrieved_source_ids": source_ids,
                "source_hit_in_8_chunk_window": bool(set(documents) & known),
                "source_recall_in_8_chunk_window": len(set(documents) & known) / len(known),
                "source_hit_in_5_chunk_window": bool(set(source_ids[:5]) & known),
                "source_recall_in_5_chunk_window": len(set(source_ids[:5]) & known) / len(known),
                "document_mrr_within_8_chunk_window": next((1 / i for i, sid in enumerate(documents, 1) if sid in known), 0),
                "retrieved_distinct_sources": len(documents),
                "retrieved_chunks": len(chunks),
                "prompt_full_chunk_ids": included,
                "prompt_chunk_mapping": [{k: v for k, v in m.items() if k != "sent_content"} for m in prompt_chunks],
                "prompt_partial_chunk_ids": [m["chunk_id"] for m in prompt_chunks if m["status"] == "partial"],
                "prompt_mapping_indeterminate": any(m["status"] == "indeterminate" for m in prompt_chunks),
                "strict_original_evidence_coverage": coverage,
                "strict_original_evidence_complete": bool(coverage) and all(coverage),
                "native_citation_ids": cited,
                "invalid_citation_ids": invalid_citations,
                "citation_mapping": citations,
                "latency_ms": row["latency_ms"],
                "provider_usage": row.get("provider_usage"),
                "query_embedding_calls": row.get("query_embedding_calls", 0),
                "other_embedding_batches": row.get("other_embedding_batches", 0),
                "sanitized_chunks": (row.get("outbound_audit") or {}).get("changed", 0),
            }
            records.append(record)
            # Keep the policy content exactly as sent. Only constant citation examples
            # following the knowledge delimiter are excluded from this review aid.
            knowledge = knowledge_prompt(system)
            packets.append(
                {
                    "question_id": qid,
                    "split": q["split"],
                    "question": q["question"],
                    "required_points": rubric_map[qid]["required_points"],
                    "gold_evidence": rubric_map[qid]["evidence"],
                    "answer": answer,
                    "actual_knowledge_prompt": knowledge,
                    "actual_provider_request": envelope,
                    "prompt_chunk_mapping": prompt_chunks,
                    "returned_reference_ids": [c.get("id", c.get("chunk_id")) for c in refs],
                    "metrics": record,
                    "response_file": str(Path(filename).resolve()),
                }
            )
    write(OUT / "mechanical_metrics.json", records)
    packet_dir = OUT / "review_packets"
    packet_dir.mkdir(exist_ok=True)
    for packet in packets:
        write(packet_dir / f"{packet['question_id']}.json", packet)
    complete = len(records) == 150 and seen == planned
    by_split = {}
    for split in ("dev", "test", "all"):
        planned_split = [qid for qid in config["question_ids"] if split == "all" or benchmark[qid]["split"] == split]
        rows = [r for r in records if split == "all" or r["split"] == split]
        by_split[split] = {
            **summarize(rows),
            "planned_questions": len(planned_split),
            "unrun_questions": sum(qid not in seen for qid in planned_split),
            "metric_denominator": "observed_questions_only; unrun IDs are listed separately",
        }
    write(
        OUT / "mechanical_summary.json",
        {
            "complete_150": complete,
            "question_count": len(records),
            "unrun_question_ids": [qid for qid in config["question_ids"] if qid not in seen],
            "by_split": by_split,
            "rubric_sha256": hashlib.sha256((REVIEW / "combined_answer_rubric_v2.json").read_bytes()).hexdigest(),
        },
    )
    print(json.dumps({"packets": len(packets), "complete_150": complete}))


def summarize(rows):
    if not rows:
        return {}
    return {
        "questions": len(rows),
        "successful": sum(r["status"] == "ok" for r in rows),
        "source_hit": sum(r["source_hit_in_8_chunk_window"] for r in rows),
        "source_recall": statistics.mean(r["source_recall_in_8_chunk_window"] for r in rows),
        "source_hit_in_5_chunk_window": sum(r["source_hit_in_5_chunk_window"] for r in rows),
        "source_recall_in_5_chunk_window": statistics.mean(r["source_recall_in_5_chunk_window"] for r in rows),
        "retrieved_distinct_sources_mean": statistics.mean(r["retrieved_distinct_sources"] for r in rows),
        "document_mrr_within_8_chunk_window": statistics.mean(r["document_mrr_within_8_chunk_window"] for r in rows),
        "strict_original_evidence_complete": sum(r["strict_original_evidence_complete"] for r in rows),
        "prompt_full_chunks_mean": statistics.mean(len(r["prompt_full_chunk_ids"]) for r in rows),
        "prompt_partial_chunks": sum(len(r["prompt_partial_chunk_ids"]) for r in rows),
        "prompt_mapping_indeterminate_questions": sum(r["prompt_mapping_indeterminate"] for r in rows),
        "latency_ms_p50": percentile([r["latency_ms"] for r in rows], 0.5),
        "latency_ms_p95": percentile([r["latency_ms"] for r in rows], 0.95),
        "provider_tokens": sum((r["provider_usage"] or {}).get("total_tokens", 0) for r in rows),
        "provider_usage_recorded_questions": sum(bool(r["provider_usage"]) for r in rows),
        "query_embedding_calls": sum(r["query_embedding_calls"] for r in rows),
        "other_embedding_batches": sum(r["other_embedding_batches"] for r in rows),
        "invalid_citation_ids": sum(len(r["invalid_citation_ids"]) for r in rows),
        "indeterminate_citation_mappings": sum(c["status"] == "indeterminate_reference_mapping" for r in rows for c in r["citation_mapping"]),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("responses", nargs="+")
    build(parser.parse_args().responses)
