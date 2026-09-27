"""Freeze a source-grouped dev/test partition before evaluating new questions."""

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
from difflib import SequenceMatcher
import hashlib
import json
from pathlib import Path
import re

from experiments.analysis.profile_dataset import fingerprint, load_records
from experiments.datasets.build_benchmark import validate_sample


def partition(samples, source_ids, edges, families, exposed_ids, target_dev=50, seed="medical-rag-policy-split-v1"):
    parent = {sid: sid for sid in source_ids}

    def find(sid):
        if sid not in parent:
            raise ValueError("Grouping references an unknown source")
        while parent[sid] != sid:
            parent[sid] = parent[parent[sid]]
            sid = parent[sid]
        return sid

    def union(members):
        roots = sorted({find(sid) for sid in members})
        for root in roots[1:]:
            parent[root] = roots[0]

    for a, b in edges:
        union([a, b])
    for members in families:
        union(members)
    question_families = defaultdict(list)
    by_id = {s["question_id"]: s for s in samples}
    if len(by_id) != len(samples) or not exposed_ids <= by_id.keys():
        raise ValueError("Question IDs must be unique and include all exposed questions")
    for sample in samples:
        if not sample["relevant_document_ids"]:
            raise ValueError("Source-group splitting requires source-backed questions")
        union(sample["relevant_document_ids"])
        question_families[sample.get("question_family_id") or sample["question_id"]].extend(sample["relevant_document_ids"])
    for members in question_families.values():
        union(members)
    components = defaultdict(list)
    for sid in sorted(parent):
        components[find(sid)].append(sid)
    groups = {}
    assignments = {}
    for sample in sorted(samples, key=lambda s: s["question_id"]):
        members = components[find(sample["relevant_document_ids"][0])]
        gid = "policy-" + hashlib.sha256("|".join(members).encode()).hexdigest()[:12]
        group = groups.setdefault(gid, {"source_document_ids": members, "question_ids": [], "forced_dev": False})
        group["question_ids"].append(sample["question_id"])
        group["forced_dev"] |= sample["question_id"] in exposed_ids
        assignments[sample["question_id"]] = gid
    selected = {gid for gid, group in groups.items() if group["forced_dev"]}
    mandatory_count = sum(len(groups[gid]["question_ids"]) for gid in selected)
    # Choose whole optional groups by deterministic subset-sum; never split a
    # mandatory group to hit a desired sample count or consult retrieval scores.
    options = {0: ()}
    for gid in sorted(groups.keys() - selected, key=lambda g: hashlib.sha256(f"{seed}:{g}".encode()).hexdigest()):
        count = len(groups[gid]["question_ids"])
        for total, ids in list(options.items()):
            options.setdefault(total + count, (*ids, gid))
    additional = min(options, key=lambda n: (abs(mandatory_count + n - target_dev), n))
    selected.update(options[additional])
    for gid, group in groups.items():
        group["split"] = "dev" if gid in selected else "test"
    return assignments, groups


def question_key(question):
    text = re.sub(r"江苏省?|苏州市?区?|徐州市?区?|20\d{2}年?|\d+", "", question)
    return re.sub(r"[^\u4e00-\u9fffA-Za-z]", "", text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--exposed-benchmark", required=True)
    parser.add_argument("--grouping-plan", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    plan = json.loads(Path(args.grouping_plan).read_text())
    source_hash, benchmark_hash = fingerprint(args.source), fingerprint(args.benchmark)
    if source_hash != plan["source_sha256"] or benchmark_hash != plan["benchmark_sha256"]:
        raise ValueError("Grouping plan belongs to another source/benchmark version")
    rows, errors, _ = load_records(args.source)
    by_source = {str(r["系统ID"]): r for _, r in rows}
    if errors or len(by_source) != len(rows):
        raise ValueError("Invalid or duplicate source records")
    samples = list(map(json.loads, Path(args.benchmark).read_text().splitlines()))
    for sample in samples:
        validate_sample(sample, by_source, source_hash)
        if sample.get("review_status") != "human_approved":
            raise ValueError("Complete label review before freezing a split")
    exposed = list(map(json.loads, Path(args.exposed_benchmark).read_text().splitlines()))
    sample_by_id = {s["question_id"]: s for s in samples}
    for old in exposed:
        current = sample_by_id[old["question_id"]]
        if any(current[key] != old[key] for key in ("question", "relevant_document_ids", "reference_answer")):
            raise ValueError("Exposed-question identity changed; audit exposure before splitting")
    assignments, groups = partition(
        samples,
        by_source,
        [(e["a"], e["b"]) for e in plan["candidate_edges"]],
        [f["documents"] for f in plan["families"]],
        {s["question_id"] for s in exposed},
        plan["target_dev_questions"],
        plan["seed"],
    )
    by_split = {split: [s for s in samples if groups[assignments[s["question_id"]]]["split"] == split] for split in ("dev", "test")}
    cross_pairs = []
    for a in by_split["dev"]:
        for b in by_split["test"]:
            ratio = SequenceMatcher(None, question_key(a["question"]), question_key(b["question"]), autojunk=False).ratio()
            if ratio >= 0.63:
                cross_pairs.append({"a": a["question_id"], "b": b["question_id"], "ratio": ratio})
    if cross_pairs:
        raise ValueError(f"Review {len(cross_pairs)} cross-split lexical near-duplicate candidates before freezing")
    signature = {
        "source_sha256": source_hash,
        "benchmark_sha256": benchmark_hash,
        "exposed_sha256": fingerprint(args.exposed_benchmark),
        "grouping_plan_sha256": fingerprint(args.grouping_plan),
        "assignments": assignments,
        "groups": groups,
    }
    version = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    output_hashes, distributions = {}, {}
    for split, values in by_split.items():
        path = out / f"{split}.jsonl"
        with path.open("x") as stream:
            for sample in values:
                stream.write(json.dumps(dict(sample, split=split, policy_group_id=assignments[sample["question_id"]], split_version=version), ensure_ascii=False) + "\n")
        output_hashes[split] = fingerprint(path)
        distributions[split] = {
            "questions": len(values),
            "source_documents": len({d for s in values for d in s["relevant_document_ids"]}),
            "policy_groups": len({assignments[s["question_id"]] for s in values}),
            "question_types": dict(Counter(s["question_type"] for s in values)),
            "regions": dict(Counter(str(s["region"]) for s in values)),
        }
    manifest = dict(
        signature,
        split_version=version,
        created_at=datetime.now(timezone.utc).isoformat(),
        code_sha256=fingerprint(__file__),
        seed=plan["seed"],
        target_dev_questions=plan["target_dev_questions"],
        output_sha256=output_hashes,
        distributions=distributions,
        exposed_question_ids=[s["question_id"] for s in exposed],
        lexical_audit={"threshold": 0.63, "cross_split_candidates": cross_pairs, "exhaustive_semantic_deduplication": False},
        test_retrieval_executed=False,
        qrels_scope="known_positive_documents_only",
    )
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    (out / "grouping_plan.json").write_bytes(Path(args.grouping_plan).read_bytes())
    print(json.dumps({"output": str(out), "split_version": version, "distributions": distributions}, ensure_ascii=False))


if __name__ == "__main__":
    main()
