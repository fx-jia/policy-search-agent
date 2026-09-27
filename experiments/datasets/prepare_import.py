"""Prepare a local-only, fingerprinted RAGFlow import proposal; never upload."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re

from experiments.analysis.profile_dataset import fingerprint, load_records, privacy_flags
from experiments.datasets.build_benchmark import validate_sample


SEED = "medical-rag-import-pilot-v1"


def export_document(row):
    """Only policy title and original body enter the proposed embedding input."""
    return f"# {row['文件名称']}\n\n{row['政策正文']}\n"


def exclusion_reasons(row):
    reasons = []
    if row.get("是否公开发布") != "是":
        reasons.append("public_status_not_yes")
    if len(row["政策正文"].strip()) < 100:
        reasons.append("body_under_100_chars")
    reasons.extend(privacy_flags(export_document(row)))
    return reasons


def select_documents(rows, gold, size):
    eligible = {sid: row for sid, row in rows.items() if not exclusion_reasons(row)}
    if not gold <= eligible.keys():
        raise ValueError("A benchmark source requires privacy/publication review")
    if size is None:
        return sorted(eligible), eligible
    if not len(gold) <= size <= len(eligible):
        raise ValueError("Pilot size must cover all known sources and fit eligible corpus")
    # Round-robin across observed region/category strata, ordered by a fixed hash.
    # This is coverage sampling for an interface pilot, not population sampling.
    strata = defaultdict(list)
    for sid in eligible.keys() - gold:
        row = eligible[sid]
        strata[(str(row.get("适用区域")), str(row.get("政策文件大类")))].append(sid)
    for values in strata.values():
        values.sort(key=lambda sid: hashlib.sha256(f"{SEED}:{sid}".encode()).hexdigest())
    chosen = set(gold)
    offset = 0
    while len(chosen) < size:
        for key in sorted(strata):
            if offset < len(strata[key]):
                chosen.add(strata[key][offset])
                if len(chosen) == size:
                    break
        offset += 1
    return sorted(chosen), eligible


def prepare(source, benchmark, output, size=200):
    numbered, errors, _ = load_records(source)
    if errors:
        raise ValueError("Invalid source JSON")
    rows = {str(row["系统ID"]): row for _, row in numbered}
    if len(rows) != len(numbered) or not all(re.fullmatch(r"[0-9]+", sid) for sid in rows):
        raise ValueError("Need unique numeric source IDs for filenames")
    source_hash = fingerprint(source)
    samples = [json.loads(line) for line in Path(benchmark).read_text().splitlines() if line.strip()]
    if not samples or len({s["question_id"] for s in samples}) != len(samples):
        raise ValueError("Need nonempty benchmark with unique questions")
    for sample in samples:
        validate_sample(sample, rows, source_hash)
        if sample.get("review_status") != "human_approved":
            raise ValueError("Benchmark must be reviewed")
    gold = {sid for s in samples for sid in s["relevant_document_ids"]}
    pilot_questions = [s["question_id"] for s in samples if s.get("split") in ("smoke", "dev")][:10]
    if len(pilot_questions) != 10:
        raise ValueError("Need ten previously exposed/development questions for the pilot")
    selected, eligible = select_documents(rows, gold, size)
    held = [{"source_document_id": sid, "reasons": exclusion_reasons(rows[sid])} for sid in sorted(rows.keys() - eligible.keys())]
    entries = []
    for sid in selected:
        row = rows[sid]
        body = export_document(row)
        entries.append(
            {
                "source_document_id": sid,
                "filename": f"{sid}.md",
                "upload_name": f"{sid}__{re.sub(r'[/\\:*?\"<>|]', '_', row['文件名称'])[:65]}.md",
                "file_sha256": hashlib.sha256(body.encode()).hexdigest(),
                "body_sha256": hashlib.sha256(row["政策正文"].encode()).hexdigest(),
                "selection_role": "known_benchmark_source" if sid in gold else "unjudged_distractor",
                "metadata": {key: row.get(key) for key in ("文件名称", "适用区域", "发文日期", "生效日期", "废止日期", "效力状态", "版本号", "文号")},
                "source_url": (row.get("数据来源路径") or {}).get("原始文件路径"),
            }
        )
    manifest = {
        "status": "proposal_only_not_uploaded",
        "source_sha256": source_hash,
        "benchmark_sha256": fingerprint(benchmark),
        "seed": SEED if size is not None else None,
        "selection_scope": "coverage_pilot" if size is not None else "all_eligible_public_main_text",
        "corpus_documents": len(rows),
        "eligible_candidates": len(eligible),
        "held_documents": len(held),
        "held_reason_counts_overlapping": dict(Counter(reason for item in held for reason in item["reasons"])),
        "selected_documents": len(selected),
        "known_benchmark_sources": len(gold),
        "unjudged_distractors": len(selected) - len(gold),
        "pilot_question_ids": pilot_questions,
        "eligible_document_ids": sorted(eligible),
        "documents": entries,
        "parser_config_proposal": {
            "parser_id": "naive",
            "chunk_token_num": 512,
            "delimiter": "\n",
            "auto_keywords": 0,
            "auto_questions": 0,
            "parent_child": {"use_parent_child": False},
            "raptor": {"use_raptor": False},
            "graphrag": {"use_graphrag": False},
        },
        "content_scope": "policy_title_and_original_main_body_only; metadata_stays_in_local_manifest_until_explicit_mapping",
        "privacy_review": "regex_screening_only_not_a_complete_anonymization_guarantee",
        "paid_api_calls": 0,
        "ragflow_document_map": None,
    }
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    (out / "documents").mkdir()
    for sid in selected:
        (out / "documents" / f"{sid}.md").write_text(export_document(rows[sid]), encoding="utf-8")
    for name, data in (("manifest.json", manifest), ("held_documents.json", held)):
        (out / name).write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {
        key: manifest[key]
        for key in ("status", "selection_scope", "corpus_documents", "eligible_candidates", "held_documents", "selected_documents", "known_benchmark_sources", "unjudged_distractors")
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--size", type=int, default=200)
    parser.add_argument("--all-eligible", action="store_true")
    args = parser.parse_args()
    print(json.dumps(prepare(args.source, args.benchmark, args.output, None if args.all_eligible else args.size), ensure_ascii=False))


if __name__ == "__main__":
    main()
