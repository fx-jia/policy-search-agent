"""Freeze a deduplicated corpus from already indexed documents; no model calls."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from experiments.analysis.profile_dataset import fingerprint


SEED = "medical-rag-existing-corpus1000-v1"


def select_existing(candidates, gold, size):
    if not gold <= candidates.keys():
        raise ValueError("Existing indexes do not cover all known evidence sources")
    if not len(gold) <= size <= len(candidates):
        raise ValueError("Corpus size must fit available indexed sources")
    extra = sorted(candidates.keys() - gold, key=lambda sid: hashlib.sha256(f"{SEED}:{sid}".encode()).hexdigest())
    return sorted(gold | set(extra[: size - len(gold)]))


def freeze(imports, benchmark, size, output):
    samples = [json.loads(line) for line in Path(benchmark).read_text().splitlines()]
    if not samples or any(s["review_status"] != "human_approved" for s in samples):
        raise ValueError("All benchmark evidence labels must be approved")
    gold = {sid for sample in samples for sid in sample["relevant_document_ids"]}
    candidates, inputs, parser, embedding_id, source_hash = {}, [], None, None, None
    for directory in imports:
        root = Path(directory)
        proposal = json.loads((root / "manifest.json").read_text())
        state = json.loads((root / "import_state.json").read_text())
        status = json.loads((root / "document_status.json").read_text())
        dataset = json.loads((root / "created_dataset.json").read_text())
        if state["proposal_sha256"] != fingerprint(root / "manifest.json"):
            raise ValueError("Import proposal changed")
        if source_hash is not None and source_hash != proposal["source_sha256"]:
            raise ValueError("Import sources differ")
        source_hash = proposal["source_sha256"]
        if embedding_id is not None and embedding_id != dataset["embedding_model"]:
            raise ValueError("Indexes use different embedding models")
        embedding_id = dataset["embedding_model"]
        entries = {entry["source_document_id"]: entry for entry in proposal["documents"]}
        for doc in status:
            if doc["run"] != "DONE" or doc["chunk_count"] < 1:
                continue
            if doc["chunk_method"] != "naive":
                raise ValueError("Unexpected chunk method")
            if parser is not None and parser != doc["parser_config"]:
                raise ValueError("Indexed parser configurations differ")
            parser = doc["parser_config"]
            sid = state["documents"][doc["id"]]
            entry = entries[sid]
            if sid in candidates:
                if candidates[sid]["file_sha256"] != entry["file_sha256"]:
                    raise ValueError("Duplicate source has different exported content")
                continue  # First listed index supplies the one fixed replica.
            candidates[sid] = {
                "source_document_id": sid,
                "document_id": doc["id"],
                "dataset_id": state["dataset_id"],
                "chunk_count": doc["chunk_count"],
                "file_sha256": entry["file_sha256"],
                "body_sha256": entry["body_sha256"],
            }
        inputs.append({"path": str(root.resolve()), "proposal_sha256": fingerprint(root / "manifest.json"), "status_sha256": fingerprint(root / "document_status.json")})
    if any(sample["dataset_version"] != source_hash for sample in samples):
        raise ValueError("Benchmark and indexed source snapshots differ")
    selected = select_existing(candidates, gold, size)
    documents = [candidates[sid] for sid in selected]
    frozen = {
        "kind": "fixed_document_scope_over_existing_indexes",
        "source_sha256": source_hash,
        "benchmark_sha256": fingerprint(benchmark),
        "selection_seed": SEED,
        "selection_rule": "all known evidence sources plus hash-selected distractors from already parsed public documents; first input wins duplicate source replicas",
        "availability_bias": "available pool is determined by earlier pilot selection and interrupted ingestion order, not representative full-corpus sampling",
        "available_unique_sources": len(candidates),
        "selected_documents": len(documents),
        "known_evidence_sources": len(gold),
        "indexed_chunks": sum(doc["chunk_count"] for doc in documents),
        "embedding_model_id": embedding_id,
        "dataset_ids": sorted({doc["dataset_id"] for doc in documents}),
        "chunking_config": {"chunk_method": "naive", "parser_config": parser, "input_format": "markdown_title_and_body", "attachments": False, "default_vlm_configured": False},
        "documents": documents,
        "inputs": inputs,
        "additional_document_embedding_calls": 0,
    }
    version = hashlib.sha256(json.dumps(frozen, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    frozen.update(knowledge_base_version=version, created_at=datetime.now(timezone.utc).isoformat())
    mapping = {
        "source_sha256": source_hash,
        "knowledge_base_version": version,
        "dataset_ids": frozen["dataset_ids"],
        "indexed_chunks": frozen["indexed_chunks"],
        "documents": {doc["document_id"]: doc["source_document_id"] for doc in documents},
        "document_datasets": {doc["document_id"]: doc["dataset_id"] for doc in documents},
    }
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    for name, data in (("frozen_corpus.json", frozen), ("document_map.json", mapping)):
        (out / name).write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    return {
        "output": str(out),
        "selected_documents": len(documents),
        "available_unique_sources": len(candidates),
        "indexed_chunks": frozen["indexed_chunks"],
        "known_evidence_sources": len(gold),
        "documents_per_dataset": dict(Counter(d["dataset_id"] for d in documents)),
        "additional_document_embedding_calls": 0,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--imports", nargs="+", required=True)
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--size", type=int, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(freeze(args.imports, args.benchmark, args.size, args.output), ensure_ascii=False))


if __name__ == "__main__":
    main()
