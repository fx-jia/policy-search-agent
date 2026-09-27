"""Run local smoke checks or explicitly authorized RAGFlow retrieval evaluation."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import re
import time

from experiments.analysis.profile_dataset import fingerprint, load_records
from experiments.datasets.build_benchmark import validate_sample
from experiments.evaluation.metrics import aggregate, retrieval_metrics
from experiments.retrieval.dense import DenseClient
from experiments.retrieval.local import LocalBM25
from experiments.retrieval.ragflow import RAGFlowClient, RetrievalError, build_payload


CONFIG_FIELDS = {
    "method",
    "base_url",
    "dataset_ids",
    "knowledge_base_version",
    "ragflow_version",
    "embedding_model",
    "rerank_model",
    "llm_model",
    "chunking_config",
    "retrieval_config",
    "generation_config",
    "source_sha256",
    "runtime_source_hashes",
    "baseline_plan_sha256",
    "scope_document_ids",
}


def verify_development_split(samples, benchmark_path, manifest_path, source_version):
    """Reject held-out questions before constructing any model client."""
    if any(sample.get("split") == "test" for sample in samples):
        raise ValueError("Held-out test execution is disabled during development")
    if not manifest_path:
        if any(sample.get("split_version") for sample in samples):
            raise ValueError("Versioned benchmark requires --split-manifest")
        return {}
    manifest = json.loads(Path(manifest_path).read_text())
    if manifest["source_sha256"] != source_version or manifest["output_sha256"]["dev"] != fingerprint(benchmark_path):
        raise ValueError("Frozen development benchmark hash mismatch")
    expected = {qid for group in manifest["groups"].values() if group["split"] == "dev" for qid in group["question_ids"]}
    if expected != {sample["question_id"] for sample in samples}:
        raise ValueError("Development question membership mismatch")
    for sample in samples:
        qid = sample["question_id"]
        group_id = manifest["assignments"][qid]
        group = manifest["groups"][group_id]
        if sample.get("split") != "dev" or sample.get("split_version") != manifest["split_version"] or sample.get("policy_group_id") != group_id or group["split"] != "dev":
            raise ValueError("Frozen policy-group assignment mismatch")
    return {"split": "dev", "split_version": manifest["split_version"], "split_manifest_sha256": fingerprint(manifest_path)}


def markdown_report(summary, manifest):
    metrics, latency = summary["metrics"], summary["latency_ms"]
    lines = [
        "# 医保检索实验结果",
        "",
        f"实验 ID：`{manifest['experiment_id']}`；方法：`{manifest['method']}`。",
        "",
        f"状态：{manifest['evaluation_status']}。尝试 {summary['attempted']}，成功 {summary['successful']}，失败 {summary['errors']}。",
        "",
        f"本次索引语料：{manifest['indexed_documents']} 篇文档；{manifest.get('indexed_chunks', '未记录')} 个 Chunk。原始语料摘要不代表全量文档均已入库。",
        "",
        "本表只测已标注来源文档的检索，不能据此宣称回答正确或政策现实有效。",
        "",
        "| 指标 | 实测值 |",
        "| --- | ---: |",
    ]
    lines.extend(f"| {key} | {value:.6f} |" for key, value in metrics.items())
    lines.extend(f"| Retrieval {key} (ms) | {value:.3f} |" for key, value in latency.items() if value is not None)
    baseline_rows = []
    for label, method in (("Dense", "dense"), ("Hybrid", "hybrid"), ("Hybrid + Rerank", "hybrid_rerank")):
        if manifest["method"] == method and summary["successful"] and metrics:
            baseline_rows.append(f"| {label} | {metrics['recall@5']:.6f} | {metrics['mrr@10']:.6f} | Not Run | {latency['mean']:.3f} ms |")
        else:
            baseline_rows.append(f"| {label} | Not Run | Not Run | Not Run | Not Run |")
    lines.extend(
        [
            "",
            "| Method | Recall@5 | MRR@10 | Answer Accuracy | Avg Retrieval Latency |",
            "| --- | --- | --- | --- | --- |",
            *baseline_rows,
            "",
            "表中只填入本次实际执行的方法，不能由单次运行推断三组正式对照已经完成。",
            "",
            "评测口径：返回 Chunk 窗口按首现位置去重为文档排名，再计算文档 Recall/Hit/MRR/nDCG。",
            "context_document_recall 单独按前 K 个 Chunk 计算，不能替代答案证据充分性。",
            "候选相关文档标签尚未完整池化，未标注文档属于未判定；分数只反映已知正例。",
            "HTTP/应用/映射错误保留为失败，对有答案题按零分计入分母，另报错误数。",
            "延迟只统计成功请求，不包含索引建立、报告写盘；单进程顺序执行，未做负载测试。",
            "生成与引用正确性：Not Run。HTTP 检索接口不返回 Token；Dense 适配的实际 Embedding Token 单独保存在逐题记录。",
            "",
        ]
    )
    return "\n".join(lines)


def run(args):
    config = json.loads(Path(args.config).read_text())
    if set(config) - CONFIG_FIELDS:
        raise ValueError("Unknown configuration fields (credentials must only be in environment)")
    records, errors, _ = load_records(args.source)
    if errors:
        raise ValueError("Source has invalid JSON records")
    source_version = fingerprint(args.source)
    if config["source_sha256"] != source_version:
        raise ValueError("Config source_sha256 mismatch")
    by_id = {str(row["系统ID"]): row for _, row in records}
    if len(by_id) != len(records):
        raise ValueError("Duplicate source IDs")
    samples = [json.loads(line) for line in Path(args.benchmark).read_text().splitlines() if line.strip()]
    if len({s["question_id"] for s in samples}) != len(samples):
        raise ValueError("Duplicate question IDs")
    if not samples or args.limit < 1:
        raise ValueError("Need nonempty benchmark and positive limit")
    split_info = verify_development_split(samples, args.benchmark, args.split_manifest, source_version)
    samples = samples[: args.limit]
    for sample in samples:
        validate_sample(sample, by_id, source_version)
    reviewed = all(s.get("review_status") == "human_approved" and s.get("reviewer") for s in samples)
    if not reviewed and not args.allow_unreviewed:
        raise ValueError("Unreviewed labels: use --allow-unreviewed for smoke checks only")
    chunking, retrieval = config["chunking_config"], config["retrieval_config"]
    if config.get("generation_config") != {"enabled": False}:
        raise ValueError("This runner only implements retrieval, not generation")
    count = retrieval["candidate_count"]
    if type(count) is not int or count < 10 or not 1 <= retrieval["context_top_k"] <= count:
        raise ValueError("Candidate window must cover @10 and context_top_k")
    started = time.perf_counter()
    if config["method"] == "local_bm25_smoke":
        expected = {"algorithm": "fixed_unicode_chars", "overlap": 0, "title": "prepend_for_scoring", "attachments": False}
        if any(chunking.get(key) != value for key, value in expected.items()):
            raise ValueError("Unsupported local chunking configuration")
        if retrieval.get("tokenizer") != "chinese_bigram_ascii_word":
            raise ValueError("Unsupported local tokenizer")
        k1, b = retrieval["bm25_k1"], retrieval["bm25_b"]
        if not all(type(value) in (int, float) and math.isfinite(value) for value in (k1, b)) or k1 <= 0 or not 0 <= b <= 1:
            raise ValueError("Require finite BM25 k1 > 0 and 0 <= b <= 1")
        if any(config.get(key) is not None for key in ("embedding_model", "rerank_model", "llm_model")):
            raise ValueError("Local BM25 uses no embedding, rerank or generation model")
        chars = chunking["chunk_chars"]
        if type(chars) is not int or chars < 1:
            raise ValueError("chunk_chars must be positive")
        client = LocalBM25(records, chars, retrieval["bm25_k1"], retrieval["bm25_b"])
        kb_version = hashlib.sha256(json.dumps({"source": source_version, "chunking": chunking}, sort_keys=True).encode()).hexdigest()
        backend_info = {"indexed_documents": len(records), "indexed_chunks": len(client.chunks), "backend": "in_process_bm25", "ragflow_api_called": False}
    else:
        if not args.allow_model_api:
            raise ValueError("Model API execution disabled; requires explicit --allow-model-api authorization")
        if len(samples) > 10 and not split_info:
            raise ValueError("API experiments beyond pilot size require a frozen development split")
        if not config.get("knowledge_base_version") or chunking.get("status") == "pending_frozen_import_manifest":
            raise ValueError("Freeze knowledge_base_version and chunking configuration before API experiments")
        if config["method"] != "dense":
            build_payload(samples[0]["question"], config)
        if not args.document_map:
            raise ValueError("A versioned document map is required")
        mapping = json.loads(Path(args.document_map).read_text())
        if mapping["source_sha256"] != source_version or mapping["knowledge_base_version"] != config["knowledge_base_version"]:
            raise ValueError("Document map version mismatch")
        if set(mapping["dataset_ids"]) != set(config["dataset_ids"]):
            raise ValueError("Document map belongs to different RAGFlow datasets")
        entries = mapping["documents"]
        if config.get("scope_document_ids") is not None and sorted(config["scope_document_ids"]) != sorted(entries):
            raise ValueError("Fixed corpus scope must equal the entire versioned document map")
        if not all(isinstance(sid, str) and sid in by_id for sid in entries.values()):
            raise ValueError("Document map contains unknown source IDs")
        if any(not set(s["relevant_document_ids"]).issubset(entries.values()) for s in samples):
            raise ValueError("Knowledge base does not contain benchmark evidence documents")
        client = DenseClient(config, mapping) if config["method"] == "dense" else RAGFlowClient(config, entries)
        kb_version = config["knowledge_base_version"]
        backend_info = {
            "backend": "ragflow_container_es_knn" if config["method"] == "dense" else "ragflow_http",
            "ragflow_api_called": config["method"] != "dense",
            "document_map_sha256": fingerprint(args.document_map),
            "document_map_snapshot": "document_map.json",
            "indexed_documents": len(set(entries.values())),
            "indexed_chunks": mapping.get("indexed_chunks"),
        }
    setup_ms = (time.perf_counter() - started) * 1000
    experiment_id = args.experiment_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", experiment_id):
        raise ValueError("Invalid experiment ID")
    out = Path(args.output_root) / experiment_id
    out.mkdir(parents=True, exist_ok=False)
    code_hashes = {str(path): fingerprint(path) for path in sorted(Path("experiments").rglob("*.py"))}
    manifest = dict(
        config,
        experiment_id=experiment_id,
        timestamp=datetime.now(timezone.utc).isoformat(),
        dataset_version=fingerprint(args.benchmark),
        source_sha256=source_version,
        knowledge_base_version=kb_version,
        benchmark_path=str(Path(args.benchmark).resolve()),
        benchmark_snapshot="benchmark.jsonl",
        question_ids=[s["question_id"] for s in samples],
        evaluation_status="human_reviewed" if reviewed else "provisional_smoke_only",
        python=platform.python_version(),
        platform=platform.platform(),
        setup_ms=setup_ms,
        code_sha256=code_hashes,
        **split_info,
        **backend_info,
    )
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    (out / "benchmark.jsonl").write_bytes(Path(args.benchmark).read_bytes())
    if config["method"] != "local_bm25_smoke":
        (out / "document_map.json").write_bytes(Path(args.document_map).read_bytes())
    rows = []
    consecutive_errors, stop_reason = 0, None
    with (out / "per_query.jsonl").open("w", encoding="utf-8") as stream:
        for sample in samples:
            start = time.perf_counter()
            status, error = "ok", None
            try:
                chunks = client.retrieve(sample["question"], count)
            except RetrievalError as exc:
                chunks, status, error = [], "error", str(exc)
            row = {
                "question_id": sample["question_id"],
                "question": sample["question"],
                "status": status,
                "error": error,
                "latency_ms": (time.perf_counter() - start) * 1000,
                "stage_latency_ms": getattr(client, "last_stage_latency_ms", None),
                "embedding_tokens": getattr(client, "last_embedding_tokens", None),
                "retrieved_chunks": chunks,
                "relevant_document_ids": sample["relevant_document_ids"],
                "metrics": retrieval_metrics(chunks, sample["relevant_document_ids"]),
                "generated_answer": None,
                "failure_type": "unjudged",
                "failure_reason": None,
            }
            rows.append(row)
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            stream.flush()
            print(json.dumps({"completed": len(rows), "total": len(samples), "question_id": sample["question_id"], "status": status}), flush=True)
            consecutive_errors = consecutive_errors + 1 if status == "error" else 0
            if error == "quota_error" or consecutive_errors >= 3:
                stop_reason = "quota_error" if error == "quota_error" else "three_consecutive_errors"
                print(json.dumps({"stopped": stop_reason}), flush=True)
                break
    if isinstance(client, DenseClient):
        client.close()
    summary = aggregate(rows)
    summary["execution_status"] = "complete" if len(rows) == len(samples) else "stopped_early"
    summary["stop_reason"] = stop_reason
    summary["subgroups"] = {}
    for field in ("question_type", "region"):
        groups = {}
        for sample, row in zip(samples, rows):
            groups.setdefault(str(sample[field]), []).append(row)
        summary["subgroups"][field] = {key: aggregate(values) for key, values in groups.items()}
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    (out / "report.md").write_text(markdown_report(summary, manifest), encoding="utf-8")
    print(json.dumps({"output": str(out), "successful": summary["successful"], "errors": summary["errors"], "metrics": summary["metrics"]}, ensure_ascii=False))
    return int(summary["errors"] > 0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="experiments/configs/local_smoke.json")
    parser.add_argument("--source", default="data/documents.jiangsu_and_2city.jsonl")
    parser.add_argument("--benchmark", default="experiments/datasets/private/benchmark_v0.jsonl")
    parser.add_argument("--document-map")
    parser.add_argument("--split-manifest")
    parser.add_argument("--output-root", default="experiments/results/private")
    parser.add_argument("--experiment-id")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--allow-unreviewed", action="store_true")
    parser.add_argument("--allow-model-api", action="store_true")
    args = parser.parse_args()
    try:
        return run(args)
    except (ValueError, KeyError, FileExistsError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
