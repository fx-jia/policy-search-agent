"""Compare frozen development runs without calling models or changing labels."""

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random
import statistics

from experiments.analysis.profile_dataset import fingerprint
from experiments.evaluation.metrics import aggregate, percentile


SHARED_FIELDS = (
    "knowledge_base_version",
    "dataset_version",
    "source_sha256",
    "dataset_ids",
    "document_map_sha256",
    "indexed_documents",
    "indexed_chunks",
    "embedding_model",
    "chunking_config",
    "generation_config",
    "split",
    "split_version",
    "split_manifest_sha256",
    "runtime_source_hashes",
    "baseline_plan_sha256",
    "scope_document_ids",
    "question_ids",
)
SHARED_RETRIEVAL = ("candidate_count", "context_top_k", "knn_top_k", "knn_num_candidates", "similarity_threshold")
METRICS = ("recall@1", "recall@5", "mrr@10", "ndcg@5", "context_document_recall@5_chunks", "returned_window_recall")


def validate_manifests(manifests):
    methods = [m["method"] for m in manifests]
    if sorted(methods) != ["dense", "hybrid", "hybrid_rerank"]:
        raise ValueError("Need exactly one Dense, Hybrid and native Hybrid+Rerank run")
    first = manifests[0]
    if first.get("split") != "dev":
        raise ValueError("Only frozen development runs can be compared here")
    for manifest in manifests[1:]:
        for field in SHARED_FIELDS:
            if manifest.get(field) != first.get(field):
                raise ValueError(f"Uncontrolled comparison: {field}")
        for field in SHARED_RETRIEVAL:
            if manifest["retrieval_config"][field] != first["retrieval_config"][field]:
                raise ValueError(f"Uncontrolled retrieval field: {field}")
    hybrid = next(m for m in manifests if m["method"] == "hybrid")
    rerank = next(m for m in manifests if m["method"] == "hybrid_rerank")
    if hybrid["retrieval_config"]["vector_similarity_weight"] != rerank["retrieval_config"]["vector_similarity_weight"]:
        raise ValueError("Hybrid configurations must use the same blend weight")


def paired_cluster_interval(reference, candidate, groups, metric, repeats=2000):
    """Resample entire policy groups, retaining within-group question dependence."""
    values = defaultdict(list)
    for base, other in zip(reference, candidate):
        if base["question_id"] != other["question_id"]:
            raise ValueError("Unpaired question sequence")
        values[groups[base["question_id"]]].append(other["metrics"][metric] - base["metrics"][metric])
    keys = sorted(values)
    if not keys:
        return None
    seed = int(hashlib.sha256(("medical-rag-bootstrap-v1:" + metric).encode()).hexdigest()[:16], 16)
    rng = random.Random(seed)
    draws = []
    for _ in range(repeats):
        draw = [value for key in rng.choices(keys, k=len(keys)) for value in values[key]]
        draws.append(statistics.mean(draw))
    return {
        "mean_delta": statistics.mean(value for values_in_group in values.values() for value in values_in_group),
        "policy_groups": len(keys),
        "replicates": repeats,
        "percentile_95_interval": [percentile(draws, 0.025), percentile(draws, 0.975)],
        "interpretation": "exploratory cluster bootstrap; few broad development groups, not a confirmatory significance test",
    }


def load_run(root):
    manifest = json.loads((root / "manifest.json").read_text())
    if fingerprint(root / "benchmark.jsonl") != manifest["dataset_version"]:
        raise ValueError("Benchmark snapshot hash mismatch")
    samples = [json.loads(line) for line in (root / "benchmark.jsonl").read_text().splitlines()]
    rows = [json.loads(line) for line in (root / "per_query.jsonl").read_text().splitlines()]
    if [row["question_id"] for row in rows] != manifest["question_ids"]:
        raise ValueError("Incomplete or reordered run")
    if any(sample["split"] != "dev" for sample in samples):
        raise ValueError("Non-development sample found")
    return manifest, samples, rows


def compare(roots, output):
    loaded = [load_run(Path(root)) for root in roots]
    validate_manifests([item[0] for item in loaded])
    samples = {sample["question_id"]: sample for sample in loaded[0][1]}
    groups = {qid: sample["policy_group_id"] for qid, sample in samples.items()}
    runs = {manifest["method"]: rows for manifest, _, rows in loaded}
    summaries = {method: aggregate(rows) for method, rows in runs.items()}
    effects = {method: {metric: paired_cluster_interval(runs["hybrid"], runs[method], groups, metric) for metric in METRICS} for method in ("dense", "hybrid_rerank")}
    diagnostics, pool = [], []
    identical_windows = 0
    paired_successful = 0
    for index, qid in enumerate(loaded[0][0]["question_ids"]):
        sample = samples[qid]
        rows = {method: values[index] for method, values in runs.items()}
        if all(rows[method]["status"] == "ok" for method in ("hybrid", "hybrid_rerank")):
            paired_successful += 1
            identical_windows += {c["id"] for c in rows["hybrid"]["retrieved_chunks"]} == {c["id"] for c in rows["hybrid_rerank"]["retrieved_chunks"]}
        observations = {}
        pooled = set(sample["relevant_document_ids"])
        appearances = defaultdict(dict)
        for method, row in rows.items():
            documents = list(dict.fromkeys(c["source_document_id"] for c in row["retrieved_chunks"]))
            for rank, did in enumerate(documents[:5], 1):
                pooled.add(did)
                appearances[did][method] = rank
            metric = row["metrics"]
            if row["status"] != "ok":
                label = "request_error"
            elif metric["returned_window_recall"] < 1:
                label = "known_gold_incomplete_in_returned_window"
            elif metric["recall@5"] < 1:
                label = "known_gold_present_but_below_document_top5"
            elif metric["context_document_recall@5_chunks"] < 1:
                label = "document_top5_hit_but_chunk_context_incomplete"
            else:
                label = "known_gold_context_hit_support_not_yet_judged"
            observations[method] = {"screening_label": label, "document_top5": documents[:5], "metrics": {key: metric[key] for key in METRICS}}
        diagnostics.append({"question_id": qid, "question": sample["question"], "policy_group_id": groups[qid], "methods": observations})
        pool.append(
            {
                "question_id": qid,
                "question": sample["question"],
                "relevant_document_ids": sample["relevant_document_ids"],
                "candidates": [
                    {"source_document_id": did, "status": "known_positive" if did in sample["relevant_document_ids"] else "pending_relevance_review", "document_ranks": appearances[did]}
                    for did in sorted(pooled)
                ],
            }
        )
    result = {
        "split": "dev",
        "split_version": loaded[0][0]["split_version"],
        "knowledge_base_version": loaded[0][0]["knowledge_base_version"],
        "questions": len(runs["hybrid"]),
        "policy_groups": len(set(groups.values())),
        "methods": summaries,
        "paired_effects_vs_hybrid": effects,
        "hybrid_rerank_returned_window_audit": {"paired_successful": paired_successful, "identical_chunk_sets": identical_windows},
        "input_artifacts": [
            {"path": str(Path(root).resolve()), "manifest_sha256": fingerprint(Path(root) / "manifest.json"), "per_query_sha256": fingerprint(Path(root) / "per_query.jsonl")} for root in roots
        ],
        "qrels_scope": "known_positive_documents_only",
        "generation": "Not Run",
    }
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    (out / "comparison.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    for name, rows in (("diagnostics.jsonl", diagnostics), ("relevance_pool.jsonl", pool)):
        (out / name).write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    lines = [
        "# 开发集检索基线对照",
        "",
        f"{result['questions']} 题，{result['policy_groups']} 个政策关联组；测试集未参与。",
        "",
        "| 方法 | 成功/总数 | 文档 Recall@5 | MRR@10 | nDCG@5 | 前5块来源覆盖 | 平均检索毫秒 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for method in ("dense", "hybrid", "hybrid_rerank"):
        summary = summaries[method]
        m = summary["metrics"]
        mean = summary["latency_ms"]["mean"]
        lines.append(
            f"| {method} | {summary['successful']}/{summary['attempted']} | {m['recall@5']:.4f} | {m['mrr@10']:.4f} | {m['ndcg@5']:.4f} | {m['context_document_recall@5_chunks']:.4f} | {mean:.1f} |"
            if mean is not None
            else f"| {method} | 0/{summary['attempted']} | 0 | 0 | 0 | 0 | — |"
        )
    lines.extend(
        [
            "",
            "分数只针对已知来源正例；未标注候选待复核。请求失败计零分，延迟只统计成功请求。",
            "",
            "Dense 使用同一运行容器的 Qwen Embedding 与 ES 纯 KNN；Hybrid 两组使用原生 HTTP 接口。调用路径不同，当前延迟是部署下观测，不是算法计算效率的公平排名。",
            "",
            "原生 Rerank 分支同时改变模型评分和词项表示权重；这三组用于基线配置比较，尚不能作为纯重排模块的单变量消融。",
            "",
            f"Hybrid 与 Rerank 成对成功 {paired_successful} 题，其中返回的 Chunk 集合完全相同 {identical_windows} 题。集合相同不代表内部打分过程相同。",
            "",
            "按政策组重采样的探索性区间保存于 comparison.json。开发侧仅 6 个宽泛关联组，不能据此作强泛化或显著性结论。",
            "",
            "diagnostics.jsonl 是已知正例与排名位置的程序筛查，不是人工失败归因。relevance_pool.jsonl 合并三种方法的文档 Top5 和已有正例；未判定候选不自动标负例。",
            "",
            "Answer Accuracy / Faithfulness / Citation Accuracy：Not Run。尚未运行生成或最终测试。",
            "",
        ]
    )
    (out / "report.md").write_text("\n".join(lines))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs=3, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = compare(args.runs, args.output)
    print(json.dumps({"output": args.output, "questions": result["questions"], "methods": list(result["methods"])}))


if __name__ == "__main__":
    main()
