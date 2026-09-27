"""Replay saved dev candidates offline; write selections before loading labels."""

import argparse
from collections import defaultdict
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random
import statistics

from experiments.evaluation.context_completion import Candidate, evidence_coverage, select_context


DEV_SHA = "6e2fa94bbf9cf4ac909ddc37b1db29896cc1d02b9ec02c31b9f303356f4cea03"
TEST_PATH = Path("experiments/datasets/private/benchmark_split_v1/test.jsonl")
TEST_SHA = "b6c7a64ec3e89787a0c0ea173acd5127e2669677acbcec920e54f580c682d185"
REVIEW_PATH = Path("experiments/datasets/private/relevance_review_v1/completed_batch_02_v1/reviewed_pairs_cumulative.jsonl")
FREEZE_PATH = Path("experiments/datasets/private/context_completion_v1/pre_evaluation_freeze.json")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_verified(path, expected):
    if digest(path) != expected:
        raise ValueError(f"Input hash changed: {path}")
    return Path(path).read_text()


def jsonlines(text):
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def write_json(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def write_jsonl(path, rows):
    with path.open("x") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def project_candidates(retrieved, snapshot):
    """Allowlist only ranking fields; never forward relevant_document_ids/answers."""
    candidates = []
    for rank, row in enumerate(retrieved, 1):
        cached = snapshot[row["id"]]
        if row["rank"] != rank or any(row[k] != cached[k] for k in ("id", "document_id", "source_document_id")):
            raise ValueError("Saved ranking and snapshot identity mismatch")
        candidates.append(Candidate(cached["id"], cached["document_id"], cached["source_document_id"], rank, cached["order"], cached["content"], cached["tokens"]))
    return candidates


def evaluate_context(question, selected, alternatives, positives, judgments):
    evidence = question["evidence"]
    if not evidence:
        raise ValueError("Cannot evaluate a question with no reference evidence")
    covered = evidence_coverage(evidence, selected)
    complete = all(covered)
    with_alternatives = complete or any(
        all(covered[i] for i in alt["required_original_indices"]) and all(evidence_coverage(alt["evidence"], selected))
        for alt in alternatives
    )
    buckets = dict.fromkeys(["known_positive_document", "unsupported_document", "partial_document", "unjudged_document"], 0)
    for c in selected:
        judgment = judgments.get((question["question_id"], c.source_document_id))
        if c.source_document_id in positives:
            bucket = "known_positive_document"
        elif judgment == "不支持":
            bucket = "unsupported_document"
        elif judgment == "部分支持":
            bucket = "partial_document"
        else:
            bucket = "unjudged_document"
        buckets[bucket] += c.tokens
    return {
        "original_complete": complete,
        "original_span_fraction": sum(covered) / len(covered),
        "evidence_covered": covered,
        "complete_with_alternatives": with_alternatives,
        "tokens": sum(c.tokens for c in selected),
        "chunk_count": len(selected),
        "tokens_by_judgment": buckets,
    }


def _alternatives(qrels, benchmark):
    # Evaluation-only mapping of already reviewed alternative paragraphs. This
    # explicitly preserves the 2026 applicability clause for med-027.
    retained = {"med-001": [], "med-027": [1]}
    result = defaultdict(list)
    for addition in qrels["additions"]:
        qid = addition["question_id"]
        if qid not in retained:
            raise ValueError("New alternative requires an explicit evidence requirement")
        if qid == "med-027" and benchmark[qid]["evidence"][1]["source_document_id"] != addition["policy_identity_context_source"]:
            raise ValueError("Alternative policy context mismatch")
        result[qid].append({"evidence": [addition["evidence"]], "required_original_indices": retained[qid]})
    return result


def _aggregate(rows):
    total_tokens = sum(r["tokens"] for r in rows)
    buckets = {k: sum(r["tokens_by_judgment"][k] for r in rows) for k in rows[0]["tokens_by_judgment"]}
    return {
        "n": len(rows),
        "saved_successes": sum(r["saved_status"] == "ok" for r in rows),
        "original_complete_count": sum(r["original_complete"] for r in rows),
        "original_complete_rate": statistics.mean(r["original_complete"] for r in rows),
        "mean_span_fraction": statistics.mean(r["original_span_fraction"] for r in rows),
        "complete_with_alternatives_count": sum(r["complete_with_alternatives"] for r in rows),
        "mean_tokens": statistics.mean(r["tokens"] for r in rows),
        "max_tokens": max(r["tokens"] for r in rows),
        "mean_chunks": statistics.mean(r["chunk_count"] for r in rows),
        "token_judgment_totals": buckets,
        "token_judgment_shares": {k: v / total_tokens if total_tokens else 0 for k, v in buckets.items()},
    }


def _paired_delta(baseline, treatment, groups, config):
    delta = {qid: int(treatment[qid]["original_complete"]) - int(baseline[qid]["original_complete"]) for qid in baseline}
    grouped = defaultdict(list)
    for qid, value in delta.items():
        grouped[groups[qid]].append(value)
    keys = sorted(grouped)
    rng = random.Random(config["bootstrap_seed"])
    estimates = []
    for _ in range(config["bootstrap_samples"]):
        values = [value for key in rng.choices(keys, k=len(keys)) for value in grouped[key]]
        estimates.append(statistics.mean(values))
    estimates.sort()
    return {
        "complete_rate_delta": statistics.mean(delta.values()),
        "improved": [qid for qid, value in delta.items() if value == 1],
        "regressed": [qid for qid, value in delta.items() if value == -1],
        "group_count": len(keys),
        "exploratory_cluster_bootstrap_95pct": [estimates[int(0.025 * len(estimates))], estimates[int(0.975 * len(estimates))]],
    }


def run(config_path, output):
    config = json.loads(Path(config_path).read_text())
    frozen = json.loads(FREEZE_PATH.read_text())
    for path, expected in frozen["files"].items():
        read_verified(path, expected)
    if digest(config_path) != frozen["files"]["experiments/configs/context_completion_v1.json"]:
        raise ValueError("Run requires the prespecified v1 configuration")
    if config["cloud_calls_allowed"] or config["test_queries_allowed"] or config["benchmark_sha256"] != DEV_SHA:
        raise ValueError("Only frozen offline development replay is permitted")
    read_verified(TEST_PATH, TEST_SHA)  # integrity only, no JSON parsing/test questions
    input_hashes = {**frozen["files"], str(FREEZE_PATH): digest(FREEZE_PATH), str(TEST_PATH): TEST_SHA}
    snapshot = json.loads(Path(config["snapshot"]).read_text())
    chunks = {r["id"]: r for r in snapshot["chunks"]}
    if len(chunks) != 3205 or any(hashlib.sha256(c["content"].encode()).hexdigest() != c["content_sha256"] for c in chunks.values()):
        raise ValueError("Chunk snapshot integrity failure")
    mapping_path = Path("experiments/datasets/private/ragflow_corpus1000_v1/document_map.json")
    read_verified(mapping_path, snapshot["metadata"]["document_map_sha256"])
    input_hashes[str(mapping_path)] = digest(mapping_path)
    # This phase loads no benchmark, qrels or review labels. Serialized ranking
    # files contain historic metrics; projection prevents these entering policy.
    selections, replay_candidates, query_ids = [], {}, None
    for method in config["methods"]:
        specification = config["rankings"][method]
        rankings = jsonlines(read_verified(specification["path"], specification["sha256"]))
        input_hashes[specification["path"]] = specification["sha256"]
        current_ids = [row["question_id"] for row in rankings]
        if len(current_ids) != 51 or len(set(current_ids)) != 51 or (query_ids is not None and current_ids != query_ids):
            raise ValueError("Saved runs have different question identities")
        query_ids = current_ids
        for row in rankings:
            candidates = project_candidates(row["retrieved_chunks"], chunks)
            if len(candidates) > config["candidate_limit"] or (row["status"] != "ok" and candidates):
                raise ValueError("Unexpected saved candidate count or failure content")
            replay_candidates[(method, row["question_id"])] = candidates
            for policy in config["policies"]:
                chosen = select_context(candidates, policy, config["budget_tokens"], config["seed_top_k"])
                if sum(c.tokens for c in chosen) > config["budget_tokens"] or len({c.id for c in chosen}) != len(chosen):
                    raise ValueError("Invalid selected context")
                selections.append({"method": method, "policy": policy, "question_id": row["question_id"], "saved_status": row["status"], "selected_ids": [c.id for c in chosen], "selected_ranks": [c.rank for c in chosen], "tokens": sum(c.tokens for c in chosen)})
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "config.json", config)
    write_jsonl(output / "selections.jsonl", selections)
    # Labels become visible only after all policy outputs have been saved.
    benchmark = {r["question_id"]: r for r in jsonlines(read_verified(config["benchmark"], DEV_SHA))}
    if list(benchmark) != query_ids or any(r["split"] != "dev" or r["review_status"] != "human_approved" for r in benchmark.values()):
        raise ValueError("Not the frozen approved dev set")
    qrels = json.loads(read_verified(config["qrels"], config["qrels_sha256"]))
    reviews = jsonlines(REVIEW_PATH.read_text())
    judgments = {(r["question_id"], r["candidate_source_document_id"]): r["judgment"] for r in reviews}
    alternatives = _alternatives(qrels, benchmark)
    input_hashes.update({config["benchmark"]: DEV_SHA, config["qrels"]: config["qrels_sha256"], str(REVIEW_PATH): digest(REVIEW_PATH)})
    evaluated = []
    for selection in selections:
        qid = selection["question_id"]
        candidates = {c.id: c for c in replay_candidates[(selection["method"], qid)]}
        chosen = [candidates[cid] for cid in selection["selected_ids"]]
        evaluation = evaluate_context(benchmark[qid], chosen, alternatives[qid], set(qrels["relevant_document_ids"][qid]), judgments)
        evaluated.append({**selection, **evaluation})
    write_jsonl(output / "per_query.jsonl", evaluated)
    groups = {qid: r["policy_group_id"] for qid, r in benchmark.items()}
    summary, paired, ceilings = {}, {}, {}
    for method in config["methods"]:
        summary[method], paired[method] = {}, {}
        by_policy = {}
        for policy in config["policies"]:
            rows = [r for r in evaluated if r["method"] == method and r["policy"] == policy]
            summary[method][policy] = _aggregate(rows)
            by_policy[policy] = {r["question_id"]: r for r in rows}
        for policy in config["policies"][1:]:
            paired[method][policy] = _paired_delta(by_policy["ranked"], by_policy[policy], groups, config)
        ceilings[method] = {qid: all(evidence_coverage(r["evidence"], replay_candidates[(method, qid)])) for qid, r in benchmark.items()}
    write_json(output / "summary.json", {"results": summary, "paired_vs_ranked": paired, "unbudgeted_candidate_reference_ceiling": ceilings})
    scripts = [Path(__file__), Path("experiments/evaluation/context_completion.py"), Path("experiments/evaluation/export_context_snapshot.py")]
    for path, expected in input_hashes.items():
        if digest(path) != expected:
            raise ValueError(f"Input changed during run: {path}")
    manifest = {
        "experiment": config["experiment"], "input_sha256": input_hashes,
        "script_sha256": {str(p): digest(p) for p in scripts},
        "output_sha256": {p.name: digest(p) for p in sorted(output.iterdir())},
        "model_calls": 0, "generation": "Not Run", "test_queries": 0,
        "index_unchanged_during_export": snapshot["metadata"]["before"] == snapshot["metadata"]["after"],
        "policies_selected_before_labels_loaded": True,
        "tokenizer_metadata": snapshot["metadata"], "human_reviewed_pairs": len(reviews),
        "question_count": len(benchmark), "question_groups": len(set(groups.values())),
    }
    write_json(output / "manifest.json", manifest)
    report = ["# 固定预算的离线上下文补全实验", "", "51题开发集；每题最多2560 cl100k参考Token；复用保存的候选。0次模型调用，99题测试集未运行。", "", "## 主要结果", "", "完整覆盖要求全部原始参考段出现，允许同来源连续块拼接。这是严格证据代理指标，不是问答正确率。", "", "| 检索 | 上下文规则 | 原始证据完整 / 51 | 允许已审核替代证据 / 51 | 平均段覆盖 | 平均Token | 平均块数 |", "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for method in config["methods"]:
        for policy in config["policies"]:
            r = summary[method][policy]
            report.append(f"| {method} | {policy} | {r['original_complete_count']} | {r['complete_with_alternatives_count']} | {r['mean_span_fraction']:.2%} | {r['mean_tokens']:.1f} | {r['mean_chunks']:.2f} |")
    report += ["", "Rerank原有11次失败按零计入，未重新请求。不同检索方法不是本轮消融变量；每种方法内的三种上下文规则才是配对对照。", "", "## 相对同预算 ranked 的变化", "", "| 检索 | 规则 | 完整率变化 | 新增覆盖题 | 退步题 | 6组bootstrap探索区间 |", "| --- | --- | ---: | --- | --- | --- |"]
    for method in config["methods"]:
        for policy in config["policies"][1:]:
            p = paired[method][policy]
            low, high = p["exploratory_cluster_bootstrap_95pct"]
            report.append(f"| {method} | {policy} | {100*p['complete_rate_delta']:+.2f} pp | {', '.join(p['improved']) or '无'} | {', '.join(p['regressed']) or '无'} | [{100*low:+.2f}, {100*high:+.2f}] pp |")
    report += ["", "## 上下文成本及已审核干扰", "", "以下为全部选中Token的加权占比。正例文档不代表其每块都回答问题，未审核也不等于无关。20个配对审核无法估计全体上下文无关率。", "", "| 检索 | 规则 | 已知正例来源 | 已审核不支持来源 | 部分支持来源 | 未审核来源 |", "| --- | --- | ---: | ---: | ---: | ---: |"]
    for method in config["methods"]:
        for policy in config["policies"]:
            s = summary[method][policy]["token_judgment_shares"]
            report.append(f"| {method} | {policy} | {s['known_positive_document']:.2%} | {s['unsupported_document']:.2%} | {s['partial_document']:.2%} | {s['unjudged_document']:.2%} |")
    report += ["", "## 复现和限制", "", "```bash", "python3 -m experiments.evaluation.context_experiment \\", "  --config experiments/configs/context_completion_v1.json \\", "  --output experiments/results/private/context_completion-dev51-replay", "```", "", "运行只读本地快照，不需要容器、网络或任何模型密钥；输出目录必须不存在。配置与选择代码已在首次评分前冻结。保留selections、per_query、summary与manifest可逐题复核。", "", "本轮只检查全文参考片段覆盖，可能把较长背景段缺失视为失败；替代证据仅有已审核两例。补全规则是在已观察的开发错误上提出，不能声称测试集收益或算法创新。候选外的邻块不会补入。cl100k预算不是Qwen实际账单或完整生成Prompt长度。"]
    (output / "report.md").write_text("\n".join(report) + "\n")
    print(json.dumps({"output": str(output), "results": summary, "paired_vs_ranked": paired}, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="experiments/configs/context_completion_v1.json")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run(args.config, Path(args.output))


if __name__ == "__main__":
    main()
