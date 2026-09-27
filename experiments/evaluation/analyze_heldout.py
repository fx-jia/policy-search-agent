"""Independent character-interval audit of the frozen held-out retrieval results."""

import argparse
from collections import defaultdict
import hashlib
import html
import json
from pathlib import Path
import re
import statistics


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def normalize(text):
    return "".join(re.sub(r"<[^>]+>", "", html.unescape(text)).split())


def audit(root, evaluation, replay):
    config = read("experiments/configs/heldout_retrieval_v1.json")
    freeze = read("experiments/datasets/private/heldout_retrieval_v1/freeze.json")
    for path, expected in freeze["files"].items():
        assert digest(path) == expected, path
    snapshot = read(config["snapshot"])
    chunks = {c["id"]: c for c in snapshot["chunks"]}
    benchmark = {q["question_id"]: q for q in rows(config["benchmark"])}
    rankings = {(r["method"], r["question_id"]): r for r in rows(root / "rankings.jsonl")}
    contexts = rows(evaluation / "per_query.jsonl")
    summary = read(evaluation / "summary.json")
    bydoc = defaultdict(list)
    for chunk in chunks.values():
        bydoc[chunk["document_id"]].append(chunk)
    sources = defaultdict(list)
    for doc_chunks in bydoc.values():
        text, intervals = "", []
        ordered = sorted(doc_chunks, key=lambda c: c["order"])
        for c in ordered:
            start = len(text)
            text += normalize(c["content"])
            intervals.append((start, len(text), c["id"]))
        sources[ordered[0]["source_document_id"]].append((text, intervals))
    options = {}
    for qid, question in benchmark.items():
        options[qid] = []
        for ref in question["evidence"]:
            needle, occurrences = normalize(ref["text"]), []
            assert needle
            for text, intervals in sources[ref["source_document_id"]]:
                position = text.find(needle)
                while position >= 0:
                    overlapping = [i for i, (a, b, _) in enumerate(intervals) if a < position + len(needle) and b > position]
                    if overlapping:
                        occurrences.append({intervals[i][2] for i in range(min(overlapping), max(overlapping) + 1)})
                    position = text.find(needle, position + 1)
            options[qid].append(occurrences)

    def coverage(qid, ids):
        return [any(required <= set(ids) for required in occurrences) for occurrences in options[qid]]

    comparisons, failures = 0, []
    for row in contexts:
        qid, ids = row["question_id"], row["selected_ids"]
        candidate_ids = [c["id"] for c in rankings[(row["method"], qid)]["retrieved_chunks"]]
        assert len(ids) == len(set(ids)) and set(ids) <= set(candidate_ids)
        assert row["tokens"] == sum(chunks[cid]["tokens"] for cid in ids) <= 2560
        checked = coverage(qid, ids)
        assert row["evidence_covered"] == checked and row["original_complete"] == all(checked)
        comparisons += len(checked)
        if not all(checked):
            missing = []
            for i, covered in enumerate(checked):
                if covered:
                    continue
                occurrences = options[qid][i]
                ref = benchmark[qid]["evidence"][i]
                required = [{"chunk_id": cid, "candidate_rank": candidate_ids.index(cid) + 1 if cid in candidate_ids else None, "selected": cid in ids, "tokens": chunks[cid]["tokens"]} for cid in sorted(set().union(*occurrences))] if occurrences else []
                category = "reference_not_located_in_snapshot" if not occurrences else "candidate_window_missing" if not any(s <= set(candidate_ids) for s in occurrences) else "context_budget_or_priority"
                missing.append({"reference_index": i, "source_document_id": ref["source_document_id"], "category": category, "required_chunks": required, "reference_text": ref["text"]})
            failures.append({"question_id": qid, "question": benchmark[qid]["question"], "method": row["method"], "policy": row["policy"], "missing": missing, "selected_ids": ids, "candidate_reference_complete": all(coverage(qid, candidate_ids))})
    for method in ["dense", "hybrid"]:
        recall5, mrr, ceiling = [], [], 0
        for qid, q in benchmark.items():
            candidate_ids = [c["id"] for c in rankings[(method, qid)]["retrieved_chunks"]]
            documents = []
            for cid in candidate_ids:
                sid = chunks[cid]["source_document_id"]
                if sid not in documents:
                    documents.append(sid)
            relevant = set(q["relevant_document_ids"])
            recall5.append(len(set(documents[:5]) & relevant) / len(relevant))
            mrr.append(next((1 / rank for rank, sid in enumerate(documents[:10], 1) if sid in relevant), 0))
            ceiling += all(coverage(qid, candidate_ids))
        assert statistics.mean(recall5) == summary["rankings"][method]["metrics"]["recall@5"]
        assert statistics.mean(mrr) == summary["rankings"][method]["metrics"]["mrr@10"]
        assert ceiling == summary["candidate_reference_ceiling"][method]
    replay_hashes = {}
    for name in ["per_query.jsonl", "summary.json", "ceilings.json", "generation_readiness.jsonl", "manifest.json"]:
        assert (evaluation / name).read_bytes() == (replay / name).read_bytes(), name
        replay_hashes[name] = digest(evaluation / name)
    before = read(root / "run_manifest.json")["index_before"]
    after = read(root / "retrieval_freeze.json")["index_after"]
    assert before == after == snapshot["metadata"]["before"]
    readiness = rows(evaluation / "generation_readiness.jsonl")
    pending = set(cid for r in readiness for cid in r["needs_review_ids"])
    denied = set(cid for r in readiness for cid in r["blocked_ids"])
    changes = []
    bykey = {(r["method"], r["policy"], r["question_id"]): r for r in contexts}
    for qid in benchmark:
        a, b = bykey[("dense", "ranked", qid)], bykey[("dense", "boundary", qid)]
        if a["original_complete"] != b["original_complete"]:
            changes.append({"question_id": qid, "question": benchmark[qid]["question"], "change": "improved" if b["original_complete"] else "regressed", "ranked_covered": a["evidence_covered"], "boundary_covered": b["evidence_covered"], "added_ids": [cid for cid in b["selected_ids"] if cid not in a["selected_ids"]], "removed_ids": [cid for cid in a["selected_ids"] if cid not in b["selected_ids"]]})
    validation = {"rankings_checked": len(rankings), "contexts_checked": len(contexts), "independent_reference_interval_checks": comparisons, "frozen_hashes_verified": len(freeze["files"]), "index_unchanged": True, "replay_byte_identical": replay_hashes, "model_calls_in_audit": 0, "all_failures_recorded": len(failures)}
    analysis = {"dense_boundary_changes": changes, "generation_readiness": {"conditions": len(readiness), "fully_previously_reviewed_conditions": sum(r["already_reviewed"] for r in readiness), "unique_chunks_needing_review": len(pending), "unique_blocked_chunks": len(denied), "pending_sources": len({chunks[cid]["source_document_id"] for cid in pending}), "generation_calls": 0}, "no_qrels_or_method_changes": True, "interpretation": "strict original-reference coverage is not answer accuracy"}
    for name, value in [("validation.json", validation), ("analysis.json", analysis)]:
        with (root / name).open("x") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
    with (root / "failure_cases.jsonl").open("x") as stream:
        for row in failures:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    return summary, analysis, validation


def render(summary, analysis, validation):
    lines = ["# 99题保留集：检索与上下文结果", "", "本轮为冻结方法后的首次保留集检索评测；99题、29政策组、相同1000篇/3205块索引。未运行生成，所有数值都不能写成问答准确率。", "", "| 检索 | 文档Recall@5 | 文档MRR@10 | ranked完整证据 | boundary完整证据 | 候选64块完整证据上限 |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for method in ["dense", "hybrid"]:
        m = summary["rankings"][method]["metrics"]
        a, b = summary["context"][method + "_ranked"], summary["context"][method + "_boundary"]
        lines.append(f"| {method} | {m['recall@5']:.2%} | {m['mrr@10']:.4f} | {a['complete']}/99 | {b['complete']}/99 | {summary['candidate_reference_ceiling'][method]}/99 |")
    lines += ["", "完整证据要求全部原始参考段在选中同来源连续Chunk中出现；它可能因长背景段或未补标的替代依据过严。文档级指标按来源首次出现去重，并只衡量已标正例。原生Hybrid保留其真实查询和排序实现，不把它当作手写BM25/Dense分数融合。", "", "| 对比 | 配对变化 | 新增完整题 | 退步题 | 29组bootstrap探索95%区间 |", "| --- | ---: | --- | --- | --- |"]
    for name, contrast in summary["paired"].items():
        lo, hi = contrast["exploratory_cluster_bootstrap_95pct"]
        lines.append(f"| {name} | {100 * contrast['complete_rate_delta']:+.2f} pp | {', '.join(contrast['improved']) or '无'} | {', '.join(contrast['regressed']) or '无'} | [{100 * lo:+.2f}, {100 * hi:+.2f}] pp |")
    lines += ["", "## 上下文开销", "", "| 条件 | 平均参考段覆盖 | 平均参考Token |", "| --- | ---: | ---: |"]
    for name, values in summary["context"].items():
        lines.append(f"| {name} | {values['mean_span_fraction']:.2%} | {values['mean_tokens']:.1f} |")
    lines += ["", "## 验证与后续", "", f"独立核验{validation['rankings_checked']}份排名、{validation['contexts_checked']}条上下文及{validation['independent_reference_interval_checks']}次字符区间覆盖。五份离线评价输出重放逐字节一致；索引前后及全部冻结输入摘要一致。全部{validation['all_failures_recorded']}条不完整上下文保存在failure_cases.jsonl，不因指标不利删题、改参考段或重跑检索。", "", f"后续两条生成条件共{analysis['generation_readiness']['conditions']}份上下文，其中{analysis['generation_readiness']['fully_previously_reviewed_conditions']}份全部在旧审核清单内；另有{analysis['generation_readiness']['unique_chunks_needing_review']}个唯一片段需要出站检查，{analysis['generation_readiness']['unique_blocked_chunks']}个唯一片段来自禁用来源。本次文档正文没有发送到模型。", "", "正式检索结果用于检验开发结论，不再用这99题选择新的权重、预算、Prompt或重排模型。后续生成评价延用冻结方案并单独报告，不能挑选较好的测试方法回填成原方案。"]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--replay", type=Path, required=True)
    args = parser.parse_args()
    summary, analysis, validation = audit(args.result, args.evaluation, args.replay)
    with (args.result / "report.md").open("x") as stream:
        stream.write(render(summary, analysis, validation))
    print(json.dumps({"analysis": analysis, "validation": validation}, ensure_ascii=False))
