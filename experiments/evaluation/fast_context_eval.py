"""Fast v2: frozen question-aware context comparison on the same dev199."""
import argparse
from collections import Counter
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
from statistics import mean
import uuid

from experiments.evaluation import answer_quality_eval as judge
from experiments.evaluation import search_agent_eval as supervisor
from experiments.evaluation.answer_quality import SYSTEM, build_packet, covered_by_sources
from experiments.evaluation.fast_rag_eval import BASE, DUMP, OUT, PRIVATE, READ, ROOT, SHA, historical, unique_rows, verify
from experiments.evaluation.fast_rag_report import _paired, cost, distribution, scores

DEST = PRIVATE / "fast-rag-v2-dev199"


def patch_runner(source):
    start, end = "def select_context(", "async def run("
    call = "select_context(initial, backend.count_tokens)"
    if source.count(start) != 1 or source.count(end) != 1 or source.count(call) != 1:
        raise ValueError("unexpected_fast_runner")
    left, right = source.index(start), source.index(end)
    if left >= right:
        raise ValueError("unexpected_fast_runner")
    source = source[:left] + "from .fast_context import select_context\n\n\n" + source[right:]
    return source.replace(call, "select_context(question, initial, backend.count_tokens, context_text)")


def prepare(out=DEST):
    if (out / "manifest.json").exists():
        raise FileExistsError("already_frozen")
    old = verify(OUT)
    config = copy.deepcopy(json.loads((OUT / "generation/config.json").read_text()))
    ids = old["question_ids"]
    if len(ids) != 199 or config["question_ids"] != ids:
        raise ValueError("cohort_changed")
    protected = [ROOT / name for name in old["input_sha256"]]
    protected += [OUT / p for p in ("manifest.json", "generation/config.json", "generation/records.jsonl", "quality/config.json", "quality/jobs.json", "quality/records.jsonl")]
    protected += [ROOT / p for p in ("experiments/search_agent/fast_context.py", "experiments/search_agent/test_fast_context.py",
                                    "experiments/evaluation/fast_context_eval.py", "experiments/evaluation/test_fast_context_eval.py",
                                    "experiments/evaluation/fast_rag_report.py", "experiments/evaluation/answer_quality.py",
                                    "experiments/evaluation/answer_quality_eval.py", "experiments/evaluation/search_agent_eval.py")]
    modules = {}
    for name, source in config["module_sources"].items():
        if name == "experiments.search_agent.fast":
            modules["experiments.search_agent.fast_context"] = (ROOT / "experiments/search_agent/fast_context.py").read_text()
            source = patch_runner(source)
        modules[name] = source
    config.update(module_sources=modules, fast_rag={"version": "v2", "max_chunks": 3, "context_tokens": 2000,
                  "selection": "query_weighted_novelty_full_articles_or_neighbor_sentences"},
                  stage_note="Same Fast v1 runner; only literal context selection replaced.",
                  container_ledger="/tmp/fast-v2-" + uuid.uuid4().hex + "/calls.jsonl")
    hashes = {str(p.relative_to(ROOT)): SHA(p) for p in sorted(set(protected))}
    config["input_hashes"] = {**config["input_hashes"], **hashes}
    config["input_hashes"].update({name.replace(".", "/") + ".py": hashlib.sha256(source.encode()).hexdigest() for name, source in modules.items()})
    gen = out / "generation"
    gen.mkdir(parents=True, exist_ok=True)
    DUMP(gen / "config.json", config)
    DUMP(gen / "manifest.json", {"config_sha256": SHA(gen / "config.json"), "planned": ids, "pilot": [],
         "direct_dev30": True, "replay_only": False, "stage_budgets": {"pilot": 0, "full": len(ids) * 60000},
         "model": config["model"], "strategy": "fast-rag-v2"})
    DUMP(out / "manifest.json", {"version": "fast-rag-v2", "frozen_at": datetime.now(timezone.utc).isoformat(),
         "question_ids": ids, "input_sha256": hashes, "generation_config_sha256": SHA(gen / "config.json"),
         "judge_model": old["judge_model"], "rubric": old["rubric"], "test_used": False,
         "protocol": "All original199 dev questions. Same model/index/retrieval/template/3-source/2000-token budget. One frozen selector, no per-question tuning or repeated generation. Three technical smoke attempts are retained.",
         "grading": "Only new Fast v2 answers sent to the same anonymous Flash rubric. Reuse all prior baseline grades; invalid or missing grades excluded, never zero.",
         "timing": "Serial generation; historical baseline timings, not randomized contemporaneous E2E controls. No router changed or trained in this batch."})
    supervisor.verify_frozen_inputs(config)
    DUMP(out / "progress.json", {"phase": "prepared", "planned": len(ids)})
    print(json.dumps({"prepared": str(out), "questions": len(ids), "model": config["model"]}), flush=True)


def verify_v2(out):
    manifest = verify(out)
    verify(OUT)
    return manifest


def usage_summary(records):
    measured = [cost(r) for r in records if isinstance(r.get("calls"), list)]
    result = {k: sum(v[k] for v in measured) for k in ("known_tokens", "unknown_calls", "unknown_reserve")}
    result["unreported_records"] = len(records) - len(measured)
    result["total_tokens"] = None if result["unknown_calls"] or result["unreported_records"] else result["known_tokens"]
    return result


def verify_quality(out):
    manifest = json.loads((out / "quality/manifest.json").read_text())
    for name, expected in manifest["input_sha256"].items():
        if SHA(ROOT / name) != expected:
            raise ValueError("grading_input_changed:" + name)
    for name in ("config", "jobs"):
        if SHA(out / "quality" / (name + ".json")) != manifest[name + "_sha256"]:
            raise ValueError("grading_input_changed:" + name)


def arm_summary(arm, common, rows, quality, questions):
    selected = [rows[arm][q] for q in common]
    unknown = [not isinstance(r.get("calls"), list) or bool(cost(r)["unknown_calls"]) for r in selected]
    result = {"quality_mean": mean(quality[arm][q] for q in common),
              "latency_ms": distribution([r["latency_ms"] for r in selected]),
              "tokens": distribution([cost(r)["known_tokens"] for r, missing in zip(selected, unknown) if not missing]),
              "unknown_usage_questions": sum(unknown), "difficulty": {}}
    for difficulty in ("Easy", "Medium", "Hard"):
        ids = [q for q in common if questions[q]["difficulty"] == difficulty]
        result["difficulty"][difficulty] = {"n": len(ids), "quality_mean": mean(quality[arm][q] for q in ids) if ids else None}
    return result


def prepare_quality(out):
    manifest = verify_v2(out)
    target = out / "quality"
    if (target / "manifest.json").exists():
        return
    rows = unique_rows([READ(out / "generation/records.jsonl")])
    if set(rows) != set(manifest["question_ids"]):
        raise ValueError("generation_incomplete")
    config = json.loads((out / "generation/config.json").read_text())
    safe = config["review"]["entries"]
    templates = {}
    for folder in (PRIVATE / "answer-quality-v2", PRIVATE / "answer-quality-dev144-flash"):
        for job in json.loads((folder / "jobs.json").read_text()):
            template = {k: v for k, v in job["packet"].items() if k not in ("answer", "citations")}
            qid = job["question_id"]
            if qid in templates and templates[qid] != template:
                raise ValueError("inconsistent_rubric")
            templates[qid] = template
    dev = {r["question_id"]: r for r in READ(BASE / "dev.jsonl")}
    jobs, exclusions = [], []
    for qid in manifest["question_ids"]:
        row = rows[qid]
        if row["status"] != "ok":
            exclusions.append({"question_id": qid, "status": row["status"]})
            continue
        for e in dev[qid]["evidence"]:
            if not covered_by_sources(e["text"], [s["sanitized_content"] for s in safe.values() if s["source_document_id"] == e["source_document_id"]]):
                raise ValueError("unverified_reference_source")
        for cid, c in row["seen_chunks"].items():
            if c["content_with_weight"] != safe[cid]["sanitized_content"] or c["docnm_kwd"] != safe[cid]["safe_title"]:
                raise ValueError("unreviewed_outbound_source")
        packet = build_packet(dev[qid], row)
        packet.update(templates[qid])
        jobs.append({"id": hashlib.sha256((qid + ":fast2").encode()).hexdigest()[:20], "question_id": qid,
                     "arm": "fast2", "stage": "extension", "packet": packet})
    random.Random(20260927).shuffle(jobs)
    qc = json.loads((OUT / "quality/config.json").read_text())
    if qc["model"] != manifest["judge_model"] or qc["system"] != SYSTEM:
        raise ValueError("judge_rubric_changed")
    qc["ledger_root"] = "/tmp/fast-v2-quality-" + uuid.uuid4().hex
    target.mkdir(parents=True, exist_ok=True)
    DUMP(target / "config.json", qc)
    DUMP(target / "jobs.json", jobs)
    protected = [out / "manifest.json", out / "generation/records.jsonl"]
    DUMP(target / "manifest.json", {"rubric": manifest["rubric"], "question_ids": manifest["question_ids"],
         "valid_answers": len(jobs), "technical_exclusions": exclusions,
         "input_sha256": {**manifest["input_sha256"], **{str(p.relative_to(ROOT)): SHA(p) for p in protected}},
         "config_sha256": SHA(target / "config.json"), "jobs_sha256": SHA(target / "jobs.json"),
         "blindness": "Anonymous single answer; no arm/id/difficulty/prior score in judge input."})


def report(out):
    manifest = verify_v2(out)
    verify_quality(out)
    questions = {r["question_id"]: r for r in READ(BASE / "dev.jsonl")}
    rows = historical()
    rows["fast"] = unique_rows([READ(OUT / "generation/records.jsonl")])
    rows["fast2"] = unique_rows([READ(out / "generation/records.jsonl")])
    current = READ(out / "quality/records.jsonl")
    quality = scores(current + READ(OUT / "quality/records.jsonl") +
                     [r for r in READ(PRIVATE / "answer-quality-dev144-flash/records.jsonl") if r["arm"] in ("rag", "ab")])
    arms, planned = ("fast2", "fast", "rag", "ab"), manifest["question_ids"]
    common = [q for q in planned if all(q in quality.get(a, {}) and rows[a][q]["status"] == "ok" for a in arms)]
    if not common:
        raise ValueError("no_common_scored_answers")
    pairs = {a: _paired([q for q in planned if q in quality.get("fast2", {}) and q in quality.get(a, {})], "fast2", a, quality, questions) for a in ("fast", "rag", "ab")}
    metrics = {"planned": len(planned), "generation_status": dict(Counter(r["status"] for r in rows["fast2"].values())),
               "grade_status": dict(Counter(r["status"] for r in current)), "common_question_ids": common,
               "pairs": pairs, "arms": {}, "new_generation_cost": {}, "new_judge_cost": {},
               "query_embedding": {"known_calls": sum(r["query_embedding_calls"] for r in rows["fast2"].values() if "query_embedding_calls" in r),
                                   "known_tokens": sum(r["query_embedding_tokens"] for r in rows["fast2"].values() if "query_embedding_tokens" in r),
                                   "unreported_records": sum("query_embedding_calls" not in r or "query_embedding_tokens" not in r for r in rows["fast2"].values())}}
    for name, records in (("new_generation_cost", list(rows["fast2"].values())), ("new_judge_cost", current)):
        metrics[name] = usage_summary(records)
    for a in arms:
        metrics["arms"][a] = arm_summary(a, common, rows, quality, questions)
    diagnostic = []
    for q in planned:
        new, old = rows["fast2"][q], rows["fast"][q]
        new_trace = next((t for t in new.get("trace", []) if t.get("event") == "fast_context"), None)
        old_trace = next((t for t in old.get("trace", []) if t.get("event") == "fast_context"), None)
        diagnostic.append({"question_id": q, "initial_same_order": bool(new_trace and old_trace and new_trace["initial_ids"] == old_trace["initial_ids"]),
                           "old_selected": [e["chunk_id"] for e in old.get("selected", [])], "new_selected": [e["chunk_id"] for e in new.get("selected", [])],
                           "scores": {a: quality.get(a, {}).get(q) for a in arms}})
    metrics["same_initial_order_count"] = sum(r["initial_same_order"] for r in diagnostic)
    metrics["selection_ms"] = distribution([r["usage"]["selection_ms"] for r in rows["fast2"].values() if "selection_ms" in r.get("usage", {})])
    DUMP(out / "metrics.json", metrics)
    DUMP(out / "diagnostics.json", diagnostic)
    lines = ["# Fast v2：问题相关原文选择开发实验", "", f"原199道开发题；四组共同有效评分{len(common)}题。质量分不是正确率。模型、原索引、Hybrid参数、回答模板、最多3来源及2000 Token预算保持一致，仅改变上下文选择。", "",
             "按照问题词项在本次8个候选中的权重、子需求覆盖和来源间覆盖增量挑选来源；原文优先完整保留，必要时按完整编号条款或命中句及相邻句抽取。保留原文身份和范围前缀，无法放下完整单元时跳过，不按字符截断。没有LLM Judge控制器、补检或额外生成。", "",
             "| 方案 | 质量均分 | 平均秒数 | P95秒数 | 平均生成Token |", "|---|---:|---:|---:|---:|"]
    for a, r in metrics["arms"].items():
        tokens = f"{r['tokens']['mean']:.0f}" if r['tokens']['mean'] is not None else "未知"
        lines.append(f"| {a} | {r['quality_mean']:.2f} | {r['latency_ms']['mean']/1000:.2f} | {r['latency_ms']['p95']/1000:.2f} | {tokens} |")
    lines += ["", "各对照使用可评分同题配对，样本数可能不同；政策组bootstrap区间未做多重比较校正。", "",
              "| v2 对照 | 有效题数 | 质量分差 | 95%区间 | 升/平/降分题 |", "|---|---:|---:|---|---|"]
    for a, p in pairs.items():
        lines.append(f"| {a} | {p['n']} | {p['delta_mean']:+.2f} | {p['cluster_bootstrap_95']} | {p['wins']}/{p['ties']}/{p['losses']} |")
    selection_time = metrics['selection_ms']['mean']
    lines += ["", f"生成状态：{metrics['generation_status']}；新评分状态：{metrics['grade_status']}。缺失、无效评分和技术失败单列，不按0分处理。",
              "", f"相对v1初检顺序一致的记录有{metrics['same_initial_order_count']}/199；逐题变化见diagnostics.json。平均本地选择用时{selection_time}毫秒。",
              "", "旧RAG/v8.1/Fast v1答案及评分均复用；耗时比较为历史对照，不能排除服务负载差异。参考上一轮错误设计的开发版本不构成独立泛化测试，也没有接入或调优Router。用量账本、分组质量和排除项见metrics.json。"]
    (out / "report.md").write_text("\n".join(lines) + "\n")
    answers = ["# Fast v2 所有开发题答案", ""]
    for q in planned:
        answers += [f"## {q}", "", questions[q]["question"], "", f"状态：{rows['fast2'][q]['status']}；质量分：{quality.get('fast2', {}).get(q, '无有效分数')}", "", rows["fast2"][q].get("answer", ""), ""]
    (out / "all_answers.md").write_text("\n".join(answers))
    verify_v2(out)
    verify_quality(out)
    DUMP(out / "integrity.json", {"inputs_unchanged": True, "protected_files": len(manifest["input_sha256"]),
                                 "protocol_sha256": SHA(out / "manifest.json"), "old_results_unchanged": True})


def execute(out=DEST):
    manifest = verify_v2(out)
    gen = out / "generation"
    if not READ(gen / "records.jsonl"):
        DUMP(out / "progress.json", {"phase": "generation_smoke", "planned": 199})
        supervisor.run_batch(gen, "full", maximum=3)
        first = READ(gen / "records.jsonl")
        if len(first) != 3 or any(r["status"] != "ok" for r in first):
            DUMP(out / "progress.json", {"phase": "inspection_required", "generated": len(first)})
            return
    DUMP(out / "progress.json", {"phase": "generation", "planned": 199})
    supervisor.run_batch(gen, "full")
    generated = len(READ(gen / "records.jsonl"))
    if generated != len(manifest["question_ids"]):
        DUMP(out / "progress.json", {"phase": "inspection_required", "generated": generated})
        return
    prepare_quality(out)
    DUMP(out / "progress.json", {"phase": "grading", "generated": generated})
    judge.run("all", out / "quality")
    graded = len(READ(out / "quality/records.jsonl"))
    planned = len(json.loads((out / "quality/jobs.json").read_text()))
    if graded != planned:
        DUMP(out / "progress.json", {"phase": "grading_inspection_required", "generated": generated, "graded": graded, "grading_planned": planned})
        return
    report(out)
    DUMP(out / "progress.json", {"phase": "complete", "generated": generated, "graded": graded})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "run", "report"))
    parser.add_argument("--output", type=Path, default=DEST)
    args = parser.parse_args()
    {"prepare": prepare, "run": execute, "report": report}[args.stage](args.output)
