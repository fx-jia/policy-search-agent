"""Exploratory, query-only local routing under grouped development validation.

No inference API, benchmark difficulty input, test data, or deployed router.
Training labels compare paired quality scores; serving features use query text only.
"""
from collections import defaultdict
import hashlib
import json
import math
import re
from statistics import mean, pstdev
import time

from experiments.evaluation.fast_rag_eval import BASE, DUMP, OUT, PRIVATE, READ, SHA, historical, unique_rows
from experiments.evaluation.fast_rag_report import _paired, cost, distribution, scores

THRESHOLDS = (.8, .9, .95)
FEATURE_NAMES = ("length", "commas", "questions", "demand_words", "compare", "parallel",
                 "calculation", "conditions", "negation", "year_count", "number_count", "source_title", "region_count")


def features(question):
    if not isinstance(question, str):
        raise TypeError("question_text_only")
    return [len(question), len(re.findall(r"[，,；;]", question)), len(re.findall(r"[？?]", question)),
            len(re.findall(r"是否|能否|可否|多少|哪些|什么|如何|怎么|怎样|谁|何时", question)),
            int(bool(re.search(r"分别|相比|对比|区别|比较|各自", question))),
            int(bool(re.search(r"同时|以及|且|并且|两种|两类|三个|两个", question))),
            int(bool(re.search(r"计算|合计|累计|剩余|超出|相加", question))),
            int(bool(re.search(r"如果|若|假如|情况下|情形|假设", question))),
            int(bool(re.search(r"不能|不可以|不适用|不得|除外|未满|不满", question))),
            len(re.findall(r"(?:19|20)\d{2}年?", question)), len(re.findall(r"\d+(?:\.\d+)?", question)),
            int("《" in question or "办法" in question or "通知" in question),
            len(set(re.findall(r"江苏|南京|苏州|无锡|常州|南通|镇江|扬州|泰州|徐州|淮安|盐城|宿迁|连云港|全国|跨省", question)))]


def assign_folds(groups, count=5):
    by_group = defaultdict(list)
    for qid, group in groups.items():
        by_group[group].append(qid)
    if len(by_group) < count:
        raise ValueError("insufficient_policy_groups")
    sizes = [0] * count
    result = {}
    # Balance by group size; hash breaks ties without looking at labels.
    ordered = sorted(by_group, key=lambda g: (-len(by_group[g]), hashlib.sha256(("20260927:" + str(g)).encode()).hexdigest()))
    for group in ordered:
        fold = min(range(count), key=lambda i: (sizes[i], i))
        sizes[fold] += len(by_group[group])
        result.update({qid: fold for qid in by_group[group]})
    return result


def _sigmoid(z):
    return 1 / (1 + math.exp(-max(-35, min(35, z))))


def probability(model, x):
    z = model["weights"][0]
    z += sum(w * (v - m) / s for w, v, m, s in zip(model["weights"][1:], x, model["mean"], model["scale"]))
    return _sigmoid(z)


def fit(x, y):
    centers = [mean(col) for col in zip(*x)]
    scales = [pstdev(col) or 1 for col in zip(*x)]
    values = [[1] + [(v - m) / s for v, m, s in zip(row, centers, scales)] for row in x]
    weights = [0.] * len(values[0])
    for _ in range(400):
        gradient = [0.] * len(weights)
        for row, target in zip(values, y):
            error = _sigmoid(sum(w * v for w, v in zip(weights, row))) - target
            for j, v in enumerate(row):
                gradient[j] += error * v
        weights = [w - .2 * (g / len(values) + (.05 * w if j else 0)) for j, (w, g) in enumerate(zip(weights, gradient))]
    return {"mean": centers, "scale": scales, "weights": weights}


def route_metrics(routes, quality, rows):
    selected = [rows[a][q] for q, a in routes.items()]
    if any(cost(r)["unknown_calls"] for r in selected):
        raise ValueError("unknown_usage_in_route_comparison")
    result = {"n": len(routes), "fast_questions": sum(a == "fast" for a in routes.values()),
              "quality_mean": mean(quality[a][q] for q, a in routes.items()),
              "latency_mean_ms": mean(r["latency_ms"] for r in selected),
              "latency_p95_ms": distribution([r["latency_ms"] for r in selected])["p95"],
              "generation_mean_tokens": mean(cost(r)["known_tokens"] for r in selected),
              "fast_harmful_routes": sum(a == "fast" and quality["fast"][q] < quality["ab"][q] for q, a in routes.items())}
    for arm in ("rag", "ab"):
        result["quality_delta_vs_" + arm] = result["quality_mean"] - mean(quality[arm][q] for q in routes)
        result["latency_reduction_vs_" + arm] = 1 - result["latency_mean_ms"] / mean(rows[arm][q]["latency_ms"] for q in routes)
    return result


def evaluate(out=OUT):
    protocol = json.loads((out / "router_protocol.json").read_text())
    if SHA(__import__("pathlib").Path(__file__)) != protocol["source_sha256"]:
        raise ValueError("router_protocol_changed")
    metrics = json.loads((out / "metrics.json").read_text())
    if not metrics["grading_complete"]:
        raise ValueError("grading_not_complete")
    ids = metrics["three_way_questions"]
    ideal = metrics["hindsight_oracle"]
    grade_rows = READ(out / "quality/records.jsonl") + [r for r in READ(PRIVATE / "answer-quality-dev144-flash/records.jsonl") if r["arm"] in ("rag", "ab")]
    quality = scores(grade_rows)
    if ideal["latency_mean_ms"] >= metrics["timings"]["rag"]["latency_ms"]["mean"] or ideal["quality_mean"] < mean(quality["rag"][q] for q in ids):
        DUMP(out / "router_metrics.json", {"evaluated": False, "reason": "No hindsight speed/quality headroom over Native RAG on current common pairs."})
        return
    questions = {r["question_id"]: r for r in READ(BASE / "dev.jsonl")}
    rows = historical()
    rows["fast"] = unique_rows([READ(out / "generation/records.jsonl")])
    folds = assign_folds({q: questions[q]["policy_group_id"] for q in ids})
    x = {q: features(questions[q]["question"]) for q in ids}
    labels = {q: int(quality["fast"][q] >= quality["ab"][q]) for q in ids}
    predictions, fold_audit = {}, []
    for fold in range(5):
        train = [q for q in ids if folds[q] != fold]
        held = [q for q in ids if folds[q] == fold]
        train_groups = {questions[q]["policy_group_id"] for q in train}
        held_groups = {questions[q]["policy_group_id"] for q in held}
        if train_groups & held_groups:
            raise ValueError("policy_group_leakage")
        model = fit([x[q] for q in train], [labels[q] for q in train])
        predictions.update({q: probability(model, x[q]) for q in held})
        fold_audit.append({"fold": fold, "train_ids": train, "held_ids": held, "train_groups": sorted(train_groups),
                           "held_groups": sorted(held_groups), "model": model})
    # Conservative, untrained structural rule is a separate baseline.
    rules = {q: "fast" if x[q][0] <= 90 and x[q][3] <= 1 and not any(x[q][4:8]) else "ab" for q in ids}
    variants = {"structural_rule": rules}
    variants.update({f"logistic_p{threshold}": {q: "fast" if predictions[q] >= threshold else "ab" for q in ids} for threshold in THRESHOLDS})
    results = {}
    for name, routes in variants.items():
        value = route_metrics(routes, quality, rows)
        routed = {q: quality[a][q] for q, a in routes.items()}
        value["paired_vs_rag"] = _paired(ids, "routed", "rag", {**quality, "routed": routed}, questions)
        value["paired_vs_ab"] = _paired(ids, "routed", "ab", {**quality, "routed": routed}, questions)
        results[name] = value
    # Time feature extraction plus the tiny CPU model; no API or retrieval.
    samples = []
    for _ in range(20):
        for q in ids:
            start = time.perf_counter_ns()
            probability(model, features(questions[q]["question"]))
            samples.append((time.perf_counter_ns() - start) / 1e6)
    result = {"evaluated": True, "n": len(ids), "scope": "Exploratory grouped development out-of-fold predictions; no test or deployment result.",
              "label": "Fast raw quality score >= v8.1 raw quality score; this is not a ground-truth correctness label.",
              "api_calls": 0, "additional_embedding_calls": 0, "runtime_ms": distribution(samples),
              "variants": results, "predictions": predictions, "folds": fold_audit,
              "limitations": "Historical answer/timing replay on three-way scored successful cases only; missing grades and technical failures excluded. All thresholds reported, no winner promoted to production."}
    DUMP(out / "router_metrics.json", result)
    lines = ["# 本地路由开发验证", "", f"{len(ids)} 道三组有效配对，按政策组5折；每题预测来自未见过该政策组标签的模型。输入仅题目文本，不使用难度、题号、参考答案或线上Judge。", "",
             "标签由两条路径的匿名质量分比较产生，存在裁判噪声。以下是离线开发诊断，不是独立测试或上线成绩。", "",
             "| 方案 | Fast题数 | 质量均分 | 对RAG分差 | 对v8.1分差 | 平均耗时(s) | 相对RAG用时减少 | Fast路由后降分题 |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name, r in results.items():
        lines.append(f"| {name} | {r['fast_questions']} | {r['quality_mean']:.2f} | {r['quality_delta_vs_rag']:+.2f} | {r['quality_delta_vs_ab']:+.2f} | {r['latency_mean_ms']/1000:.2f} | {r['latency_reduction_vs_rag']:.1%} | {r['fast_harmful_routes']} |")
    lines += ["", f"特征提取与线性分类的本机平均耗时 {result['runtime_ms']['mean']:.4f} ms，P95 {result['runtime_ms']['p95']:.4f} ms；额外模型/Embedding调用均为0。表中路径耗时基于历史实测，不含这项微小路由开销。",
              "", "全部阈值均展示；不能事后挑一个最有利点并称为最终测试。置信区间、各折训练/验证题号、权重和预测概率见router_metrics.json。未导出生产模型、未接入线上。"]
    (out / "router_report.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"router_evaluated": len(ids), "local_mean_ms": result["runtime_ms"]["mean"], "variants": list(results)}))


if __name__ == "__main__":
    evaluate()
