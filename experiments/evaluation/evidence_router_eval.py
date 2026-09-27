"""Frozen development replay of pre-answer, evidence-aware local routing."""
import argparse
from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path
from statistics import mean
import time

from experiments.evaluation.fast_rag_eval import BASE, DUMP, OUT, PRIVATE, READ, ROOT, SHA, _sources, historical, unique_rows, verify
from experiments.evaluation.fast_rag_report import _paired, distribution, scores
from experiments.evaluation.fast_rag_router import FEATURE_NAMES as QUERY_NAMES
from experiments.evaluation.fast_rag_router import THRESHOLDS, features, fit, probability, route_metrics
from experiments.search_agent.evidence import Evidence, validate_views
from experiments.search_agent.routing import FEATURE_NAMES, evidence_features, evidence_rule

DEST = PRIVATE / "evidence-router-v1-dev165"


def runtime_inputs(row):
    """Recover only the initial retrieval and Fast's literal context, never outcomes."""
    events = [event for event in row["trace"] if event.get("event") == "fast_context"]
    if len(events) != 1:
        raise ValueError("missing_or_duplicate_initial_trace")
    ids = events[0]["initial_ids"]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate_initial_chunk")
    initial = []
    for position, cid in enumerate(ids):
        chunk = row["seen_chunks"][cid]
        initial.append(Evidence(cid, chunk["doc_id"], chunk["docnm_kwd"], chunk["content_with_weight"], position))
    lookup = {e.id: e for e in initial}
    chosen = []
    for item in row["selected"]:
        source = lookup.get(item["chunk_id"])
        if source is None or source.document_id != item["document_id"]:
            raise ValueError("selection_outside_initial_evidence")
        chosen.append(replace(source, text=item["text"], spans=tuple(tuple(s) for s in item["spans"]),
                              truncated=item["truncated"]))
    validate_views(chosen, initial)
    return row["question"], initial, chosen


def grouped_predictions(x, labels, groups, folds):
    if not set(x) == set(labels) == set(groups) == set(folds):
        raise ValueError("inconsistent_cohort")
    predictions, audit = {}, []
    for fold in sorted(set(folds.values())):
        train = [q for q in x if folds[q] != fold]
        held = [q for q in x if folds[q] == fold]
        train_groups, held_groups = {groups[q] for q in train}, {groups[q] for q in held}
        if train_groups & held_groups:
            raise ValueError("policy_group_leakage")
        if not train or not held:
            raise ValueError("empty_partition")
        model = fit([x[q] for q in train], [labels[q] for q in train])
        predictions.update({q: probability(model, x[q]) for q in held})
        audit.append({"fold": fold, "train_ids": train, "held_ids": held, "train_groups": sorted(train_groups),
                      "held_groups": sorted(held_groups), "model": model})
    return predictions, audit


def prepare(out=DEST):
    if (out / "protocol.json").exists():
        raise FileExistsError("already_frozen")
    verify(OUT)
    old = json.loads((OUT / "router_metrics.json").read_text())
    metrics = json.loads((OUT / "metrics.json").read_text())
    ids = metrics["three_way_questions"]
    folds = {q: f["fold"] for f in old["folds"] for q in f["held_ids"]}
    dev = {q["question_id"] for q in READ(BASE / "dev.jsonl")}
    if len(ids) != 165 or set(ids) != set(folds) or not set(ids) <= dev:
        raise ValueError("unexpected_fixed_development_cohort")
    protected = [Path(__file__), BASE / "dev.jsonl", OUT / "metrics.json", OUT / "manifest.json",
                 OUT / "router_metrics.json", OUT / "router_protocol.json", OUT / "generation/records.jsonl",
                 OUT / "generation/config.json", OUT / "quality/records.jsonl",
                 PRIVATE / "answer-quality-dev144-flash/records.jsonl"]
    protected += [p / name for folders in _sources().values() for p in folders for name in ("records.jsonl", "config.json")]
    protected += [ROOT / p for p in (
        "experiments/search_agent/routing.py", "experiments/search_agent/evidence.py",
        "experiments/search_agent/test_routing.py", "experiments/evaluation/test_evidence_router_eval.py",
        "experiments/evaluation/fast_rag_router.py", "experiments/evaluation/fast_rag_report.py",
        "experiments/evaluation/fast_rag_eval.py", "experiments/evaluation/search_agent_eval.py")]
    protocol = {"version": "evidence-router-v1", "frozen_at": datetime.now(timezone.utc).isoformat(),
                "question_ids": ids, "fold_assignments": folds,
                "features": {"query": list(QUERY_NAMES), "evidence": list(FEATURE_NAMES),
                             "combined": list(QUERY_NAMES) + list(FEATURE_NAMES)},
                "thresholds": list(THRESHOLDS), "training": {"steps": 400, "learning_rate": .2, "l2": .05},
                "label": "Saved Fast quality >= saved v8.1 quality; not correctness ground truth.",
                "design": "Fixed old 165 successful scored triplets, same five policy-group folds, train-only scaling. All variants reported.",
                "scope": "Exploratory development after inspecting earlier errors; folds do not undo prior adaptive feature design. No independent test claim.",
                "serving_input": "Question, initial retrieval bodies/titles and selected literal spans only. No ID/difficulty/reference/answer/grade feature.",
                "cost": "Replay chosen historical full-path timing, once per question; add local measured routing overhead. Not a live shared-retrieval execution.",
                "new_model_calls": 0, "new_embedding_calls": 0, "test_used": False,
                "input_sha256": {str(p.relative_to(ROOT)): SHA(p) for p in sorted(set(protected))}}
    out.mkdir(parents=True, exist_ok=True)
    DUMP(out / "protocol.json", protocol)
    print(json.dumps({"prepared": str(out), "n": len(ids)}))


def _verify(out):
    protocol = json.loads((out / "protocol.json").read_text())
    for name, expected in protocol["input_sha256"].items():
        if SHA(ROOT / name) != expected:
            raise ValueError("frozen_input_changed:" + name)
    return protocol


def evaluate(out=DEST):
    protocol = _verify(out)
    ids, folds = protocol["question_ids"], protocol["fold_assignments"]
    questions = {r["question_id"]: r for r in READ(BASE / "dev.jsonl")}
    groups = {q: questions[q]["policy_group_id"] for q in ids}
    rows = historical()
    rows["fast"] = unique_rows([READ(OUT / "generation/records.jsonl")])
    quality = scores(READ(OUT / "quality/records.jsonl") + [r for r in READ(PRIVATE / "answer-quality-dev144-flash/records.jsonl") if r["arm"] in ("rag", "ab")])
    if any(rows[a][q]["status"] != "ok" or q not in quality[a] for a in ("fast", "rag", "ab") for q in ids):
        raise ValueError("incomplete_triplet")
    # This extraction has no access to labels; full inference inputs are source-validated.
    runtime = {q: runtime_inputs(rows["fast"][q]) for q in ids}
    ev = {q: evidence_features(*runtime[q]) for q in ids}
    if any(runtime[q][0] != questions[q]["question"] for q in ids):
        raise ValueError("question_mismatch")
    x = {"query": {q: features(runtime[q][0]) for q in ids},
         "evidence": {q: list(ev[q].values()) for q in ids}}
    x["combined"] = {q: x["query"][q] + x["evidence"][q] for q in ids}
    labels = {q: int(quality["fast"][q] >= quality["ab"][q]) for q in ids}
    predicted, audits = {}, {}
    for family, values in x.items():
        predicted[family], audits[family] = grouped_predictions(values, labels, groups, folds)
    old = json.loads((OUT / "router_metrics.json").read_text())
    if any(abs(predicted["query"][q] - old["predictions"][q]) > 1e-12 for q in ids):
        raise ValueError("query_control_not_reproduced")
    variants = {"evidence_rule": {q: evidence_rule(ev[q]) for q in ids}}
    variants.update({f"{family}_p{threshold}": {q: "fast" if predicted[family][q] >= threshold else "ab" for q in ids}
                     for family in x for threshold in THRESHOLDS})
    # Includes question feature extraction, local lexical processing and model inference.
    # Serialization and source-validation are offline log-reconstruction costs, excluded.
    timings = {family: {q: [] for q in ids} for family in (*x, "evidence_rule")}
    models = {family: {a["fold"]: a["model"] for a in audits[family]} for family in x}
    for _ in range(5):
        for q in ids:
            for family in timings:
                start = time.perf_counter_ns()
                if family == "query":
                    value = features(runtime[q][0])
                else:
                    value = evidence_features(*runtime[q])
                    if family == "evidence_rule":
                        evidence_rule(value)
                    else:
                        value = (features(runtime[q][0]) if family == "combined" else []) + list(value.values())
                if family != "evidence_rule":
                    probability(models[family][folds[q]], value)
                timings[family][q].append((time.perf_counter_ns() - start) / 1e6)
    results, failures = {}, {}
    for name, routes in variants.items():
        family = name.split("_p")[0]
        value = route_metrics(routes, quality, rows)
        routed = {q: quality[a][q] for q, a in routes.items()}
        for arm in ("rag", "ab"):
            value["paired_vs_" + arm] = _paired(ids, "routed", arm, {**quality, "routed": routed}, questions)
        latency = {q: rows[a][q]["latency_ms"] + mean(timings[family][q]) for q, a in routes.items()}
        value["latency_with_router_ms"] = distribution(list(latency.values()))
        value["difficulty"] = {d: {"n": sum(questions[q]["difficulty"] == d for q in ids),
                                   "quality_mean": mean(routed[q] for q in ids if questions[q]["difficulty"] == d)}
                               for d in ("Easy", "Medium", "Hard")}
        if family in ("evidence", "combined"):
            control = variants["query_p" + name.split("_p")[1]]
            control_scores = {q: quality[a][q] for q, a in control.items()}
            value["paired_vs_query_same_threshold"] = _paired(ids, "routed", "query", {"routed": routed, "query": control_scores}, questions)
        value["routes"] = routes
        results[name] = value
        failures[name] = [{"question_id": q, "question": runtime[q][0], "fast_score": quality["fast"][q],
                           "agent_score": quality["ab"][q], "evidence_features": ev[q]}
                          for q, a in routes.items() if a == "fast" and quality[a][q] < quality["ab"][q]]
    data = {"n": len(ids), "policy_groups": len(set(groups.values())), "new_api_calls": 0, "new_embedding_calls": 0,
            "query_control_reproduced": True, "protocol_sha256": SHA(out / "protocol.json"),
            "baselines": {a: route_metrics({q: a for q in ids}, quality, rows) for a in ("fast", "rag", "ab")},
            "variants": results, "predictions": predicted, "folds": audits,
            "runtime_ms": {family: distribution([v for values in per_q.values() for v in values]) for family, per_q in timings.items()},
            "excluded_from_199": 199 - len(ids), "limitations": protocol["scope"] + " Missing grades/technical failures excluded, not zero; timings use historical replay."}
    DUMP(out / "metrics.json", data)
    DUMP(out / "harmful_routes.json", failures)
    DUMP(out / "features.json", {q: {"query": x["query"][q], "evidence": ev[q], "fold": folds[q]} for q in ids})
    _verify(out)
    DUMP(out / "integrity.json", {"inputs_unchanged": True, "checked_files": len(protocol["input_sha256"]),
                                "protocol_sha256": SHA(out / "protocol.json"), "query_control_reproduced": True,
                                "new_api_calls": 0, "new_embedding_calls": 0})
    _report(out, data)
    print(json.dumps({"completed": str(out), "n": len(ids), "variants": len(variants)}, ensure_ascii=False))


def _report(out, data):
    lines = ["# 证据感知轻量路由：开发集离线验证", "",
             "在原来的165道三组有效配对题、22个政策组上，沿用同一5折划分。先冻结协议和输入哈希，再运行比较。每题预测来自未见过该政策组训练标签的模型。",
             "", "这是看过前一轮错误后设计特征的开发实验，不是新的独立测试。199题中的34题因缺少三组有效评分或技术失败未进入比较，没有将缺分当作0分。", "",
             "## 具体改动", "",
             "原路由只看问题文本。本轮增加20个证据信号：问题需求词在全部8块与保留3块中的覆盖差、子需求覆盖差、显式年份/地区/条款/参保对象是否只在未保留内容出现、来源标题匹配和截断比例。词项采用中文二元组及英文/数字词项，权重仅来自本次候选，不使用参考答案、难度、题号或答案质量作为运行输入。",
             "", "对比固定规则、证据特征逻辑回归、问题加证据逻辑回归。标签为历史Fast质量分是否不低于v8.1；缩放和训练均只使用每折训练部分。分类输出不是校准后的正确概率。阈值0.8/0.9/0.95全部展示，没有挑选后部署。原问题路由逐题预测已完整复现。", "",
             "## 质量与成本", "", "质量为0—100量表分，不是正确率。路由在Fast和v8.1之间选择；Native RAG作为速度和质量参照。", "",
             "| 方案 | Fast题数 | 质量均分 | 对RAG分差 | 平均秒数（含路由） | P95秒数 | 平均生成Token | Fast降分题 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name, r in data["baselines"].items():
        lines.append(f"| 始终{name} | {r['fast_questions']} | {r['quality_mean']:.2f} | {r['quality_delta_vs_rag']:+.2f} | {r['latency_mean_ms']/1000:.3f} | {r['latency_p95_ms']/1000:.3f} | {r['generation_mean_tokens']:.0f} | {r['fast_harmful_routes']} |")
    for name, r in data["variants"].items():
        t = r["latency_with_router_ms"]
        lines.append(f"| {name} | {r['fast_questions']} | {r['quality_mean']:.2f} | {r['quality_delta_vs_rag']:+.2f} | {t['mean']/1000:.3f} | {t['p95']/1000:.3f} | {r['generation_mean_tokens']:.0f} | {r['fast_harmful_routes']} |")
    lines += ["", "Fast降分题指走Fast且其历史质量分低于v8.1的题，不等于判定答案完全错误。时延使用所选路径原有实测总时延，仅计一次，并加入本地路由测量；没有真实执行共享初检的在线Adaptive链路，也没有同期随机计时。", "",
              "## 配对不确定性", "", "政策组bootstrap 2000次，区间未做多重比较校正。", "",
              "| 方案 | 对Native RAG质量分差 | 95%区间 | 对v8.1分差 | 95%区间 |", "|---|---:|---|---:|---|"]
    for name, r in data["variants"].items():
        a, b = r["paired_vs_rag"], r["paired_vs_ab"]
        lines.append(f"| {name} | {a['delta_mean']:+.2f} | {a['cluster_bootstrap_95']} | {b['delta_mean']:+.2f} | {b['cluster_bootstrap_95']} |")
    lines += ["", "## 路由本身的开销", "", "每题测量5次，含文本特征、原文视图校验和分类；不含日志反序列化。额外生成及Embedding调用均为0。", ""]
    for name, r in data["runtime_ms"].items():
        lines.append(f"- {name}：平均 {r['mean']:.3f} ms，P95 {r['p95']:.3f} ms。")
    lines += ["", "词项覆盖只是廉价风险信号：同义表述、否定范围、数字关系及跨条款推理仍可能漏判；标题匹配也不能证明原文足够。协议和来源校验见protocol.json、integrity.json；逐题特征、折内模型、路由、分组质量及错误分流见features.json、metrics.json、harmful_routes.json。无生产模型导出、无在线接入。"]
    (out / "report.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "evaluate"))
    parser.add_argument("--output", type=Path, default=DEST)
    args = parser.parse_args()
    (prepare if args.stage == "prepare" else evaluate)(args.output)
