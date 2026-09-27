"""Frozen Fast RAG development comparison using saved native/agent answers."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import random
import uuid

from experiments.evaluation import answer_quality_eval as quality
from experiments.evaluation import search_agent_eval as supervisor
from experiments.evaluation.answer_quality import RUBRIC_VERSION, SYSTEM, WEIGHTS, build_packet, covered_by_sources

ROOT = Path(__file__).resolve().parents[2]
PRIVATE = ROOT / "experiments/results/private"
BASE = PRIVATE / "benchmark-v2-baselines-v2-nonthinking"
STUDY = PRIVATE / "policy-search-agent-v8_1-study/extension"
DEV144 = PRIVATE / "policy-search-agent-v8_1-dev144"
OUT = PRIVATE / "fast-rag-v1-dev199"
READ = supervisor.readl
DUMP = supervisor.dump
SHA = supervisor.digest


def unique_rows(groups):
    rows = {}
    for group in groups:
        for row in group:
            qid = row["question_id"]
            if qid in rows:
                raise ValueError("duplicate_answer:" + qid)
            rows[qid] = row
    return rows


def paired_ids(dev, rag, agent):
    if not set(agent) <= set(dev) or not set(rag) <= set(dev):
        raise ValueError("outside_development")
    if not set(agent) <= set(rag):
        raise ValueError("missing_historical_rag")
    return sorted(agent)


def freeze_runtime(sources, fast_source):
    frozen = {}
    for name, source in sources.items():
        if name == "experiments.search_agent.runtime":
            old = "from .core import AgentLimits, run"
            if source.count(old) != 1:
                raise ValueError("unexpected_runtime_entrypoint")
            frozen["experiments.search_agent.fast"] = fast_source
            source = source.replace(old, "from .core import AgentLimits\nfrom .fast import run")
        frozen[name] = source
    return frozen


def _sources():
    rag = [BASE / "rag"]
    rag += [PRIVATE / name / "shard-0/rag" for name in ("policy-search-agent-v7-dev60-serial", "policy-search-agent-v7-dev60b-serial")]
    rag += [STUDY / "rag"]
    return {"rag": rag, "ab": [STUDY / "ab", DEV144]}


def historical():
    return {arm: unique_rows(READ(p / "records.jsonl") for p in paths) for arm, paths in _sources().items()}


def prepare(out=OUT):
    if (out / "manifest.json").exists() or (out / "generation/config.json").exists():
        raise FileExistsError("already_frozen")
    dev = {r["question_id"]: r for r in READ(BASE / "dev.jsonl")}
    rows = historical()
    ids = paired_ids(dev, rows["rag"], rows["ab"])
    if len(ids) != 199:
        raise ValueError("unexpected_development_pair_count")
    source = DEV144 / "config.json"
    original = json.loads(source.read_text())
    config = copy.deepcopy(original)
    comparisons = []
    protected = [source, BASE / "dev.jsonl", BASE / "test.jsonl", Path(__file__),
                 ROOT / "experiments/search_agent/fast.py", ROOT / "experiments/evaluation/search_agent_eval.py"]
    for arm, folders in _sources().items():
        for folder in folders:
            other = json.loads((folder / "config.json").read_text())
            for key in ("model", "model_thinking", "scope_document_ids", "index_snapshot", "chat_config", "provider_config"):
                if other[key] != original[key]:
                    raise ValueError("historical_configuration_mismatch:" + key)
            shared = set(other["runtime_code_hashes"]) & set(original["runtime_code_hashes"])
            if any(other["runtime_code_hashes"][k] != original["runtime_code_hashes"][k] for k in shared):
                raise ValueError("historical_runtime_mismatch")
            comparisons.append({"arm": arm, "source": str(folder.relative_to(ROOT)),
                                "same_model_index_retrieval": True,
                                "historical_missing_code_hashes": sorted(set(original["runtime_code_hashes"]) - shared)})
            protected += [folder / "config.json", folder / "records.jsonl"]
    for folder in (PRIVATE / "answer-quality-v2", PRIVATE / "answer-quality-dev144-flash"):
        protected += [folder / name for name in ("config.json", "manifest.json", "jobs.json", "records.jsonl")]
    runtime = freeze_runtime(original["module_sources"], (ROOT / "experiments/search_agent/fast.py").read_text())
    config.update(module_sources=runtime, question_ids=ids,
                  questions=[{"question_id": qid, "question": dev[qid]["question"]} for qid in ids],
                  stage_note="Fast v1: same initial retrieval and native answer template; 3 chunks / 2000 source tokens; no controller",
                  fast_rag={"version": "v1", "max_chunks": 3, "context_tokens": 2000, "selection": "ranked_complete_paragraph_prefix"},
                  container_ledger=f"/tmp/fast-rag-{uuid.uuid4().hex}/calls.jsonl")
    config["limits"]["run_tokens"] = len(ids) * 60000
    # The existing supervisor verifies frozen module text, not current working files.
    config["input_hashes"] = {str(p.relative_to(ROOT)): SHA(p) for p in set(protected)}
    for name, text in runtime.items():
        config["input_hashes"][name.replace(".", "/") + ".py"] = hashlib.sha256(text.encode()).hexdigest()
    gen = out / "generation"
    gen.mkdir(parents=True, exist_ok=True)
    DUMP(gen / "config.json", config)
    DUMP(gen / "manifest.json", {"config_sha256": SHA(gen / "config.json"), "planned": ids, "pilot": [],
         "direct_dev30": True, "replay_only": False, "stage_budgets": {"pilot": 0, "full": len(ids) * 60000},
         "model": config["model"], "strategy": "fast-rag-v1"})
    manifest = {"version": "fast-rag-v1", "question_ids": ids, "comparisons": comparisons,
                "input_sha256": {str(p.relative_to(ROOT)): SHA(p) for p in set(protected)},
                "generation_config_sha256": SHA(gen / "config.json"),
                "judge_model": "deepseek-v4.1-flash", "rubric": RUBRIC_VERSION,
                "test_used": False, "interpretation": "Development regression; historical baseline timings are not randomized contemporaneous controls.",
                "grading": "Regrade dev55 RAG/ab on Flash; reuse dev144 Flash raw grades. Missing/invalid grades excluded, never zero.",
                "router": "Hindsight oracle is a feasibility bound, not a deployed routing result. No routing thresholds trained on test95."}
    DUMP(out / "manifest.json", manifest)
    DUMP(out / "progress.json", {"phase": "prepared", "planned": len(ids)})
    supervisor.verify_frozen_inputs(config)
    print(json.dumps({"prepared": str(out), "questions": len(ids), "model": config["model"]}), flush=True)


def verify(out):
    manifest = json.loads((out / "manifest.json").read_text())
    for name, expected in manifest["input_sha256"].items():
        if SHA(ROOT / name) != expected:
            raise ValueError("frozen_input_changed:" + name)
    if SHA(out / "generation/config.json") != manifest["generation_config_sha256"]:
        raise ValueError("generation_config_changed")
    return manifest


def prepare_quality(out=OUT):
    manifest = verify(out)
    target = out / "quality"
    if (target / "manifest.json").exists():
        return
    records = unique_rows([READ(out / "generation/records.jsonl")])
    if set(records) != set(manifest["question_ids"]):
        raise ValueError("generation_incomplete")
    config = json.loads((out / "generation/config.json").read_text())
    safe = config["review"]["entries"]
    old_jobs = json.loads((PRIVATE / "answer-quality-v2/jobs.json").read_text())
    later_jobs = json.loads((PRIVATE / "answer-quality-dev144-flash/jobs.json").read_text())
    templates = {}
    for job in old_jobs + later_jobs:
        qid = job["question_id"]
        template = {k: v for k, v in job["packet"].items() if k not in ("answer", "citations")}
        if qid in templates and templates[qid] != template:
            raise ValueError("inconsistent_historical_rubric:" + qid)
        templates[qid] = template
    dev = {q["question_id"]: q for q in READ(BASE / "dev.jsonl")}
    jobs, exclusions = [], []
    for qid in manifest["question_ids"]:
        row = records[qid]
        if row["status"] != "ok":
            exclusions.append({"question_id": qid, "arm": "fast", "status": row["status"]})
            continue
        for evidence in dev[qid]["evidence"]:
            texts = [s["sanitized_content"] for s in safe.values() if s["source_document_id"] == evidence["source_document_id"]]
            if not covered_by_sources(evidence["text"], texts):
                raise ValueError("reference_source_not_verified")
        for cid, chunk in row["seen_chunks"].items():
            if chunk["content_with_weight"] != safe[cid]["sanitized_content"] or chunk["docnm_kwd"] != safe[cid]["safe_title"]:
                raise ValueError("unreviewed_outbound_source")
        packet = build_packet(dev[qid], row)
        packet.update(templates[qid])
        jobs.append({"id": hashlib.sha256((qid + ":fast").encode()).hexdigest()[:20],
                     "question_id": qid, "arm": "fast", "stage": "extension", "packet": packet})
    for job in old_jobs:
        if job["arm"] in ("rag", "ab"):
            jobs.append({**copy.deepcopy(job), "stage": "extension"})
    random.Random(20260927).shuffle(jobs)
    target.mkdir(parents=True, exist_ok=True)
    qc = json.loads((PRIVATE / "answer-quality-dev144-flash/config.json").read_text())
    if qc["model"] != manifest["judge_model"] or qc["system"] != SYSTEM:
        raise ValueError("judge_rubric_changed")
    if json.loads((PRIVATE / "answer-quality-v2/config.json").read_text())["system"] != SYSTEM:
        raise ValueError("historical_rubric_changed")
    qc["ledger_root"] = "/tmp/fast-rag-quality-" + uuid.uuid4().hex
    DUMP(target / "config.json", qc)
    DUMP(target / "jobs.json", jobs)
    protected = [out / "manifest.json", out / "generation/records.jsonl", Path(__file__),
                 ROOT / "experiments/evaluation/answer_quality.py", ROOT / "experiments/evaluation/answer_quality_eval.py"]
    DUMP(target / "manifest.json", {"rubric": RUBRIC_VERSION, "weights": WEIGHTS, "question_ids": manifest["question_ids"],
         "valid_answers": len(jobs), "technical_exclusions": exclusions,
         "input_sha256": {**manifest["input_sha256"], **{str(p.relative_to(ROOT)): SHA(p) for p in protected}},
         "config_sha256": SHA(target / "config.json"), "jobs_sha256": SHA(target / "jobs.json"),
         "blindness": "Anonymous single answer, no arm/id/difficulty/old score sent to judge."})
    print(json.dumps({"grading_jobs": len(jobs), "generation_exclusions": len(exclusions)}), flush=True)


def execute(out=OUT):
    manifest = verify(out)
    gen = out / "generation"
    # The first three are an integration check inside this same frozen 199-question run.
    # They remain in the final comparison, and are never regenerated.
    if not READ(gen / "records.jsonl"):
        DUMP(out / "progress.json", {"phase": "generation_smoke", "planned": len(manifest["question_ids"])})
        supervisor.run_batch(gen, "full", maximum=3)
        first = READ(gen / "records.jsonl")
        if len(first) != 3 or any(r["status"] != "ok" for r in first):
            DUMP(out / "progress.json", {"phase": "inspection_required", "completed": len(first)})
            return
    DUMP(out / "progress.json", {"phase": "generation", "planned": len(manifest["question_ids"])})
    supervisor.run_batch(gen, "full")
    rows = READ(gen / "records.jsonl")
    if len(rows) != len(manifest["question_ids"]):
        DUMP(out / "progress.json", {"phase": "inspection_required", "completed": len(rows)})
        return
    prepare_quality(out)
    DUMP(out / "progress.json", {"phase": "grading", "generated": len(rows)})
    quality.run("all", out / "quality")
    attempted = len(READ(out / "quality/records.jsonl"))
    planned = len(json.loads((out / "quality/jobs.json").read_text()))
    DUMP(out / "progress.json", {"phase": "ready_for_report" if attempted == planned else "grading_inspection_required",
                                "generated": len(rows), "graded_attempts": attempted, "grading_planned": planned})
    if attempted == planned:
        from experiments.evaluation.fast_rag_report import write_report
        write_report(out)
        DUMP(out / "progress.json", {"phase": "report_ready", "generated": len(rows), "graded_attempts": attempted})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "run", "prepare-quality", "grade"))
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args.output)
    elif args.action == "prepare-quality":
        prepare_quality(args.output)
    elif args.action == "grade":
        quality.run("all", args.output / "quality")
    else:
        execute(args.output)
