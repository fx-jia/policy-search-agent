"""Freeze anonymous grading packets and score saved development answers."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
from threading import Event, Lock

from experiments.evaluation.answer_quality import RUBRIC_VERSION, SYSTEM, WEIGHTS, build_packet, covered_by_sources, validate_result
from experiments.evaluation.native_baseline import DOCKER

ROOT = Path(__file__).resolve().parents[2]
STUDY = ROOT / "experiments/results/private/policy-search-agent-v8_1-study"
BASE = STUDY.parent / "benchmark-v2-baselines-v2-nonthinking"
OUT = STUDY.parent / "answer-quality-v2"
ARMS = ("rag", "low", "v7", "a", "ab")
HERE = Path(__file__).parent
SCOPE = {
    "mb2-024": "题干必答为医疗机构条件和全省互认。正文额外文号/标题等错误仍记录在质量评分中。",
    "mb2-110": "题干问是否自动纳入和能否地方扩围；企业自愿是背景条件，不强制其作为必答点。",
    "mb2-112": "题干问40%上限与医院药店渠道自付一致性；不强制额外说明分别统一职工和居民保障。",
    "mb2-165": "题干问不得自行扩围、方案批准和备案；新增1城市是背景，不强制额外回答。",
}


def readl(path):
    return [json.loads(s) for s in path.read_text().splitlines()] if path.exists() else []


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def stop_required(result):
    ledger = result.get("ledger", {})
    return bool(ledger.get("pending_calls") or ledger.get("fatal_reason") or result.get("usage_unknown")
                or result.get("status") in ("provider_error", "host_timeout", "worker_error"))


def validate_prior_attempts(attempts, records):
    attempted = [r["id"] for r in attempts]
    recorded = [r["id"] for r in records]
    if len(attempted) != len(set(attempted)) or len(recorded) != len(set(recorded)):
        raise ValueError("duplicate_prior_attempt")
    if set(attempted) != set(recorded):
        raise ValueError("incomplete_prior_attempt; reconcile remote ledger before resuming")
    if any(stop_required(r) for r in records):
        raise ValueError("unresolved_prior_usage; inspect durable records before resuming")
    return set(attempted)


def prepare():
    if (OUT / "manifest.json").exists():
        raise ValueError("already_frozen")
    OUT.mkdir(parents=True, exist_ok=True)
    selection = json.loads((STUDY / "selection.json").read_text())
    ids = selection["extension"]
    questions = {q["question_id"]: q for q in readl(BASE / "dev.jsonl")}
    rows = {a: {r["question_id"]: r for r in readl(STUDY / "extension" / a / "records.jsonl")} for a in ARMS}
    c = json.loads((STUDY / "extension/ab/config.json").read_text())
    safe = {cid: e for cid, e in c["review"]["entries"].items()}
    jobs = []
    exclusions = []
    for qid in ids:
        q = questions[qid]
        for evidence in q["evidence"]:
            candidates = [e["sanitized_content"] for e in safe.values() if e["source_document_id"] == evidence["source_document_id"]]
            if not covered_by_sources(evidence["text"], candidates):
                raise ValueError("reference_not_in_reviewed_text:" + qid)
        for arm in ARMS:
            row = rows[arm][qid]
            if row["status"] != "ok":
                exclusions.append({"question_id": qid, "arm": arm, "status": row["status"]})
                continue
            # Check all outbound observed texts against the approved sanitized corpus.
            for cid, chunk in row["seen_chunks"].items():
                entry = safe[cid]
                if chunk["content_with_weight"] != entry["sanitized_content"] or chunk["docnm_kwd"] != entry["safe_title"]:
                    raise ValueError("outbound_snapshot_mismatch:" + cid)
            packet = build_packet(q, row, SCOPE.get(qid, "只要求题干范围；题干明确指定年份/地区时按指定政策判断。"))
            identity = hashlib.sha256((qid + ":" + arm).encode()).hexdigest()[:20]
            jobs.append({"id": identity, "question_id": qid, "arm": arm, "packet": packet})
    rng = random.Random(20260926)
    calibration = ["mb2-" + n for n in ("024", "082", "086", "110", "112", "165")]
    pool = [q for q in ids if q not in calibration and all(rows[a][q]["status"] == "ok" for a in ARMS)]
    calibration += rng.sample(pool, 6)
    rng.shuffle(jobs)
    for job in jobs:
        job["stage"] = "calibration" if job["question_id"] in calibration else "extension"
    protected = [BASE / "dev.jsonl", BASE / "test.jsonl", STUDY / "human_adjudications.json", STUDY / "review_queue.md"]
    protected += [STUDY / "extension" / a / "records.jsonl" for a in ARMS]
    protected += [HERE / name for name in ("answer_quality.py", "answer_quality_eval.py", "answer_quality_worker.py", "native_agent_guard.py")]
    config = {"model": "deepseek-v4-pro-0813", "credential_model": "qwen3.7-flash-2026-07-15", "chat_id": c["chat_id"], "system": SYSTEM,
              "ledger_source": (HERE / "native_agent_guard.py").read_text(),
              "worker_source": (HERE / "answer_quality_worker.py").read_text(),
              "ledger_root": "/tmp/policy-answer-quality-v2", "concurrency": 4}
    dump(OUT / "config.json", config)
    dump(OUT / "jobs.json", jobs)
    dump(OUT / "manifest.json", {"rubric": RUBRIC_VERSION, "weights": WEIGHTS, "question_ids": ids,
         "calibration_ids": calibration, "valid_answers": len(jobs), "technical_exclusions": exclusions,
         "input_sha256": {str(p.relative_to(ROOT)): digest(p) for p in protected},
         "jobs_sha256": digest(OUT / "jobs.json"), "config_sha256": digest(OUT / "config.json"),
         "scope": "Exploratory quality grading of existing dev55; no new answers, retrieval or embedding; no heldout95.",
         "calibration": "Six human-reviewed cases plus six seeded random all-arm successful cases. Existing binary labels are not numeric quality gold labels.",
         "blindness": "One anonymous answer per request; no arm, difficulty, prior grades, preference or trace in provider packets."})
    print(json.dumps({"answers": len(jobs), "calibration": calibration, "technical_exclusions": len(exclusions)}, ensure_ascii=False))


def run(stage, out=OUT):
    manifest = json.loads((out / "manifest.json").read_text())
    for name, sha in manifest["input_sha256"].items():
        if digest(ROOT / name) != sha:
            raise ValueError("frozen_input_changed:" + name)
    for name in ("jobs", "config"):
        assert digest(out / (name + ".json")) == manifest[name + "_sha256"]
    config = json.loads((out / "config.json").read_text())
    jobs = json.loads((out / "jobs.json").read_text())
    attempted = validate_prior_attempts(readl(out / "attempts.jsonl"), readl(out / "records.jsonl"))
    jobs = [j for j in jobs if j["id"] not in attempted and (stage == "all" or j["stage"] == stage)]
    lock, stop = Lock(), Event()

    def append(name, row):
        with lock, (out / name).open("a") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())

    def execute(job):
        if stop.is_set():
            return
        append("attempts.jsonl", {"id": job["id"], "stage": job["stage"]})
        payload = {k: v for k, v in config.items() if k not in ("worker_source", "ledger_root", "concurrency")}
        payload.update(job={"id": job["id"], "packet": job["packet"]}, ledger_path=config["ledger_root"] + "/" + job["id"] + ".jsonl")
        try:
            proc = subprocess.run(DOCKER + [config["worker_source"]], input=json.dumps(payload, ensure_ascii=False), text=True, capture_output=True, timeout=75)
            lines = [s.removeprefix("QUALITY_RESULT:") for s in proc.stdout.splitlines() if s.startswith("QUALITY_RESULT:")]
            result = json.loads(lines[-1]) if lines else {"status": "worker_error", "exit_code": proc.returncode}
        except subprocess.TimeoutExpired:
            result = {"status": "host_timeout", "usage_unknown": True}
        result.update(id=job["id"], question_id=job["question_id"], arm=job["arm"], stage=job["stage"])
        if result["status"] == "ok":
            try:
                result["grade"] = validate_result(json.loads(result["output"]), job["packet"])
                if result.get("finish_reason") != "stop":
                    raise ValueError("output_truncated")
            except (ValueError, TypeError, KeyError) as exc:
                result["status"] = "invalid_grade"
                result["validation_error"] = str(exc)[:150]
        if stop_required(result):
            stop.set()
        append("records.jsonl", result)
        with lock:
            print(json.dumps({k: result.get(k) for k in ("id", "status", "latency_ms", "validation_error")}), flush=True)

    with ThreadPoolExecutor(max_workers=config["concurrency"]) as pool:
        for future in as_completed([pool.submit(execute, j) for j in jobs]):
            future.result()
    print(json.dumps({"attempts": len(readl(out / "attempts.jsonl")), "records": len(readl(out / "records.jsonl")), "stopped": stop.is_set()}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "run"))
    parser.add_argument("--stage", choices=("calibration", "extension", "all"), default="calibration")
    parser.add_argument("--output", type=Path, default=OUT, help="frozen grading batch to run")
    args = parser.parse_args()
    prepare() if args.action == "prepare" else run(args.stage, args.output.resolve())
