"""Freeze and supervise isolated Search Agent runs; never modifies native baselines."""
import argparse
import copy
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import queue
import subprocess
import threading
import time
import uuid

from experiments.evaluation.benchmark_v2_baselines import (
    STOP_WORKER, receive_line, runtime_questions, validate_ready_ledger, validate_result,
)
from experiments.evaluation.native_baseline import DOCKER
from experiments.evaluation.native_agent_guard import CallLedger
from experiments.search_agent.core import AgentLimits

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "experiments/results/private/benchmark-v2-baselines-v2-nonthinking"
DEFAULT_OUT = ROOT / "experiments/results/private/policy-search-agent-v1"
PILOT = ["mb2-052", "mb2-020", "mb2-018", "mb2-120", "mb2-157", "mb2-285"]
MODULES = ["experiments.evaluation.native_outbound", "experiments.evaluation.native_agent_guard",
           "experiments.evaluation.native_agent_transport", "experiments.evaluation.native_agent_worker",
           "experiments.search_agent.evidence", "experiments.search_agent.cache", "experiments.search_agent.adapter",
           "experiments.search_agent.core", "experiments.search_agent.transport", "experiments.search_agent.embedding", "experiments.search_agent.runtime"]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def readl(path):
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def append(path, value):
    with path.open("a") as stream:
        stream.write(json.dumps(value, ensure_ascii=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def bootstrap_source():
    return '''import asyncio,json,sys,types
config=json.loads(sys.stdin.readline())
for name in ("experiments","experiments.evaluation","experiments.search_agent"):
    m=types.ModuleType(name);m.__path__=[];sys.modules[name]=m
for name,source in config["module_sources"].items():
    m=types.ModuleType(name);m.__file__="<frozen-"+name+">";m.__package__=name.rpartition(".")[0]
    sys.modules[name]=m;exec(compile(source,m.__file__,"exec"),m.__dict__)
asyncio.run(sys.modules["experiments.search_agent.runtime"].execute(config))
'''


def check_stage(stage, pilot, records):
    if stage == "full":
        by_id = {r["question_id"]: r for r in records}
        selected = [by_id[qid] for qid in pilot if qid in by_id]
        failures = [r for r in selected if r["status"] != "ok"]
        if len(selected) != len(pilot) or len(failures) > 1 or any(r.get("stop_reason") != "provider_usage_unknown" for r in failures):
            raise ValueError("pilot_not_technically_valid; inspect before expansion")
        committed_tokens(selected)


def committed_tokens(records):
    total = 0
    for row in records:
        for call in row.get("calls", []):
            value = call["usage"]["total_tokens"] if "usage" in call else call.get("reserve_tokens")
            if type(value) is not int or value < 0 or ("usage" not in call and value == 0):
                raise ValueError("invalid_call_accounting")
            total += value
    return total


def diagnostic_commitment(out):
    rows = []
    for name in ("provider_diagnostic.json", "native_provider_diagnostic.json"):
        path = out / name
        if path.exists():
            row = json.loads(path.read_text())
            rows.append(row if "calls" in row else {"calls": [row]})
    return committed_tokens(rows)


def remaining_ids(planned, attempted, records):
    if len(attempted) != len(set(attempted)) or len(records) != len({r["question_id"] for r in records}):
        raise ValueError("duplicate_attempt")
    if set(attempted) != {r["question_id"] for r in records}:
        raise ValueError("incomplete_prior_attempt")
    committed_tokens(records)
    return [qid for qid in planned if qid not in attempted]


def verify_frozen_inputs(config, root=ROOT):
    frozen = {name.replace(".", "/") + ".py": source for name, source in config["module_sources"].items()}
    supervisor = "experiments/evaluation/search_agent_eval.py"
    for name, sha in config["input_hashes"].items():
        if name == supervisor:
            continue  # Host scheduling can change; the inference code below remains frozen.
        actual = hashlib.sha256(frozen[name].encode()).hexdigest() if name in frozen else digest(root / name)
        if actual != sha:
            raise ValueError("frozen_input_changed:" + name)


def bounded_worker_stop(worker, stop_remote):
    if worker.stdin and not worker.stdin.closed:
        worker.stdin.close()
    try:
        worker.wait(timeout=2)
    except subprocess.TimeoutExpired:
        try:
            stop_remote()
        finally:
            worker.kill()
            worker.wait(timeout=5)


def prior_generation_tokens(paths):
    total = 0
    for path in set(paths):
        rows = readl(path / "records.jsonl")
        if not rows:
            raise ValueError("prior_run_requires_completed_accounting")
        total += committed_tokens(rows) + diagnostic_commitment(path)
    return total


def _stage_budgets(spent, *, direct_dev30):
    budgets = {"pilot": max(0, 150000 - spent), "full": 450000 - spent}
    if (budgets["full"] if direct_dev30 else min(budgets.values())) <= 0:
        raise ValueError("study_budget_exhausted")
    return budgets


def _track_control_event(state, event):
    if event.get("event") == "controller_error":
        state["controller_error"] = str(event["error"])[:200]
    elif event.get("event") == "fallback":
        state["fallback_used"] = True


def prepare(out, *, replay=False, ablation=None, question=None, prior_runs=(), connect_retries=0, direct_dev30=False):
    if (out / "config.json").exists():
        raise FileExistsError("run_already_frozen")
    frozen = json.loads((BASE / "frozen.json").read_text())
    for name, sha in frozen["source_hashes"].items():
        if digest(ROOT / name) != sha:
            raise ValueError("benchmark_input_changed")
    planned = PILOT + [q for q in frozen["dev30"] if q not in PILOT]
    questions = runtime_questions(readl(BASE / "dev.jsonl"), planned)
    if question:
        qid = "adhoc-" + hashlib.sha256(question.encode()).hexdigest()[:12]
        planned, questions = [qid], [{"question_id": qid, "question": question}]
    prior_runs = list(dict.fromkeys(p.resolve() for p in prior_runs))
    spent = prior_generation_tokens(prior_runs)
    budgets = _stage_budgets(spent, direct_dev30=direct_dev30)
    config = copy.deepcopy(json.loads((BASE / "rag/config.json").read_text()))
    for key in ("sources", "input_sha256", "questions", "question_ids"):
        config.pop(key, None)
    if type(connect_retries) is not int or not 0 <= connect_retries <= 2:
        raise ValueError("invalid_connect_retries")
    config.update(question_ids=planned, questions=questions, reasoning=0, embedding_model="qwen3.7-text-embedding", connect_retries=connect_retries,
                  limits={"run_tokens": budgets["full"], "question_tokens": 60000, "question_calls": 4, "query_embedding_calls": 3},
                  question_timeout_seconds=60, container_ledger=f"/tmp/{out.name}-{uuid.uuid4().hex}/calls.jsonl",
                  agent_limits=asdict(AgentLimits(enhance=not replay, allow_search=ablation != "no_search", select_context=ablation != "native_context")),
                  model_thinking=False, native_patch=None,
                  stage_note="independent agent; frozen native baselines untouched; no gold in runtime")
    # Captured read-only from the installed adapter before this experiment.
    config["runtime_code_hashes"]["rag/llm/embedding_model.py"] = "dbf8eed5cabeca94f6070b729644f31583f401068c6b582a6a2f2c662727cd03"
    snapshot_path = ROOT / "experiments/datasets/private/context_completion_v1/chunks.json"
    config["snapshot_rows"] = [{k: row[k] for k in ("id", "document_id", "order", "content_sha256")} for row in json.loads(snapshot_path.read_text())["chunks"]]
    config["module_sources"] = {name: (ROOT / (name.replace(".", "/") + ".py")).read_text() for name in MODULES}
    if replay:
        records = {r["question_id"]: r for r in readl(BASE / "rag/records.jsonl")}
        config["replay"] = {qid: {"initial": records[qid]["ordinary_context"], "expected_request": records[qid]["final_request"]} for qid in planned}
    protected = [BASE / name for name in ("frozen.json", "dev.jsonl", "test.jsonl", "boundary.jsonl", "assessments.jsonl", "evaluation_overrides.json", "human_reviews.json")]
    protected += [BASE / arm / name for arm in ("rag", "low") for name in ("config.json", "records.jsonl", "events.jsonl", "attempts.jsonl")]
    protected += [path / "records.jsonl" for path in prior_runs]
    protected += [path / name for path in prior_runs for name in ("provider_diagnostic.json", "native_provider_diagnostic.json") if (path / name).exists()]
    source_paths = [ROOT / (name.replace(".", "/") + ".py") for name in MODULES] + [Path(__file__), snapshot_path]
    config["input_hashes"] = {str(p.relative_to(ROOT)): digest(p) for p in source_paths + protected}
    out.mkdir(parents=True, exist_ok=True)
    dump(out / "config.json", config)
    dump(out / "manifest.json", {"config_sha256": digest(out / "config.json"), "pilot": PILOT if not question else planned,
                                "planned": planned, "replay_only": replay, "ablation": ablation,
                                "stage_budgets": budgets, "prior_generation_commitment": spent, "direct_dev30": direct_dev30,
                                "prior_runs": [str(p.relative_to(ROOT)) for p in prior_runs], "model": config["model"]})
    print(json.dumps({"prepared": str(out), "questions": len(planned), "replay": replay}, ensure_ascii=False))


def run_batch(out, stage="pilot", maximum=None):
    manifest = json.loads((out / "manifest.json").read_text())
    if digest(out / "config.json") != manifest["config_sha256"]:
        raise ValueError("frozen_config_changed")
    config = json.loads((out / "config.json").read_text())
    verify_frozen_inputs(config)
    records = readl(out / "records.jsonl")
    attempted = [x["question_id"] for x in readl(out / "attempts.jsonl")]
    if stage != "replay" and not (stage == "full" and manifest.get("direct_dev30")):
        check_stage(stage, manifest["pilot"], records)
    elif stage == "replay" and not config.get("replay"):
        raise ValueError("replay_stage_requires_replay_configuration")
    planned = manifest["pilot"] if stage == "pilot" else config["question_ids"]
    ids = remaining_ids(planned, attempted, records)
    if maximum:
        ids = ids[:maximum]
    if not ids:
        print("No unattempted questions in this stage.")
        return
    config["limits"]["run_tokens"] = manifest["stage_budgets"]["pilot" if stage == "pilot" else "full"]
    stage_limit = config["limits"]["run_tokens"]
    extra = diagnostic_commitment(out)
    append(out / "stage_events.jsonl", {"stage": stage, "question_ids": ids, "run_limit": stage_limit,
                                      "supervisor_sha256": digest(Path(__file__)), "prior_commitment": committed_tokens(records),
                                      "diagnostic_commitment": extra, "runtime_config_sha256": manifest["config_sha256"]})
    qmap = {q["question_id"]: q for q in config["questions"]}
    index = 0
    with (out / "stderr.log").open("a") as err, (out / "noise.log").open("a") as noise:
        while index < len(ids):
            available = stage_limit - committed_tokens(records) - extra
            if available <= 0:
                print("Stage budget exhausted; remaining questions unattempted.", flush=True)
                return
            token = uuid.uuid4().hex
            config["worker_token"] = token
            config["limits"]["run_tokens"] = available
            config["container_ledger"] = f"/tmp/{out.name}-{token}/calls.jsonl"
            append(out / "stage_events.jsonl", {"event": "worker_start", "remaining_tokens": available,
                                              "ledger": config["container_ledger"], "next_question": ids[index]})
            command = DOCKER[:3] + ["-e", "BENCHMARK_RUN_TOKEN=" + token] + DOCKER[3:] + [bootstrap_source()]
            with subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=err, text=True, bufsize=1) as worker:
                lines = queue.Queue()
                def read_stdout():
                    try:
                        for line in worker.stdout:
                            lines.put(line)
                    finally:
                        lines.put(None)
                threading.Thread(target=read_stdout, daemon=True).start()
                worker.stdin.write(json.dumps(config, ensure_ascii=False) + "\n")
                worker.stdin.flush()
                deadline, sent, started = time.monotonic() + 120, False, None
                control_state = {}

                def dispatch():
                    nonlocal sent, started, deadline
                    control_state.clear()
                    qid = ids[index]
                    append(out / "attempts.jsonl", {"question_id": qid})
                    sent, started = True, time.monotonic()
                    deadline = started + 60
                    worker.stdin.write(json.dumps(qmap[qid], ensure_ascii=False) + "\n")
                    worker.stdin.flush()

                try:
                    while True:
                        line = receive_line(lines, deadline)
                        if line is None:
                            raise RuntimeError("worker_exited; inspect private stderr")
                        if not line.startswith("AGENT_RESULT:"):
                            noise.write(line)
                            noise.flush()
                            continue
                        row = json.loads(line[len("AGENT_RESULT:"):])
                        append(out / "events.jsonl", row)
                        if sent and row.get("question_id") == ids[index]:
                            _track_control_event(control_state, row)
                        if row.get("event") == "ready":
                            if row["worker_token"] != token:
                                raise ValueError("worker_identity_mismatch")
                            validate_ready_ledger(row["ledger"], [])
                            dispatch()
                        elif row.get("event") == "question_complete":
                            done = row["record"]
                            validate_result(ids[index], done)
                            append(out / "records.jsonl", done)
                            records.append(done)
                            print(json.dumps({k: done[k] for k in ("question_id", "status", "stop_reason", "latency_ms", "query_embedding_calls", "ledger")}, ensure_ascii=False), flush=True)
                            sent = False
                            index += 1
                            if index == len(ids) or done.get("stop_reason") in {"run_token_budget", "usage_exceeded_reservation"}:
                                return
                            if done["status"] != "ok":
                                break  # End this worker; retain its reservation and move to the next question.
                            dispatch()
                except TimeoutError:
                    killed = subprocess.run(DOCKER + [STOP_WORKER], input=json.dumps({"token": token, "ledger": config["container_ledger"]}), text=True, capture_output=True, timeout=15, check=True)
                    snapshot = json.loads(killed.stdout)
                    append(out / "host_deadlines.jsonl", {"question_id": ids[index] if sent else None, **snapshot})
                    if sent:
                        audit = out / (ids[index] + ".deadline-ledger.jsonl")
                        audit.write_text("".join(json.dumps(r) + "\n" for r in snapshot["ledger_events"]))
                        ledger = CallLedger(audit, run_limit=config["limits"]["run_tokens"], question_limit=60000, call_limit=4)
                        append(out / "records.jsonl", {"question_id": ids[index], "status": "host_timeout", "answer": "", "ledger": ledger.summary(),
                                                       "latency_ms": round((time.monotonic() - started) * 1000),
                                                       "calls": [c for c in ledger.calls.values() if c["question_id"] == ids[index]],
                                                       "error": "host deadline; unknown usage retained; no replay",
                                                       "control_trace_complete": False, **control_state})
                    return
                finally:
                    def stop_remote():
                        stopped = subprocess.run(DOCKER + [STOP_WORKER], input=json.dumps({"token": token, "ledger": config["container_ledger"]}), text=True, capture_output=True, timeout=15, check=True)
                        append(out / "shutdown_terminations.jsonl", json.loads(stopped.stdout))
                    bounded_worker_stop(worker, stop_remote)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "run"])
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--replay", action="store_true")
    parser.add_argument("--ablation", choices=["no_search", "native_context"])
    parser.add_argument("--question", help="single question; not combined with benchmark inputs")
    parser.add_argument("--stage", choices=["pilot", "full", "replay"], default="pilot")
    parser.add_argument("--max-questions", type=int)
    parser.add_argument("--prior-run", type=Path, action="append", default=[], help="deduct prior usage, unknown reservations and diagnostics from the study budget")
    parser.add_argument("--connect-retries", type=int, choices=(0, 1, 2), default=0)
    parser.add_argument("--direct-dev30", action="store_true", help="explicitly authorized full development comparison after prior technical diagnosis")
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args.output.resolve(), replay=args.replay, ablation=args.ablation, question=args.question, prior_runs=args.prior_run,
                connect_retries=args.connect_retries, direct_dev30=args.direct_dev30)
    else:
        run_batch(args.output.resolve(), args.stage, args.max_questions)


if __name__ == "__main__":
    main()
