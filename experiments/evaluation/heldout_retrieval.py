"""Once-frozen held-out retrieval and context evaluation; never generates answers."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import statistics
import subprocess
import time

from experiments.evaluation.context_completion import evidence_coverage, select_context
from experiments.evaluation.context_experiment import TEST_PATH, TEST_SHA, _paired_delta, digest, jsonlines, project_candidates, read_verified, write_json, write_jsonl
from experiments.evaluation.generation_pilot import safe_chunk
from experiments.evaluation.metrics import retrieval_metrics
from experiments.qa import _index_state, validate_question
from experiments.retrieval.dense import DenseClient
from experiments.retrieval.ragflow import RetrievalError


ROOT = Path("experiments/datasets/private/heldout_retrieval_v1")
CONFIG = Path("experiments/configs/heldout_retrieval_v1.json")
PROTOCOL = Path("docs/heldout_retrieval_protocol_v1.md")
SOURCES = ["experiments/evaluation/heldout_retrieval.py", "experiments/evaluation/context_completion.py", "experiments/evaluation/context_experiment.py", "experiments/evaluation/metrics.py", "experiments/evaluation/index_snapshot.py", "experiments/evaluation/generation_pilot.py", "experiments/qa.py", "experiments/retrieval/dense.py", "experiments/retrieval/dense_worker.py", "experiments/retrieval/ragflow.py", "experiments/retrieval/hybrid_worker.py"]


def query_view(samples, manifest):
    expected = {qid for group in manifest["groups"].values() if group["split"] == "test" for qid in group["question_ids"]}
    if len(samples) != len(expected) or {r["question_id"] for r in samples} != expected:
        raise ValueError("Held-out question membership mismatch")
    for row in samples:
        group = manifest["assignments"][row["question_id"]]
        if row.get("split") != "test" or row.get("split_version") != manifest["split_version"] or row.get("policy_group_id") != group or row.get("review_status") != "human_approved":
            raise ValueError("Held-out split or review status mismatch")
        validate_question(row["question"])
    return [{"question_id": row["question_id"], "question": row["question"]} for row in samples]


def should_stop(error, consecutive):
    return error in {"quota_error", "http_400", "http_401", "http_403", "http_404", "call_limit", "worker_protocol", "dense_worker_timeout", "dense_worker_transport_error", "dense_worker_response_mismatch", "dense_worker_unavailable", "hybrid_worker_unavailable"} or consecutive >= 3


class _HybridClient(DenseClient):
    def __init__(self, config, mapping):
        self.config, self.failed = config, False
        script = Path("experiments/retrieval/hybrid_worker.py").read_text()
        self.process = subprocess.Popen(["docker", "exec", "-i", "docker-ragflow-cpu-1", "/ragflow/.venv/bin/python", "-u", "-c", script], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1)
        try:
            ready = self._request({"config": config, "mapping": mapping, "adapter_source": Path("experiments/retrieval/ragflow.py").read_text()}, timeout=60)
            if ready.get("status") != "ready":
                raise RuntimeError("Native Hybrid client initialization failed")
        except Exception:
            self.close()
            raise

    def retrieve(self, question, limit):
        if self.failed or limit != 64:
            raise RetrievalError("hybrid_worker_unavailable")
        import hashlib

        identity = hashlib.sha256(question.encode()).hexdigest()
        try:
            response = self._request({"question": question, "request_id": identity}, timeout=40)
            if response.get("request_id") != identity:
                raise RetrievalError("worker_protocol")
        except RetrievalError:
            self.failed = True
            self.close()
            raise
        if response["status"] != "ok":
            raise RetrievalError(response["error"])
        return response["chunks"]


def freeze():
    config = json.loads(CONFIG.read_text())
    for path, expected in config["input_sha256"].items():
        if digest(path) != expected:
            raise ValueError("Frozen input hash mismatch: " + path)
    # Do not parse held-out questions before method/code/protocol are frozen.
    ROOT.mkdir(parents=True, exist_ok=False)
    files = {**config["input_sha256"], **{p: digest(p) for p in [str(CONFIG), str(PROTOCOL), *SOURCES]}}
    for path in [str(CONFIG), str(PROTOCOL), *SOURCES]:
        target = ROOT / "code" / path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    write_json(ROOT / "freeze.json", {"created_at": datetime.now(timezone.utc).isoformat(), "files": files, "test_questions_read": False, "max_queries": 198, "generation_calls": 0, "method_selection": "Dense + boundary + short IDs fixed before held-out access"})
    print(json.dumps({"frozen": str(ROOT), "test_questions_read": False, "model_calls": 0}))


def verify():
    freeze_data = json.loads((ROOT / "freeze.json").read_text())
    for path, expected in freeze_data["files"].items():
        if digest(path) != expected:
            raise ValueError("Frozen method/input changed: " + path)
    return json.loads(CONFIG.read_text())


def run(output, allow_model_api, allow_heldout):
    if not allow_model_api or not allow_heldout:
        raise ValueError("Requires --allow-model-api and --allow-heldout")
    config = verify()
    if (ROOT / "execution_lock.json").exists():
        raise ValueError("Held-out run already started; do not silently rerun or tune")
    snapshot = json.loads(Path(config["snapshot"]).read_text())
    chunks = {c["id"]: c for c in snapshot["chunks"]}
    mapping = json.loads(Path(config["document_map"]).read_text())
    method_configs = {m: json.loads(Path(p).read_text()) for m, p in config["methods"].items()}
    for current in method_configs.values():
        if sorted(current["scope_document_ids"]) != sorted(mapping["documents"]) or current["retrieval_config"]["candidate_count"] != 64 or current["rerank_model"]:
            raise ValueError("Method changed frozen corpus/candidate scope")
    before = _index_state(method_configs["dense"])
    if before != snapshot["metadata"]["before"]:
        raise ValueError("Index changed since development")
    expected_runtime = method_configs["dense"]["runtime_source_hashes"]
    script = "import hashlib,json; from pathlib import Path; paths=" + repr(list(expected_runtime)) + "; print(json.dumps({p:hashlib.sha256((Path('/ragflow')/p).read_bytes()).hexdigest() for p in paths}))"
    runtime_read = subprocess.run(["docker", "exec", "docker-ragflow-cpu-1", "/ragflow/.venv/bin/python", "-c", script], capture_output=True, text=True, timeout=30)
    if runtime_read.returncode or json.loads(runtime_read.stdout) != expected_runtime:
        raise ValueError("Running RAGFlow code differs from frozen development baseline")
    output.mkdir(parents=True, exist_ok=False)
    clients = {}
    try:
        clients["dense"] = DenseClient(method_configs["dense"], mapping)
        clients["hybrid"] = _HybridClient(method_configs["hybrid"], mapping)
        manifest = json.loads(Path(config["split_manifest"]).read_text())
        queries = query_view(jsonlines(read_verified(TEST_PATH, TEST_SHA)), manifest)
        if len(queries) != 99:
            raise ValueError("Expected exactly 99 frozen test questions")
        write_json(ROOT / "execution_lock.json", {"started_at": datetime.now(timezone.utc).isoformat(), "output": str(output), "freeze_sha256": digest(ROOT / "freeze.json")})
        write_jsonl(output / "queries.jsonl", queries)
        write_json(output / "run_manifest.json", {"freeze_sha256": digest(ROOT / "freeze.json"), "index_before": before, "verified_runtime_hashes": expected_runtime, "planned_questions": 99, "planned_requests": 198, "generation_calls": 0, "document_embedding_calls": 0, "rerank_calls": 0, "order": "alternate dense/hybrid order by query", "query_projection": "question_id and question only; no relevance/answer fields"})
        calls, consecutive, stopped, records = 0, 0, None, []
        with (output / "rankings.jsonl").open("x") as stream:
            for i, query in enumerate(queries):
                for method in (["dense", "hybrid"] if i % 2 == 0 else ["hybrid", "dense"]):
                    client = clients[method]
                    status, error, retrieved = "not_run_after_error", stopped, []
                    start = time.perf_counter()
                    if not stopped:
                        calls += 1
                        try:
                            retrieved = client.retrieve(query["question"], 64)
                            project_candidates(retrieved, chunks)
                            if len(retrieved) > 64:
                                raise ValueError("Oversized candidate window")
                            status, error = "ok", None
                        except RetrievalError as exc:
                            status, error, retrieved = "error", str(exc), []
                        except Exception:
                            status, error, retrieved = "error", "worker_protocol", []
                        consecutive = consecutive + 1 if status != "ok" else 0
                        if should_stop(error, consecutive):
                            stopped = error or "three_consecutive_errors"
                    row = {**query, "method": method, "status": status, "error": error, "retrieved_chunks": retrieved, "latency_ms": (time.perf_counter() - start) * 1000 if status != "not_run_after_error" else None, "embedding_tokens": getattr(client, "last_embedding_tokens", None) if status == "ok" else None, "stage_latency_ms": getattr(client, "last_stage_latency_ms", None) if status == "ok" else None}
                    records.append(row)
                    stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                    stream.flush()
                    print(json.dumps({"question_id": query["question_id"], "method": method, "status": status, "calls": calls}), flush=True)
    finally:
        for client in clients.values():
            client.close()
    after = _index_state(method_configs["dense"])
    if after != before:
        write_json(output / "integrity_failure.json", {"index_before": before, "index_after": after})
        raise ValueError("Index changed during test run; results invalid")
    selections = []
    for row in records:
        candidates = project_candidates(row["retrieved_chunks"], chunks)
        for policy in config["policies"]:
            selected = select_context(candidates, policy, config["budget_tokens"], config["seed_top_k"])
            selections.append({"question_id": row["question_id"], "method": row["method"], "policy": policy, "saved_status": row["status"], "selected_ids": [c.id for c in selected], "selected_ranks": [c.rank for c in selected], "tokens": sum(c.tokens for c in selected)})
    write_jsonl(output / "selections.jsonl", selections)
    write_json(output / "retrieval_freeze.json", {"rankings_sha256": digest(output / "rankings.jsonl"), "selections_sha256": digest(output / "selections.jsonl"), "queries_sha256": digest(output / "queries.jsonl"), "actual_retrieval_requests": calls, "stop_reason": stopped, "index_after": after, "generation_calls": 0, "labels_used_to_select": False})
    print(json.dumps({"completed": str(output), "requests": calls, "stop_reason": stopped, "generation_calls": 0}))


def evaluate(result, output):
    config = verify()
    frozen = json.loads((result / "retrieval_freeze.json").read_text())
    for name in ["rankings", "selections", "queries"]:
        if digest(result / (name + ".jsonl")) != frozen[name + "_sha256"]:
            raise ValueError("Retrieval output changed")
    if frozen["stop_reason"]:
        raise ValueError("Retrieval stopped early; do not report a completed 99-query score")
    benchmark = {r["question_id"]: r for r in jsonlines(read_verified(TEST_PATH, TEST_SHA))}
    rankings = jsonlines((result / "rankings.jsonl").read_text())
    selections = jsonlines((result / "selections.jsonl").read_text())
    snapshot = {c["id"]: c for c in json.loads(Path(config["snapshot"]).read_text())["chunks"]}
    candidates = {(r["question_id"], r["method"]): project_candidates(r["retrieved_chunks"], snapshot) for r in rankings}
    evaluated, ceilings, ranking_summary = [], {}, {}
    for row in selections:
        question = benchmark[row["question_id"]]
        chosen = [c for cid in row["selected_ids"] for c in candidates[(row["question_id"], row["method"])] if c.id == cid]
        covered = evidence_coverage(question["evidence"], chosen)
        evaluated.append({**row, "evidence_covered": covered, "original_complete": all(covered), "original_span_fraction": statistics.mean(covered), "policy_group_id": question["policy_group_id"]})
    for method in config["methods"]:
        rows = [r for r in rankings if r["method"] == method]
        scores = [retrieval_metrics(r["retrieved_chunks"], benchmark[r["question_id"]]["relevant_document_ids"]) for r in rows]
        ranking_summary[method] = {"successes": sum(r["status"] == "ok" for r in rows), "questions": len(rows), "metrics": {key: statistics.mean(s[key] for s in scores) for key in scores[0]}, "mean_successful_latency_ms": statistics.mean(r["latency_ms"] for r in rows if r["status"] == "ok"), "reported_embedding_tokens": sum(r["embedding_tokens"] for r in rows if type(r["embedding_tokens"]) is int) if any(type(r["embedding_tokens"]) is int for r in rows) else None}
        ceilings[method] = {qid: all(evidence_coverage(q["evidence"], candidates[(qid, method)])) for qid, q in benchmark.items()}
    groups = {qid: q["policy_group_id"] for qid, q in benchmark.items()}
    by_condition = {(method, policy): {r["question_id"]: r for r in evaluated if r["method"] == method and r["policy"] == policy} for method in config["methods"] for policy in config["policies"]}
    summary = {f"{method}_{policy}": {"complete": sum(r["original_complete"] for r in rows.values()), "total": len(rows), "mean_span_fraction": statistics.mean(r["original_span_fraction"] for r in rows.values()), "mean_tokens": statistics.mean(r["tokens"] for r in rows.values())} for (method, policy), rows in by_condition.items()}
    contrasts = {"dense_boundary_vs_dense_ranked": _paired_delta(by_condition[("dense", "ranked")], by_condition[("dense", "boundary")], groups, config), "dense_boundary_vs_hybrid_ranked": _paired_delta(by_condition[("hybrid", "ranked")], by_condition[("dense", "boundary")], groups, config), "hybrid_boundary_vs_hybrid_ranked": _paired_delta(by_condition[("hybrid", "ranked")], by_condition[("hybrid", "boundary")], groups, config)}
    review = json.loads(Path(config["privacy_review"]).read_text())
    blocked = set(review["blocked_source_ids"]) | {"79870"}
    readiness = []
    for row in selections:
        if (row["method"], row["policy"]) not in [("dense", "boundary"), ("hybrid", "ranked")]:
            continue
        unreviewed, denied = [], []
        for cid in row["selected_ids"]:
            try:
                safe_chunk(snapshot[cid], review["chunks"], blocked)
            except ValueError:
                (denied if snapshot[cid]["source_document_id"] in blocked else unreviewed).append(cid)
        readiness.append({"question_id": row["question_id"], "method": row["method"], "policy": row["policy"], "already_reviewed": not unreviewed and not denied, "needs_review_ids": unreviewed, "blocked_ids": denied})
    output.mkdir(parents=True, exist_ok=False)
    write_jsonl(output / "per_query.jsonl", evaluated)
    write_json(output / "summary.json", {"context": summary, "rankings": ranking_summary, "paired": contrasts, "candidate_reference_ceiling": {m: sum(v.values()) for m, v in ceilings.items()}, "generation": "Not Run", "test_questions": 99, "policy_groups": len(set(groups.values())), "stage": "held-out retrieval and context; no answer accuracy"})
    write_json(output / "ceilings.json", ceilings)
    write_jsonl(output / "generation_readiness.jsonl", readiness)
    write_json(output / "manifest.json", {"method_freeze_sha256": digest(ROOT / "freeze.json"), "retrieval_freeze_sha256": digest(result / "retrieval_freeze.json"), "test_sha256": TEST_SHA, "evaluation_source_sha256": digest(__file__), "model_calls": 0, "outputs": {p.name: digest(p) for p in sorted(output.iterdir())}})
    print(json.dumps({"context": summary, "paired": contrasts, "generation": "Not Run"}, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["freeze", "run", "evaluate"])
    parser.add_argument("--output")
    parser.add_argument("--result")
    parser.add_argument("--allow-model-api", action="store_true")
    parser.add_argument("--allow-heldout", action="store_true")
    args = parser.parse_args()
    if args.action == "freeze":
        freeze()
    elif args.action == "run":
        run(Path(args.output), args.allow_model_api, args.allow_heldout)
    else:
        evaluate(Path(args.result), Path(args.output))


if __name__ == "__main__":
    main()
