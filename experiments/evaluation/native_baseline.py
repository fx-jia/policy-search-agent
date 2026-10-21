"""Freeze and execute the 1000-document native baseline in the installed container."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "experiments/results/private/native-corpus1000-v2"
PREFLIGHT = REPO / "experiments/results/private/native-corpus1000-v2-preflight"
WORKER = Path(__file__).with_name("native_baseline_worker.py")
GATE = Path(__file__).with_name("native_outbound.py")
MODEL = "qwen3.7-flash-2026-07-15"
DOCKER = ["docker", "exec", "-i", "docker-ragflow-cpu-1", "/ragflow/.venv/bin/python", "-c"]

SNAPSHOT = r'''
import hashlib,json,sys
from pathlib import Path
import pymysql,yaml
chat_id=sys.stdin.readline().strip()
c=yaml.safe_load(Path('/ragflow/conf/service_conf.yaml').read_text())['mysql']
db=pymysql.connect(host=c['host'],port=int(c['port']),user=c['user'],password=c['password'],database=c['name'],autocommit=True)
with db.cursor() as cur:
 cur.execute('SELECT tenant_llm_id,kb_ids,top_n,top_k,similarity_threshold,vector_similarity_weight,rerank_id,llm_setting,prompt_config FROM dialog WHERE id=%s',(chat_id,))
 mid,kbs,topn,topk,threshold,weight,rerank,llm,prompt=cur.fetchone()
 cur.execute('SELECT model_name,extra FROM tenant_model WHERE id=%s',(mid,))
 name,extra=cur.fetchone()
 dbconfig={'model':name,'model_id':mid,'dataset_ids':sorted(json.loads(kbs)),'top_n':topn,'top_k':topk,'similarity_threshold':float(threshold),'vector_similarity_weight':float(weight),'rerank_id':rerank,'llm_setting':json.loads(llm),'prompt_config':json.loads(prompt),'model_context_budget':json.loads(extra)['max_tokens']}
db.close()
files=['api/apps/restful_apis/chat_api.py','api/db/services/dialog_service.py','rag/prompts/generator.py','api/db/services/llm_service.py','rag/llm/chat_model.py']
hashes={n:hashlib.sha256((Path('/ragflow')/n).read_bytes()).hexdigest() for n in files}
print(json.dumps({'chat_config':dbconfig,'runtime_code_hashes':hashes},ensure_ascii=False))
'''


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    with Path(path).open("x") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def approved_chat_id():
    chat_id = os.environ.get("POLICY_APPROVED_CHAT_ID", "")
    if not re.fullmatch(r"[0-9a-f]{32}", chat_id):
        raise ValueError("POLICY_APPROVED_CHAT_ID must configure an approved 32-character identifier")
    return chat_id


def prepare():
    chat_id = approved_chat_id()
    OUT.mkdir(parents=True, exist_ok=True)
    target = OUT / "prepared.json"
    if target.exists():
        raise FileExistsError(target)
    corpus = REPO / "experiments/datasets/private/ragflow_corpus1000_v1/document_map.json"
    review_path = PREFLIGHT / "reviewed_outbound_chunks.json"
    split_root = REPO / "experiments/datasets/private/benchmark_split_v1"
    mapping = json.loads(corpus.read_text())
    questions = []
    for split, expected in (("dev", 51), ("test", 99)):
        rows = [json.loads(line) for line in (split_root / f"{split}.jsonl").read_text().splitlines()]
        assert len(rows) == expected
        questions.extend({"question_id": r["question_id"], "question": r["question"], "split": split} for r in rows)
    assert len(mapping["documents"]) == 1000 and mapping["indexed_chunks"] == 3205
    assert len({q["question_id"] for q in questions}) == 150
    snapshot = subprocess.run(DOCKER + [SNAPSHOT], input=chat_id + "\n", text=True, capture_output=True, check=True)
    frozen = json.loads(snapshot.stdout)
    assert frozen["chat_config"]["model"] == MODEL
    assert frozen["chat_config"]["top_n"] == 8
    assert frozen["chat_config"]["dataset_ids"] == sorted(mapping["dataset_ids"])
    frozen.update({"chat_id": chat_id, "model": MODEL, "scope_document_ids": sorted(mapping["documents"]), "document_map": mapping, "review": json.loads(review_path.read_text()), "gate_source": GATE.read_text(), "question_ids": [q["question_id"] for q in questions], "questions": questions, "stop_at_tokens": 900000})
    paths = [corpus, review_path, split_root / "dev.jsonl", split_root / "test.jsonl", WORKER, GATE, Path(__file__)]
    frozen["input_sha256"] = {str(p.relative_to(REPO)): sha(p) for p in paths}
    frozen["scope_note"] = "All fixed 1000 documents, native hybrid top8 chunks, original system prompt; uniformly reviewed outbound sanitization. 51 development and 99 previously inspected test questions. No new holdout claim."
    dump(target, frozen)
    print(json.dumps({"prepared": 150, "scope_documents": 1000, "model": MODEL}))


def check_inputs(config):
    for name, expected in config["input_sha256"].items():
        if sha(REPO / name) != expected:
            raise ValueError(f"Frozen input changed: {name}")


def freeze(preflight_path):
    config = json.loads((OUT / "prepared.json").read_text())
    check_inputs(config)
    rows = [json.loads(line) for line in Path(preflight_path).read_text().splitlines()]
    requests = {}
    for row in rows:
        if row["status"] != "ok" or not row.get("outbound_request") or not row.get("outbound_audit"):
            raise ValueError("Preflight is incomplete")
        qid = row["question_id"]
        if qid in requests:
            raise ValueError("Duplicate preflight question")
        if row.get("generation_attempted"):
            raise ValueError("Preflight unexpectedly generated")
        requests[qid] = row["outbound_request"]
    if set(requests) != set(config["question_ids"]):
        raise ValueError("Preflight question set mismatch")
    config["frozen_requests"] = requests
    config["preflight_sha256"] = sha(preflight_path)
    dump(OUT / "generation_frozen.json", config)
    print(json.dumps({"frozen_requests": len(requests), "max_model_calls": len(requests)}))


def run(mode, name, limit=None):
    chat_id = approved_chat_id()
    config_path = OUT / ("generation_frozen.json" if mode == "generate" else "prepared.json")
    config = json.loads(config_path.read_text())
    if config.get("chat_id") != chat_id:
        raise PermissionError("frozen chat differs from the approved dialog")
    check_inputs(config)
    config["mode"] = mode
    questions = config["questions"][:limit]
    if mode == "generate":
        # A permanent attempt ledger prevents accidental cloud replays across runs.
        previous = set()
        for file in OUT.glob("*.attempted.jsonl"):
            previous.update(json.loads(line)["question_id"] for line in file.read_text().splitlines())
        questions = [q for q in questions if q["question_id"] not in previous]
        known_usage = {}
        for file in OUT.glob("*.jsonl"):
            if file.name.endswith(".attempted.jsonl"):
                continue
            for line in file.read_text().splitlines():
                row = json.loads(line)
                if row.get("generation_attempted"):
                    tokens = (row.get("provider_usage") or {}).get("total_tokens")
                    if not tokens:
                        raise ValueError("Cannot resume with unknown prior generation usage")
                    if row["question_id"] in known_usage:
                        raise ValueError("Duplicate prior generation")
                    known_usage[row["question_id"]] = int(tokens)
        if previous != set(known_usage):
            raise ValueError("Uncertain previous attempt; inspect before resuming")
        config["initial_provider_tokens"] = sum(known_usage.values())
    dump(OUT / f"{name}.lock.json", {"mode": mode, "config_sha256": sha(config_path), "question_ids": [q["question_id"] for q in questions]})
    with (OUT / f"{name}.stderr.log").open("x") as err, (OUT / f"{name}.noise.log").open("x") as noise, (OUT / f"{name}.jsonl").open("x") as output, (OUT / f"{name}.attempted.jsonl").open("x") as ledger:
        with subprocess.Popen(DOCKER + [WORKER.read_text()], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=err, text=True, bufsize=1) as worker:
            def receive():
                for line in worker.stdout:
                    if line.startswith("NATIVE_RESULT:"):
                        return json.loads(line[len("NATIVE_RESULT:"):])
                    noise.write(line)
                    noise.flush()
                raise RuntimeError("Worker exited; inspect private stderr log")

            worker.stdin.write(json.dumps(config, ensure_ascii=False) + "\n")
            worker.stdin.flush()
            ready = receive()
            if ready.get("status") != "ready":
                raise ValueError("Native worker failed to initialize")
            print(json.dumps(ready), flush=True)
            for item in questions:
                if mode == "generate":
                    ledger.write(json.dumps({"question_id": item["question_id"]}) + "\n")
                    ledger.flush()
                    os.fsync(ledger.fileno())
                worker.stdin.write(json.dumps(item, ensure_ascii=False) + "\n")
                worker.stdin.flush()
                result = receive()
                if result.get("question_id") != item["question_id"]:
                    raise ValueError("Question response mismatch")
                output.write(json.dumps(result, ensure_ascii=False) + "\n")
                output.flush()
                os.fsync(output.fileno())
                print(json.dumps({k: result.get(k) for k in ("question_id", "status", "latency_ms", "generation_calls_so_far", "provider_total_tokens_so_far")}), flush=True)
                if mode == "generate" and result["status"] != "ok":
                    break
            worker.stdin.close()
            worker.wait(timeout=30)
            if worker.returncode:
                raise RuntimeError("Worker process failed; inspect private stderr log")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "preflight", "freeze", "generate"))
    parser.add_argument("--name", default="preflight")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.action == "prepare":
        prepare()
    elif args.action == "freeze":
        freeze(OUT / f"{args.name}.jsonl")
    else:
        run(args.action, args.name, args.limit)
