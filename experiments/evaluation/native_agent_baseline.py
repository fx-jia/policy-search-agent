"""Freeze and run the installed native Agent with bounded, audited model use."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

from experiments.evaluation.native_baseline import SNAPSHOT, CHAT_ID, MODEL, DOCKER

REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).parent
PREFLIGHT = REPO / "experiments/results/private/native-agent-medium-v1-preflight"
IDS = ["med-043", "med-029", "med-115", "med-022", "med-002", "med-017"]

EXTRA_SNAPSHOT = r'''
from elasticsearch import Elasticsearch
from elasticsearch.helpers import scan
mapping=json.loads(sys.stdin.read())
c=yaml.safe_load(Path('/ragflow/conf/service_conf.yaml').read_text()); m=c['mysql']
db=pymysql.connect(host=m['host'],port=int(m['port']),user=m['user'],password=m['password'],database=m['name'])
with db.cursor() as cur:
 cur.execute('SELECT tenant_id FROM knowledgebase WHERE id=%s',(mapping['dataset_ids'][0],)); tenant=cur.fetchone()[0]
 cur.execute('SELECT p.provider_name,i.extra FROM tenant_model m JOIN tenant_model_provider p ON p.id=m.provider_id JOIN tenant_model_instance i ON i.id=m.instance_id WHERE m.id=%s',(dbconfig['model_id'],))
 factory,instance_extra=cur.fetchone()
 frozen['provider_config']={'factory':factory,'api_base':(json.loads(instance_extra) if instance_extra else {}).get('base_url','')}
db.close()
s=c['es']; es=Elasticsearch(s['hosts'].split(','),basic_auth=(s['username'],s['password']),verify_certs=True,request_timeout=30,max_retries=0)
index='ragflow_'+tenant
kbfilter={'terms':{'kb_id':mapping['dataset_ids']}}
compiled=es.count(index=index,query={'bool':{'filter':[kbfilter,{'exists':{'field':'compile_kwd'}}]}})['count']
if compiled: raise ValueError('Compiled outputs require a separate review')
query={'query':{'bool':{'filter':[kbfilter,{'terms':{'doc_id':list(mapping['documents'])}}]}},'_source':['doc_id','kb_id','content_with_weight','docnm_kwd']}
chunks={}
for hit in scan(es,index=index,query=query):
 row=hit['_source']; cid=hit['_id']
 chunks[cid]={'doc_id':row['doc_id'],'title':row['docnm_kwd'],'sha256':hashlib.sha256(row['content_with_weight'].encode()).hexdigest()}
es.close()
frozen['index_snapshot']={'compiled_rows':compiled,'chunks':chunks}
for path in Path('/ragflow/rag/advanced_rag').rglob('*.py'):
 frozen['runtime_code_hashes'][str(path.relative_to('/ragflow'))]=hashlib.sha256(path.read_bytes()).hexdigest()
frozen['runtime_code_hashes']['rag/nlp/search.py']=hashlib.sha256(Path('/ragflow/rag/nlp/search.py').read_bytes()).hexdigest()
print(json.dumps(frozen,ensure_ascii=False))
'''


def build_config(snapshot, questions, review, document_map):
    by_id = {q["question_id"]: q for q in questions}
    if any(qid not in by_id for qid in IDS):
        raise ValueError("missing_planned_question")
    if snapshot["chat_config"]["model"] != MODEL:
        raise ValueError("unexpected_model")
    return {
        **snapshot, "chat_id": CHAT_ID, "model": MODEL, "reasoning": 2,
        "scope_document_ids": sorted(document_map["documents"]), "document_map": document_map,
        "review": review, "question_ids": IDS,
        "questions": [{"question_id": qid, "question": by_id[qid]["question"]} for qid in IDS],
        "limits": {"run_tokens": 180000, "question_tokens": 60000, "question_calls": 16, "query_embedding_calls": 16},
        "provider_hosts": ["dashscope.aliyuncs.com"],
        "container_ledger": "/tmp/native-agent-medium-v1/calls.jsonl",
        "configuration_changes": ["worker_only_is_tools_true", "provider_max_output_1200", "cold_json_cache", "uniform_outbound_sanitization", "physical_request_budget"],
        "historical_native_report": "experiments/results/private/native-corpus1000-v2/semantic_summary.json",
    }


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_title_review(review, live_chunks, document_map):
    entries = review.get("entries", {})
    if review.get("unresolved") or set(entries) != set(document_map["documents"]):
        raise ValueError("incomplete_document_title_review")
    for doc_id, entry in entries.items():
        if entry.get("source_document_id") != document_map["documents"][doc_id]:
            raise ValueError("document_title_source_changed")
        for field in ("raw_title", "safe_title"):
            if not isinstance(entry.get(field), str) or hashlib.sha256(entry[field].encode()).hexdigest() != entry.get(field + "_sha256"):
                raise ValueError("document_title_hash_changed")
    for chunk in live_chunks.values():
        if entries.get(chunk["doc_id"], {}).get("raw_title") != chunk["title"]:
            raise ValueError("document_title_index_changed")


def prepare(output):
    output = Path(output)
    target = output / "config.json"
    if target.exists():
        raise FileExistsError(target)
    mapping_path = REPO / "experiments/datasets/private/ragflow_corpus1000_v1/document_map.json"
    mapping = json.loads(mapping_path.read_text())
    questions_path = REPO / "experiments/datasets/private/benchmark_split_v1/dev.jsonl"
    questions = [json.loads(line) for line in questions_path.read_text().splitlines()]
    review_path = REPO / "experiments/results/private/native-corpus1000-v2-preflight/reviewed_outbound_chunks.json"
    extension_path = PREFLIGHT / "review_extensions.json"
    title_review_path = PREFLIGHT / "reviewed_document_titles.json"
    title_review = json.loads(title_review_path.read_text())
    review = json.loads(review_path.read_text())
    extensions = json.loads(extension_path.read_text())
    if extensions["unresolved"] or set(review["entries"]) & set(extensions["entries"]):
        raise ValueError("unresolved_or_overlapping_review")
    review["entries"].update(extensions["entries"])
    review["status"] = "exact_reviewed_subset_new_chunks_block"
    review["expected_chunks"] = len(review["entries"])
    base = SNAPSHOT.replace("print(json.dumps({'chat_config':dbconfig,'runtime_code_hashes':hashes},ensure_ascii=False))", "frozen={'chat_config':dbconfig,'runtime_code_hashes':hashes}")
    result = subprocess.run(DOCKER + [base + EXTRA_SNAPSHOT], input=CHAT_ID + "\n" + json.dumps(mapping), text=True, capture_output=True)
    if result.returncode:
        output.mkdir(parents=True, exist_ok=True)
        (output / "snapshot_error.log").write_text(result.stderr)
        raise RuntimeError("Read-only snapshot failed; inspect private snapshot_error.log")
    snapshot = json.loads(result.stdout)
    chunks = json.loads((REPO / "experiments/datasets/private/context_completion_v1/chunks.json").read_text())["chunks"]
    live = snapshot["index_snapshot"]["chunks"]
    if len(live) != 3205 or len(mapping["documents"]) != 1000 or set(live) != {c["id"] for c in chunks}:
        raise ValueError("index_scope_changed")
    for chunk in chunks:
        actual = live[chunk["id"]]
        if actual["doc_id"] != chunk["document_id"] or actual["sha256"] != chunk["content_sha256"]:
            raise ValueError("index_content_changed")
    for cid, entry in review["entries"].items():
        if live[cid]["title"] != entry["raw_title"] or live[cid]["sha256"] != entry["raw_sha256"]:
            raise ValueError("review_content_or_title_changed")
    validate_title_review(title_review, live, mapping)
    config = build_config(snapshot, questions, review, mapping)
    config["document_title_review"] = title_review
    config["container_ledger"] = f"/tmp/{output.name}/calls.jsonl"
    modules = ["native_outbound", "native_agent_guard", "native_agent_transport"]
    config["sources"] = {name: (HERE / f"{name}.py").read_text() for name in modules}
    paths = [mapping_path, questions_path, review_path, extension_path, title_review_path, *(HERE / f"{name}.py" for name in modules), HERE / "native_agent_worker.py", Path(__file__)]
    config["input_sha256"] = {str(p.relative_to(REPO)): digest(p) for p in paths}
    output.mkdir(parents=True, exist_ok=True)
    with target.open("x") as handle:
        json.dump(config, handle, ensure_ascii=False, indent=2)
    print(json.dumps({"status": "prepared", "questions": len(IDS), "documents": 1000, "chunks": 3205, "reviewed_chunks": len(review["entries"]), "compiled_rows": 0}))
    return config


def run(output, max_questions=None):
    output = Path(output)
    config = json.loads((output / "config.json").read_text())
    for name, expected in config["input_sha256"].items():
        if digest(REPO / name) != expected:
            raise ValueError(f"frozen_input_changed: {name}")
    attempts_path = output / "attempts.jsonl"
    prior = [json.loads(line) for line in attempts_path.read_text().splitlines()] if attempts_path.exists() else []
    records_path = output / "records.jsonl"
    records = [json.loads(line) for line in records_path.read_text().splitlines()] if records_path.exists() else []
    if {r["question_id"] for r in prior} != {r["question_id"] for r in records} or any(r["status"] != "ok" for r in records):
        raise ValueError("prior_attempt_incomplete_or_failed_requires_inspection")
    seen = {q["question_id"] for q in prior}
    questions = [q for q in config["questions"] if q["question_id"] not in seen][:max_questions]
    if not questions:
        print(json.dumps({"status": "no_unattempted_questions"}))
        return
    with (output / "stderr.log").open("a") as err, (output / "events.jsonl").open("a") as events, (output / "noise.log").open("a") as noise, attempts_path.open("a") as attempts, records_path.open("a") as result_file:
        with subprocess.Popen(DOCKER + [(HERE / "native_agent_worker.py").read_text()], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=err, text=True, bufsize=1) as worker:
            def save(handle, row):
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            def receive():
                for line in worker.stdout:
                    if line.startswith("AGENT_RESULT:"):
                        row = json.loads(line[len("AGENT_RESULT:"):])
                        save(events, row)
                        return row
                    noise.write(line)
                    noise.flush()
                raise RuntimeError("Native Agent worker exited; inspect private logs")
            worker.stdin.write(json.dumps(config, ensure_ascii=False) + "\n")
            worker.stdin.flush()
            ready = receive()
            if ready.get("event") != "ready":
                raise ValueError("worker_not_ready")
            print(json.dumps(ready), flush=True)
            for question in questions:
                save(attempts, {"question_id": question["question_id"]})
                worker.stdin.write(json.dumps(question, ensure_ascii=False) + "\n")
                worker.stdin.flush()
                while True:
                    row = receive()
                    if row.get("event") == "question_complete":
                        record = row["record"]
                        save(result_file, record)
                        print(json.dumps({"question_id": record["question_id"], "status": record["status"], "agent_graph_called": record.get("agent_graph_called", False), "ledger": record["ledger"]}), flush=True)
                        break
                    if row.get("event") == "provider_finished":
                        print(json.dumps({"event": "provider_finished", "call_id": row["call_id"], "usage": row["usage"]}), flush=True)
                if record["status"] != "ok":
                    break
            worker.stdin.close()
            worker.wait(timeout=30)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["prepare", "run"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-questions", type=int)
    args = parser.parse_args()
    prepare(args.output) if args.mode == "prepare" else run(args.output, args.max_questions)


if __name__ == "__main__":
    main()
