"""Upload, parse, inspect and freeze the prepared local RAGFlow pilot corpus."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time
import uuid
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import ProxyHandler, Request, build_opener

from experiments.analysis.profile_dataset import fingerprint
from experiments.retrieval.ragflow import NoRedirect


def write_json(path, value, exclusive=False):
    if exclusive:
        with path.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
    else:
        temp = path.with_suffix(".tmp")
        temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temp.replace(path)


class API:
    def __init__(self):
        self.base = "http://127.0.0.1:9380/api/v1"
        self.token = os.environ["RAGFLOW_API_KEY"].strip()
        if not self.token:
            raise ValueError("Empty RAGFLOW_API_KEY")
        self.opener = build_opener(ProxyHandler({}), NoRedirect())

    def request(self, method, path, payload=None, content_type="application/json"):
        if isinstance(payload, (dict, list)):
            payload = json.dumps(payload, ensure_ascii=False).encode()
        req = Request(self.base + path, data=payload, method=method, headers={"Authorization": "Bearer " + self.token, "Content-Type": content_type})
        try:
            with self.opener.open(req, timeout=60) as response:
                result = json.load(response)
        except HTTPError as exc:
            raise RuntimeError(f"HTTP {exc.code}") from None
        except (URLError, TimeoutError, OSError):
            raise RuntimeError("Transport error; inspect server state before retrying writes") from None
        if type(result.get("code")) is not int or result["code"] != 0:
            raise RuntimeError(f"RAGFlow application error {result.get('code')}; inspect local server diagnostics")
        return result.get("data")

    def documents(self, dataset):
        docs, page = [], 1
        while True:
            result = self.request("GET", f"/datasets/{dataset}/documents?" + urlencode({"page": page, "page_size": 100, "orderby": "name", "desc": "false"}))
            docs.extend(result["docs"])
            if len(docs) >= result["total"]:
                break
            if not result["docs"]:
                raise ValueError("Unexpected empty document page")
            page += 1
        if len({d["id"] for d in docs}) != len(docs):
            raise ValueError("Duplicate IDs during document pagination")
        return docs

    def upload(self, dataset, entries, root):
        boundary = "medical-rag-" + uuid.uuid4().hex
        parts = []
        for entry in entries:
            name = entry["upload_name"]
            if any(c in name for c in '\r\n"'):
                raise ValueError("Invalid upload name")
            parts.extend(
                [
                    f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="{name}"\r\nContent-Type: text/markdown\r\n\r\n'.encode(),
                    (root / "documents" / entry["filename"]).read_bytes(),
                    b"\r\n",
                ]
            )
        parts.append(f"--{boundary}--\r\n".encode())
        return self.request("POST", f"/datasets/{dataset}/documents", b"".join(parts), f"multipart/form-data; boundary={boundary}")


def reconcile(entries, docs):
    expected = {e["upload_name"]: e for e in entries}
    if len(expected) != len(entries):
        raise ValueError("Duplicate proposed upload names")
    names = [d["name"] for d in docs]
    if len(set(names)) != len(names) or set(names) - expected.keys():
        raise ValueError("Unexpected or duplicate documents in experiment dataset")
    return {d["id"]: expected[d["name"]]["source_document_id"] for d in docs}


def complete_parse(api, dataset_id, root, entries, max_pending=5, timeout_seconds=1800):
    attempts = Counter()
    journal = root / "parse_requests.jsonl"
    if journal.exists():
        for line in journal.read_text().splitlines():
            attempts.update(json.loads(line)["document_ids"])
    deadline = time.monotonic() + timeout_seconds
    previous = None
    while time.monotonic() < deadline:
        docs = api.documents(dataset_id)
        reconcile(entries, docs)
        if len(docs) != len(entries):
            raise ValueError("Complete upload before starting the parse queue")
        states = Counter(d.get("run") for d in docs)
        if states != previous:
            print(json.dumps({"states": dict(states), "chunks": sum(d.get("chunk_count", 0) for d in docs)}), flush=True)
            previous = states
        if states.get("DONE", 0) == len(docs):
            return docs
        active = sum(states.get(s, 0) for s in ("RUNNING", "SCHEDULE"))
        pending = [d["id"] for d in docs if d.get("run") in ("UNSTART", "FAIL") and attempts[d["id"]] < 3][: max(0, max_pending - active)]
        if pending:
            api.request("POST", f"/datasets/{dataset_id}/documents/parse", {"document_ids": pending})
            attempts.update(pending)
            with journal.open("a") as stream:
                stream.write(json.dumps({"time": datetime.now(timezone.utc).isoformat(), "document_ids": pending}) + "\n")
        elif not active:
            raise RuntimeError("Remaining documents require inspection after three parse attempts")
        time.sleep(10)
    raise RuntimeError("Parse queue time limit reached; resume after inspecting status")


def run(args):
    root = Path(args.proposal)
    manifest = json.loads((root / "manifest.json").read_text())
    for entry in manifest["documents"]:
        filename = Path(entry["filename"])
        if filename.name != str(filename) or fingerprint(root / "documents" / filename) != entry["file_sha256"]:
            raise ValueError("Proposed document path/hash mismatch")
    api = API()
    state_path = root / "import_state.json"
    manifest_hash = fingerprint(root / "manifest.json")
    if state_path.exists():
        state = json.loads(state_path.read_text())
        if state["proposal_sha256"] != manifest_hash:
            raise ValueError("Import proposal changed since dataset creation")
    else:
        if args.action != "upload":
            raise ValueError("Upload must initialize the experiment dataset first")
        if not args.embedding_model:
            raise ValueError("Specify the configured embedding model ID when creating a dataset")
        datasets = api.request("GET", "/datasets?page_size=100")
        if any(item["name"] == args.name for item in datasets):
            raise ValueError("Dataset name already exists without a local receipt; inspect before adopting")
        parser_config = dict(manifest["parser_config_proposal"])
        parser_config.pop("parser_id", None)
        parser_config.update({"enable_metadata": False, "enable_children": False, "filename_embd_weight": 0.1})
        dataset = api.request(
            "POST",
            "/datasets",
            {"name": args.name, "embedding_model": args.embedding_model, "permission": "me", "language": "Chinese", "chunk_method": "naive", "parser_config": parser_config},
        )
        state = {"dataset_id": dataset["id"], "dataset_name": args.name, "proposal_sha256": manifest_hash, "created_at": datetime.now(timezone.utc).isoformat(), "documents": {}}
        write_json(root / "created_dataset.json", dataset, exclusive=True)
        write_json(state_path, state)
    dataset_id = state["dataset_id"]
    docs = api.documents(dataset_id)
    mapping = reconcile(manifest["documents"], docs)
    for did, sid in state["documents"].items():
        if mapping.get(did) != sid:
            raise ValueError("Previously recorded upload disappeared or changed")
    if args.action == "upload":
        uploaded = set(mapping.values())
        remaining = [e for e in manifest["documents"] if e["source_document_id"] not in uploaded]
        for offset in range(0, len(remaining), 10):
            batch = remaining[offset : offset + 10]
            result = api.upload(dataset_id, batch, root)
            additions = reconcile(batch, result)
            if len(additions) != len(batch):
                raise ValueError("Upload response incomplete; reconcile before resuming")
            mapping.update(additions)
            state["documents"] = mapping
            write_json(state_path, state)
            print(json.dumps({"uploaded": len(mapping), "expected": len(manifest["documents"])}), flush=True)
        docs = api.documents(dataset_id)
        mapping = reconcile(manifest["documents"], docs)
    elif args.action == "parse":
        selected_status = "FAIL" if args.retry_failed else "UNSTART"
        pending = [d["id"] for d in docs if d.get("run") == selected_status][: args.batch_size]
        if pending:
            api.request("POST", f"/datasets/{dataset_id}/documents/parse", {"document_ids": pending})
            with (root / "parse_requests.jsonl").open("a") as stream:
                stream.write(json.dumps({"time": datetime.now(timezone.utc).isoformat(), "document_ids": pending}) + "\n")
        print(json.dumps({"parse_submitted": len(pending)}), flush=True)
        docs = api.documents(dataset_id)
    elif args.action == "complete":
        docs = complete_parse(api, dataset_id, root, manifest["documents"], args.max_pending, args.timeout_seconds)
    summary = {"dataset_id": dataset_id, "documents": len(docs), "states": dict(Counter(d.get("run") for d in docs)), "chunks": sum(d.get("chunk_count", 0) for d in docs)}
    state["documents"] = mapping
    state["last_status"] = summary
    write_json(state_path, state)
    # Avoid persisting progress messages that can include provider error details.
    snapshot = [{k: d.get(k) for k in ("id", "name", "run", "progress", "chunk_count", "token_count", "parser_config", "chunk_method")} for d in docs]
    write_json(root / "document_status.json", snapshot)
    if args.action == "freeze":
        if len(docs) != len(manifest["documents"]) or any(d.get("run") != "DONE" or d.get("chunk_count", 0) < 1 for d in docs):
            raise ValueError("All expected documents must finish parsing with nonempty chunks")
        dataset = api.request("GET", f"/datasets/{dataset_id}")
        frozen = {"source_sha256": manifest["source_sha256"], "proposal_sha256": manifest_hash, "dataset_ids": [dataset_id], "dataset": dataset, "documents": snapshot}
        version = hashlib.sha256(json.dumps(frozen, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        write_json(root / "frozen_knowledge_base.json", dict(frozen, knowledge_base_version=version), exclusive=True)
        write_json(
            root / "document_map.json",
            {
                "source_sha256": manifest["source_sha256"],
                "knowledge_base_version": version,
                "dataset_ids": [dataset_id],
                "indexed_chunks": summary["chunks"],
                "documents": mapping,
            },
            exclusive=True,
        )
    print(json.dumps(summary), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("upload", "parse", "complete", "status", "freeze"))
    parser.add_argument("--proposal", required=True)
    parser.add_argument("--name", default="医保-检索实验-pilot200-v1")
    parser.add_argument("--embedding-model")
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--max-pending", type=int, default=5)
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    args = parser.parse_args()
    if not 1 <= args.batch_size <= 50:
        parser.error("batch-size must be in [1, 50]")
    if not 1 <= args.max_pending <= 50 or not 1 <= args.timeout_seconds <= 14400:
        parser.error("Require 1 <= max-pending <= 50 and timeout-seconds <= 14400")
    try:
        run(args)
    except (ValueError, KeyError, RuntimeError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
