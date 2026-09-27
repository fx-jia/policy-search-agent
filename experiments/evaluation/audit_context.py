"""Read actual top-k chunks locally and check literal reference-span presence.

This is an inspection aid, not a semantic support or answer-accuracy metric.
It issues only chunk GET requests; it never reruns retrieval or calls a model.
"""

import argparse
import hashlib
import html
import json
from pathlib import Path
import re
from urllib.parse import quote, urlencode

from experiments.datasets.import_ragflow import API, write_json


def normalize(text):
    text = re.sub(r"</?[A-Za-z][^>]*>", "", html.unescape(text))
    return re.sub(r"\s+", "", text)


def find_evidence(evidence, chunks):
    return [[c["id"] for c in chunks if c["source_document_id"] == e["source_document_id"] and normalize(e["text"]) and normalize(e["text"]) in normalize(c["content"])] for e in evidence]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True)
    args = parser.parse_args()
    root = Path(args.experiment)
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest["base_url"] != "http://127.0.0.1:9380":
        parser.error("This inspector only supports the local RAGFlow instance")
    map_path = root / manifest["document_map_snapshot"]
    if hashlib.sha256(map_path.read_bytes()).hexdigest() != manifest["document_map_sha256"]:
        parser.error("Document map snapshot changed")
    document_datasets = json.loads(map_path.read_text())["document_datasets"]
    benchmark_bytes = (root / "benchmark.jsonl").read_bytes()
    if hashlib.sha256(benchmark_bytes).hexdigest() != manifest["dataset_version"]:
        parser.error("Benchmark snapshot changed")
    benchmark = {s["question_id"]: s for s in map(json.loads, benchmark_bytes.splitlines())}
    rows = [json.loads(line) for line in (root / "per_query.jsonl").read_text().splitlines()]
    if [r["question_id"] for r in rows] != manifest["question_ids"]:
        parser.error("Question sequence changed")
    api, cache, audits = API(), {}, []
    top_k = manifest["retrieval_config"]["context_top_k"]
    with (root / "context_chunks.jsonl").open("x", encoding="utf-8") as stream:
        for row in rows:
            chunks = []
            for candidate in row["retrieved_chunks"][:top_k]:
                cid, did = candidate["id"], candidate["document_id"]
                dataset = document_datasets[did]
                if dataset not in manifest["dataset_ids"]:
                    raise ValueError("Chunk dataset differs from frozen corpus")
                key = (did, cid)
                if key not in cache:
                    result = api.request("GET", f"/datasets/{quote(dataset, safe='')}/documents/{quote(did, safe='')}/chunks?" + urlencode({"id": cid}))
                    found = result["chunks"]
                    if len(found) != 1 or found[0]["id"] != cid or found[0]["document_id"] != did:
                        raise ValueError("Chunk identity mismatch")
                    cache[key] = found[0]["content"]
                chunks.append(dict(candidate, content=cache[key]))
            matches = find_evidence(benchmark[row["question_id"]]["evidence"], chunks)
            audits.append({"question_id": row["question_id"], "reference_span_chunk_ids": matches, "all_reference_spans_matched": bool(matches) and all(matches)})
            stream.write(json.dumps({"question_id": row["question_id"], "chunks": chunks}, ensure_ascii=False) + "\n")
    write_json(
        root / "context_audit.json",
        {
            "context_top_k": top_k,
            "unique_chunk_reads": len(cache),
            "method": "same-source literal span match after whitespace and HTML-tag normalization; unmatched spans require inspection",
            "rows": audits,
        },
        exclusive=True,
    )
    print(json.dumps({"queries": len(audits), "all_reference_spans_matched": sum(a["all_reference_spans_matched"] for a in audits), "unique_chunk_reads": len(cache)}))


if __name__ == "__main__":
    main()
