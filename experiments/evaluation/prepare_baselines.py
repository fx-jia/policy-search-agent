"""Create baseline configs from the frozen scope of already indexed documents."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from experiments.analysis.profile_dataset import fingerprint


RUNTIME_PATHS = ("api/apps/restful_apis/chunk_api.py", "rag/nlp/search.py", "rag/llm/embedding_model.py", "rag/llm/rerank_model.py", "rag/utils/es_conn.py", "rag/svr/task_executor_limiter.py")


def prepare(root):
    root = Path(root)
    plan = json.loads((root / "baseline_plan_v1.json").read_text())
    frozen = json.loads((root / "frozen_corpus.json").read_text())
    mapping = json.loads((root / "document_map.json").read_text())
    if fingerprint(root / "frozen_corpus.json") != plan["corpus_scope_sha256"]:
        raise ValueError("Corpus plan changed")
    if frozen["knowledge_base_version"] != mapping["knowledge_base_version"]:
        raise ValueError("Frozen map mismatch")
    if len(frozen["documents"]) != frozen["selected_documents"] or any(d["chunk_count"] < 1 for d in frozen["documents"]):
        raise ValueError("Corpus is not fully indexed")
    chunking = frozen["chunking_config"]
    parser = chunking["parser_config"]
    if (
        chunking["chunk_method"] != "naive"
        or parser["chunk_token_num"] != 512
        or parser.get("auto_keywords")
        or parser.get("auto_questions")
        or parser.get("enable_children")
        or parser.get("parent_child", {}).get("use_parent_child")
    ):
        raise ValueError("Unexpected chunking configuration")
    script = "from pathlib import Path; import json,hashlib; names=" + repr(RUNTIME_PATHS) + "; print(json.dumps({p:hashlib.sha256((Path('/ragflow')/p).read_bytes()).hexdigest() for p in names}))"
    result = subprocess.run(["docker", "exec", "docker-ragflow-cpu-1", "/ragflow/.venv/bin/python", "-c", script], capture_output=True, text=True, check=True)
    hashes = json.loads(result.stdout)
    config = {
        "method": "hybrid",
        "base_url": "http://127.0.0.1:9380",
        "dataset_ids": frozen["dataset_ids"],
        "knowledge_base_version": frozen["knowledge_base_version"],
        "source_sha256": frozen["source_sha256"],
        "baseline_plan_sha256": fingerprint(root / "baseline_plan_v1.json"),
        "scope_document_ids": sorted(mapping["documents"]),
        "ragflow_version": "v0.27.2; image sha256:53153e3f7cfb05ab3d5f58204a060a9a45d9af2d963b1ed8ab993dda94d9e278; QWen transport retry and session reuse v2; independent embedding concurrency 3",
        "runtime_source_hashes": hashes,
        "embedding_model": plan["embedding_model"],
        "rerank_model": None,
        "llm_model": None,
        "chunking_config": chunking,
        "retrieval_config": {},
        "generation_config": {"enabled": False},
    }
    output = root / "baseline_configs_v1"
    output.mkdir(exist_ok=False)
    for method in plan["method_order"]:
        current = dict(
            config,
            method=method,
            rerank_model=plan["rerank_id"] if method == "hybrid_rerank" else None,
            retrieval_config=dict(plan["shared_retrieval"], vector_similarity_weight=plan["vector_similarity_weight"][method]),
        )
        (output / f"{method}.json").write_text(json.dumps(current, ensure_ascii=False, indent=2) + "\n")
    (output / "manifest.json").write_text(
        json.dumps(
            {
                "baseline_plan_sha256": fingerprint(root / "baseline_plan_v1.json"),
                "config_sha256": {method: fingerprint(output / f"{method}.json") for method in plan["method_order"]},
                "prepare_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            },
            indent=2,
        )
        + "\n"
    )
    return str(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", required=True)
    args = parser.parse_args()
    print(json.dumps({"output": prepare(args.corpus)}))


if __name__ == "__main__":
    main()
