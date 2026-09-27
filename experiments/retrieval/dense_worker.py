"""Private stdin/stdout worker, executed inside the existing RAGFlow container.

Uses that runtime's QWenEmbed and existing ES index; never changes the index.
Only the KNN clause scores results. No text query or relevance labels enter it.
"""

import contextlib
import json
import logging
import math
from pathlib import Path
import sys
import time
import warnings


def corpus_filter(dataset_ids, document_ids=None):
    filters = [{"terms": {"kb_id": dataset_ids}}]
    if document_ids is not None:
        if not document_ids:
            raise ValueError("Empty fixed corpus scope")
        filters.append({"terms": {"doc_id": document_ids}})
    return {"bool": {"filter": filters, "must_not": [{"range": {"available_int": {"lt": 1}}}]}}


def knn_body(vector, dataset_ids, retrieval, document_ids=None):
    if not vector or not all(math.isfinite(value) for value in vector):
        raise ValueError("Invalid query vector")
    return {
        "size": retrieval["candidate_count"],
        "_source": ["doc_id", "kb_id"],
        "knn": {
            "field": f"q_{len(vector)}_vec",
            "query_vector": vector,
            "k": retrieval["knn_top_k"],
            "num_candidates": retrieval["knn_num_candidates"],
            "similarity": retrieval["similarity_threshold"],
            "filter": corpus_filter(dataset_ids, document_ids),
        },
    }


def initialize(config, expected_documents, expected_chunks):
    import pymysql
    import yaml
    from elasticsearch import Elasticsearch
    from rag.llm.embedding_model import QWenEmbed

    if config["method"] != "dense" or not config["dataset_ids"] or config.get("rerank_model"):
        raise ValueError("Unsupported Dense configuration")
    settings = yaml.safe_load(Path("/ragflow/conf/service_conf.yaml").read_text())
    mysql = settings["mysql"]
    dataset_ids = config["dataset_ids"]
    placeholders = ",".join(["%s"] * len(dataset_ids))
    db = pymysql.connect(host=mysql["host"], port=int(mysql["port"]), user=mysql["user"], password=mysql["password"], database=mysql["name"], cursorclass=pymysql.cursors.DictCursor)
    with db.cursor() as cursor:
        cursor.execute(f"SELECT tenant_id,embd_id FROM knowledgebase WHERE id IN ({placeholders})", dataset_ids)
        kbs = cursor.fetchall()
        if len(kbs) != len(dataset_ids) or len({(kb["tenant_id"], kb["embd_id"]) for kb in kbs}) != 1:
            raise ValueError("Dense datasets must share one tenant and embedding model")
        kb = kbs[0]
        cursor.execute("SELECT m.model_name,i.api_key,i.extra FROM tenant_model m JOIN tenant_model_instance i ON m.instance_id=i.id WHERE m.id=%s", (kb["embd_id"],))
        model = cursor.fetchone()
        cursor.execute(f"SELECT id,run,chunk_num FROM document WHERE kb_id IN ({placeholders})", dataset_ids)
        docs = cursor.fetchall()
    db.close()
    scope = config.get("scope_document_ids")
    if scope is not None:
        if set(scope) != set(expected_documents):
            raise ValueError("Fixed scope differs from document map")
        docs = [doc for doc in docs if doc["id"] in set(scope)]
    if {row["id"] for row in docs} != set(expected_documents) or any(row["run"] != "3" or row["chunk_num"] < 1 for row in docs) or sum(row["chunk_num"] for row in docs) != expected_chunks:
        raise ValueError("Knowledge base changed or is not fully indexed")
    if model["model_name"] != config["embedding_model"] or model["model_name"] != "qwen3.7-text-embedding":
        raise ValueError("Dense worker requires the verified QWen embedding configuration")
    extra = json.loads(model["extra"]) if isinstance(model["extra"], str) else model["extra"]
    embedding = QWenEmbed(model["api_key"], model["model_name"], base_url=(extra or {}).get("base_url"))
    es_config = settings["es"]
    es = Elasticsearch(es_config["hosts"].split(","), basic_auth=(es_config["username"], es_config["password"]), verify_certs=True, request_timeout=30, max_retries=0)
    index = f"ragflow_{kb['tenant_id']}"
    properties = es.indices.get_mapping(index=index)[index]["mappings"]["properties"]
    fields = {name: value for name, value in properties.items() if name.startswith("q_") and name.endswith("_vec") and value.get("type") == "dense_vector"}
    count = es.count(index=index, query=corpus_filter(dataset_ids, scope))["count"]
    if count != expected_chunks:
        raise ValueError("ES chunk count differs from frozen corpus")
    return embedding, es, index, dataset_ids, fields


def retrieve(question, config, state):
    embedding, es, index, dataset_ids, fields = state
    started = time.perf_counter()
    vector, tokens = embedding.encode_queries(question)
    vector = [float(value) for value in vector]
    embedding_ms = (time.perf_counter() - started) * 1000
    field = fields.get(f"q_{len(vector)}_vec", {})
    if field.get("similarity", "cosine") != "cosine" or field.get("dims") != len(vector):
        raise ValueError("Index vector dimensions or similarity mismatch")
    started = time.perf_counter()
    response = es.search(index=index, body=knn_body(vector, dataset_ids, config["retrieval_config"], config.get("scope_document_ids")))
    search_ms = (time.perf_counter() - started) * 1000
    if response.get("timed_out") or response.get("_shards", {}).get("failed", 0):
        raise RuntimeError("Incomplete ES response")
    chunks = []
    for hit in response["hits"]["hits"]:
        source = hit["_source"]
        if not set(dataset_ids).intersection([source["kb_id"]] if isinstance(source["kb_id"], str) else source["kb_id"]):
            raise ValueError("Out-of-corpus result")
        # ES cosine KNN score is (1 + cosine) / 2, no other scoring clause.
        cosine = 2 * hit["_score"] - 1
        chunks.append({"id": hit["_id"], "document_id": source["doc_id"], "similarity": cosine, "vector_similarity": cosine, "term_similarity": None})
    return {"code": 0, "data": {"chunks": chunks}, "stage_latency_ms": {"query_embedding": embedding_ms, "es_knn": search_ms}, "embedding_tokens": tokens}


def main():
    logging.disable(logging.CRITICAL)
    warnings.filterwarnings("ignore")
    request = json.loads(sys.stdin.readline())
    try:
        # Framework imports can print diagnostics. The protocol never emits them.
        with contextlib.redirect_stdout(sys.stderr):
            state = initialize(request["config"], request["documents"], request["indexed_chunks"])
    except Exception as exc:
        print(json.dumps({"status": "init_error", "error_type": type(exc).__name__}), flush=True)
        return 1
    print(json.dumps({"status": "ready"}), flush=True)
    for line in sys.stdin:
        item = {}
        try:
            item = json.loads(line)
            with contextlib.redirect_stdout(sys.stderr):
                response = retrieve(item["question"], request["config"], state)
        except Exception as exc:
            response = {"code": 1, "error_type": type(exc).__name__}
            if any(word in str(exc).lower() for word in ("quota", "arrearage", "insufficient balance", "freetier", "free tier")):
                response["error_category"] = "quota"
        response["request_id"] = item.get("request_id")
        print(json.dumps(response), flush=True)
    state[1].close()
    if hasattr(state[0], "close"):
        state[0].close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
