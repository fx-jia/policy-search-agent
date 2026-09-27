"""POST /api/v1/retrieval adapter checked against the running v0.27.2 source."""

import json
import math
import os
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener


class RetrievalError(RuntimeError):
    pass


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # Never forward the Authorization header to an unexpected destination.
        return None


def build_payload(question, config):
    method = config["method"]
    if method not in ("hybrid", "hybrid_rerank"):
        raise ValueError("This endpoint has no verified strict Dense/keyword-only switch")
    retrieval = config["retrieval_config"]
    count = retrieval["candidate_count"]
    knn = retrieval["knn_top_k"]
    ann = retrieval["knn_num_candidates"]
    rerank_id = config.get("rerank_model")
    if method == "hybrid_rerank" and not rerank_id:
        raise ValueError("hybrid_rerank requires an explicit rerank model ID")
    if method == "hybrid" and rerank_id:
        raise ValueError("hybrid must not enable a rerank model")
    if not all(type(value) is int for value in (count, knn, ann)) or not 10 <= count <= 100 or knn < count or ann < knn:
        raise ValueError("Require 10 <= candidate_count <= 100, knn_top_k >= count, ANN >= KNN")
    for field in ("vector_similarity_weight", "similarity_threshold"):
        value = retrieval[field]
        if type(value) not in (float, int) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"{field} must be a finite number in [0, 1]")
    if not isinstance(config.get("dataset_ids"), list) or not config["dataset_ids"] or not all(isinstance(sid, str) and sid for sid in config["dataset_ids"]):
        raise ValueError("dataset_ids is required")
    payload = {
        "question": question,
        "dataset_ids": config["dataset_ids"],
        "page": 1,
        "page_size": count,
        "knn_top_k": knn,
        "knn_num_candidates": ann,
        "rerank_candidates_count": count,
        "vector_similarity_weight": retrieval["vector_similarity_weight"],
        "similarity_threshold": retrieval["similarity_threshold"],
        "rerank_id": rerank_id or "",
        "keyword": False,
        "highlight": False,
        "cross_languages": [],
        "use_kg": False,
        "toc_enhance": False,
        "include_knowledge_compilation": False,
    }
    scope = config.get("scope_document_ids")
    if scope is not None:
        if not isinstance(scope, list) or not scope or not all(isinstance(did, str) and did for did in scope) or len(set(scope)) != len(scope):
            raise ValueError("Fixed corpus scope requires unique document IDs")
        payload["document_ids"] = scope
    return payload


def parse_response(response, document_map):
    if not isinstance(response, dict) or type(response.get("code")) is not int or response["code"] != 0:
        message = str(response.get("message", "")).lower() if isinstance(response, dict) else ""
        if (isinstance(response, dict) and response.get("error_category") == "quota") or any(word in message for word in ("quota", "arrearage", "insufficient balance", "freetier", "free tier")):
            raise RetrievalError("quota_error")
        raise RetrievalError("ragflow_application_error")
    data = response.get("data")
    if not isinstance(data, dict) or not isinstance(data.get("chunks"), list):
        raise RetrievalError("invalid_chunks_schema")
    output, seen = [], set()
    for rank, chunk in enumerate(data["chunks"], 1):
        if not isinstance(chunk, dict) or not all(isinstance(chunk.get(k), str) and chunk[k] for k in ("id", "document_id")):
            raise RetrievalError("missing_chunk_or_document_id")
        if chunk["id"] in seen:
            raise RetrievalError("duplicate_chunk_id")
        seen.add(chunk["id"])
        did = chunk["document_id"]
        if did not in document_map:
            raise RetrievalError("unmapped_document_id")
        item = {"rank": rank, "id": chunk["id"], "document_id": did, "source_document_id": document_map[did]}
        for field in ("similarity", "term_similarity", "vector_similarity"):
            value = chunk.get(field)
            if value is not None and (type(value) not in (int, float) or not math.isfinite(value)):
                raise RetrievalError("invalid_score")
            item[field] = value
        # Bodies may contain PII: persist IDs and scores, not arbitrary response text.
        output.append(item)
    return output


class RAGFlowClient:
    def __init__(self, config, document_map):
        self.config, self.document_map = config, document_map
        base = config["base_url"].rstrip("/")
        url = urlparse(base)
        if url.username or url.password or url.query or url.fragment or url.path:
            raise ValueError("base_url must be an origin without credentials, path or query")
        if url.scheme != "https" and not (url.scheme == "http" and url.hostname in ("localhost", "127.0.0.1", "::1")):
            raise ValueError("Remote API origins must use HTTPS")
        self.url = base + "/api/v1/retrieval"
        self.token = os.environ.get("RAGFLOW_API_KEY")
        if not self.token:
            raise ValueError("Set RAGFLOW_API_KEY in your environment; never put it in config files")
        self.opener = build_opener(ProxyHandler({}), NoRedirect())

    def retrieve(self, question, limit):
        payload = build_payload(question, self.config)
        if limit != payload["page_size"]:
            raise ValueError("Candidate window mismatch")
        request = Request(self.url, data=json.dumps(payload).encode(), method="POST", headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.token}"})
        try:
            with self.opener.open(request, timeout=30) as response:
                data = json.load(response)
        except HTTPError as exc:
            raise RetrievalError(f"http_{exc.code}") from None
        except (URLError, TimeoutError, OSError):
            raise RetrievalError("transport_error") from None
        except (ValueError, UnicodeError):
            raise RetrievalError("invalid_json_response") from None
        return parse_response(data, self.document_map)
