"""Run the installed native chat route with a reviewed outbound-content gate.

Executed in the existing RAGFlow container. It changes no server/index files.
The only pipeline hook substitutes approved sanitized text before kb_prompt.
"""

import copy
import hashlib
import json


def make_payload(config, question):
    return {
        "chat_id": config["chat_id"],
        "messages": [{"role": "user", "content": question["question"]}],
        "doc_ids": ",".join(config["scope_document_ids"]),
        "stream": False,
        "pass_all_history_messages": True,
        "store_history_messages": False,
    }


def validate_outbound(expected, actual):
    if expected != actual:
        raise ValueError("outbound_request_differs_from_frozen_preflight")


def check_budget(spent, reserve, limit):
    if spent + reserve > limit:
        raise ValueError("run_token_budget_reached")


def usage_tokens(usage):
    if not isinstance(usage, dict) or int(usage.get("total_tokens") or 0) <= 0:
        raise ValueError("provider_usage_unavailable_stop_before_next_call")
    return int(usage["total_tokens"])


def valid_response(http_status, body, state, allowed):
    data = body.get("data") or {}
    answer = data.get("answer") or ""
    refs = (data.get("reference") or {}).get("chunks") or []
    return bool(
        http_status == 200
        and body.get("code") == 0
        and isinstance(answer, str)
        and answer.strip()
        and not answer.startswith("**ERROR**")
        and state.get("outbound_request")
        and all(c.get("document_id") in allowed for c in refs)
        and all(c.get("doc_id") in allowed for c in state.get("retrieved_chunks", []))
    )


class CallGuard:
    def __init__(self, question_ids):
        self.allowed = set(question_ids)
        self.claimed = set()

    @property
    def calls(self):
        return len(self.claimed)

    def claim(self, question_id):
        if question_id not in self.allowed or question_id in self.claimed:
            raise ValueError("unplanned_or_duplicate_model_call")
        self.claimed.add(question_id)


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


async def execute(config):
    import asyncio
    import contextlib
    import io
    import logging
    import os
    from pathlib import Path
    import sys
    import time

    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    os.environ["LLM_MAX_RETRIES"] = "0"
    logging.disable(logging.CRITICAL)
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        from api.apps import app
        from api.db.services import dialog_service
        from api.db.services.llm_service import LLMBundle
        import pymysql
        import yaml

    namespace = {"__name__": "native_outbound_gate"}
    exec(compile(config["gate_source"], "native_outbound.py", "exec"), namespace)
    sanitize_kbinfos = namespace["sanitize_kbinfos"]
    original_kb_prompt = dialog_service.kb_prompt
    original_chat = LLMBundle.async_chat
    original_encode_queries = LLMBundle.encode_queries
    original_encode = LLMBundle.encode
    current = {}
    guard = CallGuard(config["question_ids"])
    total_usage = config.get("initial_provider_tokens", 0)

    settings = yaml.safe_load(Path("/ragflow/conf/service_conf.yaml").read_text())["mysql"]
    db = pymysql.connect(host=settings["host"], port=int(settings["port"]), user=settings["user"], password=settings["password"], database=settings["name"], autocommit=True)
    with db.cursor() as cur:
        cur.execute("SELECT tenant_id,tenant_llm_id,kb_ids,top_n,top_k,similarity_threshold,vector_similarity_weight,rerank_id,llm_setting,prompt_config FROM dialog WHERE id=%s", (config["chat_id"],))
        row = cur.fetchone()
        if not row:
            raise ValueError("missing_chat")
        tenant, model_id, kbs, top_n, top_k, threshold, weight, rerank, llm_setting, prompt_config = row
        cur.execute("SELECT model_name,extra FROM tenant_model WHERE id=%s", (model_id,))
        model_name, model_extra = cur.fetchone()
        actual_config = {
            "model": model_name,
            "model_id": model_id,
            "dataset_ids": sorted(json.loads(kbs)),
            "top_n": top_n,
            "top_k": top_k,
            "similarity_threshold": float(threshold),
            "vector_similarity_weight": float(weight),
            "rerank_id": rerank,
            "llm_setting": json.loads(llm_setting),
            "prompt_config": json.loads(prompt_config),
            "model_context_budget": json.loads(model_extra)["max_tokens"],
        }
        validate_outbound(config["chat_config"], actual_config)
        cur.execute("SELECT token FROM api_token WHERE tenant_id=%s AND (dialog_id IS NULL OR dialog_id='') ORDER BY create_time DESC LIMIT 1", (tenant,))
        secret = cur.fetchone()
        if not secret:
            raise ValueError("missing_api_token")
        token = secret[0]
    db.close()

    # Disable optional external telemetry in this isolated evaluation process.
    dialog_service.TenantLangfuseService.filter_by_tenant = lambda **kwargs: []

    def gated_kb_prompt(kbinfos, *args, **kwargs):
        current["retrieved_chunks"] = [
            {k: copy.deepcopy(c[k]) for k in ("chunk_id", "id", "doc_id", "docnm_kwd", "content_with_weight", "kb_id", "similarity", "term_similarity", "vector_similarity", "url", "document_metadata") if k in c}
            for c in kbinfos.get("chunks", [])
        ]
        try:
            safe, audit = sanitize_kbinfos(kbinfos, config["review"]["entries"], config["document_map"])
        except ValueError as exc:
            current["review_error"] = str(exc)
            raise
        current["outbound_audit"] = audit
        current["outbound_chunks"] = [
            {k: copy.deepcopy(c[k]) for k in ("chunk_id", "id", "doc_id", "docnm_kwd", "content_with_weight") if k in c}
            for c in safe.get("chunks", [])
        ]
        return original_kb_prompt(safe, *args, **kwargs)

    def counted_queries(self, query):
        current["query_embedding_calls"] = current.get("query_embedding_calls", 0) + 1
        return original_encode_queries(self, query)

    def counted_encode(self, texts):
        current["other_embedding_batches"] = current.get("other_embedding_batches", 0) + 1
        return original_encode(self, texts)

    async def captured_chat(self, system, history, gen_conf=None, **kwargs):
        nonlocal total_usage
        if current.get("outbound_request") is not None:
            raise ValueError("multiple_generation_calls_for_one_question")
        envelope = {"model": self.model_config["llm_name"], "system": system, "history": copy.deepcopy(history), "gen_conf": copy.deepcopy(gen_conf or {})}
        if envelope["model"] != config["model"] or not current.get("outbound_audit"):
            raise ValueError("unexpected_model_or_missing_outbound_review")
        current["outbound_request"] = envelope
        current["outbound_sha256"] = fingerprint(envelope)
        if config["mode"] == "preflight":
            return "本地预检占位。##0$$"
        expected = config["frozen_requests"].get(current["question_id"])
        validate_outbound(expected, envelope)
        check_budget(total_usage, config["chat_config"]["model_context_budget"], config.get("stop_at_tokens", 900000))
        guard.claim(current["question_id"])
        current["generation_attempted"] = True
        answer = await original_chat(self, system, history, gen_conf or {}, **kwargs)
        usage = getattr(self.mdl, "last_usage", None)
        current["provider_usage"] = copy.deepcopy(usage) if isinstance(usage, dict) else None
        total_usage += usage_tokens(usage)
        return answer

    dialog_service.kb_prompt = gated_kb_prompt
    LLMBundle.async_chat = captured_chat
    LLMBundle.encode_queries = counted_queries
    LLMBundle.encode = counted_encode

    def emit(value):
        print("NATIVE_RESULT:" + json.dumps(value, ensure_ascii=False), flush=True)

    code_hashes = {name: hashlib.sha256((Path("/ragflow") / name).read_bytes()).hexdigest() for name in config["runtime_code_hashes"]}
    validate_outbound(config["runtime_code_hashes"], code_hashes)
    emit({"status": "ready", "mode": config["mode"], "model": model_name, "scope_documents": len(config["scope_document_ids"]), "top_n": top_n})
    client = app.test_client()
    for line in sys.stdin:
        item = json.loads(line)
        if item["question_id"] not in config["question_ids"]:
            raise ValueError("unplanned_question")
        current.clear()
        current["question_id"] = item["question_id"]
        start = time.monotonic()
        try:
            response = await asyncio.wait_for(client.post("/api/v1/chat/completions", json=make_payload(config, item), headers={"Authorization": "Bearer " + token}), timeout=150)
            body = await response.get_json()
            ok = valid_response(response.status_code, body, current, set(config["scope_document_ids"]))
            status = "ok" if ok else "api_error"
            if current.get("review_error"):
                status = "outbound_review_required"
            record = {"question_id": item["question_id"], "status": status, "http_status": response.status_code, "body": body, **copy.deepcopy(current)}
        except Exception as exc:
            record = {"question_id": item["question_id"], "status": "worker_error", "error_type": type(exc).__name__, "error": str(exc)[:200], **copy.deepcopy(current)}
        record.update({"latency_ms": round((time.monotonic() - start) * 1000, 1), "generation_calls_so_far": guard.calls, "provider_total_tokens_so_far": total_usage})
        emit(record)
        if config["mode"] == "generate" and record["status"] != "ok":
            emit({"status": "stopped_after_error", "question_id": item["question_id"], "generation_calls": guard.calls})
            break
        if config["mode"] == "generate" and total_usage >= config.get("stop_at_tokens", 900000):
            emit({"status": "stopped_at_token_budget", "generation_calls": guard.calls})
            break


def main():
    import asyncio
    import sys

    config = json.loads(sys.stdin.readline())
    if config["mode"] not in {"preflight", "generate"}:
        raise ValueError("invalid_mode")
    asyncio.run(execute(config))


if __name__ == "__main__":
    main()
