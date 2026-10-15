"""Isolated installed-RAGFlow adapter. Importing this module performs no I/O."""
import copy
from dataclasses import asdict
import hashlib
import json
from threading import RLock

from .adapter import Snapshot, check_final_request, judge_messages, native_kbinfos
from .cache import RequestCache
from .core import AgentLimits, run
from .evidence import DecisionError, Evidence, validate_views
from .embedding import single_request


def build_snapshot(config, rows):
    evidence = []
    for row in rows:
        entry = config["review"]["entries"][row["id"]]
        if row["document_id"] not in config["scope_document_ids"] or config["document_map"]["documents"][row["document_id"]] != entry["source_document_id"]:
            raise ValueError("snapshot_source_mismatch")
        if row["content_sha256"] != entry["raw_sha256"] or hashlib.sha256(entry["sanitized_content"].encode()).hexdigest() != entry["safe_sha256"]:
            raise ValueError("snapshot_content_changed")
        native = {"chunk_id": row["id"], "doc_id": row["document_id"], "docnm_kwd": entry["safe_title"],
                  "content_with_weight": entry["sanitized_content"], "kb_id": config["chat_config"]["dataset_ids"][0],
                  "content_ltks": "", "positions": [], "important_kwd": [], "img_id": ""}
        evidence.append(Evidence(row["id"], row["document_id"], entry["safe_title"], entry["sanitized_content"], row["order"], native))
    if set(config["review"]["entries"]) != {e.id for e in evidence}:
        raise ValueError("snapshot_review_coverage_mismatch")
    return Snapshot(evidence)


class VectorCache:
    def __init__(self, limit):
        self.values = {}
        self.lock = RLock()
        self.limit = limit
        self.calls = 0
        self.hits = 0
        self.tokens = 0

    def get(self, key, calculate):
        with self.lock:
            if key in self.values:
                self.hits += 1
                return copy.deepcopy(self.values[key][0]), 0
            if self.calls >= self.limit:
                raise ValueError("query_embedding_call_limit")
            self.calls += 1
            result = calculate()
            self.tokens += int(result[1])
            self.values[key] = copy.deepcopy(result)
            return result


class RuntimeBackend:
    def __init__(self, config, env, snapshot, current):
        self.config, self.env, self.snapshot, self.current = config, env, snapshot, current
        self.count_tokens = env["count_tokens"]
        self.cache = RequestCache()
        self.observed = {}
        self.selected = None
        self.replay = None
        self.metrics = {}
        self.native_initial = None
        self.native_final = False

    def record_control_event(self, event):
        self.env["event"](event)

    def remaining_generation_calls(self):
        ledger = self.env["ledger"]
        ledger.assert_open()
        used = sum(row["question_id"] == ledger.question_id for row in ledger.calls.values())
        return max(0, ledger.limits[2] - used)

    def evidence(self, result):
        safe = self.env["safe_result"](result, review=self.config["review"]["entries"], document_map=self.config["document_map"], title_review=self.config["document_title_review"]["entries"])
        items = []
        for c in safe["chunks"]:
            cid = c.get("chunk_id", c.get("id"))
            source = self.snapshot.by_id[cid]
            if c["doc_id"] != source.document_id or c["content_with_weight"] != source.text:
                raise ValueError("runtime_source_changed")
            # Keep one canonical native row per chunk; query-dependent scores are trace data.
            item = self.observed.get(cid) or Evidence(cid, c["doc_id"], c["docnm_kwd"], source.text, source.order, copy.deepcopy(c))
            self.observed[cid] = item
            items.append(item)
        return safe, items

    async def search(self, query, document_ids=None):
        scope = document_ids or self.config["scope_document_ids"]
        allowed = set(self.config["scope_document_ids"])
        if not scope or not set(scope) <= allowed:
            raise ValueError("invalid_document_scope")
        if document_ids and not set(scope) <= {e.document_id for e in self.observed.values()}:
            raise ValueError("unseen_document_scope")
        key = (query, tuple(sorted(scope)))

        async def retrieve():
            if self.config.get("replay"):
                if self.native_initial is not None:
                    raise ValueError("offline_replay_cannot_search")
                raw = copy.deepcopy(self.current["replay"]["initial"])
            else:
                c = self.config["chat_config"]
                raw = await self.env["retrieval"](
                    query, self.env["embedding"], self.env["tenant_ids"], c["dataset_ids"], 1, c["top_n"],
                    c["similarity_threshold"], c["vector_similarity_weight"], doc_ids=scope,
                    knn_top_k=c["top_k"], aggs=True, rerank_mdl=self.env["reranker"],
                    rank_feature=self.env["label_question"](query, self.env["kbs"]),
                    rerank_candidates_count=self.env["rerank_candidates"])
                raw["chunks"] = self.env["children"](raw["chunks"], self.env["tenant_ids"])
            safe, items = self.evidence(raw)
            if self.native_initial is None:
                self.native_initial = copy.deepcopy(safe)
            self.env["event"]({"event": "search_result", "query": query, "chunk_ids": [e.id for e in items]})
            return items
        result = await self.cache.get(key, retrieve)
        self.metrics["search_cache_hits"] = self.cache.hits
        return result

    async def neighbors(self, seed, direction):
        if seed.id not in self.observed:
            raise ValueError("unseen_neighbor_seed")
        result = self.snapshot.neighbors(seed, direction)
        for e in result:
            self.observed.setdefault(e.id, e)
        return [self.observed[e.id] for e in result]

    def neighbor_directions(self, seed):
        return [direction for direction in ("previous", "next")
                if any(e.id not in self.observed for e in self.snapshot.neighbors(seed, direction))]

    async def resolve(self, context):
        from .resolution import resolution_messages, validate_resolution

        if self.remaining_generation_calls() < 2:
            raise ValueError("final_call_reserved")
        system, history = resolution_messages(context)
        budget = self.config.get("conversation", {}).get("history_tokens", 3000)
        if self.count_tokens(system + history[0]["content"]) > budget:
            raise DecisionError("conversation_context_budget")
        self.current["stage"] = "resolve"
        self.current["approved_resolution_messages"] = [{"role": "system", "content": system}] + history
        answer = await self.env["chat"].async_chat(system, history, {"temperature": 0, "top_p": 1, "max_tokens": 600,
                                                                  "response_format": {"type": "json_object"}})
        self.env["ledger"].assert_open()
        self.env["event"]({"event": "resolution_raw_response", "text": answer})
        text = answer.strip()
        if text.startswith("```json") and text.endswith("```"):
            text = text[7:-3].strip()
        try:
            value = json.loads(text)
        except json.JSONDecodeError as exc:
            raise DecisionError("invalid_resolution_json") from exc
        resolution = validate_resolution(value, context)
        self.env["event"]({"event": "resolution", "value": asdict(resolution)})
        return resolution

    async def decide(self, question, views, previous):
        validate_views(views, list(self.observed.values()))
        system, history = judge_messages(question, views, previous,
                                         neighbors={e.id: self.neighbor_directions(e) for e in views},
                                         gap_contract=self.config.get("agent_limits", {}).get("gap_contract", True))
        if self.count_tokens(system + history[0]["content"]) > 7000:
            raise ValueError("judge_input_budget")
        self.current["stage"] = "judge"
        self.current["approved_judge_messages"] = [{"role": "system", "content": system}] + history
        answer = await self.env["chat"].async_chat(system, history, {"temperature": 0, "top_p": 1, "max_tokens": 1200,
                                                                  "response_format": {"type": "json_object"}})
        self.env["ledger"].assert_open()
        self.env["event"]({"event": "judge_raw_response", "text": answer})
        text = answer.strip()
        if text.startswith("```json") and text.endswith("```"):
            text = text[7:-3].strip()
        try:
            value = json.loads(text)
        except json.JSONDecodeError as exc:
            raise DecisionError("invalid_json") from exc
        self.env["event"]({"event": "judge_result", "value": value})
        return value

    async def generate(self, question, selected, *, native=False):
        self.env["ledger"].assert_open()
        if self.env["ledger"].summary()["pending_calls"]:
            raise ValueError("unresolved_provider_usage")
        validate_views(selected, list(self.observed.values()))
        self.selected = selected
        self.native_final = native
        self.current["stage"] = "final"
        # The native route consumes the selected evidence; it must not retrieve again.
        self.replay = copy.deepcopy(self.native_initial) if native else native_kbinfos(selected)
        if not native:
            for chunk in self.replay["chunks"]:
                chunk["content_ltks"] = self.env["tokenize"](chunk["content_with_weight"])
                chunk["content_sm_ltks"] = self.env["fine_tokenize"](chunk["content_ltks"])
        self.current["replay_retrieval_calls"] = 0
        self.current["kb_prompt_reviewed"] = False
        response = await self.env["client"].post("/api/v1/chat/completions", json=self.env["payload"](self.config, {"question": question}), headers={"Authorization": "Bearer " + self.env["token"]})
        raw = (await response.get_data()).decode()
        if self.current.get("preflight_verified"):
            return {"answer": "本地请求等价校验通过（未调用模型）", "reference": {}}
        result = self.env["collect_sse"]([raw])
        self.env["ledger"].assert_open()
        if response.status_code != 200 or result["error"] or not result["stream_complete"]:
            raise ValueError("native_final_route_failed")
        self.current["native_sse"] = raw
        return result


def _check_resolution_request(body, current):
    if body.get("messages") != current["approved_resolution_messages"]:
        raise ValueError("resolution_request_changed")
    if body.get("response_format") != {"type": "json_object"}:
        raise ValueError("resolution_json_mode_dropped")
    if body.get("max_tokens") != 600:
        raise ValueError("resolution_output_budget_changed")


async def _serve(config, env, snapshot, current, emit, lines):
    """Run serial requests in the isolated process that owns the patched adapters."""
    import re
    import time
    import uuid

    conversational = "conversation" in config
    sessions = {}
    if conversational:
        from .conversation import Conversation, ConversationLimits

        conversation_limits = ConversationLimits(**config["conversation"])
        if config.get("replay"):
            raise ValueError("conversation_cannot_use_single_question_replay")
    ledger = env["ledger"]
    for line in lines:
        item = json.loads(line)
        if conversational:
            session_id = item.get("session_id")
            if not isinstance(session_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", session_id):
                raise ValueError("invalid_session_id")
            if item.get("command") == "clear" and set(item) == {"command", "session_id"}:
                conversation = sessions.pop(session_id, None)
                if conversation:
                    await conversation.clear()
                emit({"event": "session_cleared", "session_id": session_id})
                continue
            if set(item) != {"session_id", "question"} or not isinstance(item["question"], str) or not item["question"].strip():
                raise ValueError("invalid_conversation_question")
            if session_id not in sessions:
                if len(sessions) >= 32:
                    raise ValueError("conversation_session_limit")
                sessions[session_id] = Conversation(session_id, limits=conversation_limits)
            question_id = "turn-" + uuid.uuid4().hex
        else:
            if item["question_id"] not in config["question_ids"] or set(item) != {"question_id", "question"}:
                raise ValueError("invalid_runtime_question")
            question_id = item["question_id"]
        current.clear()
        current.update(question_id=question_id, vectors=VectorCache(3), stage="search", embedding_physical={})
        if config.get("replay"):
            current["replay"] = config["replay"][question_id]
        ledger.start(question_id)
        if conversational:
            emit({"event": "turn_started", "question_id": question_id, "session_id": session_id})
        backend = RuntimeBackend(config, env, snapshot, current)
        current["backend"] = backend
        started = time.monotonic()
        try:
            limits = AgentLimits(**config["agent_limits"])
            result = await sessions[session_id].ask(item["question"], backend, limits) if conversational else await run(item["question"], backend, limits)
        finally:
            await backend.cache.close()
        row = asdict(result)
        if ledger.summary()["pending_calls"] and not ledger.fatal_reason:
            ledger.abort("provider_usage_unknown")
        if current["embedding_physical"].get("unknown_embedding_usage") and not ledger.fatal_reason:
            ledger.abort("embedding_usage_unknown")
        fatal = ledger.fatal_reason
        if fatal:
            row["status"] = "budget_censored" if "budget" in fatal or "call_limit" in fatal else "guard_stopped"
            row["stop_reason"] = fatal
        row.update(question_id=question_id, question=item["question"], latency_ms=round((time.monotonic() - started) * 1000),
                   ledger=ledger.summary(), calls=[copy.deepcopy(c) for c in ledger.calls.values() if c["question_id"] == question_id],
                   query_embedding_calls=current["vectors"].calls, embedding_cache_hits=current["vectors"].hits,
                   query_embedding_tokens=current["vectors"].tokens, citation_embedding_calls=current.get("citation_embedding_calls", 0),
                   embedding_physical=current["embedding_physical"],
                   seen_chunks={cid: {"doc_id": e.document_id, "docnm_kwd": e.title, "content_with_weight": e.text} for cid, e in backend.observed.items()},
                   final_request=current.get("final_request"), events=current.get("events", []), preflight_verified=current.get("preflight_verified", False),
                   request_difference=current.get("request_difference"))
        emit({"event": "question_complete", "record": row})
        if fatal or (not conversational and row["status"] != "ok"):
            break


async def execute(config):
    import contextlib
    import functools
    import io
    import logging
    import os
    from pathlib import Path
    import sys

    from experiments.evaluation.native_agent_guard import CallLedger
    from experiments.evaluation.native_agent_worker import (
        collect_sse, completion_options, make_payload, runtime_model_config, safe_result,
        validate_chat_config, validate_runtime_embedding,
    )
    from .transport import monitored_send

    if os.environ.get("BENCHMARK_RUN_TOKEN") != config["worker_token"]:
        raise ValueError("worker_token_mismatch")
    if os.getpgid(0) != os.getpid():
        os.setsid()
    os.environ["LLM_MAX_RETRIES"] = "0"
    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    logging.disable(logging.CRITICAL)
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        from api.apps import app
        from api.db.services import dialog_service
        from api.db.services.dialog_service import DialogService
        from api.db.services.llm_service import LLMBundle
        from common import settings
        from rag.prompts import generator
        from rag.nlp import rag_tokenizer
        from rag.llm import embedding_model
        import httpx
        import litellm
        import pymysql
        import yaml

    for path, digest in config["runtime_code_hashes"].items():
        if hashlib.sha256((Path("/ragflow") / path).read_bytes()).hexdigest() != digest:
            raise ValueError("installed_runtime_changed:" + path)
    dbconf = yaml.safe_load(Path("/ragflow/conf/service_conf.yaml").read_text())["mysql"]
    db = pymysql.connect(host=dbconf["host"], port=int(dbconf["port"]), user=dbconf["user"], password=dbconf["password"], database=dbconf["name"])
    with db.cursor() as cur:
        cur.execute("SELECT tenant_id,tenant_llm_id,kb_ids,top_n,top_k,similarity_threshold,vector_similarity_weight,rerank_id,llm_setting,prompt_config,meta_data_filter FROM dialog WHERE id=%s", (config["chat_id"],))
        tenant, mid, kbs, top_n, top_k, threshold, weight, rerank, llm, prompt, metadata = cur.fetchone()
        if metadata and json.loads(metadata):
            raise ValueError("metadata_not_allowed")
        cur.execute("SELECT model_name,extra FROM tenant_model WHERE id=%s", (mid,))
        model_name, extra = cur.fetchone()
        actual = {"model": model_name, "model_id": mid, "dataset_ids": sorted(json.loads(kbs)), "top_n": top_n, "top_k": top_k,
                  "similarity_threshold": float(threshold), "vector_similarity_weight": float(weight), "rerank_id": rerank,
                  "llm_setting": json.loads(llm), "prompt_config": json.loads(prompt), "model_context_budget": json.loads(extra)["max_tokens"]}
        validate_chat_config(config["chat_config"], actual)
        cur.execute("SELECT token FROM api_token WHERE tenant_id=%s AND (dialog_id IS NULL OR dialog_id='') ORDER BY create_time DESC LIMIT 1", (tenant,))
        token = cur.fetchone()[0]
    db.close()
    if not config.get("replay"):
        from elasticsearch import Elasticsearch
        from elasticsearch.helpers import scan
        service = yaml.safe_load(Path("/ragflow/conf/service_conf.yaml").read_text())
        esconf = service["es"]
        es = Elasticsearch(esconf["hosts"].split(","), basic_auth=(esconf["username"], esconf["password"]), verify_certs=True, request_timeout=15, max_retries=0)
        try:
            index = "ragflow_" + tenant
            kbfilter = {"terms": {"kb_id": actual["dataset_ids"]}}
            compiled = es.count(index=index, query={"bool": {"filter": [kbfilter, {"exists": {"field": "compile_kwd"}}]}})["count"]
            query = {"query": {"bool": {"filter": [kbfilter, {"terms": {"doc_id": config["scope_document_ids"]}}]}},
                     "_source": ["doc_id", "docnm_kwd", "content_with_weight"]}
            chunks = {}
            for hit in scan(es, index=index, query=query):
                value = hit["_source"]
                chunks[hit["_id"]] = {"doc_id": value["doc_id"], "title": value["docnm_kwd"], "sha256": hashlib.sha256(value["content_with_weight"].encode()).hexdigest()}
            if {"compiled_rows": compiled, "chunks": chunks} != config["index_snapshot"]:
                raise ValueError("live_index_differs_from_frozen_baseline")
        finally:
            es.close()
    dialog_service.TenantLangfuseService.filter_by_tenant = lambda **kw: []
    generator.get_llm_cache = lambda *a, **k: None
    generator.set_llm_cache = lambda *a, **k: None
    resolver = dialog_service.get_model_config_by_id

    def resolve(*args, **kwargs):
        value = resolver(*args, **kwargs)
        return runtime_model_config(value, config["model"]) if value.get("llm_name") == actual["model"] else value

    dialog_service.get_model_config_by_id = resolve
    models = dialog_service.get_models

    def get_models(*args, **kwargs):
        result = models(*args, **kwargs)
        chat = result[3]
        if chat.model_config["llm_name"] != config["model"] or type(chat.mdl).__name__ != "LiteLLMBase" or chat.model_config.get("api_base", "") != config["provider_config"]["api_base"]:
            raise ValueError("provider_model_or_adapter_changed")
        chat.model_config["is_tools"] = chat.is_tools = chat.mdl.is_tools = True
        return result

    dialog_service.get_models = get_models
    found, dialog = DialogService.get_by_id(config["chat_id"])
    if not found:
        raise ValueError("missing_dialog")
    kbs, embedding, reranker, chat, _ = get_models(dialog)
    if embedding.model_config["llm_name"] != config["embedding_model"]:
        raise ValueError("embedding_model_changed")
    snapshot = build_snapshot(config, config["snapshot_rows"])
    limit = config["limits"]
    ledger = CallLedger(config["container_ledger"], run_limit=limit["run_tokens"], question_limit=limit["question_tokens"], call_limit=limit["question_calls"])
    current = {}

    if type(embedding.mdl).__name__ != "QWenEmbed":
        raise ValueError("unverified_embedding_adapter")

    def embed_request(self, inputs, text_type):
        ledger.assert_open()
        if config.get("replay"):
            raise ValueError("embedding_forbidden_in_replay")
        return single_request(self, inputs, text_type, embedding_model.dashscope.TextEmbedding.call,
                              embedding_model._dashscope_native_api_url_scope, current["embedding_physical"],
                              connect_retries=config.get("connect_retries", 0))

    embedding_model.QWenEmbed._request = embed_request

    def emit(value):
        print("AGENT_RESULT:" + json.dumps(value, ensure_ascii=False), flush=True)

    def event(value):
        row = {"question_id": current.get("question_id"), "stage": current.get("stage"), **value}
        current.setdefault("events", []).append(row)
        emit(row)

    original_query = LLMBundle.encode_queries
    original_encode = LLMBundle.encode

    def encode_queries(self, query):
        if config.get("replay"):
            raise ValueError("embedding_forbidden_in_replay")
        if len(query.encode()) > 4096:
            raise ValueError("query_too_long")
        identity = tuple(str(self.model_config.get(k, "")) for k in ("llm_name", "api_base", "llm_factory", "model_type", "max_tokens", "dimensions"))
        result = current["vectors"].get(identity + (query,), lambda: original_query(self, query))
        event({"event": "query_embedding", "query": query, "physical_calls": current["vectors"].calls, "cache_hits": current["vectors"].hits, "tokens": result[1]})
        return result

    def encode(self, texts):
        import inspect
        if config.get("replay"):
            raise ValueError("embedding_forbidden_in_replay")
        frame = inspect.currentframe().f_back
        validate_runtime_embedding(frame.f_globals.get("__name__"), frame.f_code.co_name, texts)
        current["citation_embedding_calls"] = current.get("citation_embedding_calls", 0) + 1
        if current["citation_embedding_calls"] > 2:
            raise ValueError("citation_embedding_limit")
        result = original_encode(self, texts)
        event({"event": "citation_embedding", "tokens": result[1]})
        return result

    LLMBundle.encode_queries, LLMBundle.encode = encode_queries, encode
    retrieval, children = settings.retriever.retrieval, settings.retriever.retrieval_by_children
    original_prompt = dialog_service.kb_prompt

    async def replay_retrieval(*args, **kwargs):
        backend = current["backend"]
        if backend.replay is None or current["stage"] != "final" or current["replay_retrieval_calls"]:
            raise ValueError("unexpected_native_retrieval")
        if set(kwargs.get("doc_ids") or []) != set(config["scope_document_ids"]):
            raise ValueError("native_scope_changed")
        current["replay_retrieval_calls"] += 1
        return copy.deepcopy(backend.replay)

    def replay_children(chunks, *args, **kwargs):
        if current.get("stage") != "final":
            raise ValueError("unexpected_native_children")
        return chunks

    def kb_prompt(info, *args, **kwargs):
        backend = current["backend"]
        if info != backend.replay:
            raise ValueError("native_evidence_mutated")
        validate_views(backend.selected, list(backend.observed.values()))
        current["kb_prompt_reviewed"] = True
        return original_prompt(info, *args, **kwargs)

    settings.retriever.retrieval, settings.retriever.retrieval_by_children = replay_retrieval, replay_children
    dialog_service.kb_prompt = kb_prompt
    completion = litellm.acompletion

    @functools.wraps(completion)
    async def complete(*args, **kwargs):
        return await completion(*args, **completion_options(kwargs, config))

    litellm.acompletion = complete

    def before_send(body):
        if current["stage"] == "resolve":
            _check_resolution_request(body, current)
        elif current["stage"] == "judge":
            if body.get("messages") != current["approved_judge_messages"]:
                raise ValueError("judge_request_changed")
            if body.get("response_format") != {"type": "json_object"}:
                raise ValueError("judge_json_mode_dropped")
        elif current["stage"] == "final":
            if not current.get("kb_prompt_reviewed") or current.get("final_request"):
                raise ValueError("missing_context_or_duplicate_final")
            if not current["backend"].native_final:
                check_final_request(body, current["backend"].selected)
            current["final_request"] = copy.deepcopy(body)
            if config.get("replay"):
                expected = current["replay"]["expected_request"]
                if body != expected:
                    current["request_difference"] = {k: {"expected": expected.get(k), "actual": body.get(k)} for k in set(body) | set(expected) if body.get(k) != expected.get(k)}
                    raise ValueError("native_request_equivalence_failed")
                current["preflight_verified"] = True
                raise ValueError("offline_preflight_no_provider_call")
        else:
            raise ValueError("unexpected_generation_phase")

    httpx.AsyncClient._send_single_request = monitored_send(httpx.AsyncClient._send_single_request, ledger, config, before_send, event)
    env = {"retrieval": retrieval, "children": children, "kbs": kbs, "embedding": embedding, "reranker": reranker, "chat": chat,
           "tenant_ids": list(set(k.tenant_id for k in kbs)), "rerank_candidates": getattr(dialog, "rerank_candidates_count", 64),
           "label_question": dialog_service.label_question, "count_tokens": generator.num_tokens_from_string,
           "tokenize": rag_tokenizer.tokenize, "fine_tokenize": rag_tokenizer.fine_grained_tokenize,
           "safe_result": safe_result, "event": event, "ledger": ledger,
           "client": app.test_client(), "token": token, "payload": make_payload, "collect_sse": collect_sse}
    emit({"event": "ready", "worker_token": config["worker_token"], "ledger": ledger.summary()})
    await _serve(config, env, snapshot, current, emit, sys.stdin)
