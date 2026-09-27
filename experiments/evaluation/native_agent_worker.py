"""Isolated adapter for the installed native Agent; no server files are edited."""

import copy
import hashlib
import json


def runtime_model_config(original, model):
    chosen = copy.deepcopy(original)
    chosen["llm_name"] = model
    return chosen


def generation_context_valid(reasoning, current):
    return bool(current.get("ordinary_context_reviewed") if reasoning == 0 else current.get("agent_initialized"))


def completion_options(options, config):
    chosen = copy.deepcopy(options)
    chosen.update(num_retries=0, max_retries=0)
    chosen["max_tokens"] = min(int(chosen.get("max_tokens") or 1200), 1200)
    if "model_thinking" in config:
        chosen["extra_body"] = {**chosen.get("extra_body", {}), "enable_thinking": config["model_thinking"]}
    if chosen.get("stream"):
        chosen["stream_options"] = {"include_usage": True}
    return chosen


def trace_sca(original, event, stage):
    """Observe native evidence decisions without changing their results."""
    import functools
    import inspect

    signature = inspect.signature(original)

    @functools.wraps(original)
    async def observed(*args, **kwargs):
        params = signature.bind(*args, **kwargs).arguments
        evidence = (getattr(params.get("tools"), "kbinfos", None) or {}).get("chunks", [])
        handle = stage.set("sca")
        try:
            event({"event": "sca_input", "question": params.get("question"),
                   "claims": copy.deepcopy(params.get("claims")),
                   "evidence": [{"chunk_id": c.get("chunk_id", c.get("id")),
                                 "content_sha256": hashlib.sha256(str(c.get("content_with_weight", c.get("content", ""))).encode()).hexdigest()}
                                for c in evidence]})
            result = await original(*args, **kwargs)
            event({"event": "sca_result", "result": copy.deepcopy(result)})
            return result
        except BaseException as exc:
            event({"event": "sca_failed", "error_type": type(exc).__name__})
            raise
        finally:
            stage.reset(handle)

    return observed


def make_payload(config, question):
    return {
        "chat_id": config["chat_id"], "reasoning": config.get("reasoning", 2),
        "messages": [{"role": "user", "content": question["question"]}],
        "doc_ids": ",".join(config["scope_document_ids"]), "stream": True,
        "internet": False, "store_history_messages": False, "pass_all_history_messages": True,
    }


def safe_result(result, *, review, document_map, title_review=None):
    gate = globals().get("_sanitize")
    if gate is None:
        from experiments.evaluation.native_outbound import sanitize_kbinfos
        gate = sanitize_kbinfos
    if not isinstance(result, dict) or not isinstance(result.get("chunks"), list):
        raise ValueError("invalid_evidence_result")
    safe = copy.deepcopy(result)
    sources = document_map["documents"]
    def sha(value):
        return hashlib.sha256(value.encode()).hexdigest()
    for chunk in safe.get("chunks", []):
        if not isinstance(chunk, dict):
            raise ValueError("invalid_evidence_chunk")
        cid = chunk.get("chunk_id", chunk.get("id"))
        entry = review.get(cid)
        if not isinstance(entry, dict) or sources.get(chunk.get("doc_id")) != entry.get("source_document_id"):
            raise ValueError("unreviewed_chunk_or_scope")
        for field in ("safe_sha256", "sanitized_content", "safe_title", "safe_title_sha256", "raw_sha256", "raw_title", "raw_title_sha256"):
            if not isinstance(entry.get(field), str):
                raise ValueError("invalid_review_field")
        for alias in ("content", "document_name", "url", "document_metadata"):
            if chunk.get(alias):
                raise ValueError("unreviewed_prompt_alias")
        content = chunk.get("content_with_weight", "")
        if not isinstance(content, str) or not isinstance(chunk.get("docnm_kwd"), str):
            raise ValueError("invalid_evidence_text")
        if (sha(content) == entry["safe_sha256"] and content == entry["sanitized_content"]
                and chunk.get("docnm_kwd") == entry["safe_title"]
                and sha(entry["safe_title"]) == entry["safe_title_sha256"]):
            continue
        gated, _ = gate({"chunks": [chunk]}, review, document_map)
        chunk.clear()
        chunk.update(gated["chunks"][0])
    by_source = {}
    for entry in review.values():
        by_source.setdefault(entry["source_document_id"], entry)
    for agg in safe.get("doc_aggs", []):
        doc_id = agg.get("doc_id")
        entry = title_review.get(doc_id) if title_review is not None else by_source.get(sources.get(doc_id))
        if (not isinstance(entry, dict) or not sources.get(doc_id) or entry.get("source_document_id") != sources[doc_id]
                or any(not isinstance(entry.get(k), str) for k in ("raw_title", "safe_title", "raw_title_sha256", "safe_title_sha256"))
                or sha(entry["raw_title"]) != entry["raw_title_sha256"] or sha(entry["safe_title"]) != entry["safe_title_sha256"]
                or agg.get("doc_name") not in (entry["raw_title"], entry["safe_title"])):
            raise ValueError("unreviewed_aggregate_title")
        agg["doc_name"] = entry["safe_title"]
    return safe


def safe_direct_rows(rows, *, review, document_map):
    if not isinstance(rows, dict):
        raise ValueError("invalid_direct_store_result")
    result = {}
    for cid, row in rows.items():
        if not isinstance(row, dict) or row.get("compile_kwd") or "content_with_weight" not in row:
            raise ValueError("unreviewed_direct_store_row")
        try:
            safe = safe_result({"chunks": [{**row, "id": cid}]}, review=review, document_map=document_map)["chunks"][0]
        except (ValueError, TypeError, KeyError, AttributeError) as exc:
            raise ValueError("unreviewed_direct_store_row") from exc
        if "id" not in row:
            safe.pop("id", None)
        result[cid] = safe
    return result


def validate_chat_config(expected, actual):
    if expected != actual:
        raise ValueError("runtime_chat_configuration_changed")


def validate_runtime_embedding(module, function, texts):
    if module == "navigation_query" and function == "scoped_query":
        if not isinstance(texts, list) or len(texts) != 1 or not isinstance(texts[0], str) or len(texts[0].encode()) > 4096:
            raise ValueError("navigation_query_embedding_limit")
        return
    if module != "rag.nlp.search" or function != "insert_citations" or not isinstance(texts, list) or not 0 < len(texts) <= 32:
        raise ValueError("non_citation_embedding_not_allowed")
    if any(not isinstance(text, str) for text in texts) or sum(len(text.encode()) for text in texts) > 12000:
        raise ValueError("citation_embedding_batch_limit")


def map_final_evidence(request, seen):
    text = "\n".join(m.get("content", "") for m in request.get("messages", []) if isinstance(m.get("content"), str))
    evidence = []
    for cid, chunk in seen.items():
        content = chunk["content_with_weight"]
        fragment = content if content and content in text else None
        match = "full"
        if fragment is None:
            lines = [line for line in content.splitlines() if len(line.strip()) >= 3 and line in text]
            if not lines or (len(lines) < 2 and max(map(len, lines)) < 12):
                continue
            fragment, match = "\n".join(lines), "partial_literal"
        evidence.append({"chunk_id": cid, "doc_id": chunk["doc_id"], "docnm_kwd": chunk["docnm_kwd"], "content_with_weight": fragment,
                         "match": match, "source_mapping": "literal_candidate_requires_citation_review"})
    return evidence


def collect_sse(chunks):
    result = {"answer": "", "reference": {}, "error": False, "stream_complete": False}
    deltas = []
    thinking = False
    final_answer = None
    for event in "".join(chunks).replace("\r\n", "\n").split("\n\n"):
        if not event.strip():
            continue
        payload = "\n".join(line[5:] for line in event.splitlines() if line.startswith("data:"))
        if not payload:
            continue
        row = json.loads(payload)
        if row.get("code") != 0:
            result["error"] = True
        data = row.get("data")
        if data is True:
            result["stream_complete"] = True
        elif isinstance(data, dict):
            if data.get("start_to_think"):
                thinking = True
            elif data.get("end_to_think"):
                thinking = False
            elif data.get("final"):
                if data.get("answer"):
                    final_answer = data["answer"]
                if "reference" in data:
                    result["reference"] = data["reference"]
            elif not thinking and isinstance(data.get("answer"), str):
                deltas.append(data["answer"])
    result["answer"] = final_answer if final_answer is not None else "".join(deltas)
    return result


def patch_compose_source(source, expected_sha256):
    if hashlib.sha256(source.encode()).hexdigest() != expected_sha256:
        raise ValueError("compose_source_changed")
    read = '    empty_result = state.get("empty_result", False)\n'
    check = '    no_evidence = abstain or empty_result or not kbinfos["chunks"]\n'
    if source.count(read) != 1 or source.count(check) != 1:
        raise ValueError("compose_patch_pattern_changed")
    return source.replace(read, "").replace(check, '    no_evidence = abstain or not kbinfos["chunks"]\n')


def prepare_compose_patch(original, native_patch, *, integrity_manifest=None, action_module=None, installer=None):
    """Choose an explicit experiment condition without changing default behavior."""
    import inspect

    if not native_patch:
        if integrity_manifest is not None:
            raise ValueError("integrity_requires_base_patch")
        return original
    if native_patch["name"] != "compose_current_evidence_v1":
        raise ValueError("unknown_native_patch")
    source = patch_compose_source(inspect.getsource(original), native_patch["original_function_sha256"])
    if hashlib.sha256(source.encode()).hexdigest() != native_patch["patched_function_sha256"]:
        raise ValueError("patched_compose_source_changed")
    if integrity_manifest is not None:
        if integrity_manifest.get("name") != "evidence_integrity_v1":
            raise ValueError("unknown_integrity_patch")
        if action_module is None or not callable(installer):
            raise ValueError("integrity_runtime_dependency_missing")
        return installer(action_module, source, integrity_manifest, compose_globals=original.__globals__)
    namespace = dict(original.__globals__)
    exec(compile(source, "<isolated-compose-current-evidence-v1>", "exec"), namespace)
    return namespace["_compose_answer_from_evidence"]


async def execute(config):
    import asyncio
    import contextlib
    from contextvars import ContextVar
    import functools
    import inspect
    import io
    import logging
    import os
    from pathlib import Path
    import sys
    import time
    import traceback

    if config.get("worker_token"):
        if os.environ.get("BENCHMARK_RUN_TOKEN") != config["worker_token"]:
            raise ValueError("worker_environment_token_mismatch")
        if os.getpgid(0) != os.getpid():
            os.setsid()
    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    os.environ["LLM_MAX_RETRIES"] = "0"
    logging.disable(logging.CRITICAL)
    integrity = config.get("evidence_integrity_patch")
    if integrity is not None:
        source = config["sources"].get("native_agent_integrity_patch")
        if not isinstance(source, str) or hashlib.sha256(source.encode()).hexdigest() != integrity.get("module_sha256"):
            raise ValueError("integrity_module_source_changed")
    loaded = {}
    for name, source in config["sources"].items():
        ns = {"__name__": name}
        exec(compile(source, name + ".py", "exec"), ns)
        loaded[name] = ns
    globals()["_sanitize"] = loaded["native_outbound"]["sanitize_kbinfos"]
    Ledger = loaded["native_agent_guard"]["CallLedger"]
    validate_scope = loaded["native_agent_guard"]["validate_scope"]
    Monitor = loaded["native_agent_transport"]["ProviderMonitor"]
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        from api.apps import app
        from api.db.services import dialog_service
        from api.db.services.llm_service import LLMBundle
        from common import settings
        from rag.advanced_rag import agentic_rag, agentic_rag_graph
        from rag.advanced_rag.harness.tools import navigation
        from rag.advanced_rag.knowlege_compile import dataset_nav
        from rag.prompts import generator
        import httpx
        import litellm
        import pymysql
        import yaml

    for name, expected in config["runtime_code_hashes"].items():
        if hashlib.sha256((Path("/ragflow") / name).read_bytes()).hexdigest() != expected:
            raise ValueError("runtime_source_changed")
    dbconf = yaml.safe_load(Path("/ragflow/conf/service_conf.yaml").read_text())["mysql"]
    db = pymysql.connect(host=dbconf["host"], port=int(dbconf["port"]), user=dbconf["user"], password=dbconf["password"], database=dbconf["name"], autocommit=True)
    with db.cursor() as cur:
        cur.execute("SELECT tenant_id,tenant_llm_id,kb_ids,top_n,top_k,similarity_threshold,vector_similarity_weight,rerank_id,llm_setting,prompt_config,meta_data_filter FROM dialog WHERE id=%s", (config["chat_id"],))
        tenant, mid, kbs, top_n, top_k, threshold, weight, rerank, llm, prompt, metadata = cur.fetchone()
        if metadata and json.loads(metadata):
            raise ValueError("metadata_auto_not_allowed")
        cur.execute("SELECT model_name,extra FROM tenant_model WHERE id=%s", (mid,))
        model_name, extra = cur.fetchone()
        actual = {"model": model_name, "model_id": mid, "dataset_ids": sorted(json.loads(kbs)), "top_n": top_n, "top_k": top_k,
                  "similarity_threshold": float(threshold), "vector_similarity_weight": float(weight), "rerank_id": rerank,
                  "llm_setting": json.loads(llm), "prompt_config": json.loads(prompt), "model_context_budget": json.loads(extra)["max_tokens"]}
        validate_chat_config(config["chat_config"], actual)
        cur.execute("SELECT token FROM api_token WHERE tenant_id=%s AND (dialog_id IS NULL OR dialog_id='') ORDER BY create_time DESC LIMIT 1", (tenant,))
        secret = cur.fetchone()
        if not secret:
            raise ValueError("missing_api_token")
        token = secret[0]
    db.close()
    dialog_service.TenantLangfuseService.filter_by_tenant = lambda **kwargs: []
    allowed = set(config["scope_document_ids"])
    stage = ContextVar("native_agent_eval_stage", default="outer_router")
    embedding_purpose = ContextVar("native_agent_embedding_purpose", default="")
    current = {}
    limit = config["limits"]
    ledger = Ledger(config["container_ledger"], run_limit=limit["run_tokens"], question_limit=limit["question_tokens"], call_limit=limit["question_calls"])

    def emit(row):
        with ledger.lock:
            print("AGENT_RESULT:" + json.dumps(row, ensure_ascii=False), flush=True)

    def event(row):
        row = {"question_id": current.get("question_id"), "stage": stage.get(), **row}
        current.setdefault("events", []).append(row)
        emit(row)

    class ErrorCapture(logging.Handler):
        def emit(self, record):
            if record.exc_info:
                event({"event": "native_exception", "exception_type": record.exc_info[0].__name__,
                       "frames": [{"file": f.filename, "line": f.lineno, "function": f.name} for f in traceback.extract_tb(record.exc_info[2])]})

    logging.disable(logging.NOTSET)
    for logger in [logging.getLogger(), *(v for v in logging.Logger.manager.loggerDict.values() if isinstance(v, logging.Logger))]:
        logger.handlers.clear()
        logger.setLevel(logging.ERROR)
        logger.propagate = True
    logging.getLogger().addHandler(ErrorCapture(level=logging.ERROR))

    def fail(reason):
        ledger.abort(reason)
        raise ValueError(reason)

    def gate(result, origin):
        ledger.assert_open()
        try:
            safe = safe_result(result, review=config["review"]["entries"], document_map=config["document_map"],
                               title_review=config.get("document_title_review", {}).get("entries"))
        except Exception as exc:
            current.setdefault("blocked_results", []).append({"origin": origin, "result": result})
            fail(str(exc) if isinstance(exc, ValueError) else "malformed_evidence_gate_error")
        for chunk in safe.get("chunks", []):
            if chunk.get("doc_id") not in allowed:
                fail("returned_document_out_of_scope")
            current.setdefault("seen_chunks", {})[chunk.get("chunk_id", chunk.get("id"))] = copy.deepcopy(chunk)
        event({"event": "evidence", "origin": origin, "chunk_ids": [c.get("chunk_id", c.get("id")) for c in safe.get("chunks", [])]})
        return safe

    original_retrieval = settings.retriever.retrieval
    retrieval_signature = inspect.signature(original_retrieval)

    @functools.wraps(original_retrieval)
    async def retrieval(*args, **kwargs):
        ledger.assert_open()
        params = retrieval_signature.bind(*args, **kwargs).arguments
        try:
            validate_scope(params.get("doc_ids"), allowed)
        except ValueError:
            fail("retrieval_scope_error")
        event({"event": "retrieval_request", "query": params.get("question"),
               "page_size": params.get("page_size"),
               "vector_similarity_weight": params.get("vector_similarity_weight"),
               "scope_documents": len(params.get("doc_ids") or [])})
        result = await original_retrieval(*args, **kwargs)
        return gate(result, "retrieval")

    original_children = settings.retriever.retrieval_by_children

    @functools.wraps(original_children)
    def children(*args, **kwargs):
        return gate({"chunks": original_children(*args, **kwargs)}, "children")["chunks"]

    original_chunks = settings.retriever.chunk_list
    chunks_signature = inspect.signature(original_chunks)

    @functools.wraps(original_chunks)
    def chunk_list(*args, **kwargs):
        ledger.assert_open()
        params = chunks_signature.bind(*args, **kwargs).arguments
        if params.get("doc_id") not in allowed:
            fail("full_document_scope_error")
        return gate({"chunks": original_chunks(*args, **kwargs)}, "chunk_list")["chunks"]

    settings.retriever.retrieval = retrieval
    settings.retriever.retrieval_by_children = children
    settings.retriever.chunk_list = chunk_list

    original_fields = settings.docStoreConn.get_fields

    @functools.wraps(original_fields)
    def get_fields(*args, **kwargs):
        rows = original_fields(*args, **kwargs)
        caller = sys._getframe(1).f_globals.get("__name__", "")
        if caller.startswith("rag.advanced_rag"):
            try:
                return safe_direct_rows(rows or {}, review=config["review"]["entries"], document_map=config["document_map"])
            except Exception:
                current.setdefault("blocked_results", []).append({"origin": "direct_store", "result": rows})
                fail("unreviewed_direct_store_row")
        return rows

    settings.docStoreConn.get_fields = get_fields
    original_get = settings.docStoreConn.get

    @functools.wraps(original_get)
    def store_get(*args, **kwargs):
        ledger.assert_open()
        result = original_get(*args, **kwargs)
        if not result:
            return result
        cid = args[0] if args else kwargs.get("chunk_id")
        try:
            return safe_direct_rows({cid: result}, review=config["review"]["entries"], document_map=config["document_map"])[cid]
        except Exception:
            current.setdefault("blocked_results", []).append({"origin": "store_get", "result": {cid: result}})
            fail("unreviewed_direct_store_get")

    settings.docStoreConn.get = store_get
    original_nav_select = navigation._ask_nav_select

    @functools.wraps(original_nav_select)
    async def nav_select(tools, query, items, *args, **kwargs):
        if items:
            fail("unreviewed_navigation_metadata")
        return await original_nav_select(tools, query, items, *args, **kwargs)

    navigation._ask_nav_select = nav_select

    def wrap_navigation_query(original):
        signature = inspect.signature(original)

        @functools.wraps(original)
        async def scoped_query(*args, **kwargs):
            params = signature.bind(*args, **kwargs).arguments
            try:
                validate_scope(params.get("doc_scope"), allowed)
            except ValueError:
                fail("navigation_query_scope_error")
            handle = embedding_purpose.set("navigation_query")
            try:
                return await original(*args, **kwargs)
            finally:
                embedding_purpose.reset(handle)
        return scoped_query

    for name in ("search_nav_tree_descent", "search_dataset_nav"):
        original = getattr(dataset_nav, name)
        wrapped = wrap_navigation_query(original)
        setattr(dataset_nav, name, wrapped)
        # Cover aliases imported before this isolated hook was installed.
        for module in list(sys.modules.values()):
            if getattr(module, "__name__", "").startswith(("api.apps.services", "rag.advanced_rag")) and getattr(module, name, None) is original:
                setattr(module, name, wrapped)

    original_models = dialog_service.get_models
    if config.get("runtime_model_override"):
        original_resolver = dialog_service.get_model_config_by_id

        @functools.wraps(original_resolver)
        def resolver(*args, **kwargs):
            resolved = original_resolver(*args, **kwargs)
            if resolved.get("llm_name") == config["chat_config"]["model"]:
                return runtime_model_config(resolved, config["model"])
            return resolved

        dialog_service.get_model_config_by_id = resolver

    @functools.wraps(original_models)
    def get_models(*args, **kwargs):
        models = original_models(*args, **kwargs)
        chat = models[3]
        if chat.model_config["llm_name"] != config["model"]:
            fail("chat_model_changed")
        if type(chat.mdl).__name__ != "LiteLLMBase" or chat.model_config.get("llm_factory") != config["provider_config"]["factory"] or chat.model_config.get("api_base", "") != config["provider_config"]["api_base"]:
            fail("provider_adapter_or_endpoint_changed")
        chat.model_config["is_tools"] = True
        chat.is_tools = True
        chat.mdl.is_tools = True
        current["model_adapter"] = type(chat.mdl).__name__
        return models

    dialog_service.get_models = get_models
    original_init = agentic_rag.RAGTools.__init__

    @functools.wraps(original_init)
    def tools_init(self, *args, **kwargs):
        if set(kwargs.get("doc_scope") or []) != allowed or kwargs.get("web_search") is not None:
            fail("agent_initial_scope_or_web_error")
        original_init(self, *args, **kwargs)
        if self.sql_kbs:
            fail("sql_source_not_reviewed")
        current["agent_initialized"] = True
        current["thinking_mode"] = kwargs.get("thinking_mode")

    agentic_rag.RAGTools.__init__ = tools_init
    original_rag = agentic_rag.RAGTools.rag

    @functools.wraps(original_rag)
    async def rag(self, *args, **kwargs):
        current["agent_graph_called"] = True
        handle = stage.set("agent_graph")
        try:
            return await original_rag(self, *args, **kwargs)
        finally:
            stage.reset(handle)

    agentic_rag.RAGTools.rag = rag
    original_compose = agentic_rag_graph._compose_answer_from_evidence
    patch = config.get("native_patch")
    action_module = None
    if integrity is not None:
        from rag.advanced_rag.harness import action_session
        action_module = action_session
    original_compose = prepare_compose_patch(
        original_compose, patch, integrity_manifest=integrity, action_module=action_module,
        installer=loaded.get("native_agent_integrity_patch", {}).get("install_integrity_patch"),
    )

    @functools.wraps(original_compose)
    async def compose(*args, **kwargs):
        state = args[0] if args else kwargs["state"]
        evidence = state.get("kbinfos") or {}
        current["compose_input"] = copy.deepcopy({key: state[key] for key in ("question", "kbinfos", "empty_result", "abstain", "partial_answer", "slot_evidence") if key in state})
        event({"event": "compose_state", "empty_result": state.get("empty_result"), "abstain": state.get("abstain"),
               "partial_answer": state.get("partial_answer"), "chunk_ids": [c.get("chunk_id", c.get("id")) for c in evidence.get("chunks", [])],
               "pre_summary_chars": len(evidence.get("pre_summary") or ""), "native_patch": patch["name"] if patch else None,
               "evidence_integrity_patch": integrity["name"] if integrity is not None else None})
        handle = stage.set("final_answer")
        try:
            return await original_compose(*args, **kwargs)
        finally:
            stage.reset(handle)

    agentic_rag_graph._compose_answer_from_evidence = compose
    from rag.advanced_rag.harness.orchestrator import sufficient_context
    sufficient_context.sufficient_context_agent = trace_sca(sufficient_context.sufficient_context_agent, event, stage)
    # Cold per-experiment JSON cache; production Redis cache is untouched.
    generator.get_llm_cache = lambda *args, **kwargs: None
    generator.set_llm_cache = lambda *args, **kwargs: None
    original_encode_queries = LLMBundle.encode_queries
    original_encode = LLMBundle.encode

    def encode_queries(self, query):
        with ledger.lock:
            ledger.assert_open()
            current["query_embedding_calls"] = current.get("query_embedding_calls", 0) + 1
            if current["query_embedding_calls"] > limit["query_embedding_calls"]:
                fail("embedding_call_limit")
        result = original_encode_queries(self, query)
        event({"event": "runtime_embedding", "purpose": "encode_queries", "texts": 1, "tokens": result[1],
               "query_sha256": hashlib.sha256(str(query).encode()).hexdigest()})
        return result

    def encode(self, texts):
        ledger.assert_open()
        caller = sys._getframe(1)
        module, function = caller.f_globals.get("__name__", ""), caller.f_code.co_name
        if embedding_purpose.get() == "navigation_query":
            module, function = "navigation_query", "scoped_query"
        try:
            validate_runtime_embedding(module, function, texts)
        except ValueError as exc:
            event({"event": "blocked_embedding", "caller_module": module, "caller_function": function})
            fail(str(exc))
        with ledger.lock:
            key = "query_embedding_calls" if module == "navigation_query" else "citation_embedding_batches"
            current[key] = current.get(key, 0) + 1
            if current[key] > (limit["query_embedding_calls"] if module == "navigation_query" else 2):
                fail("embedding_call_limit")
        result = original_encode(self, texts)
        event({"event": "runtime_embedding", "purpose": module, "texts": len(texts), "tokens": result[1]})
        return result

    LLMBundle.encode_queries = encode_queries
    LLMBundle.encode = encode
    original_kb_prompt = dialog_service.kb_prompt

    @functools.wraps(original_kb_prompt)
    def kb_prompt(kbinfos, *args, **kwargs):
        safe = gate(kbinfos, "kb_prompt")
        current["ordinary_context_reviewed"] = True
        current["ordinary_context"] = copy.deepcopy(safe)
        return original_kb_prompt(safe, *args, **kwargs)

    dialog_service.kb_prompt = kb_prompt
    original_completion = litellm.acompletion

    @functools.wraps(original_completion)
    async def completion(*args, **kwargs):
        ledger.assert_open()
        return await original_completion(*args, **completion_options(kwargs, config))

    litellm.acompletion = completion

    def before_send(body):
        ledger.assert_open()
        if not generation_context_valid(config.get("reasoning", 2), current):
            fail("native_generation_context_missing")
        if "model_thinking" in config and body.get("enable_thinking") is not config["model_thinking"]:
            fail("provider_thinking_configuration_mismatch")
        if config.get("dry_run"):
            event({"event": "local_preflight", "provider_model": body.get("model"), "stream": body.get("stream"), "max_tokens": body.get("max_tokens"), "enable_thinking": body.get("enable_thinking"), "tool_names": [t.get("function", {}).get("name") for t in body.get("tools", [])]})
            fail("local_preflight_no_provider_call")
        if stage.get() == "final_answer" or config.get("reasoning", 2) == 0:
            current["final_request"] = copy.deepcopy(body)

    monitor = Monitor(ledger, model=config["model"], hosts=config["provider_hosts"], before_send=before_send, event=event)
    httpx.AsyncClient._send_single_request = monitor.wrap_send(httpx.AsyncClient._send_single_request)
    emit({"event": "ready", "model": config["model"], "scope_documents": len(allowed), "ledger": ledger.summary(), "worker_token": config.get("worker_token"), "worker_pid": os.getpid()})
    client = app.test_client()
    for line in sys.stdin:
        item = json.loads(line)
        if item["question_id"] not in config["question_ids"]:
            raise ValueError("unplanned_question")
        current.clear()
        current["question_id"] = item["question_id"]
        ledger.start(item["question_id"])
        started = time.monotonic()
        result = {}
        try:
            async def request():
                response = await client.post("/api/v1/chat/completions", json=make_payload(config, item), headers={"Authorization": "Bearer " + token})
                raw = (await response.get_data()).decode()
                current["native_sse"] = raw
                return response.status_code, collect_sse([raw])
            http_status, result = await asyncio.wait_for(request(), timeout=float(config.get("question_timeout_seconds", 420)))
            status = "ok" if http_status == 200 and result["stream_complete"] and not result["error"] and (config.get("reasoning", 2) == 0 or current.get("agent_graph_called")) and result["answer"].strip() else "native_route_error"
        except TimeoutError:
            ledger.abort("question_wall_time_limit")
            status = "timeout"
        except Exception as exc:
            status = "worker_error"
            current["error_type"] = type(exc).__name__
            current["error"] = str(exc)[:180]
        fatal = ledger.summary()["fatal_reason"]
        if fatal:
            status = "timeout" if fatal == "question_wall_time_limit" else ("budget_censored" if "budget" in fatal or "call_limit" in fatal else "guard_stopped")
        final_evidence = map_final_evidence(current.get("final_request", {}), current.get("seen_chunks", {}))
        row = {**copy.deepcopy(current), **result, "status": status, "question": item["question"], "latency_ms": round((time.monotonic() - started) * 1000),
               "calls": [copy.deepcopy(c) for c in ledger.calls.values() if c["question_id"] == item["question_id"]],
               "final_evidence": final_evidence, "ledger": ledger.summary()}
        emit({"event": "question_complete", "record": row})
        if status != "ok":
            break


if __name__ == "__main__":
    import asyncio
    import sys
    asyncio.run(execute(json.loads(sys.stdin.readline())))
