"""Authorized, audited policy conversation service with request-local models."""
import asyncio
import copy
from dataclasses import asdict, replace
from datetime import date, datetime
import inspect
import json
import os
from pathlib import Path
import re
import time
from uuid import uuid4
from zoneinfo import ZoneInfo

from experiments.evaluation.native_agent_guard import CallLedger
from experiments.search_agent.conversation import Conversation, ConversationResult
from experiments.search_agent.core import AgentLimits
from experiments.search_agent.policy_backend import PolicyBackend
from experiments.search_agent.policy_transport import AuditedCalls, AsyncAuditTransport, SyncAuditTransport
from experiments.search_agent.resolution import Resolution, resolution_messages, validate_resolution


async def _await(value):
    return await value if inspect.isawaitable(value) else value


def _append(path, value):
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _model_proxy():
    proxy = os.environ.get("POLICY_MODEL_PROXY")
    if proxy is not None and proxy != "http://host.docker.internal:7897":
        raise ValueError("invalid_policy_model_proxy")
    return proxy


class _PublishedQueryEmbedding:
    def __init__(self, bundle, state):
        self._bundle, self._state = bundle, state

    def encode_queries(self, query):
        dimension = self._state.get("dimension")
        if type(dimension) is not int or dimension <= 0:
            raise ValueError("published_embedding_dimension_required")
        vector, tokens = self._bundle.encode_queries(query)
        if len(vector) != dimension or getattr(vector, "ndim", 1) != 1 or any(isinstance(value, (list, tuple)) for value in vector):
            raise ValueError("query_embedding_dimension_mismatch")
        return vector, tokens


class PolicySearchService:
    """One authorized session; every ask binds fresh models, audit and release.

    Optional dependency callbacks isolate the external database/model boundary.
    request_factory(kb, audit) returns resolve/chat/retrieve/count_tokens/kb_prompt
    and close callbacks. Production defaults lazily use existing RAGFlow services.
    Output directories belong to one exact tenant/KB/session/budget identity.
    """

    def __init__(self, tenant_id, kb_id, session_id, output_dir, budget=60000, *,
                 publication=None, kb_reader=None, request_factory=None):
        if any(not isinstance(value, str) or not value.strip() for value in (tenant_id, kb_id, session_id)):
            raise ValueError("authenticated_scope_and_session_required")
        if type(budget) is not int or not 0 < budget <= 60000:
            raise ValueError("invalid_session_budget")
        self.tenant_id, self.kb_id, self.session_id = tenant_id, kb_id, session_id
        self.budget = budget
        self.output = Path(output_dir)
        self.output.mkdir(parents=True, exist_ok=True)
        (self.output / "ledgers").mkdir(exist_ok=True)
        identity = {"tenant_id": tenant_id, "kb_id": kb_id, "session_id": session_id, "budget": budget}
        self._ledger_identity = identity
        manifest = self.output / "session.json"
        if manifest.exists():
            if json.loads(manifest.read_text()) != identity:
                raise ValueError("session_output_identity_mismatch")
        else:
            with manifest.open("x", encoding="utf-8") as handle:
                json.dump(identity, handle, ensure_ascii=False)
        self.conversation = Conversation(session_id)
        self.publication, self._kb_reader, self._request_factory = publication, kb_reader, request_factory
        self._lock = asyncio.Lock()
        self._blocked = ""
        self._unresolved_ledger_paths = set()
        self._required_ledger_paths = set()
        self._used = 0
        self._last_as_of = None
        records = self.output / "records.jsonl"
        if records.exists():
            lines = records.read_text().splitlines()
            if lines:
                for line in lines:
                    row = json.loads(line)
                    result, recorded_ledger = row.get("result") or {}, row.get("ledger") or {}
                    usage = result.get("usage") or {}
                    qid = row.get("question_id")
                    if qid is not None:
                        if not isinstance(qid, str) or not qid or Path(qid).name != qid:
                            raise ValueError("invalid_question_record")
                        self._required_ledger_paths.add(str((self.output / "ledgers" / (qid + ".jsonl")).resolve()))
                    if (not recorded_ledger and usage.get("unknown_usage") is True) or recorded_ledger.get("pending_calls") or recorded_ledger.get("fatal_reason") in {"provider_usage_unknown", "embedding_usage_unknown", "provider_usage_unresolved"}:
                        if not isinstance(qid, str) or not qid or Path(qid).name != qid:
                            raise ValueError("invalid_unresolved_question_record")
                        self._unresolved_ledger_paths.add(str((self.output / "ledgers" / (qid + ".jsonl")).resolve()))
                state = json.loads(lines[-1]).get("conversation", {})
                self.conversation._turns = state.get("turns", [])
                self.conversation._pending = state.get("pending")
                self.conversation._history_truncated = state.get("history_truncated", False)
                self._last_as_of = state.get("as_of")
        self._refresh_usage()

    def _refresh_usage(self):
        self._used = 0
        unresolved, reconciled, present = set(), set(), set()
        invalid = False
        for path in (self.output / "ledgers").glob("*.jsonl"):
            key = str(path.resolve())
            present.add(key)
            try:
                ledger = CallLedger(path, run_limit=self.budget, question_limit=60000, call_limit=7,
                                    identity=self._ledger_identity, artifact_root=self.output)
                summary = ledger.summary()
            except (OSError, ValueError, TypeError, KeyError, AttributeError):
                invalid = True
                unresolved.add(key)
                continue
            self._used += summary["provider_total_tokens"]
            if summary["pending_calls"] or summary["fatal_reason"] in {"provider_usage_unknown", "embedding_usage_unknown", "provider_usage_unresolved"}:
                unresolved.add(key)
                self._unresolved_ledger_paths.add(key)
            elif summary.get("preflight_not_sent_reconciliation"):
                reconciled.add(key)
        self._required_ledger_paths.update(present)
        blocked = unresolved | (self._unresolved_ledger_paths - reconciled) | (self._required_ledger_paths - present)
        self._blocked = "usage_reconciliation_invalid" if invalid else ("unresolved_provider_usage" if blocked else "")

    async def authorize(self):
        if self._kb_reader is None:
            def read():
                from api.db.db_models import DB
                from api.db.services.knowledgebase_service import KnowledgebaseService
                with DB.connection_context():
                    exists, value = KnowledgebaseService.get_by_id(self.kb_id)
                    return value.to_dict() if exists else None
            kb = await asyncio.to_thread(read)
        else:
            kb = await self._invoke(self._kb_reader, self.kb_id)
        if not isinstance(kb, dict) or kb.get("id") != self.kb_id:
            raise PermissionError("knowledgebase_not_found")
        if kb.get("tenant_id") != self.tenant_id:
            raise PermissionError("knowledgebase_tenant_mismatch")
        return copy.deepcopy(kb)

    async def read_state(self):
        """Read completed topic results and accounting at an idle owner boundary.

        The HTTP owner caches this value during ask; reading an active ledger
        reservation here would incorrectly classify a normal call as unresolved.
        This method performs publication reads only and never selects models.
        """
        async with self._lock:
            self._refresh_usage()
            records = []
            path = self.output / "records.jsonl"
            if path.exists():
                for line in path.read_text().splitlines():
                    row = json.loads(line)
                    if row.get("event") == "clear":
                        records = []
                    elif row.get("question_id") and isinstance(row.get("result"), dict):
                        records.append(row)
            point = self._last_as_of or datetime.now(ZoneInfo("Asia/Shanghai")).date().isoformat()
            source, source_ready = {}, False
            try:
                snapshot = await asyncio.wait_for(self._snapshot_loader(point), timeout=10)
                source = {key: copy.deepcopy(snapshot[key]) for key in
                          ("release_id", "coverage", "pending", "scanned_at") if key in snapshot}
                source["as_of"] = point
                source_ready = bool(snapshot.get("release_id") and snapshot.get("doc_ids"))
                scanned = snapshot.get("scanned_at")
                if scanned is not None:
                    try:
                        source["stale"] = (datetime.now(ZoneInfo("UTC")) - datetime.fromisoformat(scanned.replace("Z", "+00:00"))).total_seconds() > 48 * 3600
                    except (TypeError, ValueError):
                        source["stale"] = None
            except Exception:
                # Catalog/database failure does not imply zero coverage or usage.
                pass
            return copy.deepcopy({"records": records, "pending": self.conversation.pending,
                                  "used_tokens": self._used, "budget": self.budget,
                                  "blocked_reason": self._blocked or ("session_token_budget_exhausted" if self._used >= self.budget else None),
                                  "source_ready": source_ready, "source_state": source})

    async def clear(self):
        """Clear retained user context while preserving all usage and fatal gates."""
        async with self._lock:
            await self.conversation.clear()
            self._last_as_of = None
            _append(self.output / "records.jsonl", {"event": "clear", "conversation": {
                "turns": [], "pending": None, "history_truncated": False, "as_of": None}})

    @staticmethod
    async def _invoke(callback, *args):
        if inspect.iscoroutinefunction(callback):
            return await callback(*args)
        return await _await(await asyncio.to_thread(callback, *args))

    @staticmethod
    def _date_terms(text):
        chinese = list(re.finditer(r"(?<!\d)((?:19|20)\d{2})年\s*(\d{1,2})月\s*(\d{1,2})日", text))
        dates = {date(*map(int, match.groups())).isoformat() for match in chinese}
        dates.update(date.fromisoformat(value).isoformat() for value in re.findall(r"(?<!\d)((?:19|20)\d{2}-\d{2}-\d{2})(?!\d)", text))
        rest = re.sub(r"(?<!\d)(?:19|20)\d{2}年\s*\d{1,2}月\s*\d{1,2}日", "", text)
        return set(re.findall(r"(?<!\d)((?:19|20)\d{2})年", rest)), dates

    def _policy_date(self, question, as_of, *, include_pending=True):
        explicit = date.fromisoformat(as_of).isoformat() if as_of else None
        # Effective questions preserve chronology; an explicit current condition
        # overrides inherited user conditions rather than becoming a false clash.
        prior, separator, current = question.rpartition("\n用户本轮原话：")
        try:
            current_years, current_dates = self._date_terms(current if separator else question)
            if separator and not current_years and not current_dates:
                current_years, current_dates = self._date_terms(prior)
            if include_pending and not current_years and not current_dates and self.conversation.pending:
                pending = self.conversation.pending
                original = "\n".join([pending["question"], *pending["messages"]])
                current_years, current_dates = self._date_terms(original)
        except ValueError:
            return explicit or self._last_as_of, "问题中的日期无效，请明确有效的政策适用日期（YYYY-MM-DD）。"
        as_of = explicit or self._last_as_of
        if as_of and ((current_years and current_years != {as_of[:4]}) or (current_dates and current_dates != {as_of})):
            return as_of, "问题中的年份或日期与指定政策时点不一致，请明确政策适用日期（YYYY-MM-DD）。"
        if as_of is None and (current_years or current_dates):
            if len(current_dates) == 1 and not current_years:
                as_of = date.fromisoformat(next(iter(current_dates))).isoformat()
            else:
                return None, "请明确政策适用日期（YYYY-MM-DD），以便查询对应历史时点。"
        return as_of or datetime.now(ZoneInfo("Asia/Shanghai")).date().isoformat(), None

    async def _snapshot_loader(self, as_of):
        def read():
            if self.publication is None:
                from api.db.db_models import DB
                from api.db.services.policy_catalog_service import PolicyCatalogService
                from api.db.services.policy_publication_service import PolicyPublicationService
                with DB.connection_context():
                    return PolicyPublicationService(PolicyCatalogService()).acquire_snapshot(self.tenant_id, self.kb_id, as_of)
            catalog = getattr(self.publication, "catalog", None)
            if catalog is not None:
                with catalog.Scope._meta.database.connection_context():
                    return self.publication.acquire_snapshot(self.tenant_id, self.kb_id, as_of)
            return self.publication.acquire_snapshot(self.tenant_id, self.kb_id, as_of)
        if self.publication is not None and inspect.iscoroutinefunction(self.publication.acquire_snapshot):
            return await self.publication.acquire_snapshot(self.tenant_id, self.kb_id, as_of)
        return await asyncio.to_thread(read)

    async def ask(self, question, as_of=None):
        if not isinstance(question, str) or not question.strip():
            raise ValueError("question_required")
        async with self._lock:
            started = time.monotonic()
            kb = await asyncio.wait_for(self.authorize(), timeout=60)
            self._refresh_usage()
            if self._blocked:
                raise ValueError(self._blocked)
            if self._used >= self.budget:
                raise ValueError("session_token_budget_exhausted")
            as_of, clarification = self._policy_date(question, as_of)
            qid = uuid4().hex
            ledger = CallLedger(self.output / "ledgers" / (qid + ".jsonl"), run_limit=self.budget - self._used,
                                question_limit=min(60000, self.budget - self._used), call_limit=7,
                                identity=self._ledger_identity, artifact_root=self.output)
            ledger.start(qid)
            self._required_ledger_paths.add(str(ledger.path.resolve()))
            audit = AuditedCalls(ledger)
            backend, dependencies = None, None
            cancelled = False
            result = ConversationResult(session_id=self.session_id, question=question)
            try:
                if clarification:
                    result.status, result.stop_reason = "needs_clarification", "missing_user_information"
                    result.answer = clarification
                    async with self.conversation._lock:
                        self.conversation._clarify(question, clarification, False)
                else:
                    async with asyncio.timeout(max(.001, 60 - (time.monotonic() - started))):
                        factory = self._request_factory or self._build_request
                        dependencies = await _await(factory(kb, audit))
                        async def resolve(context):
                            value = await dependencies["resolve"](context)
                            resolution = value if isinstance(value, Resolution) else validate_resolution(value, context)
                            if resolution.kind == "question":
                                _, conflict = self._policy_date(resolution.question, as_of, include_pending=False)
                                if conflict:
                                    return Resolution("clarification", clarification=conflict, user_context=resolution.user_context, reset=resolution.reset)
                            return resolution
                        async def load_snapshot(point):
                            snapshot = await self._snapshot_loader(point)
                            if dependencies.get("validate_snapshot"):
                                await _await(dependencies["validate_snapshot"](snapshot))
                            return snapshot
                        backend = PolicyBackend(load_snapshot, dependencies["retrieve"], resolve,
                                                dependencies["chat"], dependencies["count_tokens"], audit=audit.audit,
                                                kb_prompt=dependencies["kb_prompt"], tenant_id=self.tenant_id, kb_id=self.kb_id,
                                                as_of=as_of, remaining_calls=audit.remaining_generation)
                        remaining = max(0.001, 60 - (time.monotonic() - started))
                        limits = replace(AgentLimits(), total_seconds=remaining, search_seconds=min(40, remaining))
                        result = await self.conversation.ask(question, backend, limits)
                        if result.status == "ok":
                            self._last_as_of = as_of
            except asyncio.CancelledError:
                cancelled = True
                result.status, result.stop_reason = "cancelled", "policy_request_cancelled"
            except Exception as exc:
                result.status, result.stop_reason = "technical_error", "policy_request_error"
                result.error = str(exc)[:200]
            finally:
                if dependencies and dependencies.get("close"):
                    try:
                        await asyncio.wait_for(_await(dependencies["close"]()), timeout=5)
                    except Exception as exc:
                        _append(self.output / "events.jsonl", {"question_id": qid, "event": "close_error", "error": str(exc)[:200]})
            summary = ledger.summary()
            if summary["pending_calls"]:
                ledger.abort("provider_usage_unknown")
                summary = ledger.summary()
            self._used += summary["provider_total_tokens"]
            if summary["pending_calls"] or summary["fatal_reason"] in {"provider_usage_unknown", "embedding_usage_unknown", "provider_usage_unresolved"}:
                self._blocked = "unresolved_provider_usage"
                self._unresolved_ledger_paths.add(str(ledger.path.resolve()))
            snapshot = (backend.snapshot if backend else None) or {}
            scanned = snapshot.get("scanned_at")
            try:
                stale = scanned is None or (datetime.now(ZoneInfo("UTC")) - datetime.fromisoformat(scanned.replace("Z", "+00:00"))).total_seconds() > 48 * 3600
            except (TypeError, ValueError):
                stale = True
            result.latency_ms = round((time.monotonic() - started) * 1000)
            value = {"question_id": qid, "result": asdict(result), "ledger": summary,
                     "calls": copy.deepcopy(list(ledger.calls.values())), "session_total_tokens": self._used,
                     "release_id": snapshot.get("release_id"), "coverage": snapshot.get("coverage", {}),
                     "pending": snapshot.get("pending", []), "scanned_at": scanned, "as_of": as_of,
                     "applicability": backend.applicability if backend else "unknown", "stale": stale,
                     "models": dependencies.get("models", {}) if dependencies else {}}
            state = {"turns": self.conversation.turns, "pending": self.conversation.pending,
                     "history_truncated": self.conversation._history_truncated, "as_of": self._last_as_of}
            _append(self.output / "records.jsonl", {**value, "conversation": state})
            for event in backend.events if backend else []:
                _append(self.output / "events.jsonl", {"question_id": qid, **event})
            if cancelled:
                raise asyncio.CancelledError()
            return value

    async def _build_request(self, kb, audit):
        """Create isolated real model bundles and wire audited native retrieval."""
        proxy = _model_proxy()
        import httpx
        from openai import AsyncOpenAI, OpenAI
        setup = asyncio.create_task(asyncio.to_thread(self._model_inputs, kb))
        try:
            inputs = await asyncio.shield(setup)
        except asyncio.CancelledError:
            async def close_late():
                try:
                    late = await setup
                    await self._close_models(*late[:2])
                except Exception as exc:
                    _append(self.output / "events.jsonl", {"event": "cancelled_model_setup_error", "error": str(exc)[:200]})
            asyncio.create_task(close_late())
            raise
        chat, embedding, chat_config, embedding_config, ChatBase, OpenAIEmbed, settings, num_tokens_from_string, kb_prompt, canonical_embedding_config, embedding_config_sha256 = inputs
        try:
            original = getattr(chat.mdl, "async_client", None)
            embed_original = getattr(embedding.mdl, "client", None)
            if (not isinstance(original, AsyncOpenAI) or not isinstance(embed_original, OpenAI)
                    or not isinstance(embedding.mdl, OpenAIEmbed)
                    or getattr(type(embedding.mdl), "_call") is not OpenAIEmbed._call
                    or getattr(type(chat.mdl), "async_chat") is not ChatBase.async_chat
                    or getattr(type(chat.mdl), "_async_chat") is not ChatBase._async_chat
                    or "qwq" in chat.mdl.model_name.lower()):
                raise ValueError("provider_without_auditable_raw_usage")
            chat.mdl.async_client = original.with_options(
                http_client=httpx.AsyncClient(transport=AsyncAuditTransport(httpx.AsyncHTTPTransport(proxy=proxy, retries=0, trust_env=False, verify=True), audit,
                                                                          "generation", str(original.base_url), chat.mdl.model_name), timeout=60, trust_env=False),
                max_retries=0)
            await original.close()
            embedding.mdl.client = embed_original.with_options(
                http_client=httpx.Client(transport=SyncAuditTransport(httpx.HTTPTransport(proxy=proxy, retries=0, trust_env=False, verify=True), audit,
                                                                    "embedding", str(embed_original.base_url), embedding.mdl.model_name), timeout=60, trust_env=False),
                max_retries=0)
            embed_original.close()
            chat.mdl.max_retries = 0
            chat.is_tools = False
        except BaseException:
            await self._close_models(chat, embedding)
            raise

        async def chat_call(system, messages, config):
            history = copy.deepcopy(messages)
            if system and history and history[0].get("role") != "system":
                history.insert(0, {"role": "system", "content": system})
            text, _ = await chat.mdl._async_chat(history, copy.deepcopy(config))
            return text

        async def resolve(context):
            system, messages = resolution_messages(context)
            if num_tokens_from_string(system + messages[0]["content"]) > 3000:
                raise ValueError("conversation_context_budget")
            text = await chat_call(system, messages, {"temperature": 0, "top_p": 1, "max_tokens": 600,
                                                     "response_format": {"type": "json_object"}})
            text = text.strip()
            if text.startswith("```json") and text.endswith("```"):
                text = text[7:-3].strip()
            return validate_resolution(json.loads(text), context)

        state = {}
        query_embedding = _PublishedQueryEmbedding(embedding, state)
        request_config = canonical_embedding_config(embedding_config)

        def validate_snapshot(snapshot):
            state["dimension"] = self._published_dimension(snapshot, request_config, canonical_embedding_config, embedding_config_sha256)

        async def retrieve(query, doc_ids):
            if not doc_ids:
                raise ValueError("empty_native_document_scope")
            if state.get("dimension") is None:
                raise ValueError("published_embedding_snapshot_not_validated")
            return await settings.retriever.retrieval(query, query_embedding, [self.tenant_id], [self.kb_id], 1, 8,
                                                     similarity_threshold=0.2, vector_similarity_weight=0.3,
                                                     doc_ids=doc_ids, knn_top_k=1024, aggs=True,
                                                     rerank_mdl=None, allow_dense_fallback=False)

        async def close():
            await self._close_models(chat, embedding)
        return {"chat": chat_call, "resolve": resolve, "retrieve": retrieve, "count_tokens": num_tokens_from_string,
                "kb_prompt": kb_prompt, "close": close, "validate_snapshot": validate_snapshot,
                "models": {"generation": chat_config["llm_name"], "embedding": embedding_config["llm_name"],
                           "embedding_ref": kb.get("tenant_embd_id") or kb["embd_id"]}}

    @staticmethod
    def _published_dimension(snapshot, request_config, canonical, fingerprint):
        request_hash = fingerprint(request_config)
        dimensions = set()
        for source in snapshot["sources"].values():
            published = source.get("embedding_config")
            if (not isinstance(published, dict) or canonical(published) != request_config
                    or source.get("embedding_config_sha256") != request_hash):
                raise ValueError("published_embedding_configuration_mismatch")
            dimension = source.get("embedding_dimension")
            if type(dimension) is not int or dimension <= 0:
                raise ValueError("published_embedding_dimension_required")
            dimensions.add(dimension)
        if len(dimensions) > 1:
            raise ValueError("published_embedding_dimension_mismatch")
        return next(iter(dimensions), None)

    def _model_inputs(self, kb):
        from api.db.db_models import DB
        from api.db.joint_services.tenant_model_service import get_tenant_default_model_by_type, get_model_config_by_id, resolve_model_config
        from api.db.services.llm_service import LLMBundle
        from api.db.services.policy_sync_service import canonical_embedding_config, embedding_config_sha256
        from common import settings
        from common.constants import LLMType
        from common.token_utils import num_tokens_from_string
        from rag.llm.chat_model import Base as ChatBase
        from rag.llm.embedding_model import OpenAIEmbed
        from rag.prompts.generator import kb_prompt

        with DB.connection_context():
            chat_ref = (kb.get("parser_config") or {}).get("llm_id")
            chat_config = (resolve_model_config(self.tenant_id, LLMType.CHAT, chat_ref) if chat_ref
                           else get_tenant_default_model_by_type(self.tenant_id, LLMType.CHAT))
            if kb.get("tenant_embd_id"):
                embedding_config = get_model_config_by_id(self.tenant_id, LLMType.EMBEDDING, kb["tenant_embd_id"])
            elif kb.get("embd_id"):
                embedding_config = resolve_model_config(self.tenant_id, LLMType.EMBEDDING, kb["embd_id"])
            else:
                raise ValueError("knowledgebase_embedding_model_required")
            chat = LLMBundle(self.tenant_id, chat_config, lang="Chinese", max_retries=0)
            try:
                embedding = LLMBundle(self.tenant_id, embedding_config, lang="Chinese", max_retries=0)
            except BaseException:
                asyncio.run(self._close_models(chat))
                raise
        return chat, embedding, chat_config, embedding_config, ChatBase, OpenAIEmbed, settings, num_tokens_from_string, kb_prompt, canonical_embedding_config, embedding_config_sha256

    @staticmethod
    async def _close_models(*models):
        for model in models:
            if model is None:
                continue
            async_client = getattr(model.mdl, "async_client", None)
            if async_client is not None:
                await async_client.close()
            client = getattr(model.mdl, "client", None)
            if client is not None:
                client.close()
            model.close()
