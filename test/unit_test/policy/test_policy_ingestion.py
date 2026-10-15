"""The native parser must bind the model frozen by policy admission."""

import ast
import asyncio
import importlib.util
import logging
import sys
from pathlib import Path
from functools import partial
from types import ModuleType, SimpleNamespace
from typing import Optional
from unittest.mock import patch

from test.unit_test.api.db.services.test_policy_catalog import CatalogFixture, capture


def load():
    path = Path(__file__).resolve().parents[3] / "api/db/services/policy_ingestion_service.py"
    spec = importlib.util.spec_from_file_location("policy_ingestion_unit", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class IngestionBindingTests(CatalogFixture):
    def setUp(self):
        super().setUp()
        self.model_config = {"llm_factory": "OpenAI", "llm_name": "embed", "api_base": "https://provider.test/v1", "max_tokens": 8192}
        row = self.observe(capture())
        self.catalog.bind_documents(self.sid, row["revision_id"], [{
            "doc_id": "doc", "role": "body", "sha256": row["assets"][0]["sha256"],
            "parse_build_key": "build-1", "embedding_model": self.model_config,
            "embd_id": "embed@OpenAI", "tenant_embd_id": "model-id",
            "parser_id": "naive", "parser_config": {"chunk_token_num": 128, "raptor": {"use_raptor": False}},
        }])
        self.metadata = {"scope_id": self.sid, "revision_id": row["revision_id"], "parse_build_key": "build-1"}
        self.task = {"source_type": "shanghai_medical/connector", "tenant_id": "tenant", "kb_id": "kb", "doc_id": "doc", "embd_id": "embed@OpenAI", "tenant_embd_id": "model-id", "parser_id": "naive", "parser_config": {"chunk_token_num": 128}}

    def validate(self, config=None, task=None):
        from test.unit_test.policy.test_policy_sync import load as load_sync

        with patch.dict(sys.modules, {"api.db.services.policy_sync_service": load_sync()}):
            return load().validate_policy_parse_task(task or self.task, config or self.model_config, catalog=self.catalog, metadata_reader=lambda _: self.metadata)

    def test_actual_model_config_matches_frozen_binding_credentials_excluded(self):
        self.validate(dict(self.model_config, api_key="never-persist-this"))

    def test_changed_model_or_endpoint_is_rejected(self):
        for field, value in (("llm_name", "different"), ("api_base", "https://elsewhere.test/v1")):
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "embedding"):
                self.validate(dict(self.model_config, **{field: value}))

    def test_foreign_tenant_or_unbound_document_is_rejected(self):
        for field, value in (("tenant_id", "intruder"), ("doc_id", "unbound"), ("source_type", "shanghai_medical/other-connector"), ("source_type", "shanghai_medical")):
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.validate(task=dict(self.task, **{field: value}))

    def test_changed_reference_or_missing_metadata_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "embedding"):
            self.validate(task=dict(self.task, tenant_embd_id="other-model"))
        self.metadata = {}
        with self.assertRaisesRegex(ValueError, "binding"):
            self.validate()

    def test_changed_parser_is_rejected_with_native_postprocessing_fields_omitted(self):
        for field, value in (("parser_id", "paper"), ("parser_config", {"chunk_token_num": 999})):
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "parser"):
                self.validate(task=dict(self.task, **{field: value}))

    def test_other_native_sources_do_not_access_policy_catalog(self):
        load().validate_policy_parse_task({"source_type": "local/upload"}, {}, metadata_reader=lambda _: self.fail("unrelated source accessed policy metadata"))

    def test_default_metadata_reader_uses_strict_failure_path(self):
        from test.unit_test.policy.test_policy_sync import load as load_sync

        def read(doc_id, *, strict=False):
            if strict:
                raise ConnectionError("metadata read unavailable")
            return self.metadata

        module = ModuleType("api.db.services.doc_metadata_service")
        module.DocMetadataService = SimpleNamespace(get_document_metadata=read)
        with patch.dict(sys.modules, {"api.db.services.policy_sync_service": load_sync(), "api.db.services.doc_metadata_service": module}):
            with self.assertRaisesRegex(ConnectionError, "read unavailable"):
                load().validate_policy_parse_task(self.task, self.model_config, catalog=self.catalog)

    def test_both_native_executors_check_binding_before_model_creation(self):
        root = Path(__file__).resolve().parents[3]
        task = dict(self.task, id="task", from_page=0, to_page=1, task_type="", language="Chinese", parser_config={}, kb_parser_config={}, llm_id="chat", name="policy.txt", parser_id="naive")
        for file, name in (("rag/svr/task_executor.py", "do_handle_task"), ("rag/svr/task_executor_refactor/task_handler.py", "_bind_embedding_model")):
            with self.subTest(executor=file):
                guards = []

                def reject(actual_task, actual_model):
                    guards.append((actual_task, actual_model))
                    raise ValueError("policy embedding configuration changed after admission")

                async def thread(function, *args):
                    return function(*args)

                tree = ast.parse((root / file).read_text())
                function = next(node for node in ast.walk(tree) if isinstance(node, ast.AsyncFunctionDef) and node.name == name)
                function.decorator_list = []
                namespace = dict(
                    Optional=Optional, partial=partial, logging=logging,
                    thread_pool_exec=thread, resolve_model_config=lambda *args: self.model_config,
                    get_model_config_by_id=lambda *args: self.model_config,
                    _embedding_config_has_missing_credentials=lambda config: False,
                    LLMType=SimpleNamespace(EMBEDDING="embedding"),
                    LLMBundle=lambda *args, **kwargs: self.fail("model constructed before build rejection"),
                    has_canceled=lambda task_id: False, set_progress=lambda *args, **kwargs: None,
                    timer=lambda: 0, rag_tokenizer=SimpleNamespace(tokenizer=SimpleNamespace(set_language=lambda _: None)),
                )
                exec(compile(ast.Module(body=[function], type_ignores=[]), str(root / file), "exec"), namespace)
                module = ModuleType("api.db.services.policy_ingestion_service")
                module.validate_policy_parse_task = reject
                context = SimpleNamespace(tenant_id="tenant", embd_id="embed@OpenAI", tenant_embd_id="model-id", language="Chinese", get=task.get, raw_task=task, progress_cb=lambda *args, **kwargs: None)
                with patch.dict(sys.modules, {"api.db.services.policy_ingestion_service": module}):
                    argument = task if name == "do_handle_task" else SimpleNamespace(_task_context=context)
                    with self.assertRaisesRegex(ValueError, "embedding configuration changed"):
                        asyncio.run(namespace[name](argument))
                self.assertEqual(guards, [(task, self.model_config)])
