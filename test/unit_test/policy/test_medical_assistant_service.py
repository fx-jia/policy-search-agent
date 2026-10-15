"""Offline command ownership and durable at-most-once execution contracts."""

import asyncio
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[3]


def load_module():
    spec = importlib.util.spec_from_file_location("medical_assistant_offline", ROOT / "api/db/services/medical_assistant_service.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PolicyFixture:
    def __init__(self, output):
        self.output = Path(output)
        self.session_id = "s"
        self.kb_id = "k"
        self.calls = []
        self.clears = 0
        self.blocked = ""
        self.used = 20
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.release.set()
        self.records = []
        self.authorizations = 0

    async def authorize(self):
        self.authorizations += 1
        return {"id": "k", "tenant_id": "t"}

    async def read_state(self):
        return {"records": self.records.copy(), "pending": None, "used_tokens": self.used, "budget": 100, "blocked_reason": self.blocked, "source_ready": True}

    async def ask(self, question):
        self.calls.append(question)
        self.started.set()
        await self.release.wait()
        row = {
            "question_id": "q" + str(len(self.calls)),
            "result": {
                "question": question,
                "status": "ok",
                "answer": "依据。##0$$",
                "effective_question": question,
                "reference": {
                    "chunks": [
                        {
                            "chunk_id": "a",
                            "doc_id": "d",
                            "docnm_kwd": "政策",
                            "content_with_weight": "原文",
                            "document_metadata": {
                                "source_url": "https://example.test/policy",
                                "published_at": "2026-01-01",
                                "temporal": {"effective_from": "2026-02-01", "status": "unknown"},
                                "local_path": "/secret",
                            },
                        }
                    ]
                },
                "trace": [
                    {"event": "decision", "value": {"needs": [{"text": "范围", "status": "supported", "evidence_ids": ["s0"]}], "reason": "hidden reasoning"}, "views": [{"secret": True}]},
                    {"event": "provider", "body": "secret"},
                ],
                "stop_reason": "sufficient",
                "latency_ms": 10,
                "fallback_used": False,
                "error": "/secret key",
            },
            "release_id": "r",
            "as_of": "2026-10-12",
            "coverage": {"complete": False},
            "pending": [],
            "scanned_at": None,
            "applicability": "unknown",
            "stale": True,
            "calls": [{"api_key": "secret"}],
        }
        self.records.append(row)
        return row

    async def clear(self):
        self.clears += 1
        self.records.clear()


class MedicalAssistantTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.module = load_module()
        self.policy = PolicyFixture(self.tmp.name)
        self.host = self.module.MedicalAssistantService(self.policy, owner_user_id="owner", access_check=lambda u, k: True)
        self.addCleanup(lambda: self.host._owner_file.close())

    async def test_submit_calls_agent_once(self):
        result = await self.host.submit(str(uuid4()), "问题")
        self.assertEqual(result["state"], "completed")
        self.assertEqual(self.policy.calls, ["问题"])
        self.assertEqual(result["turn"]["citations"][0]["document_id"], "d")
        self.assertEqual(result["turn"]["citations"][0]["effective_from"], "2026-02-01")
        self.assertNotIn("secret", json.dumps(result))
        self.assertNotIn("reason", result["turn"]["trace"][0])
        self.assertEqual(len(result["turn"]["trace"]), 1)

    async def test_duplicate_id_returns_original_result(self):
        rid = str(uuid4())
        first = await self.host.submit(rid, "问题")
        self.assertEqual(await self.host.submit(rid, "问题"), first)
        self.assertEqual(self.policy.calls, ["问题"])

    async def test_same_id_different_body_rejected(self):
        rid = str(uuid4())
        await self.host.submit(rid, "问题")
        with self.assertRaises(ValueError):
            await self.host.submit(rid, "另一个问题")
        self.assertEqual(len(self.policy.calls), 1)

    async def test_concurrent_turn_rejected(self):
        self.policy.release.clear()
        rid = str(uuid4())
        caller = asyncio.create_task(self.host.submit(rid, "问题"))
        await self.policy.started.wait()
        self.assertEqual(await self.host.submit(rid, "问题"), {"state": "running", "request_id": rid})
        with self.assertRaises(ValueError):
            await self.host.submit(str(uuid4()), "另一个问题")
        self.policy.release.set()
        await caller
        self.assertEqual(self.policy.calls, ["问题"])

    async def test_foreign_user_rejected(self):
        with self.assertRaises(PermissionError):
            await self.host.authorize("foreign")
        self.assertEqual(self.policy.authorizations, 0)
        await self.host.authorize("owner")
        self.assertEqual(self.policy.authorizations, 1)

    async def test_clear_preserves_block_and_budget(self):
        self.policy.blocked = "unresolved_provider_usage"
        state = await self.host.clear(str(uuid4()))
        self.assertEqual(state["state"], "blocked")
        self.assertFalse(state["can_submit"])
        self.assertEqual(self.policy.used, 20)
        with self.assertRaises(ValueError):
            await self.host.submit(str(uuid4()), "问题")
        self.assertEqual(self.policy.calls, [])

    async def test_read_state_while_running(self):
        self.policy.release.clear()
        caller = asyncio.create_task(self.host.submit(str(uuid4()), "问题"))
        await self.policy.started.wait()
        self.policy.blocked = "unresolved_provider_usage"  # active reservation is not an idle terminal gate
        state = await asyncio.wait_for(self.host.snapshot(), 0.1)
        self.assertEqual(state["state"], "running")
        self.assertIsNone(state["blocked_reason"])
        with self.assertRaises(ValueError):
            await self.host.clear(str(uuid4()))
        self.policy.release.set()
        await caller
        self.assertEqual((await self.host.snapshot())["state"], "blocked")

    async def test_disconnected_caller_keeps_server_execution(self):
        self.policy.release.clear()
        rid = str(uuid4())
        caller = asyncio.create_task(self.host.submit(rid, "问题"))
        await self.policy.started.wait()
        caller.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await caller
        self.policy.release.set()
        await self.host._task
        self.assertEqual((await self.host.submit(rid, "问题"))["state"], "completed")
        self.assertEqual(len(self.policy.calls), 1)

    async def test_restart_restores_result_and_unconfirmed_command(self):
        rid = str(uuid4())
        result = await self.host.submit(rid, "问题")
        missing = str(uuid4())
        with (Path(self.tmp.name) / "web_commands.jsonl").open("a") as handle:
            handle.write(json.dumps({"event": "accepted", "request_id": missing, "kind": "turn", "question": "失联问题"}) + "\n")
        self.host._owner_file.close()
        fresh = self.module.MedicalAssistantService(self.policy, owner_user_id="owner", access_check=lambda u, k: True)
        self.addCleanup(fresh._owner_file.close)
        self.assertEqual((await fresh.snapshot())["state"], "unconfirmed")
        self.assertEqual(await fresh.submit(rid, "问题"), result)
        self.assertEqual(await fresh.submit(missing, "失联问题"), {"state": "unconfirmed", "request_id": missing})
        self.assertEqual(len(self.policy.calls), 1)

    async def test_restart_unconfirmed_clear_returns_full_session_without_repeating(self):
        rid = str(uuid4())
        with (Path(self.tmp.name) / "web_commands.jsonl").open("a") as handle:
            handle.write(json.dumps({"event": "accepted", "request_id": rid, "kind": "clear", "question": None}) + "\n")
        self.host._owner_file.close()
        fresh = self.module.MedicalAssistantService(self.policy, owner_user_id="owner", access_check=lambda u, k: True)
        self.addCleanup(fresh._owner_file.close)
        result = await fresh.clear(rid)
        self.assertEqual(result.get("session_id"), "s")
        self.assertEqual(result["state"], "unconfirmed")
        self.assertEqual(result["current_request_id"], rid)
        self.assertFalse(result["can_submit"])
        self.assertEqual(self.policy.clears, 0)

    async def test_old_clear_does_not_clear_later_turn(self):
        clear_id = str(uuid4())
        await self.host.clear(clear_id)
        await self.host.submit(str(uuid4()), "新问题")
        await self.host.clear(clear_id)
        self.assertEqual(len((await self.host.snapshot())["turns"]), 1)
        self.assertEqual(self.policy.clears, 1)

    async def test_second_owner_cannot_open_same_session(self):
        with self.assertRaises(ValueError):
            self.module.MedicalAssistantService(self.policy, owner_user_id="owner", access_check=lambda u, k: True)

    async def test_malformed_command_rejected_before_execution(self):
        for rid, text in [("not-uuid", "问题"), (str(uuid4()), " "), (str(uuid4()), "x" * 10001)]:
            with self.assertRaises(ValueError):
                await self.host.submit(rid, text)
        self.assertEqual(self.policy.calls, [])


class PolicyReadStateTests(unittest.IsolatedAsyncioTestCase):
    async def test_read_state_retains_real_records_and_usage_after_clear(self):
        asyncio.get_running_loop().slow_callback_duration = 1
        from test.unit_test.policy.test_policy_search_service import load_service, ProviderFixture

        with tempfile.TemporaryDirectory() as directory:
            fixture = ProviderFixture()
            service = load_service()("t", "k", "s", directory, publication=fixture, kb_reader=fixture.kb_reader, request_factory=fixture.factory)
            await service.ask("费用？", as_of="2026-10-10")
            state = await service.read_state()
            self.assertEqual(state["used_tokens"], 80)
            self.assertEqual(len(state["records"]), 1)
            self.assertTrue(state["source_ready"])
            await service.clear()
            state = await service.read_state()
            self.assertEqual(state["records"], [])
            self.assertEqual(state["used_tokens"], 80)


class DeploymentBindingTests(unittest.IsolatedAsyncioTestCase):
    async def test_missing_config_never_creates_a_session(self):
        from unittest.mock import patch

        module = load_module()
        with patch.dict("os.environ", {}, clear=True):
            self.assertIsNone(await module.get_medical_assistant_service())
        self.assertIsNone(module._host)

    async def test_config_requires_existing_exact_session_manifest(self):
        from unittest.mock import patch

        module = load_module()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "missing-session"
            config = {"owner_user_id": "owner", "tenant_id": "t", "kb_id": "k", "session_id": "s", "output_dir": str(output), "budget": 100}
            path = root / "config.json"
            path.write_text(json.dumps(config))
            with patch.dict("os.environ", {"MEDICAL_ASSISTANT_CONFIG": str(path)}):
                with self.assertRaises(ValueError):
                    await module.get_medical_assistant_service()
                self.assertFalse(output.exists())
                output.mkdir()
                (output / "session.json").write_text(json.dumps({"tenant_id": "t", "kb_id": "k", "session_id": "other", "budget": 100}))
                with self.assertRaises(ValueError):
                    await module.get_medical_assistant_service()
                config["budget"] = True
                path.write_text(json.dumps(config))
                with self.assertRaises(ValueError):
                    await module.get_medical_assistant_service()
                config["budget"] = 100
                config["extra"] = "unsupported"
                path.write_text(json.dumps(config))
                with self.assertRaises(ValueError):
                    await module.get_medical_assistant_service()
            self.assertFalse((output / "web_commands.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
