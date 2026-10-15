"""Exercise the real worker protocol and core with offline model/HTTP substitutes."""
import json
import io
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from experiments.evaluation.native_agent_guard import CallLedger
from experiments.search_agent import runtime
from experiments.search_agent.evidence import Evidence


class RequestTests(unittest.TestCase):
    def test_resolution_egress_rejects_changed_messages_json_mode_or_output_budget(self):
        current = {"approved_resolution_messages": [{"role": "user", "content": "审查过的输入"}]}
        body = {"messages": current["approved_resolution_messages"], "response_format": {"type": "json_object"}, "max_tokens": 600}
        runtime._check_resolution_request(body, current)
        for changed in ({**body, "messages": [{"role": "user", "content": "未经审查"}]},
                        {**body, "response_format": None}, {**body, "max_tokens": 1200}):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                runtime._check_resolution_request(changed, current)


class OfflineChat:
    def __init__(self, ledger):
        self.ledger = ledger

    async def async_chat(self, system, messages, options):
        data = json.loads(messages[0]["content"])
        cid = f"{self.ledger.question_id}:{len(self.ledger.calls) + 1}"
        self.ledger.reserve(cid, 100, {"offline_test": True})
        self.ledger.finish(cid, {"total_tokens": 10})
        if "current" in data:
            question = data["current"]["content"]
            if question == "医保条件？":
                result = {"kind": "clarification", "clarification": "请说明地区。", "user_context": [], "reset": False}
            elif question == "上海" and data["pending"]:
                result = {"kind": "question", "clarification": "", "user_context": [{"message_id": "pending-question", "quote": "医保条件"}], "reset": False}
            elif question == "退休人员呢？":
                result = {"kind": "question", "clarification": "",
                          "user_context": [{"message_id": data["history"][-2]["id"], "quote": "上海"}], "reset": False}
            else:
                result = {"kind": "question", "clarification": "", "user_context": [], "reset": False}
            return json.dumps(result, ensure_ascii=False)
        sid = data["evidence"][0]["segments"][0]["id"]
        return json.dumps({"constraints": [], "needs": [{"text": "参保条件", "status": "supported", "evidence_ids": [sid]}],
                           "action": {"type": "finish"}, "reason": "已有原文"}, ensure_ascii=False)


class OfflineBackend(runtime.RuntimeBackend):
    async def search(self, question, document_ids=None):
        item = Evidence("a", "d", "政策", "参保人不收费。", 0)
        self.observed[item.id] = item
        self.env["queries"].append(question)
        return [item]

    def neighbor_directions(self, seed):
        return []

    async def generate(self, question, selected, *, native=False):
        self.env["final_questions"].append(question)
        ledger = self.env["ledger"]
        cid = f"{ledger.question_id}:{len(ledger.calls) + 1}"
        ledger.reserve(cid, 100, {"offline_test": True})
        ledger.finish(cid, {"total_tokens": 10})
        return {"answer": "不收费。[ID:0]", "reference": {"chunks": [{"id": e.id} for e in selected]}}


class WorkerTests(unittest.IsolatedAsyncioTestCase):
    async def execute_lines(self, items, *, conversation=True):
        with tempfile.TemporaryDirectory() as temp:
            ledger = CallLedger(Path(temp) / "calls.jsonl", run_limit=10000, question_limit=60000, call_limit=4)
            config = {"agent_limits": {}, "question_ids": ["single"]}
            if conversation:
                config["conversation"] = {"max_rounds": 6, "history_tokens": 3000, "resolver_seconds": 10}
            env = {"ledger": ledger, "chat": OfflineChat(ledger), "count_tokens": len, "event": lambda e: None,
                   "queries": [], "final_questions": []}
            current, events = {}, []
            with patch.object(runtime, "RuntimeBackend", OfflineBackend):
                await runtime._serve(config, env, None, current, events.append, [json.dumps(i, ensure_ascii=False) for i in items])
            return events, env, ledger.summary()

    async def test_followup_receives_fresh_evidence_and_one_shared_accounting_ledger(self):
        events, env, summary = await self.execute_lines([{"session_id": "s", "question": "上海居民医保条件？"}, {"session_id": "s", "question": "退休人员呢？"}])
        rows = [e["record"] for e in events if e["event"] == "question_complete"]
        self.assertEqual([r["status"] for r in rows], ["ok", "ok"])
        self.assertNotEqual(rows[0]["question_id"], rows[1]["question_id"])
        self.assertIn("上海居民医保条件？", env["queries"][1])
        self.assertIn("退休人员呢？", env["queries"][1])
        self.assertIn("用户本轮原话：退休人员呢？", env["final_questions"][1])
        self.assertEqual([len(r["calls"]) for r in rows], [3, 3])
        self.assertEqual(summary["provider_total_tokens"], 60)

    async def test_clarification_then_short_reply_completes_the_original_request(self):
        events, env, _ = await self.execute_lines([{"session_id": "s", "question": "医保条件？"}, {"session_id": "s", "question": "上海"}])
        rows = [e["record"] for e in events if e["event"] == "question_complete"]
        self.assertEqual(rows[0]["status"], "needs_clarification")
        self.assertEqual(len(rows[0]["calls"]), 1)
        self.assertEqual(rows[1]["status"], "ok")
        self.assertIn("医保条件？", env["queries"][0])
        self.assertIn("上海", env["queries"][0])
        self.assertEqual(len(env["queries"]), 1)

    async def test_clear_and_independent_session_do_not_reuse_old_context(self):
        events, env, _ = await self.execute_lines([{"session_id": "s", "question": "上海医保条件？"}, {"command": "clear", "session_id": "s"},
                                                {"session_id": "s", "question": "杭州医保条件？"}, {"session_id": "other", "question": "医保条件？"}])
        self.assertIn({"event": "session_cleared", "session_id": "s"}, events)
        self.assertEqual([e["record"]["status"] for e in events if e["event"] == "question_complete"], ["ok", "ok", "needs_clarification"])
        self.assertEqual(env["queries"], ["上海医保条件？", "杭州医保条件？"])

    async def test_existing_single_question_worker_does_not_run_resolution(self):
        events, _, summary = await self.execute_lines([{"question_id": "single", "question": "上海医保条件？"}], conversation=False)
        row = events[0]["record"]
        self.assertEqual(row["status"], "ok")
        self.assertEqual(summary["provider_calls"], 2)

    async def test_unknown_embedding_usage_stops_before_the_next_turn(self):
        class UnknownEmbedding(OfflineBackend):
            async def search(self, question, document_ids=None):
                self.current["embedding_physical"].update(query_physical_calls=1, unknown_embedding_usage=True)
                raise ValueError("embedding_http_error:500")

        with tempfile.TemporaryDirectory() as temp:
            ledger = CallLedger(Path(temp) / "calls.jsonl", run_limit=10000, question_limit=60000, call_limit=4)
            config = {"agent_limits": {}, "conversation": {"max_rounds": 6, "history_tokens": 3000, "resolver_seconds": 10}}
            env = {"ledger": ledger, "chat": OfflineChat(ledger), "count_tokens": len, "event": lambda e: None}
            events = []
            items = [{"session_id": "s", "question": q} for q in ("first", "second")]
            with patch.object(runtime, "RuntimeBackend", UnknownEmbedding):
                await runtime._serve(config, env, None, {}, events.append, [json.dumps(i) for i in items])
            rows = [e["record"] for e in events if e["event"] == "question_complete"]
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["stop_reason"], "embedding_usage_unknown")
            self.assertTrue(rows[0]["embedding_physical"]["unknown_embedding_usage"])
            self.assertEqual(ledger.fatal_reason, "embedding_usage_unknown")


class CLIIntegrationTests(unittest.TestCase):
    def test_cli_and_real_worker_complete_clarification_then_followup(self):
        from experiments.search_agent.cli import run_interactive

        worker_source = """
import asyncio, json, sys
from pathlib import Path
from experiments.evaluation.native_agent_guard import CallLedger
from experiments.search_agent import runtime
from experiments.search_agent.test_conversation_runtime import OfflineBackend, OfflineChat
config = json.loads(sys.stdin.readline())
ledger = CallLedger(Path(config['container_ledger']), run_limit=60000, question_limit=60000, call_limit=4)
def emit(row):
    print('AGENT_RESULT:' + json.dumps(row, ensure_ascii=False), flush=True)
env = {'ledger': ledger, 'chat': OfflineChat(ledger), 'count_tokens': len,
       'event': lambda e: None, 'queries': [], 'final_questions': []}
runtime.RuntimeBackend = OfflineBackend
emit({'event': 'ready', 'worker_token': config['worker_token'], 'ledger': ledger.summary()})
asyncio.run(runtime._serve(config, env, None, {}, emit, sys.stdin))
"""
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            ledger_path = output / "remote.jsonl"
            processes = []
            config = {"worker_token": "offline-test", "container_ledger": str(ledger_path), "agent_limits": {},
                      "limits": {"run_tokens": 60000}, "conversation": {"max_rounds": 6, "history_tokens": 3000, "resolver_seconds": 10}}

            def spawn(command, **options):
                process = subprocess.Popen([sys.executable, "-u", "-c", worker_source], **options)
                processes.append(process)
                return process

            def stop(config):
                for process in processes:
                    if process.poll() is None:
                        process.kill()
                        process.wait(timeout=3)
                return {"terminated_pids": [], "ledger_events": [json.loads(line) for line in ledger_path.read_text().splitlines()]}

            stream = io.StringIO()
            status = run_interactive(config, output, input_stream=io.StringIO("医保条件？\n上海\n退休人员呢？\n/exit\n"),
                                     output_stream=stream, process_factory=spawn, stop_worker=stop)
            rows = [json.loads(line) for line in (output / "records.jsonl").read_text().splitlines()]
            self.assertEqual(status, 0)
            self.assertEqual([r["status"] for r in rows], ["needs_clarification", "ok", "ok"])
            self.assertIn("上海", rows[-1]["effective_question"])
            self.assertIn("退休人员呢？", rows[-1]["effective_question"])
            self.assertEqual(json.loads((output / "ledger_summary.json").read_text())["provider_total_tokens"], 70)
            self.assertIn("请说明地区", stream.getvalue())


if __name__ == "__main__":
    unittest.main()
