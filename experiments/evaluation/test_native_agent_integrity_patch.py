"""Exercise isolated changes against exported installed functions, without services."""

import asyncio
import copy
import hashlib
import importlib.util
import inspect
import json
import linecache
import logging
import re
import runpy
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[2]
FIXTURE = Path(__file__).with_name("fixtures") / "native_agent_installed_functions.json"


def digest(source):
    return hashlib.sha256(source.encode()).hexdigest()


@pytest.fixture
def api():
    path = Path(__file__).with_name("native_agent_integrity_patch.py")
    assert path.exists(), "integrity patch implementation missing"
    spec = importlib.util.spec_from_file_location("native_agent_integrity_patch", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def installed():
    rows = json.loads(FIXTURE.read_text())
    for row in rows.values():
        assert digest(row["source"]) == row["sha256"]
    return rows


def base_compose(installed):
    from experiments.evaluation.native_agent_worker import patch_compose_source

    row = installed["_compose_answer_from_evidence"]
    return patch_compose_source(row["source"], row["sha256"])


@pytest.mark.parametrize("name", ["admit", "list_chunks", "compose_registry"])
def test_changed_source_fingerprint_is_rejected(api, installed, name):
    original = {
        "admit": installed["_admit_evidence"]["source"],
        "list_chunks": installed["_exec_list_chunks"]["source"],
        "compose_registry": base_compose(installed),
    }[name]
    with pytest.raises(ValueError, match="source_changed"):
        getattr(api, f"patch_{name}_source")(original + "\n", digest(original))


@pytest.mark.parametrize(
    "name,anchor",
    [
        ("admit", "    if len(_pool) >= _EVIDENCE_POOL_CAP:\n"),
        ("list_chunks", "        if _admit_evidence(kbinfos, kb_seen, c, out, ids, seen, include_doc_id=False):\n"),
        ("compose_registry", '    return {"final_answer": ""}\n'),
    ],
)
@pytest.mark.parametrize("occurrences", [0, 2])
def test_missing_or_duplicate_anchors_are_rejected(api, installed, name, anchor, occurrences):
    original = {
        "admit": installed["_admit_evidence"]["source"],
        "list_chunks": installed["_exec_list_chunks"]["source"],
        "compose_registry": base_compose(installed),
    }[name]
    changed = original.replace(anchor, anchor * occurrences)
    with pytest.raises(ValueError, match="anchor_changed"):
        getattr(api, f"patch_{name}_source")(changed, digest(changed))


def test_registry_patch_requires_shared_empty_flag_fix(api, installed):
    row = installed["_compose_answer_from_evidence"]
    with pytest.raises(ValueError, match="compose_base_fix_required"):
        api.patch_compose_registry_source(row["source"], row["sha256"])


@pytest.fixture
def patched_actions(api, installed, monkeypatch):
    helpers = runpy.run_path(str(ROOT / "rag/advanced_rag/harness/chunk_utils.py"))
    search = ModuleType("rag.advanced_rag.harness.tools.search")
    search.__dict__.update({name: helpers[name] for name in ("_chunk_id", "_chunk_text", "_doc_id", "_dataset_id")})
    search._is_table_chunk = lambda chunk: False

    async def list_chunks(tools, doc_id):
        return {"chunks": tools.loaded}

    search.list_chunks = list_chunks
    monkeypatch.setitem(sys.modules, search.__name__, search)
    namespace = {
        "_LOG": logging.getLogger(__name__),
        "_EVIDENCE_POOL_CAP": 2,
        "_EVIDENCE_POOL_STATE": {"full_logged": False},
        "ToolOutcome": SimpleNamespace,
        "ERROR": "error",
        "_seed_evidence": lambda tools: tools.kbinfos,
        "_search_outcome": lambda out, ids, new: {"payload": out, "ids": ids, "new_evidence": new},
    }
    for name, patcher in [("_admit_evidence", api.patch_admit_source), ("_exec_list_chunks", api.patch_list_chunks_source)]:
        row = installed[name]
        exec(compile(patcher(row["source"], row["sha256"]), name, "exec"), namespace)
    return namespace


@pytest.mark.parametrize("at_capacity", [False, True])
def test_deep_read_upgrades_same_identity_without_changing_rank_or_original(patched_actions, at_capacity):
    original = {"chunk_id": "a", "doc_id": "d", "kb_id": "k", "content": "introduction", "similarity": 0.8}
    other = {"chunk_id": "b", "doc_id": "d", "content_with_weight": "other", "similarity": 0.4}
    pool = [original] + ([other] if at_capacity else [])
    full = {"id": "a", "doc_id": "d", "content_with_weight": "introduction; payment is 380.", "similarity": 0.1}
    tools = SimpleNamespace(kbinfos={"chunks": pool}, loaded=[full])
    result = asyncio.run(patched_actions["_exec_list_chunks"](tools, "d"))
    upgraded = tools.kbinfos["chunks"][0]
    assert upgraded["content_with_weight"] == "introduction; payment is 380."
    assert "content" not in upgraded
    assert upgraded["similarity"] == 0.8
    assert upgraded["kb_id"] == "k"
    assert upgraded["chunk_id"] == "a"
    assert len(pool) == (2 if at_capacity else 1)
    assert result["new_evidence"] == 1
    assert original["content"] == "introduction"
    assert asyncio.run(patched_actions["_exec_list_chunks"](tools, "d"))["new_evidence"] == 0


@pytest.mark.parametrize(
    "incoming",
    [
        {"id": "a", "doc_id": "other", "content_with_weight": "longer body from different document"},
        {"id": "a", "doc_id": "d", "kb_id": "other", "content_with_weight": "longer body from different dataset"},
        {"id": "a", "doc_id": "d", "content_with_weight": "short"},
    ],
)
def test_deep_read_does_not_downgrade_or_cross_source(patched_actions, incoming):
    original = {"chunk_id": "a", "doc_id": "d", "kb_id": "k", "content_with_weight": "known original text"}
    tools = SimpleNamespace(kbinfos={"chunks": [original]}, loaded=[incoming])
    result = asyncio.run(patched_actions["_exec_list_chunks"](tools, "d"))
    assert tools.kbinfos["chunks"] == [original]
    assert result["new_evidence"] == 0


def test_new_identity_still_respects_installed_pool_cap(patched_actions):
    original = [{"chunk_id": cid, "doc_id": "d", "content_with_weight": cid} for cid in ("a", "b")]
    tools = SimpleNamespace(kbinfos={"chunks": copy.deepcopy(original)}, loaded=[{"id": "c", "doc_id": "d", "content_with_weight": "full"}])
    result = asyncio.run(patched_actions["_exec_list_chunks"](tools, "d"))
    assert tools.kbinfos["chunks"] == original
    assert result["new_evidence"] == 0


def test_search_does_not_upgrade_text(patched_actions):
    original = {"chunk_id": "a", "doc_id": "d", "content_with_weight": "old"}
    pool = {"chunks": [original]}
    assert not patched_actions["_admit_evidence"](pool, {"a"}, dict(original, content_with_weight="longer search excerpt"), [], [], set())
    assert pool["chunks"] == [original]


@pytest.fixture
def compose_environment(installed, monkeypatch):
    generator = ModuleType("rag.prompts.generator")
    render_globals = {"logging": logging, "re": re, "num_tokens_from_string": len}
    for name in ("get_value", "kb_prompt"):
        exec(compile(installed[name]["source"], name, "exec"), render_globals)
    generator.get_value = render_globals["get_value"]
    generator.citation_prompt = lambda prompts: "Cite the evidence."
    monkeypatch.setitem(sys.modules, generator.__name__, generator)
    modules = {
        "rag.advanced_rag.agentic_rag": {"_EVIDENCE_BUDGET_TOKENS": 8000},
        "rag.advanced_rag.harness.prompts.report_prompt": {"FINAL_ANSWER_SYSTEM": "Answer from evidence. {cite_rules}", "PARTIAL_ANSWER_PREAMBLE": "Some information is missing."},
    }
    for name, fields in modules.items():
        module = ModuleType(name)
        module.__dict__.update(fields)
        monkeypatch.setitem(sys.modules, name, module)
    return {
        "AgenticState": dict,
        "asyncio": asyncio,
        "_LOG": logging.getLogger(__name__),
        "_snip": str,
        "kb_prompt": render_globals["kb_prompt"],
        "form_message": lambda system, user: [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "message_fit_in": lambda messages, budget: (0, messages),
    }


def invoke_compose(source, namespace, pool, budget=8192, abstain=False):
    scope = dict(namespace)
    exec(compile(source, "compose", "exec"), scope)
    requests = []

    async def stream_answer(system, history, gen_conf):
        requests.append({"system": system, "history": history, "gen_conf": gen_conf})
        yield "Answer [ID:0]"

    tools = SimpleNamespace(chat_mdl=SimpleNamespace(max_length=budget, async_chat_streamly_delta=stream_answer), user_defined_prompts={}, empty_response="No evidence.")
    state = {"question": "Which policy?", "kbinfos": pool, "empty_result": True, "abstain": abstain}
    result = asyncio.run(scope["_compose_answer_from_evidence"](state, tools, asyncio.Queue(), {"temperature": 0.3}))
    return result, tools, requests


def test_compose_registry_matches_six_real_prompt_blocks_without_prompt_changes(api, installed, compose_environment):
    pool = {
        "chunks": [
            {"chunk_id": cid, "content_with_weight": f"Evidence {cid}", "similarity": score}
            for cid, score in [("a", 0.4), ("b", 0.9), ("c", 0.2), ("d", 0.7), ("e", 0.8), ("f", 0.1), ("g", 0.6), ("h", 0.5)]
        ],
        "doc_aggs": [],
        "pre_summary": "Research draft.",
    }
    original = copy.deepcopy(pool)
    base = base_compose(installed)
    patched = api.patch_compose_registry_source(base, digest(base))
    _, _, old_requests = invoke_compose(base, compose_environment, pool)
    result, tools, requests = invoke_compose(patched, compose_environment, pool)
    assert requests == old_requests
    blocks = re.findall(r"\nID: (\d+)\n└── Content:\n(Evidence [a-h])", requests[0]["history"][0]["content"])
    assert blocks == [("0", "Evidence b"), ("1", "Evidence e"), ("2", "Evidence d"), ("3", "Evidence g"), ("4", "Evidence h"), ("5", "Evidence a")]
    registry = result["kbinfos"]["chunks"]
    for index, content in blocks:
        assert registry[int(index)]["content_with_weight"] == content
    assert tools.kbinfos is result["kbinfos"]
    assert pool == original
    assert result["kbinfos"] is not pool


def test_compose_registry_excludes_empty_and_budget_trimmed_chunks(api, installed, compose_environment):
    pool = {
        "chunks": [
            {"chunk_id": "empty", "content": "", "content_with_weight": "hidden", "similarity": 0.9},
            {"chunk_id": "b", "content_with_weight": "Evidence b", "similarity": 0.8},
            {"chunk_id": "e", "content_with_weight": "Evidence e", "similarity": 0.7},
            {"chunk_id": "overflow", "content_with_weight": "x" * 500, "similarity": 0.6},
        ],
        "doc_aggs": [],
    }
    base = base_compose(installed)
    result, tools, requests = invoke_compose(api.patch_compose_registry_source(base, digest(base)), compose_environment, pool, budget=100)
    assert [c["chunk_id"] for c in result["kbinfos"]["chunks"]] == ["b", "e"]
    assert tools.kbinfos is result["kbinfos"]
    assert re.findall(r"\nID: (\d+)", requests[0]["history"][0]["content"]) == ["0", "1"]


@pytest.mark.parametrize("pool,abstain", [({"chunks": [], "doc_aggs": []}, False), ({"chunks": [{"content_with_weight": "known"}], "doc_aggs": []}, True)])
def test_compose_refusal_semantics_are_preserved(api, installed, compose_environment, pool, abstain):
    base = base_compose(installed)
    result, _, requests = invoke_compose(api.patch_compose_registry_source(base, digest(base)), compose_environment, pool, abstain=abstain)
    assert result == {"final_answer": "No evidence."}
    assert requests == []


def install_manifest(api, installed):
    admit = installed["_admit_evidence"]
    list_chunks = installed["_exec_list_chunks"]
    compose = installed["_compose_answer_from_evidence"]
    compose_base = base_compose(installed)
    return {
        "functions": {
            "_admit_evidence": {
                "installed_source_sha256": admit["sha256"],
                "patch_input_sha256": admit["sha256"],
                "patched_source_sha256": digest(api.patch_admit_source(admit["source"], admit["sha256"])),
            },
            "_exec_list_chunks": {
                "installed_source_sha256": list_chunks["sha256"],
                "patch_input_sha256": list_chunks["sha256"],
                "patched_source_sha256": digest(api.patch_list_chunks_source(list_chunks["source"], list_chunks["sha256"])),
            },
            "_compose_answer_from_evidence": {
                "installed_source_sha256": compose["sha256"],
                "patch_input_sha256": digest(compose_base),
                "patched_source_sha256": digest(api.patch_compose_registry_source(compose_base, digest(compose_base))),
            },
        }
    }


@pytest.fixture
def installed_action_module(installed, monkeypatch):
    """Use the frozen installed source without writing a production file."""
    helpers = runpy.run_path(str(ROOT / "rag/advanced_rag/harness/chunk_utils.py"))
    search = ModuleType("rag.advanced_rag.harness.tools.search")
    search.__dict__.update({name: helpers[name] for name in ("_chunk_id", "_chunk_text", "_doc_id", "_dataset_id")})
    search._is_table_chunk = lambda chunk: False

    async def list_chunks(tools, doc_id):
        return {"chunks": tools.loaded}

    search.list_chunks = list_chunks
    monkeypatch.setitem(sys.modules, search.__name__, search)
    module = ModuleType("frozen_action_session")
    module.__dict__.update(
        {
            "_LOG": logging.getLogger(__name__),
            "_EVIDENCE_POOL_CAP": 2,
            "_EVIDENCE_POOL_STATE": {"full_logged": False},
            "ToolOutcome": SimpleNamespace,
            "ERROR": "error",
            "_seed_evidence": lambda tools: tools.kbinfos,
            "_search_outcome": lambda out, ids, new: {"payload": out, "ids": ids, "new_evidence": new},
        }
    )
    filename = "<frozen-action-session-integrity-fixture>"
    source = "from __future__ import annotations\n" + installed["_admit_evidence"]["source"] + "\n" + installed["_exec_list_chunks"]["source"]
    linecache.cache[filename] = (len(source), None, source.splitlines(True), filename)
    exec(compile(source, filename, "exec"), module.__dict__)
    yield module
    linecache.cache.pop(filename, None)


def test_install_is_explicit_and_deep_read_resolves_new_admit(api, installed, installed_action_module, compose_environment):
    original_admit = installed_action_module._admit_evidence
    original_list = installed_action_module._exec_list_chunks
    assert "full_text" not in inspect.signature(original_admit).parameters

    compose = api.install_integrity_patch(
        installed_action_module,
        base_compose(installed),
        install_manifest(api, installed),
        compose_globals=compose_environment,
    )

    assert inspect.iscoroutinefunction(compose)
    assert installed_action_module._admit_evidence is not original_admit
    assert installed_action_module._exec_list_chunks is not original_list
    assert installed_action_module._admit_evidence.__globals__ is installed_action_module._exec_list_chunks.__globals__
    assert installed_action_module._exec_list_chunks.__globals__["_admit_evidence"] is installed_action_module._admit_evidence
    assert installed_action_module._admit_evidence.__globals__ is not installed_action_module.__dict__
    original = {"chunk_id": "a", "doc_id": "d", "content_with_weight": "short", "similarity": 0.8}
    tools = SimpleNamespace(kbinfos={"chunks": [original]}, loaded=[{"id": "a", "doc_id": "d", "content_with_weight": "short; canonical full text"}])
    assert asyncio.run(installed_action_module._exec_list_chunks(tools, "d"))["new_evidence"] == 1
    assert tools.kbinfos["chunks"][0]["content_with_weight"] == "short; canonical full text"
    assert tools.kbinfos["chunks"][0]["similarity"] == 0.8
    assert original["content_with_weight"] == "short"
    pool = {"chunks": [{"chunk_id": "b", "doc_id": "d", "content_with_weight": "short"}]}
    assert not installed_action_module._admit_evidence(pool, {"b"}, {"id": "b", "doc_id": "d", "content_with_weight": "longer search excerpt"}, [], [], set())
    assert pool["chunks"][0]["content_with_weight"] == "short"


@pytest.mark.parametrize(
    "name,field",
    [
        ("_admit_evidence", "installed_source_sha256"),
        ("_admit_evidence", "patch_input_sha256"),
        ("_admit_evidence", "patched_source_sha256"),
        ("_exec_list_chunks", "installed_source_sha256"),
        ("_exec_list_chunks", "patch_input_sha256"),
        ("_exec_list_chunks", "patched_source_sha256"),
        ("_compose_answer_from_evidence", "installed_source_sha256"),
        ("_compose_answer_from_evidence", "patch_input_sha256"),
        ("_compose_answer_from_evidence", "patched_source_sha256"),
    ],
)
def test_install_hash_mismatch_leaves_module_unchanged(api, installed, installed_action_module, compose_environment, name, field):
    original = (installed_action_module._admit_evidence, installed_action_module._exec_list_chunks)
    manifest = install_manifest(api, installed)
    manifest["functions"][name][field] = "0" * 64
    with pytest.raises(ValueError, match="integrity"):
        api.install_integrity_patch(installed_action_module, base_compose(installed), manifest, compose_globals=compose_environment)
    assert (installed_action_module._admit_evidence, installed_action_module._exec_list_chunks) == original


def test_install_rejects_compose_without_shared_empty_fix_without_mutating_action(api, installed, installed_action_module, compose_environment):
    original = (installed_action_module._admit_evidence, installed_action_module._exec_list_chunks)
    manifest = install_manifest(api, installed)
    manifest["functions"]["_compose_answer_from_evidence"]["patch_input_sha256"] = installed["_compose_answer_from_evidence"]["sha256"]
    with pytest.raises(ValueError, match="compose_base_fix_required"):
        api.install_integrity_patch(installed_action_module, installed["_compose_answer_from_evidence"]["source"], manifest, compose_globals=compose_environment)
    assert (installed_action_module._admit_evidence, installed_action_module._exec_list_chunks) == original


def test_install_late_compose_compile_failure_leaves_action_unchanged(api, installed, installed_action_module):
    original = (installed_action_module._admit_evidence, installed_action_module._exec_list_chunks)
    with pytest.raises(NameError, match="AgenticState"):
        api.install_integrity_patch(
            installed_action_module,
            base_compose(installed),
            install_manifest(api, installed),
            compose_globals={},
        )
    assert (installed_action_module._admit_evidence, installed_action_module._exec_list_chunks) == original


def test_install_accepts_prepatched_compose_function(api, installed, installed_action_module, compose_environment):
    base = base_compose(installed)
    filename = "<shared-empty-compose-fixture>"
    linecache.cache[filename] = (len(base), None, base.splitlines(True), filename)
    scope = dict(compose_environment)
    scope["AgenticState"] = dict
    scope["asyncio"] = asyncio
    exec(compile(base, filename, "exec"), scope)
    try:
        result = api.install_integrity_patch(installed_action_module, scope["_compose_answer_from_evidence"], install_manifest(api, installed))
        assert inspect.iscoroutinefunction(result)
        assert result.__globals__ is not scope
    finally:
        linecache.cache.pop(filename, None)
