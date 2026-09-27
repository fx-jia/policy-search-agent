"""The experiment patch switch stays off unless an exact condition enables it."""

import asyncio
import hashlib
import inspect
from types import SimpleNamespace

import pytest

from experiments.evaluation import native_agent_worker as worker


async def _compose_answer_from_evidence(state):
    kbinfos = state["kbinfos"]
    abstain = state.get("abstain", False)
    empty_result = state.get("empty_result", False)
    no_evidence = abstain or empty_result or not kbinfos["chunks"]
    return no_evidence


def base_manifest():
    source = inspect.getsource(_compose_answer_from_evidence)
    digest = hashlib.sha256(source.encode()).hexdigest()
    patched = worker.patch_compose_source(source, digest)
    return {"name": "compose_current_evidence_v1", "original_function_sha256": digest,
            "patched_function_sha256": hashlib.sha256(patched.encode()).hexdigest()}


def test_default_condition_returns_same_native_function():
    assert worker.prepare_compose_patch(_compose_answer_from_evidence, None) is _compose_answer_from_evidence


def test_shared_base_condition_preserves_existing_fix():
    patched = worker.prepare_compose_patch(_compose_answer_from_evidence, base_manifest())
    for chunks, abstain, expected in [([{}], False, False), ([], False, True), ([{}], True, True)]:
        assert asyncio.run(patched({"kbinfos": {"chunks": chunks}, "empty_result": True, "abstain": abstain})) is expected
    assert "empty_result" in inspect.getsource(_compose_answer_from_evidence)


def test_integrity_requires_base_condition():
    with pytest.raises(ValueError, match="integrity_requires_base_patch"):
        worker.prepare_compose_patch(_compose_answer_from_evidence, None, integrity_manifest={"name": "evidence_integrity_v1"})


@pytest.mark.parametrize("missing", ["action_module", "installer"])
def test_integrity_rejects_missing_runtime_dependency(missing):
    args = {"action_module": SimpleNamespace(), "installer": lambda *a, **kw: None}
    args.pop(missing)
    with pytest.raises(ValueError, match="integrity_runtime_dependency_missing"):
        worker.prepare_compose_patch(_compose_answer_from_evidence, base_manifest(),
                                     integrity_manifest={"name": "evidence_integrity_v1"}, **args)


def test_integrity_receives_only_verified_base_source_and_original_globals():
    module, marker = SimpleNamespace(), object()
    manifest = {"name": "evidence_integrity_v1", "functions": {}}
    calls = []

    def install(action_module, source, passed_manifest, *, compose_globals):
        calls.append(source)
        assert action_module is module and passed_manifest is manifest
        assert compose_globals is _compose_answer_from_evidence.__globals__
        assert '    no_evidence = abstain or not kbinfos["chunks"]\n' in source
        assert 'state.get("empty_result"' not in source
        return marker

    assert worker.prepare_compose_patch(_compose_answer_from_evidence, base_manifest(), integrity_manifest=manifest,
                                        action_module=module, installer=install) is marker
    assert len(calls) == 1


@pytest.mark.parametrize("field", ["original_function_sha256", "patched_function_sha256"])
def test_base_hash_failure_never_calls_integrity_installer(field):
    manifest = base_manifest()
    manifest[field] = "0" * 64
    calls = []
    with pytest.raises(ValueError, match="source_changed"):
        worker.prepare_compose_patch(_compose_answer_from_evidence, manifest,
                                     integrity_manifest={"name": "evidence_integrity_v1"},
                                     action_module=SimpleNamespace(), installer=lambda *a, **kw: calls.append(a))
    assert calls == []


def test_unknown_integrity_condition_is_rejected():
    with pytest.raises(ValueError, match="unknown_integrity_patch"):
        worker.prepare_compose_patch(_compose_answer_from_evidence, base_manifest(), integrity_manifest={"name": "other"})
