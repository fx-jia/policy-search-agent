"""Checked, isolated source deltas for the installed Agent's evidence path."""

import ast
import hashlib
import inspect


def _check_source(source, expected_sha256):
    if not isinstance(source, str) or hashlib.sha256(source.encode()).hexdigest() != expected_sha256:
        raise ValueError("integrity_source_changed")


def _replace_once(source, old, new, label):
    if source.count(old) != 1:
        raise ValueError(f"integrity_anchor_changed:{label}")
    return source.replace(old, new, 1)


def _validated_function(source, name):
    tree = ast.parse(source)
    if len(tree.body) != 1 or not isinstance(tree.body[0], (ast.FunctionDef, ast.AsyncFunctionDef)) or tree.body[0].name != name:
        raise ValueError("integrity_function_changed")
    compile(tree, f"<isolated-integrity:{name}>", "exec")
    return source


def patch_admit_source(source, expected_sha256):
    """Upgrade a known excerpt only when the caller identifies a full-text read."""
    _check_source(source, expected_sha256)
    source = _replace_once(
        source,
        "def _admit_evidence(kbinfos, kb_seen, c, out, ids, seen, include_doc_id=True) -> bool:\n",
        "def _admit_evidence(kbinfos, kb_seen, c, out, ids, seen, include_doc_id=True, *, full_text=False) -> bool:\n",
        "admit_signature",
    )
    source = _replace_once(
        source,
        "    if len(_pool) >= _EVIDENCE_POOL_CAP:\n",
        "    if existing_index is None and len(_pool) >= _EVIDENCE_POOL_CAP:\n",
        "admit_cap",
    )
    pool_line = '    _pool = kbinfos.get("chunks", []) if isinstance(kbinfos, dict) else (getattr(kbinfos, "chunks", []) or [])\n'
    source = _replace_once(
        source,
        pool_line,
        "    from rag.advanced_rag.harness.tools.search import _chunk_id, _chunk_text, _dataset_id, _doc_id, _is_table_chunk\n\n"
        "    cid = _chunk_id(c)\n"
        "    if not cid or cid in seen:\n"
        "        return False\n" + pool_line + "    existing_index = next((i for i, old in enumerate(_pool) if _chunk_id(old) == cid), None)\n",
        "admit_identity",
    )
    source = _replace_once(
        source,
        "    from rag.advanced_rag.harness.tools.search import _chunk_id, _chunk_text, _doc_id, _is_table_chunk\n\n    cid = _chunk_id(c)\n    if cid in seen:\n        return False\n",
        "    upgraded = False\n"
        "    if full_text and existing_index is not None:\n"
        "        old = _pool[existing_index]\n"
        "        if not _doc_id(c) or _doc_id(c) != _doc_id(old):\n"
        "            return False\n"
        "        if _dataset_id(c) and _dataset_id(old) and _dataset_id(c) != _dataset_id(old):\n"
        "            return False\n"
        "        if len(_chunk_text(c)) > len(_chunk_text(old)):\n"
        "            replacement = dict(old, **c)\n"
        '            replacement["content_with_weight"] = _chunk_text(c)\n'
        '            replacement.pop("content", None)\n'
        '            replacement.pop("text", None)\n'
        '            for key in ("similarity", "score", "vector_similarity", "term_similarity"):\n'
        "                if key in old:\n"
        "                    replacement[key] = old[key]\n"
        "            _pool[existing_index] = replacement\n"
        "            upgraded = True\n",
        "admit_upgrade",
    )
    source = _replace_once(source, "        return True\n    return False\n", "        return True\n    return upgraded\n", "admit_progress")
    return _validated_function(source, "_admit_evidence")


def patch_list_chunks_source(source, expected_sha256):
    """Mark only the existing list_chunks admission as an explicit deep read."""
    _check_source(source, expected_sha256)
    source = _replace_once(
        source,
        "        if _admit_evidence(kbinfos, kb_seen, c, out, ids, seen, include_doc_id=False):\n",
        "        if _admit_evidence(kbinfos, kb_seen, c, out, ids, seen, include_doc_id=False, full_text=True):\n",
        "list_chunks_admit",
    )
    return _validated_function(source, "_exec_list_chunks")


def patch_compose_registry_source(source, expected_sha256):
    """Apply after the shared empty-flag patch; preserve installed prompt bytes."""
    _check_source(source, expected_sha256)
    if 'state.get("empty_result"' in source or source.count('    no_evidence = abstain or not kbinfos["chunks"]\n') != 1:
        raise ValueError("compose_base_fix_required")
    import_line = "    from rag.advanced_rag.agentic_rag import _EVIDENCE_BUDGET_TOKENS\n"
    source = _replace_once(source, import_line, import_line + "    from rag.prompts.generator import get_value\n", "compose_import")
    render_line = "    evidence_blocks = kb_prompt(evidence_kbinfos, min(tools.chat_mdl.max_length, _EVIDENCE_BUDGET_TOKENS))\n"
    source = _replace_once(
        source,
        render_line,
        render_line + '    cite_chunks = [c for c in cite_chunks if get_value(c, "content", "content_with_weight")][: len(evidence_blocks)]\n' + '    evidence_kbinfos["chunks"] = cite_chunks\n',
        "compose_registry",
    )
    fit_line = "    _, msg = message_fit_in(form_message(system, user_content), min(tools.chat_mdl.max_length, _EVIDENCE_BUDGET_TOKENS))\n"
    source = _replace_once(source, fit_line, fit_line + "    tools.kbinfos = evidence_kbinfos\n", "compose_tools")
    source = _replace_once(source, '    return {"final_answer": ""}\n', '    return {"final_answer": "", "kbinfos": evidence_kbinfos}\n', "compose_state")
    return _validated_function(source, "_compose_answer_from_evidence")


def _manifest_function(manifest, name):
    try:
        row = manifest["functions"][name]
        hashes = {key: row[key] for key in ("installed_source_sha256", "patch_input_sha256", "patched_source_sha256")}
    except (KeyError, TypeError):
        raise ValueError("integrity_manifest_invalid") from None
    if any(not isinstance(value, str) or len(value) != 64 for value in hashes.values()):
        raise ValueError("integrity_manifest_invalid")
    return hashes


def _function_source(function):
    try:
        return inspect.getsource(function)
    except (OSError, TypeError):
        raise ValueError("integrity_function_source_unavailable") from None


def _check_shared_compose_base(source, installed_sha256):
    """Verify that the sole prior compose delta is the shared empty-result fix."""
    read = '    empty_result = state.get("empty_result", False)\n'
    old_check = '    no_evidence = abstain or empty_result or not kbinfos["chunks"]\n'
    fixed_check = '    no_evidence = abstain or not kbinfos["chunks"]\n'
    abstain = '    abstain = state.get("abstain", False)\n'
    if read in source or old_check in source or source.count(fixed_check) != 1 or source.count(abstain) != 1:
        raise ValueError("compose_base_fix_required")
    original = source.replace(abstain, abstain + read, 1).replace(fixed_check, old_check, 1)
    _check_source(original, installed_sha256)


def install_integrity_patch(action_module, compose_base, manifest, *, compose_globals=None):
    """Install verified evidence functions together; return the new compose function.

    ``compose_base`` is the source or function *after* the shared empty-result
    fix. A source string needs its graph globals via ``compose_globals``.
    No module binding changes until all source, hash and compilation checks pass.
    """
    admit_row = _manifest_function(manifest, "_admit_evidence")
    list_row = _manifest_function(manifest, "_exec_list_chunks")
    compose_row = _manifest_function(manifest, "_compose_answer_from_evidence")

    admit_source = _function_source(action_module._admit_evidence)
    list_source = _function_source(action_module._exec_list_chunks)
    for source, row in ((admit_source, admit_row), (list_source, list_row)):
        _check_source(source, row["installed_source_sha256"])
        _check_source(source, row["patch_input_sha256"])
    patched_admit = patch_admit_source(admit_source, admit_row["patch_input_sha256"])
    patched_list = patch_list_chunks_source(list_source, list_row["patch_input_sha256"])
    _check_source(patched_admit, admit_row["patched_source_sha256"])
    _check_source(patched_list, list_row["patched_source_sha256"])

    if isinstance(compose_base, str):
        if not isinstance(compose_globals, dict):
            raise ValueError("integrity_compose_globals_required")
        compose_source = compose_base
        graph_globals = compose_globals
    else:
        compose_source = _function_source(compose_base)
        graph_globals = compose_base.__globals__ if compose_globals is None else compose_globals
        if not isinstance(graph_globals, dict):
            raise ValueError("integrity_compose_globals_required")
    _check_source(compose_source, compose_row["patch_input_sha256"])
    _check_shared_compose_base(compose_source, compose_row["installed_source_sha256"])
    patched_compose = patch_compose_registry_source(compose_source, compose_row["patch_input_sha256"])
    _check_source(patched_compose, compose_row["patched_source_sha256"])

    action_scope = dict(action_module.__dict__)
    exec(compile(patched_admit + "\n" + patched_list, "<isolated-integrity:actions>", "exec"), action_scope)
    graph_scope = dict(graph_globals)
    exec(compile(patched_compose, "<isolated-integrity:compose>", "exec"), graph_scope)
    new_admit = action_scope["_admit_evidence"]
    new_list = action_scope["_exec_list_chunks"]
    new_compose = graph_scope["_compose_answer_from_evidence"]
    if new_list.__globals__["_admit_evidence"] is not new_admit:
        raise ValueError("integrity_action_binding_changed")
    action_module.__dict__.update({"_admit_evidence": new_admit, "_exec_list_chunks": new_list})
    return new_compose
