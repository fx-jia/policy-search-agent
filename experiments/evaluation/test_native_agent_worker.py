import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from experiments.evaluation.test_native_outbound import RAW, SAFE, inputs


def test_common_nonthinking_request_preserves_tools_and_is_explicit(api):
    original = {"stream": True, "max_tokens": 4096, "tools": [{"type": "function"}],
                "extra_body": {"enable_thinking": True, "other": 1}}
    result = api.completion_options(original, {"model_thinking": False})
    assert result["extra_body"] == {"enable_thinking": False, "other": 1}
    assert result["max_tokens"] == 1200 and result["tools"] == original["tools"]
    assert result["stream_options"]["include_usage"] is True
    assert original["extra_body"]["enable_thinking"] is True
    assert api.completion_options({"max_tokens": 400}, {})["max_tokens"] == 400


@pytest.fixture
def api():
    path = Path(__file__).with_name("native_agent_worker.py")
    assert path.exists(), "native_agent_worker implementation missing"
    spec = importlib.util.spec_from_file_location("native_agent_worker", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_no_gold_in_streamed_agent_payload(api):
    p = api.make_payload({"chat_id": "c", "scope_document_ids": ["a", "b"], "reasoning": 2}, {"question": "何时执行？", "answer": "gold"})
    assert p["reasoning"] == 2 and p["stream"] is True
    assert p["doc_ids"] == "a,b"
    assert p["store_history_messages"] is False
    assert p["internet"] is False
    assert "gold" not in str(p)


def test_repeated_gate_and_parent_replacement(api):
    data, review, mapping = inputs()
    safe = api.safe_result(data, review=review, document_map=mapping)
    assert safe["chunks"][0]["content_with_weight"] == SAFE
    assert api.safe_result(safe, review=review, document_map=mapping) == safe
    replaced = copy.deepcopy(safe)
    replaced["chunks"][0]["content_with_weight"] = RAW + " another patient's story"
    with pytest.raises(ValueError):
        api.safe_result(replaced, review=review, document_map=mapping)


def test_aggregated_titles_are_reviewed(api):
    data, review, mapping = inputs()
    data["doc_aggs"] = [{"doc_id": "d1", "doc_name": review["c1"]["raw_title"], "count": 1}]
    safe = api.safe_result(data, review=review, document_map=mapping)
    assert safe["doc_aggs"][0]["doc_name"] == review["c1"]["safe_title"]
    data["doc_aggs"][0]["doc_name"] = "unreviewed personal case title"
    with pytest.raises(ValueError):
        api.safe_result(data, review=review, document_map=mapping)


def test_title_only_review_allows_metadata_but_not_unreviewed_chunk_text(api):
    data, review, mapping = inputs()
    mapping["documents"]["d3"] = "s3"
    title = "公开政策通知.md"
    titles = {"d3": {"source_document_id": "s3", "raw_title": title, "safe_title": title,
                     "raw_title_sha256": hashlib.sha256(title.encode()).hexdigest(), "safe_title_sha256": hashlib.sha256(title.encode()).hexdigest()}}
    data["doc_aggs"] = [{"doc_id": "d3", "doc_name": title, "count": 1}]
    safe = api.safe_result(data, review=review, document_map=mapping, title_review=titles)
    assert safe["doc_aggs"] == data["doc_aggs"]
    data["chunks"].append({"id": "unreviewed", "doc_id": "d3", "docnm_kwd": title, "content_with_weight": "unreviewed text"})
    with pytest.raises(ValueError, match="unreviewed_chunk"):
        api.safe_result(data, review=review, document_map=mapping, title_review=titles)
    data["chunks"].pop()
    titles["d3"]["safe_title_sha256"] = "mismatch"
    with pytest.raises(ValueError, match="aggregate_title"):
        api.safe_result(data, review=review, document_map=mapping, title_review=titles)


def test_sse_preserves_answer_deltas_when_final_footer_is_empty(api):
    events = [{"code": 0, "data": {"start_to_think": True, "answer": ""}},
              {"code": 0, "data": {"answer": "草稿"}},
              {"code": 0, "data": {"end_to_think": True, "answer": ""}},
              {"code": 0, "data": {"answer": "最终"}},
              {"code": 0, "data": {"answer": "答案"}},
              {"code": 0, "data": {"answer": "", "final": True, "reference": {"chunks": []}}}, {"code": 0, "data": True}]
    wire = "".join("data:" + json.dumps(e, ensure_ascii=False) + "\n\n" for e in events)
    result = api.collect_sse([wire[:12], wire[12:]])
    assert result["answer"] == "最终答案"
    assert result["stream_complete"] is True
    events.insert(-1, {"code": 500, "data": {"answer": "**ERROR**"}})
    result = api.collect_sse(["".join("data:" + json.dumps(e) + "\n\n" for e in events)])
    assert result["error"] is True


def test_malformed_evidence_is_a_gate_error(api):
    for value in (None, [], 42):
        data, review, mapping = inputs()
        data["chunks"][0]["content_with_weight"] = value
        with pytest.raises(ValueError):
            api.safe_result(data, review=review, document_map=mapping)


def test_direct_store_compiled_content_blocks(api):
    _, review, mapping = inputs()
    with pytest.raises(ValueError, match="direct_store"):
        api.safe_direct_rows({"new": {"doc_id": "d1", "compile_kwd": "claim", "content_with_weight": "unreviewed claim"}}, review=review, document_map=mapping)
    assert api.safe_direct_rows({}, review=review, document_map=mapping) == {}


def test_final_prompt_mapping_handles_multiline_and_partial(api):
    request = {"messages": [{"role": "system", "content": "[ID:0] 第一行\n第二行"}]}
    seen = {"c": {"doc_id": "d", "content_with_weight": "第一行\n第二行", "docnm_kwd": "t"}}
    assert api.map_final_evidence(request, seen)[0]["match"] == "full"
    seen["c"]["content_with_weight"] += "\n第三行"
    assert api.map_final_evidence(request, seen)[0]["match"] == "partial_literal"


def test_changed_runtime_configuration_is_rejected(api):
    expected = {"top_n": 8, "prompt_config": {"system": "a"}, "model": "m"}
    api.validate_chat_config(expected, copy.deepcopy(expected))
    with pytest.raises(ValueError):
        api.validate_chat_config(expected, {**expected, "top_n": 12})


def test_only_bounded_runtime_citation_embedding_is_allowed(api):
    api.validate_runtime_embedding("rag.nlp.search", "insert_citations", ["已生成的回答片段"])
    for module, function, texts in [("rag.svr.task_executor", "embedding", ["doc"]), ("rag.nlp.search", "insert_citations", ["x"] * 33), ("rag.nlp.search", "insert_citations", ["x" * 12001])]:
        with pytest.raises(ValueError):
            api.validate_runtime_embedding(module, function, texts)


def test_navigation_query_embedding_is_not_ingestion(api):
    api.validate_runtime_embedding("navigation_query", "scoped_query", ["江苏2023年居民医保缴费标准"])
    with pytest.raises(ValueError):
        api.validate_runtime_embedding("navigation_query", "scoped_query", ["first", "second"])


def test_compose_patch_is_exact_and_preserves_real_empty_and_abstain(api):
    source = '''async def compose(state):
    kbinfos = state["kbinfos"]
    abstain = state.get("abstain", False)
    empty_result = state.get("empty_result", False)
    no_evidence = abstain or empty_result or not kbinfos["chunks"]
    return no_evidence
'''
    expected = hashlib.sha256(source.encode()).hexdigest()
    patched = api.patch_compose_source(source, expected)
    namespace = {}
    exec(patched, namespace)
    import asyncio
    for chunks, abstain, refuses in [([{}], False, False), ([], False, True), ([{}], True, True)]:
        assert asyncio.run(namespace["compose"]({"kbinfos": {"chunks": chunks}, "empty_result": True, "abstain": abstain})) is refuses
    with pytest.raises(ValueError, match="source"):
        api.patch_compose_source(source + "\n", expected)
    with pytest.raises(ValueError, match="pattern"):
        api.patch_compose_source(patched, hashlib.sha256(patched.encode()).hexdigest())


def test_sca_trace_preserves_result_and_restores_stage(api):
    import asyncio
    from contextvars import ContextVar
    from types import SimpleNamespace

    stage = ContextVar('test_sca_stage', default='agent_graph')
    events = []
    result = {'is_sufficient': False, 'claims': {'a': {'missing': ['期限']}}}
    claims = [('a', '已有草稿', [0])]
    tool = SimpleNamespace(kbinfos={'chunks': [{'chunk_id': 'c', 'content_with_weight': '已审核政策'}]})

    async def original(tools, question, claims):
        assert tools is tool and question == '多久？'
        assert stage.get() == 'sca'
        return result

    wrapped = api.trace_sca(original, events.append, stage)
    assert asyncio.run(wrapped(tool, '多久？', claims=claims)) is result
    assert stage.get() == 'agent_graph'
    assert events[0]['event'] == 'sca_input'
    assert events[0]['claims'] == claims
    assert events[0]['evidence'][0]['chunk_id'] == 'c'
    assert events[-1]['result'] == result
    result['claims'].clear()
    assert events[-1]['result']['claims']


def test_sca_trace_preserves_failure_without_exception_text(api):
    import asyncio
    from contextvars import ContextVar
    from types import SimpleNamespace

    stage = ContextVar('test_sca_failure', default='agent_graph')
    events = []

    async def original(tools, question, claims):
        raise RuntimeError('do not log arbitrary credentials')

    wrapped = api.trace_sca(original, events.append, stage)
    with pytest.raises(RuntimeError):
        asyncio.run(wrapped(SimpleNamespace(kbinfos={}), 'q', []))
    assert stage.get() == 'agent_graph'
    assert events[-1] == {'event': 'sca_failed', 'error_type': 'RuntimeError'}
    assert 'credentials' not in str(events)

def test_ordinary_rag_requires_reviewed_context_but_agents_require_real_graph(api):
    assert hasattr(api,'generation_context_valid'), 'mode-aware generation guard missing'
    assert api.generation_context_valid(0, {'ordinary_context_reviewed': True})
    assert not api.generation_context_valid(0, {})
    assert not api.generation_context_valid(1, {'ordinary_context_reviewed': True})
    assert api.generation_context_valid(1, {'agent_initialized': True})
    assert api.generation_context_valid(3, {'agent_initialized': True})

def test_model_override_is_copy_only_and_does_not_change_credentials(api):
    assert hasattr(api,'runtime_model_config'), 'process-only model selection missing'
    original={'llm_name':'configured','api_key':'secret','api_base':'https://provider','max_tokens':8192}
    chosen=api.runtime_model_config(original, 'dated-model')
    assert chosen['llm_name']=='dated-model' and chosen['api_key']=='secret'
    assert original['llm_name']=='configured'
