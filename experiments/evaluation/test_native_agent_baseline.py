import importlib.util
import hashlib
from pathlib import Path

import pytest


@pytest.fixture
def api():
    path = Path(__file__).with_name("native_agent_baseline.py")
    assert path.exists(), "native_agent_baseline implementation missing"
    spec = importlib.util.spec_from_file_location("native_agent_baseline", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_gold_and_other_splits_do_not_enter_config(api):
    ids = ['med-043', 'med-029', 'med-115', 'med-022', 'med-002', 'med-017']
    qs = [{"question_id": q, "question": "医保规定？", "gold": "secret", "split": "dev"} for q in reversed(ids)]
    c = api.build_config({"chat_config": {"model": api.MODEL}, "runtime_code_hashes": {}}, qs, {"entries": {}}, {"documents": {"d": "s"}})
    assert c["question_ids"] == ids
    assert all(set(q) == {"question_id", "question"} for q in c["questions"])
    assert "secret" not in str(c)
    assert c["limits"]["run_tokens"] == 180000


def test_incomplete_planned_set_does_not_silently_shrink(api):
    with pytest.raises(ValueError, match="question"):
        api.build_config({"chat_config": {"model": api.MODEL}}, [], {"entries": {}}, {"documents": {}})


def test_title_review_matches_scope_and_live_titles(api):
    title = "公开政策通知"
    digest = hashlib.sha256(title.encode()).hexdigest()
    review = {"entries": {"d": {"source_document_id": "s", "raw_title": title, "safe_title": title,
                               "raw_title_sha256": digest, "safe_title_sha256": digest}}, "unresolved": []}
    live = {"c": {"doc_id": "d", "title": title}}
    mapping = {"documents": {"d": "s"}}
    api.validate_title_review(review, live, mapping)
    with pytest.raises(ValueError, match="title"):
        api.validate_title_review(review, {"c": {"doc_id": "d", "title": "changed"}}, mapping)
    with pytest.raises(ValueError, match="title"):
        api.validate_title_review(review, live, {"documents": {"d": "different"}})
    with pytest.raises(ValueError, match="title"):
        api.validate_title_review({**review, "unresolved": ["d"]}, live, mapping)
