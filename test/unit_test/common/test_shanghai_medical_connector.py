"""Offline HTTP-boundary tests for Shanghai's official policy collector."""

import hashlib
import importlib.util
import io
import socket
import sys
import zipfile
from pathlib import Path
from types import ModuleType

import pydantic
import pytest
import requests
import xxhash

from common import ssrf_guard

ROOT = Path(__file__).resolve().parents[3]
BASE = "https://ybj.sh.gov.cn"
LIST = BASE + "/gfxwj/index.html"
DETAIL = BASE + "/gfxwj/20260911/example.html"
OLDER = BASE + "/gfxwj/20200101/older.html"
ATTACH = BASE + "/cmsres/example.pdf"


def _load_collector():
    path = ROOT / "common/data_source/shanghai_medical_connector.py"
    if not path.exists():
        return None
    names = ("common.data_source", "anthropic")
    saved = {key: value for key, value in sys.modules.items() if any(key == name or key.startswith(name + ".") for name in names)}
    package = ModuleType("common.data_source")
    package.__path__ = [str(path.parent)]
    sys.modules["common.data_source"] = package
    if "anthropic" not in sys.modules:
        anthropic = ModuleType("anthropic")
        anthropic.BaseModel = pydantic.BaseModel
        sys.modules["anthropic"] = anthropic
    try:
        spec = importlib.util.spec_from_file_location("_shanghai_collector_test", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        for key in list(sys.modules):
            if any(key == name or key.startswith(name + ".") for name in names):
                sys.modules.pop(key, None)
        sys.modules.update(saved)


collector = _load_collector()


def test_collector_has_build_interface():
    assert collector is not None, "ShanghaiMedicalConnector is not implemented"
    assert callable(collector.ShanghaiMedicalConnector.build_connector)


def listing(url=DETAIL, next_url=None):
    following = f'<a rel="next" href="{next_url}">下一页</a>' if next_url else ""
    return f'<div class="policy-title"><a href="{url}">医保通知</a></div>{following}'


def article(text="第一条 医保待遇按规定支付。", attachments=(), extra=""):
    links = "".join(f'<p><a href="{link}">附件</a></p>' for link in attachments)
    return f"""<meta name="ArticleTitle" content="医保通知"><meta name="PubDate" content="2026-09-11">
    <div class="xxgk">文号：沪医保规〔2026〕10号 成文日期：2026-09-09</div>
    <div class="article-content"><p>{text}</p>{links}</div>
    <div class="article-content phone"><p>{text}</p>{links}</div><footer>{extra}</footer>"""


@pytest.fixture
def http(monkeypatch):
    responses = {BASE + "/robots.txt": (200, "User-agent: *\nAllow: /", "text/plain")}
    calls = []
    monkeypatch.delenv("ALLOW_ANY_HOST", raising=False)
    monkeypatch.setattr(ssrf_guard, "_orig_getaddrinfo", lambda host, port, *a, **k: [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("1.1.1.1", port or 0))])

    def send(session, prepared, **kwargs):
        calls.append((prepared.url, dict(prepared.headers), kwargs))
        value = responses[prepared.url]
        if isinstance(value, list):
            value = value.pop(0)
        if isinstance(value, Exception):
            raise value
        status, body, headers = value
        response = requests.Response()
        response.status_code = status
        response.url = prepared.url
        response.headers.update({"Content-Type": headers} if isinstance(headers, str) else headers)
        response.raw = io.BytesIO(body.encode() if isinstance(body, str) else body)
        return response

    monkeypatch.setattr(requests.Session, "send", send)
    return responses, calls


@pytest.fixture
def build(http, monkeypatch):
    if collector is None:
        pytest.skip("Collector behavior depends on missing implementation")
    clock = [0.0]
    monkeypatch.setattr(collector.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(collector.time, "sleep", lambda seconds: clock.__setitem__(0, clock[0] + seconds))

    def create(**options):
        config = {"list_urls": [LIST], "parse_build_key": "parse-v1", "seed_urls": [], **options}
        connector = collector.ShanghaiMedicalConnector.build_connector(config)
        captures = []

        def sink(capture):
            captures.append(capture)
            return {"revision_id": "catalog-row"}

        connector.capture_sink = sink
        return connector, captures

    return create


def docs(connector):
    return [document for batch in connector.poll_source(9_999_999_999, 10_000_000_000) for document in batch]


def test_pagination_ignores_dates_and_deduplicates_urls(build, http):
    pages, _ = http
    pages[LIST] = (200, listing(next_url="index_2.html"), "text/html")
    pages[BASE + "/gfxwj/index_2.html"] = (200, listing() + listing(OLDER), "text/html")
    pages[DETAIL] = (200, article(), "text/html")
    pages[OLDER] = (200, article("旧日期但新发现的医保条款"), "text/html")
    connector, captures = build()
    result = docs(connector)
    assert len(result) == len(captures) == 2
    assert result[0].metadata["published_at"] == "2026-09-11"
    assert result[0].metadata["signed_at"] == "2026-09-09"
    assert result[0].metadata["document_number"] == "沪医保规〔2026〕10号"
    assert result[0].metadata["revision_id"] == "catalog-row"
    assert result[0].blob.decode().count("第一条") == 1
    assert result[0].fingerprint == xxhash.xxh128(result[0].blob).hexdigest()
    assert captures[0]["raw_html"] and captures[0]["assets"][0]["sha256"] == hashlib.sha256(result[0].blob).hexdigest()


def test_real_static_pagination_template_is_bounded(build, http):
    pages, _ = http
    template = "<script>totalPage: 2; location.href = 'index_' + i + '.html';</script>"
    pages[LIST] = (200, listing() + template, "text/html")
    pages[BASE + "/gfxwj/index_2.html"] = (200, listing(OLDER) + template, "text/html")
    pages[DETAIL] = pages[OLDER] = (200, article(), "text/html")
    assert len(docs(build()[0])) == 2


def test_template_change_preserves_identity_and_same_url_body_change_does_not(build, http):
    pages, _ = http
    pages[LIST] = (200, listing(), "text/html")
    connector, captures = build()
    pages[DETAIL] = (200, article(extra="模板一"), "text/html")
    first = docs(connector)[0]
    pages[DETAIL] = (200, article(extra="模板二"), "text/html")
    assert docs(connector)[0].id == first.id
    pages[DETAIL] = (200, article("第一条 已修改待遇。"), "text/html")
    assert docs(connector)[0].id != first.id
    assert len(captures) == 3


def test_attachment_changes_revision_even_when_parent_is_unchanged(build, http):
    pages, _ = http
    pages[LIST] = (200, listing(), "text/html")
    pages[DETAIL] = (200, article(attachments=[ATTACH]), "text/html")
    pages[ATTACH] = (200, b"%PDF-1.7\noriginal", "application/pdf")
    connector, captures = build()
    first = docs(connector)
    pages[ATTACH] = (200, b"%PDF-1.7\nchanged", "application/pdf")
    second = docs(connector)
    assert len(first) == len(second) == 2
    assert {doc.id for doc in first}.isdisjoint({doc.id for doc in second})
    assert captures[0]["revision_hash"] != captures[1]["revision_hash"]


@pytest.mark.parametrize("failure", [(404, b"missing", "text/plain"), (200, b"HTML error", "application/pdf"), (200, b"%PDF-1.7", "text/html")])
def test_failed_attachment_is_captured_without_parse_documents(build, http, failure):
    pages, _ = http
    pages[LIST] = (200, listing(), "text/html")
    pages[DETAIL] = (200, article(attachments=[ATTACH]), "text/html")
    pages[ATTACH] = failure
    connector, captures = build()
    assert docs(connector) == []
    assert len(captures) == 1
    assert captures[0]["revision_hash"] is None
    assert captures[0]["errors"] and captures[0]["raw_html"]
    assert any(asset["url"] == ATTACH for asset in captures[0]["assets"])

    assert connector.discovery_complete
    assert connector.observation_complete
    assert not connector.scan_complete


@pytest.mark.parametrize("unsafe", ["http://ybj.sh.gov.cn/cmsres/x.pdf", "https://evil.example/x.pdf", BASE + "/other/x.pdf", BASE + "/cmsres/../other/x.pdf"])
def test_unsafe_attachment_becomes_failed_candidate_before_network(build, http, unsafe):
    pages, calls = http
    pages[LIST] = (200, listing(), "text/html")
    pages[DETAIL] = (200, article(attachments=[unsafe]), "text/html")
    connector, captures = build()
    assert docs(connector) == []
    assert captures[0]["revision_hash"] is None
    assert unsafe not in [call[0] for call in calls]

    assert connector.discovery_complete
    assert connector.observation_complete
    assert not connector.scan_complete


def test_private_dns_and_cross_host_redirects_are_blocked(build, http, monkeypatch):
    pages, calls = http
    connector, _ = build()
    monkeypatch.setattr(ssrf_guard, "_orig_getaddrinfo", lambda *a, **k: [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 0))])
    with pytest.raises(ValueError, match="non-public"):
        docs(connector)
    assert not calls
    monkeypatch.setattr(ssrf_guard, "_orig_getaddrinfo", lambda *a, **k: [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("1.1.1.1", 0))])
    pages[LIST] = (302, b"", {"Location": "https://evil.example/index.html"})
    with pytest.raises(ValueError):
        docs(connector)
    assert "https://evil.example/index.html" not in [call[0] for call in calls]


def test_robots_denial_stops_without_fetching_list(build, http):
    pages, calls = http
    pages[BASE + "/robots.txt"] = (200, "User-agent: *\nDisallow: /gfxwj/", "text/plain")
    with pytest.raises(ValueError, match="robots"):
        docs(build()[0])
    assert LIST not in [call[0] for call in calls]


def test_size_limits_and_cancellation_stop_scan(build, http):
    pages, calls = http
    pages[LIST] = (200, "x" * 1025, "text/html")
    with pytest.raises(ValueError, match="size"):
        docs(build(max_html_bytes=1024)[0])
    calls.clear()
    connector, _ = build()
    connector.should_stop = lambda: True
    with pytest.raises(ValueError, match="cancel"):
        docs(connector)
    assert not calls


def test_pagination_loop_and_page_limit_never_report_complete(build, http):
    pages, _ = http
    pages[LIST] = (200, listing(next_url=LIST), "text/html")
    with pytest.raises(ValueError, match="pagination"):
        docs(build()[0])
    pages[LIST] = (200, listing(next_url="index_2.html"), "text/html")
    with pytest.raises(ValueError, match="limit"):
        docs(build(max_pages_per_column=1)[0])


def test_parse_build_changes_asset_ids(build, http):
    pages, _ = http
    pages[LIST] = (200, listing(), "text/html")
    pages[DETAIL] = (200, article(), "text/html")
    assert docs(build()[0])[0].id != docs(build(parse_build_key="parse-v2")[0])[0].id


def test_configuration_rejects_extra_credentials_unsafe_sources_and_limits(build):
    connector, _ = build()
    with pytest.raises(ValueError):
        connector.load_credentials({"token": "not-needed"})
    for invalid in ({"list_urls": ["https://evil.example/index.html"]}, {"request_interval_seconds": 0}, {"max_pages_per_column": 101}, {"parse_build_key": ""}, {"sync_deleted_files": True}):
        with pytest.raises(ValueError):
            build(**invalid)


def test_docx_signature_requires_actual_office_archive(build, http):
    pages, _ = http
    target = BASE + "/cmsres/valid.docx"
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("[Content_Types].xml", "<Types/>")
        archive.writestr("word/document.xml", "<document/>")
    pages[LIST] = (200, listing(), "text/html")
    pages[DETAIL] = (200, article(attachments=[target]), "text/html")
    pages[target] = (200, buffer.getvalue(), "application/vnd.openxmlformats-officedocument.wordprocessingml.document")
    connector, _ = build()
    assert docs(connector)[1].extension == ".docx"


def test_failed_format_preserves_downloaded_bytes_for_audit(build, http):
    pages, _ = http
    pages[LIST] = (200, listing(), "text/html")
    pages[DETAIL] = (200, article(attachments=[ATTACH]), "text/html")
    pages[ATTACH] = (200, b"damaged source", "application/pdf")
    connector, captures = build()
    assert docs(connector) == []
    attachment = captures[0]["assets"][1]
    assert attachment["blob"] == b"damaged source"
    assert attachment["sha256"] == hashlib.sha256(b"damaged source").hexdigest()

    assert connector.discovery_complete
    assert connector.observation_complete
    assert not connector.scan_complete


def test_conditional_requests_reuse_raw_capture_but_force_weekly_refresh(build, http, monkeypatch):
    pages, calls = http
    pages[LIST] = (200, listing(), "text/html")
    pages[DETAIL] = (200, article(), {"Content-Type": "text/html", "ETag": '"v1"'})
    connector, captures = build()
    wall_time = [1_800_000_000.0]
    monkeypatch.setattr(collector.time, "time", lambda: wall_time[0])
    first = docs(connector)
    pages[DETAIL] = (304, b"", {})
    assert docs(connector)[0].id == first[0].id
    assert captures[1]["raw_html"] == captures[0]["raw_html"]
    assert [call for call in calls if call[0] == DETAIL][-1][1]["If-None-Match"] == '"v1"'
    wall_time[0] += 7 * 86400 + 1
    pages[DETAIL] = (200, article("缓存标记未变，但正文变化"), {"Content-Type": "text/html", "ETag": '"v1"'})
    assert docs(connector)[0].id != first[0].id
    assert "If-None-Match" not in [call for call in calls if call[0] == DETAIL][-1][1]


def test_retry_after_is_obeyed_and_403_is_not_retried(build, http):
    pages, calls = http
    pages[LIST] = [(429, b"", {"Retry-After": "8"}), (200, listing(), "text/html")]
    pages[DETAIL] = (200, article(), "text/html")
    connector, _ = build()
    assert len(docs(connector)) == 1
    assert collector.time.monotonic() >= 8
    pages[LIST] = (403, b"blocked", "text/html")
    calls.clear()
    with pytest.raises(ValueError):
        docs(connector)
    assert len([call for call in calls if call[0] == LIST]) == 1


def test_limits_persist_deferred_candidates_and_do_not_finish_scan(build, http):
    pages, _ = http
    pages[LIST] = (200, listing() + listing(OLDER), "text/html")
    pages[DETAIL] = pages[OLDER] = (200, article(), "text/html")
    connector, captures = build(max_new_policies=1)
    generator = connector.load_from_state()
    result = []
    for batch in generator:
        result.extend(batch)
    assert len(result) == 1
    assert len(captures) == 2
    assert captures[1]["source_url"] == OLDER and captures[1]["revision_hash"] is None
    assert connector.backlog == [OLDER]
    assert connector.scan_complete is False

    assert connector.discovery_complete
    assert connector.observation_complete
    assert not connector.scan_complete
    assert connector.retryable_errors == []


def test_js_only_download_is_needs_review(build, http):
    pages, _ = http
    pages[LIST] = (200, listing(), "text/html")
    pages[DETAIL] = (200, article().replace("第一条 医保待遇按规定支付。", '第一条<button onclick="download()">下载附件</button>'), "text/html")
    connector, captures = build()
    assert docs(connector) == []
    assert captures[0]["revision_hash"] is None
    assert any("unsupported_attachment" in error for error in captures[0]["errors"])

    assert connector.discovery_complete
    assert connector.observation_complete
    assert not connector.scan_complete


def test_nested_mobile_duplicate_is_not_a_second_body(build, http):
    pages, _ = http
    pages[LIST] = (200, listing(), "text/html")
    html = '<meta name="ArticleTitle" content="医保通知"><div class="article-content"><div class="TRS_Editor">第一条 医保待遇。</div><div class="TRS_Editor phone">第一条 医保待遇。</div></div>'
    pages[DETAIL] = (200, html, "text/html")
    assert docs(build()[0])[0].blob.decode().count("第一条") == 1


def test_sink_metadata_cannot_replace_official_policy_identity(build, http):
    pages, _ = http
    pages[LIST] = (200, listing(), "text/html")
    pages[DETAIL] = (200, article(), "text/html")
    connector, _ = build()
    connector.capture_sink = lambda capture: {"source_url": "https://evil.example", "revision_hash": "fake"}
    document = docs(connector)[0]
    assert document.metadata["source_url"] == DETAIL
    assert document.metadata["revision_hash"] != "fake"


def test_official_detail_sibling_attachment_and_document_number(build, http):
    pages, _ = http
    pages[LIST] = (200, listing(), "text/html")
    # Minimal structure from official 2026-09-11 detail, excluding full policy text.
    html = """<meta name="ArticleTitle" content="医保通知"><meta name="PubDate" content="2026年09月11日">
    <div class="content-container"><div class="content-agency">沪医保规〔2026〕9 号</div>
    <div class="content-detail"><p>医保待遇原文</p><p style="text-align: right;">上海市医疗保障局</p><p style="text-align: right;">2026年9月9日</p></div>
    <div class="content-links"><a href="/zcjd/20260911/explanation.html">政策解读</a></div>
    <div class="content-attaches"><a href="/cmsres/example.pdf">申请表.pdf</a></div></div>
    <div id="ivs_content"><p>医保待遇原文</p></div><div id="enclosure"><a href="/cmsres/example.pdf">申请表.pdf</a></div>"""
    pages[DETAIL] = (200, html, "text/html")
    pages[ATTACH] = (200, b"%PDF-1.7\napplication", "application/pdf")
    connector, captures = build()
    result = docs(connector)
    assert len(result) == 2
    assert result[0].metadata["document_number"] == "沪医保规〔2026〕9号"
    assert result[0].metadata["signed_at"] == "2026-09-09"
    assert result[0].metadata["published_at"] == "2026-09-11"
    assert captures[0]["related_urls"] == [BASE + "/zcjd/20260911/explanation.html"]
    assert result[0].blob.decode().count("医保待遇原文") == 1


def test_cancel_closes_transport_and_blocks_remaining_work(build, http, monkeypatch):
    _, calls = http
    connector, _ = build()
    closed = []
    monkeypatch.setattr(connector._session, "close", lambda: closed.append(True))
    connector.cancel()
    assert closed == [True]
    with pytest.raises(ValueError, match="cancel"):
        docs(connector)
    assert not calls


def test_unfetched_attachment_is_missing_bytes_not_an_empty_archive(build, http):
    pages, _ = http
    pages[LIST] = (200, listing(), "text/html")
    pages[DETAIL] = (200, article(attachments=[ATTACH]), "text/html")
    pages[ATTACH] = (404, b"missing", "text/plain")
    connector, captures = build()
    assert docs(connector) == []
    assert captures[0]["assets"][1]["blob"] is None
    assert captures[0]["assets"][1]["sha256"] is None

    assert connector.discovery_complete
    assert connector.observation_complete
    assert not connector.scan_complete


def test_generic_change_notice_with_covered_document_number_is_collected(build, http):
    pages, _ = http
    notice = BASE + "/gfxwj/20261001/extension.html"
    title = "关于延长有关规范性文件有效期的通知"
    pages[LIST] = (200, listing().replace("医保通知", title).replace(DETAIL, notice), "text/html")
    pages[notice] = (200, article("《已有办法》（沪医保规〔2026〕9 号）按本通知载明的期限执行。").replace("医保通知", title), "text/html")
    connector, captures = build(covered_document_numbers=["沪医保规〔2026〕9号"])
    result = docs(connector)
    assert len(result) == 1
    assert captures[0]["source_url"] == notice
    assert captures[0]["covered_document_numbers_matched"] == ["沪医保规〔2026〕9号"]
    relation = captures[0]["relations"][0]
    assert relation["kind"] == "association_reference"
    assert relation["document_number"] == "沪医保规〔2026〕9号"
    assert "沪医保规〔2026〕9 号" in relation["source_span"]
    assert relation["status"] == "needs_review"
    assert all(item["kind"] not in {"replaces", "extends_validity", "repeals"} for item in captures[0]["relations"])


def test_ambiguous_generic_change_notice_is_archived_as_review_candidate(build, http):
    pages, _ = http
    title = "关于废止部分规范性文件的通知"
    pages[LIST] = (200, listing().replace("医保通知", title), "text/html")
    pages[DETAIL] = (200, article("有关文件的处理情况详见本文，未记录具体文号。").replace("医保通知", title), "text/html")
    connector, captures = build(covered_document_numbers=["沪医保规〔2026〕9号"])
    assert docs(connector) == []
    assert len(captures) == 1
    assert captures[0]["raw_html"]
    assert captures[0]["revision_hash"] is None
    assert captures[0]["covered_document_numbers_matched"] == []
    assert captures[0]["errors"]
    assert not connector.scan_complete

    assert connector.discovery_complete
    assert connector.observation_complete
    assert not connector.scan_complete


def test_change_notice_discovery_uses_existing_new_policy_limit(build, http):
    pages, calls = http
    notice = BASE + "/gfxwj/20261001/extension.html"
    pages[LIST] = (200, listing() + listing(notice).replace("医保通知", "关于延长规范性文件有效期的通知"), "text/html")
    pages[DETAIL] = (200, article(), "text/html")
    connector, captures = build(max_new_policies=1)
    assert len(docs(connector)) == 1
    assert len(captures) == 2
    assert captures[1]["source_url"] == notice
    assert captures[1]["revision_hash"] is None
    assert notice not in [call[0] for call in calls]

    assert connector.discovery_complete
    assert connector.observation_complete
    assert not connector.scan_complete


@pytest.mark.parametrize(
    "bad_next",
    [
        (403, b"blocked", "text/html"),
        (200, b"not an HTML list", "text/plain"),
        (200, "<p>template without policy URLs</p>", "text/html"),
    ],
)
def test_preflight_rejects_failed_or_malformed_next_page(build, http, bad_next):
    pages, _ = http
    pages[LIST] = (200, listing(next_url="index_2.html"), "text/html")
    pages[BASE + "/gfxwj/index_2.html"] = bad_next
    with pytest.raises(ValueError):
        build()[0].validate_connector_settings()


def test_preflight_checks_only_first_and_one_next_page(build, http):
    pages, calls = http
    pages[LIST] = (200, listing(next_url="index_2.html"), "text/html")
    pages[BASE + "/gfxwj/index_2.html"] = (200, listing(OLDER, next_url="index_3.html"), "text/html")
    build()[0].validate_connector_settings()
    assert [call[0] for call in calls] == [BASE + "/robots.txt", LIST, BASE + "/gfxwj/index_2.html"]


@pytest.mark.parametrize("next_listing", [listing(OLDER, next_url=LIST), listing(OLDER, next_url="index_2.html"), listing(next_url="index_2.html")])
def test_preflight_rejects_next_page_loops_and_repeated_content(build, http, next_listing):
    pages, _ = http
    pages[LIST] = (200, listing(next_url="index_2.html"), "text/html")
    pages[BASE + "/gfxwj/index_2.html"] = (200, next_listing, "text/html")
    with pytest.raises(ValueError, match="pagination"):
        build()[0].validate_connector_settings()


def test_preflight_rejects_repeated_policy_list_despite_template_change(build, http):
    pages, _ = http
    pages[LIST] = (200, listing(next_url="index_2.html") + "<footer>第一页</footer>", "text/html")
    pages[BASE + "/gfxwj/index_2.html"] = (200, listing(next_url="index_3.html") + "<footer>第二页</footer>", "text/html")
    with pytest.raises(ValueError, match="pagination"):
        build()[0].validate_connector_settings()


def test_generic_change_number_matching_keeps_other_scope_in_review(build, http):
    pages, _ = http
    title = "关于修改有关规范性文件的通知"
    pages[LIST] = (200, listing().replace("医保通知", title), "text/html")
    pages[DETAIL] = (200, article("沪医保规〔2026〕19号的条款有调整。").replace("医保通知", title), "text/html")
    connector, captures = build(covered_document_numbers=["沪医保规〔2026〕9号"])
    assert docs(connector) == []
    assert captures[0]["covered_document_numbers_matched"] == []
    assert captures[0]["relations"] == []

    assert connector.discovery_complete
    assert connector.observation_complete
    assert not connector.scan_complete


def _covered_policy_pages(http, count=101):
    pages, _ = http
    urls = [BASE + f"/gfxwj/20261001/policy-{index}.html" for index in range(count)]
    pages[LIST] = (200, "".join(listing(url) for url in urls), "text/html")
    for index, url in enumerate(urls):
        pages[url] = (200, article(f"第{index}份医保待遇原文。"), "text/html")
    return urls


def _published_sink(captures, ingestion_urls=(), published_build="parse-v1"):
    # This is the external catalog callback contract. Real publication identity
    # proofs are covered by catalog/sync-service tests rather than faked here.
    ingestion_urls = set(ingestion_urls)

    def sink(capture):
        required = capture["source_url"] in ingestion_urls or capture["parse_build_key"] != published_build
        if required and len(capture["assets"]) > capture["_asset_budget_remaining"]:
            capture["errors"].append("asset_limit: deferred")
            capture["revision_hash"] = None
        captures.append(capture)
        return {"policy_ingestion_required": "1" if required else "0"}

    return sink


def test_101_published_unchanged_policies_complete_two_full_rechecks(build, http):
    urls = _covered_policy_pages(http)
    connector, _ = build(covered_urls=urls)
    captures = []
    connector.capture_sink = _published_sink(captures)
    for _ in range(2):
        assert docs(connector) == []
        assert connector.scan_complete
        assert connector.backlog == []
        assert connector.errors == []
    assert len(captures) == 202
    assert all(capture["raw_html"] and capture["revision_hash"] for capture in captures)
    assert all(capture["_asset_budget_remaining"] == 100 for capture in captures)


def test_101_unchanged_policies_leave_budget_for_one_changed_revision(build, http):
    urls = _covered_policy_pages(http, count=102)
    http[0][urls[-1]] = (200, article("已变更的医保待遇条款。"), "text/html")
    connector, _ = build(covered_urls=urls, max_assets=1)
    captures = []
    connector.capture_sink = _published_sink(captures, ingestion_urls=[urls[-1]])
    result = docs(connector)
    assert len(result) == 1
    assert result[0].metadata["source_url"] == urls[-1]
    assert "已变更的医保待遇条款。" in result[0].blob.decode()
    assert result[0].metadata["policy_ingestion_required"] == "1"
    assert connector.scan_complete
    assert len(captures) == 102
    assert captures[-1]["revision_hash"]


def test_changed_group_over_budget_is_archived_once_as_deferred(build, http):
    urls = _covered_policy_pages(http, count=103)
    connector, _ = build(covered_urls=urls, max_assets=1)
    captures = []
    connector.capture_sink = _published_sink(captures, ingestion_urls=urls[-2:])
    result = []
    for batch in connector.load_from_state():
        result.extend(batch)
    assert len(result) == 1 and result[0].metadata["source_url"] == urls[-2]
    assert len(captures) == 103
    deferred = captures[-1]
    assert deferred["source_url"] == urls[-1]
    assert deferred["_asset_budget_remaining"] == 0
    assert deferred["raw_html"] and deferred["assets"][0]["blob"]
    assert deferred["revision_hash"] is None
    assert deferred["errors"] == ["asset_limit: deferred"]
    assert connector.backlog == [urls[-1]]
    assert not connector.scan_complete

    assert connector.discovery_complete
    assert connector.observation_complete
    assert not connector.scan_complete
    assert connector.retryable_errors == []


def test_new_parse_build_requires_assets_even_for_unchanged_policies(build, http):
    urls = _covered_policy_pages(http)
    connector, _ = build(covered_urls=urls, parse_build_key="parse-v2")
    captures = []
    connector.capture_sink = _published_sink(captures, published_build="parse-v1")
    result = []
    for batch in connector.load_from_state():
        result.extend(batch)
    assert len(result) == 100
    assert all(document.metadata["parse_build_key"] == "parse-v2" for document in result)
    assert len(captures) == 101
    assert captures[-1]["revision_hash"] is None
    assert connector.backlog == [urls[-1]]

    assert connector.discovery_complete
    assert connector.observation_complete
    assert not connector.scan_complete
    assert connector.retryable_errors == []


def test_no_sink_still_enforces_parse_asset_limit(build, http):
    urls = _covered_policy_pages(http, count=2)
    connector, _ = build(covered_urls=urls, max_assets=1)
    connector.capture_sink = None
    result = []
    for batch in connector.load_from_state():
        result.extend(batch)
    assert len(result) == 1
    assert connector.backlog == [urls[-1]]

    assert connector.discovery_complete
    assert connector.observation_complete
    assert not connector.scan_complete
    assert connector.retryable_errors == []


def test_sink_cannot_bypass_budget_by_omitting_deferral(build, http):
    urls = _covered_policy_pages(http, count=2)
    connector, _ = build(covered_urls=urls, max_assets=1)
    captures = []
    connector.capture_sink = lambda capture: captures.append(capture) or {"policy_ingestion_required": "1"}
    result = []
    with pytest.raises(ValueError, match="asset_limit"):
        for batch in connector.load_from_state():
            result.extend(batch)
    assert len(result) == 1
    assert len(captures) == 2
    assert captures[-1]["revision_hash"]
    assert connector.backlog == [urls[-1]]


def test_partial_observations_keep_failure_and_review_counts_but_finish_task(build, http):
    pages, _ = http
    review = BASE + "/gfxwj/20261001/review.html"
    fourth = BASE + "/gfxwj/20261001/deferred.html"
    title = "关于废止有关文件的通知"
    pages[LIST] = (200, listing() + listing(OLDER) + listing(review).replace("医保通知", title) + listing(fourth), "text/html")
    pages[DETAIL] = (200, article(), "text/html")
    pages[OLDER] = (200, article(attachments=[ATTACH]), "text/html")
    pages[ATTACH] = (404, b"missing", "text/plain")
    pages[review] = (200, article("有关事项须核对文号。").replace("医保通知", title), "text/html")
    connector, captures = build(max_new_policies=3)
    result = docs(connector)
    assert len(result) == 1 and len(captures) == 4
    assert connector.discovery_complete and connector.observation_complete
    assert not connector.scan_complete
    assert connector.scan_health["discovered_count"] == 4
    assert connector.scan_health["observed_count"] == 4
    assert connector.scan_health["sealed_count"] == 1
    assert connector.scan_health["failed_count"] == 1
    assert connector.scan_health["needs_review_count"] == 1
    assert connector.scan_health["deferred_count"] == 1
    assert connector.scan_health["ingestion_asset_count"] == 1
    assert len(connector.retryable_errors) == 1
    assert "HTTP failure" in connector.retryable_errors[0]
    assert connector.backlog == [fourth]


def test_archive_failure_flushes_previous_valid_batch_then_remains_fatal(build, http):
    pages, _ = http
    pages[LIST] = (200, listing() + listing(OLDER), "text/html")
    pages[DETAIL] = pages[OLDER] = (200, article(), "text/html")
    connector, _ = build(batch_size=2)
    captures = []

    def sink(capture):
        if capture["source_url"] == OLDER:
            capture["errors"].append("archive_failed: object-store readback mismatch")
            capture["revision_hash"] = None
        captures.append(capture)
        return {"policy_ingestion_required": "1"}

    connector.capture_sink = sink
    result = []
    with pytest.raises(ValueError, match="archive_failed"):
        for batch in connector.load_from_state():
            result.extend(batch)
    assert len(result) == 1 and len(captures) == 2
    assert connector.discovery_complete
    assert not connector.observation_complete and not connector.scan_complete
    assert connector.scan_health["observation_complete"] is False
    assert connector.scan_health["observed_count"] == 2
    assert connector.retryable_errors == ["archive_failed: object-store readback mismatch"]


def test_detail_dns_rebinding_is_fatal_after_successful_discovery(build, http, monkeypatch):
    pages, calls = http
    pages[LIST] = (200, listing(), "text/html")
    pages[DETAIL] = (200, article(), "text/html")
    original_send = requests.Session.send

    def send(session, prepared, **kwargs):
        response = original_send(session, prepared, **kwargs)
        if prepared.url == LIST:
            monkeypatch.setattr(ssrf_guard, "_orig_getaddrinfo", lambda *a, **k: [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 0))])
        return response

    monkeypatch.setattr(requests.Session, "send", send)
    connector, captures = build()
    with pytest.raises(ValueError, match="non-public"):
        docs(connector)
    assert connector.discovery_complete
    assert not connector.observation_complete
    assert not captures
    assert DETAIL not in [call[0] for call in calls]


def test_complete_health_is_reset_before_a_failed_second_discovery(build, http):
    pages, _ = http
    pages[LIST] = (200, listing(), "text/html")
    pages[DETAIL] = (200, article(), "text/html")
    connector, _ = build()
    assert len(docs(connector)) == 1
    assert connector.scan_health["scan_complete"]
    pages[LIST] = (403, b"blocked", "text/html")
    with pytest.raises(ValueError):
        docs(connector)
    assert not connector.discovery_complete and not connector.observation_complete and not connector.scan_complete
    assert connector.scan_health["observed_count"] == 0


def test_task_deadline_during_detail_processing_stays_fatal(build, http, monkeypatch):
    pages, _ = http
    pages[LIST] = (200, listing(), "text/html")
    pages[DETAIL] = (200, article(), "text/html")
    original_send = requests.Session.send

    def send(session, prepared, **kwargs):
        response = original_send(session, prepared, **kwargs)
        if prepared.url == DETAIL:
            connector._started = -10000
        return response

    connector, captures = build()
    monkeypatch.setattr(requests.Session, "send", send)
    with pytest.raises(ValueError, match="task timeout"):
        docs(connector)
    assert connector.discovery_complete and not connector.observation_complete
    assert not captures


def test_sink_metadata_protocol_error_cannot_finish_observations(build, http):
    pages, _ = http
    pages[LIST] = (200, listing(), "text/html")
    pages[DETAIL] = (200, article(), "text/html")
    connector, _ = build()
    connector.capture_sink = lambda capture: None
    with pytest.raises(ValueError, match="string metadata"):
        docs(connector)
    assert connector.discovery_complete and not connector.observation_complete


def test_attachment_request_timeout_is_persisted_retryable_policy_gap(build, http):
    pages, _ = http
    pages[LIST] = (200, listing(), "text/html")
    pages[DETAIL] = (200, article(attachments=[ATTACH]), "text/html")
    pages[ATTACH] = requests.Timeout("attachment read timed out")
    connector, captures = build()
    assert docs(connector) == []
    assert connector.discovery_complete and connector.observation_complete and not connector.scan_complete
    assert len(captures) == 1 and captures[0]["revision_hash"] is None
    assert len(connector.retryable_errors) == 1
    assert "timed out" in connector.retryable_errors[0]


def test_complete_health_counts_published_reuse_without_ingestion(build, http):
    urls = _covered_policy_pages(http, count=2)
    connector, _ = build(covered_urls=urls)
    captures = []
    connector.capture_sink = _published_sink(captures)
    assert docs(connector) == []
    assert connector.discovery_complete and connector.observation_complete and connector.scan_complete
    assert connector.scan_health["sealed_count"] == connector.scan_health["unchanged_count"] == 2
    assert connector.scan_health["ingestion_asset_count"] == 0
    assert connector.scan_health["retryable_error_count"] == 0


@pytest.mark.parametrize(
    "proxy",
    [
        "socks5h://host.docker.internal:7897",
        "http://host.docker.internal:7897",
        "https://host.docker.internal:7897",
        "socks5://evil.example:7897",
        "socks5://127.0.0.1:7897",
        "socks5://host.docker.internal:1080",
        "socks5://host.docker.internal",
        "socks5://host.docker.internal:7897/path",
        "socks5://host.docker.internal:7897?target=other",
        "socks5://host.docker.internal:7897#other",
        "socks5://@host.docker.internal:7897",
        "socks5://:@host.docker.internal:7897",
    ],
)
def test_socks_transport_rejects_remote_dns_and_unapproved_endpoints(build, proxy):
    with pytest.raises(ValueError):
        build(socks5_proxy_url=proxy)


def test_socks_credentials_are_rejected_without_leaking_input(build):
    private_value = "private-test-secret"
    url = "socks5://ignored-user:" + private_value + "@host.docker.internal:7897"
    with pytest.raises(ValueError) as error:
        build(socks5_proxy_url=url)
    assert private_value not in str(error.value)
    assert "ignored-user" not in str(error.value)


def test_explicit_socks_transport_retains_dns_pin_and_verified_tls(build, http, monkeypatch):
    pages, calls = http
    pages[LIST] = (200, listing(), "text/html")
    pages[DETAIL] = (200, article(), "text/html")
    monkeypatch.setenv("HTTPS_PROXY", "http://ambient.example:9999")
    monkeypatch.setattr(ssrf_guard, "_orig_getaddrinfo", lambda host, port, *a, **k: [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("1.1.1.1" if port is None else "8.8.8.8", port or 0))])
    original_send = requests.Session.send
    target_ips = []

    def send(session, prepared, **kwargs):
        target_ips.append(socket.getaddrinfo("ybj.sh.gov.cn", 443)[0][4][0])
        return original_send(session, prepared, **kwargs)

    monkeypatch.setattr(requests.Session, "send", send)
    connector, _ = build(socks5_proxy_url="socks5://host.docker.internal:7897")
    assert len(docs(connector)) == 1
    assert connector._session.trust_env is False
    assert target_ips == ["1.1.1.1"] * 3
    assert all(call[2]["proxies"] == {"https": "socks5://host.docker.internal:7897"} for call in calls)
    assert all(call[2]["verify"] is True for call in calls)
    assert all(not call[2]["allow_redirects"] for call in calls)


def test_default_transport_ignores_ambient_proxy(build, http, monkeypatch):
    pages, calls = http
    pages[LIST] = (200, listing(), "text/html")
    pages[DETAIL] = (200, article(), "text/html")
    monkeypatch.setenv("HTTPS_PROXY", "http://ambient.example:9999")
    assert len(docs(build()[0])) == 1
    assert all(call[2]["proxies"] == {} for call in calls)
    assert all(call[2]["verify"] is True for call in calls)


def test_unreadable_list_error_identifies_the_actual_page(build, http):
    pages, _ = http
    following = BASE + "/gfxwj/index_2.html"
    pages[LIST] = (200, listing(next_url=following), "text/html")
    pages[following] = (200, "<title>返回首页</title><p>当前栏目无可读取政策链接</p>", "text/html")
    connector, _ = build()
    with pytest.raises(ValueError) as error:
        docs(connector)
    assert "no readable links" in str(error.value)
    assert following in str(error.value)
    assert not connector.discovery_complete


def test_transient_tls_connection_errors_retry_with_verified_transport(build, http):
    pages, calls = http
    pages[LIST] = [requests.exceptions.SSLError("UNEXPECTED_EOF_WHILE_READING"), requests.ConnectionError("transient reset"), (200, listing(), "text/html")]
    pages[DETAIL] = (200, article(), "text/html")
    connector, _ = build()
    assert len(docs(connector)) == 1
    assert len([call for call in calls if call[0] == LIST]) == 3
    assert all(call[2]["verify"] is True for call in calls)


def test_connection_error_retries_are_bounded_and_leave_discovery_incomplete(build, http):
    pages, calls = http
    pages[LIST] = requests.ConnectionError("connection reset")
    connector, _ = build()
    with pytest.raises(ValueError, match="HTTP failure"):
        docs(connector)
    assert len([call for call in calls if call[0] == LIST]) == 3
    assert not connector.discovery_complete
    assert len(connector.retryable_errors) == 1


def _historical_listing(url, title="历史规范性文件", total=3, current=1):
    # Minimal phone/desktop structure from gfxwj/index_8.html's real historical links.
    return f'''<meta name="ColumnName" content="规范性文件"><title>规范性文件</title>
    <div class="policy-title"><a href="{url}" title="{title}" id="ivs_title">{title}</a></div>
    <div id="medicalDynamicContentDiv"><span class="span-title"><a href="{url}" id="ivs_title">{title}</a></span></div>
    <script>$(".paginationPhone").pagination({{totalPage: {total}, callBack: function(i) {{location.href = 'index_' + i + '.html';}}}});
    $(".paginationPhone").pagination("setPage", {current}, {total});</script>'''


def test_historical_policy_dom_traverses_mixed_and_all_historical_pages_without_fetch(build, http):
    pages, calls = http
    historical = [BASE + "/zxzc/20181110/0038-2915.html", BASE + "/zxzc/20180724/0038-4629.html", BASE + "/zxzc/20161107/0038-2928.html"]
    pages[LIST] = (200, _historical_listing(historical[0]) + listing(), "text/html")
    second = BASE + "/gfxwj/index_2.html"
    third = BASE + "/gfxwj/index_3.html"
    pages[second] = (200, _historical_listing(historical[1], current=2), "text/html")
    pages[third] = (200, _historical_listing(historical[2], current=3), "text/html")
    pages[DETAIL] = (200, article(), "text/html")
    connector, captures = build(max_new_policies=1)
    result = docs(connector)
    assert len(result) == 1
    assert connector.discovery_complete and connector.observation_complete and not connector.scan_complete
    assert len(captures) == 4
    assert connector.scan_health["out_of_scope_count"] == 3
    assert connector.scan_health["observed_count"] == 4
    assert connector.retryable_errors == []
    assert not any(url in [call[0] for call in calls] for url in historical)
    excluded = [capture for capture in captures if capture["source_url"] in historical]
    assert len(excluded) == 3
    for capture in excluded:
        assert capture["revision_hash"] is None and capture["raw_html"] == b""
        assert capture["title"] == "历史规范性文件"
        assert "needs_review: historical source path outside configured scope" in capture["errors"]
        assert capture["relations"][0]["kind"] == "discovered_on"
        assert capture["relations"][0]["source_list_url"] == capture["source_list_url"]
    assert excluded[-1]["source_list_url"] == third


@pytest.mark.parametrize(
    "historical",
    [
        "https://evil.example/zxzc/20181110/0038-2915.html",
        "http://ybj.sh.gov.cn/zxzc/20181110/0038-2915.html",
        BASE + "/other/20181110/0038-2915.html",
        BASE + "/zxzc/20181110/../0038-2915.html",
    ],
)
def test_historical_discovery_never_admits_arbitrary_source_paths(build, http, historical):
    pages, _ = http
    pages[LIST] = (200, _historical_listing(historical, total=1), "text/html")
    with pytest.raises(ValueError, match="no readable links"):
        docs(build()[0])


def test_historical_footer_links_do_not_prove_a_readable_policy_page(build, http):
    pages, _ = http
    historical = BASE + "/zxzc/20181110/0038-2915.html"
    pages[LIST] = (200, f'<title>模板</title><footer><a href="{historical}">历史规范性文件</a></footer>', "text/html")
    with pytest.raises(ValueError, match="no readable links"):
        docs(build()[0])
    with pytest.raises(ValueError):
        build(seed_urls=[historical])
    with pytest.raises(ValueError):
        build(covered_urls=[historical])


def test_actual_interpretation_last_page_is_observed_outside_fetch_scope(build, http):
    pages, calls = http
    last = BASE + "/zcjd/index_11.html"
    historical = BASE + "/zxgkzfxx/20080401/0038-2887.html"
    title = "关于本市2008医保年度城镇职工有关医保待遇标准调整的问答"
    html = f'''<meta name="ColumnName" content="政策解读"><title>政策解读</title>
    <div class="policy-title"><a href="{historical}" title="{title}" id="ivs_title">{title}</a></div>
    <script>$(".paginationPhone").pagination({{totalPage: 11, callBack: function(i) {{location.href = 'index_' + i + '.html';}}}});
    $(".paginationPhone").pagination("setPage", 11, 11);</script>'''
    pages[last] = (200, html, "text/html")
    connector, captures = build(list_urls=[last])
    assert docs(connector) == []
    assert connector.discovery_complete and connector.observation_complete and not connector.scan_complete
    assert connector.scan_health["out_of_scope_count"] == 1
    assert captures[0]["source_kind"] == "zcjd"
    assert captures[0]["title"] == title
    assert captures[0]["source_url"] == historical
    assert captures[0]["source_list_url"] == last
    assert captures[0]["relations"] == [{"kind": "discovered_on", "source_list_url": last, "status": "needs_review"}]
    assert historical not in [call[0] for call in calls]
    with pytest.raises(ValueError):
        build(seed_urls=[historical])
    with pytest.raises(ValueError):
        build(covered_urls=[historical])


def test_discovered_historical_path_does_not_enable_fetch_redirects(build, http):
    pages, calls = http
    historical = BASE + "/zxgkzfxx/20080401/0038-2887.html"
    pages[LIST] = (200, listing(), "text/html")
    pages[DETAIL] = (302, b"", {"Location": historical})
    with pytest.raises(ValueError, match="allowlist"):
        docs(build()[0])
    assert historical not in [call[0] for call in calls]


LONG_POLICY_TITLE = "上海市医疗保障局关于延长《关于本市城镇从事自由职业人员、无雇工的个体工商户以及其他灵活就业人员享受基本医疗保险待遇设置等待期的通知》有效期的通知"


def test_long_chinese_filename_fits_native_bytes_without_changing_policy_content(build, http):
    pages, _ = http
    pages[LIST] = (200, listing(), "text/html")
    body = "原文待遇与等待期条款完整保留。"
    pages[DETAIL] = (200, article(body).replace('content="医保通知"', f'content="{LONG_POLICY_TITLE}"'), "text/html")
    connector, captures = build()
    document = docs(connector)[0]
    assert len(document.semantic_identifier.encode("utf-8")) <= 255
    capture = captures[0]
    suffix = f"-{capture['policy_key'][:12]}-{capture['revision_hash'][:12]}-9934fa04-body.txt"
    assert document.semantic_identifier == LONG_POLICY_TITLE[:70] + suffix
    assert document.metadata["title"] == capture["title"] == LONG_POLICY_TITLE
    assert document.metadata["source_url"] == DETAIL
    assert document.metadata["revision_hash"] == capture["revision_hash"]
    assert document.blob == (LONG_POLICY_TITLE + "\n\n" + body).encode()
    assert document.id == "shanghai_medical:" + ":".join((capture["policy_key"], capture["revision_hash"], "parse-v1", "body", hashlib.sha256(document.blob).hexdigest()))


@pytest.mark.parametrize("title, visible", [
    ("汉" * 70 + "A", "汉" * 70 + "A"),  # 211 title bytes + 44 suffix bytes = 255.
    ("汉" * 70 + "AB", "汉" * 70 + "A"),
    ("A" + "😀" * 60, "A" + "😀" * 52),
    ("abc" * 60, "abc" * 40),  # Preserve the existing 120-character cap.
    ("短标题", "短标题"),
])
def test_asset_filename_byte_boundary_preserves_complete_unicode_and_short_names(title, visible):
    name = collector._asset_filename(title, "p" * 64, "r" * 64, "parse-v1", "body", ".txt")
    assert name == visible + "-pppppppppppp-rrrrrrrrrrrr-9934fa04-body.txt"
    assert len(name.encode("utf-8")) <= 255
    assert name.encode("utf-8").decode("utf-8") == name


@pytest.mark.parametrize("extension", [".pdf", ".docx"])
def test_asset_filename_preserves_full_attachment_role_and_extension(extension):
    name = collector._asset_filename("汉" * 120, "p" * 64, "r" * 64, "parse-v1", "attachment:0123456789abcdef", extension)
    assert name == "汉" * 62 + "-pppppppppppp-rrrrrrrrrrrr-9934fa04-attachment:0123456789abcdef" + extension
    assert len(name.encode("utf-8")) <= 255


def test_asset_filename_truncation_does_not_merge_distinct_stable_suffixes():
    title = "汉" * 120
    inputs = [
        ("p" * 64, "r" * 64, "parse-v1", "body"),
        ("q" * 64, "r" * 64, "parse-v1", "body"),
        ("p" * 64, "s" * 64, "parse-v1", "body"),
        ("p" * 64, "r" * 64, "parse-v2", "body"),
        ("p" * 64, "r" * 64, "parse-v1", "attachment:0123456789abcdef"),
    ]
    names = {collector._asset_filename(title, *values, ".txt") for values in inputs}
    assert len(names) == len(inputs)
    assert all(len(name.encode("utf-8")) <= 255 for name in names)


def test_asset_filename_rejects_suffix_that_cannot_fit_native_limit():
    with pytest.raises(ValueError, match="filename suffix"):
        collector._asset_filename("标题", "p" * 64, "r" * 64, "parse-v1", "x" * 216, ".txt")
