"""Bounded collection of immutable Shanghai official medical-policy captures."""

import hashlib
import io
import json
import posixpath
import re
import time
import threading
import zipfile
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any, Callable
from urllib.parse import unquote, urljoin, urlsplit, urlunsplit
from urllib.robotparser import RobotFileParser

import bs4
import requests
import xxhash
from pydantic import BaseModel, ConfigDict, Field, field_validator

from api.constants import FILE_NAME_LEN_LIMIT
from common.data_source.config import DocumentSource
from common.data_source.interfaces import LoadConnector, PollConnector
from common.data_source.models import Document
from common.ssrf_guard import assert_url_is_safe, pin_dns

_BASE = "https://ybj.sh.gov.cn"
_COLUMNS = ("gfxwj", "qtwj", "zcjd")
_LIST_URLS = [f"{_BASE}/{column}/index.html" for column in _COLUMNS]
_SEED_URLS = [
    f"{_BASE}/gfxwj/20260911/f57a8b26aa414a15a57d1040c1024600.html",
    f"{_BASE}/zcjd/20260911/e36f80c1d81a422384db6261d96de59d.html",
    f"{_BASE}/qtwj/20260914/52a884336736421ba9ace49d84ddbc80.html",
]
_USER_AGENT = "RAGFlow-ShanghaiMedical/1.0"
_DETAIL_PATH = re.compile(r"^/(gfxwj|qtwj|zcjd)/\d{8}/[^/]+\.html$")
_HISTORICAL_PATH = re.compile(r"^/(?:zxzc|zxgkzfxx)/\d{8}/[^/]+\.html$")
_LIST_PATH = re.compile(r"^/(gfxwj|qtwj|zcjd)/index(?:_\d+)?\.html$")
_DATE = r"(\d{4})[-年](\d{1,2})[-月](\d{1,2})日?"
_CHANGE_NOTICE = re.compile(r"延长|延期|废止|废除|修订|修改|修正|调整|失效|有效期|停止执行|终止|撤销|继续实施|清理")


class _FatalScanError(ValueError):
    """A source safety, deadline, or persistence failure cannot finish the job."""


def _asset_filename(title: str, policy_key: str, revision_hash: str, parse_build_key: str, role: str, extension: str) -> str:
    suffix = f"-{policy_key[:12]}-{revision_hash[:12]}-{hashlib.sha256(parse_build_key.encode()).hexdigest()[:8]}-{role}{extension}"
    title_bytes = FILE_NAME_LEN_LIMIT - len(suffix.encode("utf-8"))
    if title_bytes < 0:
        raise ValueError("Policy filename suffix exceeds native filename limit")
    return title[:120].encode("utf-8")[:title_bytes].decode("utf-8", errors="ignore") + suffix


def _official_url(url: str, kind: str) -> str:
    parsed = urlsplit(url.strip())
    if parsed.scheme != "https" or parsed.hostname != "ybj.sh.gov.cn" or parsed.port not in (None, 443) or parsed.username or parsed.password:
        raise ValueError("Only official HTTPS ybj.sh.gov.cn sources are allowed")
    path = unquote(parsed.path)
    if "\\" in path or posixpath.normpath(path) != path or parsed.query:
        raise ValueError("Source URL has an unsafe path or query")
    valid = (
        (kind == "list" and _LIST_PATH.fullmatch(path))
        or (kind == "detail" and _DETAIL_PATH.fullmatch(path))
        or (kind == "attachment" and path.startswith("/cmsres/"))
        or (kind == "robots" and path == "/robots.txt")
    )
    if not valid:
        raise ValueError(f"Source URL is outside the {kind} allowlist")
    return urlunsplit(("https", "ybj.sh.gov.cn", parsed.path, "", ""))


class ShanghaiMedicalConnectorConfig(BaseModel):
    """The same offline validation schema used by connector creation and collection."""

    model_config = ConfigDict(extra="forbid", strict=True, hide_input_in_errors=True)
    list_urls: list[str] = Field(default_factory=lambda: list(_LIST_URLS), min_length=1, max_length=3)
    seed_urls: list[str] = Field(default_factory=lambda: list(_SEED_URLS), max_length=200)
    covered_urls: list[str] = Field(default_factory=list, max_length=200)
    covered_document_numbers: list[str] = Field(default_factory=list, max_length=200)
    topics: list[str] = Field(default_factory=lambda: ["医疗保险", "医保", "生育", "个人账户", "长期护理", "医疗互助", "门诊大病"], max_length=40)
    parse_build_key: str = Field(min_length=1, max_length=128)
    parse_generation: int = Field(default=0, ge=0, le=1000000)
    batch_size: int = Field(default=2, ge=1, le=100)
    max_pages_per_column: int = Field(default=100, ge=1, le=100)
    max_new_policies: int = Field(default=40, ge=1, le=40)
    max_assets: int = Field(default=100, ge=1, le=100)
    max_policies: int = Field(default=200, ge=1, le=200)
    request_interval_seconds: float = Field(default=2.0, ge=2, le=60)
    request_timeout_seconds: float = Field(default=15.0, gt=0, le=15)
    max_html_bytes: int = Field(default=5 * 1024 * 1024, ge=1024, le=5 * 1024 * 1024)
    max_attachment_bytes: int = Field(default=20 * 1024 * 1024, ge=1024, le=20 * 1024 * 1024)
    max_archive_bytes: int = Field(default=200 * 1024 * 1024, ge=1024, le=200 * 1024 * 1024)
    task_timeout_seconds: int = Field(default=3600, ge=1, le=3600)
    sync_deleted_files: bool = False
    socks5_proxy_url: str | None = Field(default=None, repr=False)

    @field_validator("socks5_proxy_url")
    @classmethod
    def validate_socks_transport(cls, value):
        if value is None:
            return None
        try:
            parsed = urlsplit(value)
            allowed = (
                parsed.scheme == "socks5"
                and parsed.hostname == "host.docker.internal"
                and parsed.port == 7897
                and parsed.username is None
                and parsed.password is None
                and not parsed.path
                and not parsed.query
                and not parsed.fragment
            )
        except ValueError:
            allowed = False
        if not allowed:
            raise ValueError("Only the credential-free Docker SOCKS5 endpoint on port 7897 is allowed")
        return "socks5://host.docker.internal:7897"

    @field_validator("list_urls", "seed_urls", "covered_urls")
    @classmethod
    def validate_urls(cls, values, info):
        kind = "list" if info.field_name == "list_urls" else "detail"
        return list(dict.fromkeys(_official_url(value, kind) for value in values))

    @field_validator("covered_document_numbers")
    @classmethod
    def normalize_document_numbers(cls, values):
        result = []
        for value in values:
            number = re.sub(r"\s+", "", value)
            if not number or len(number) > 128:
                raise ValueError("covered document numbers must be nonempty and at most 128 characters")
            result.append(number)
        return list(dict.fromkeys(result))

    @field_validator("parse_build_key")
    @classmethod
    def validate_build_key(cls, value):
        if not value.strip():
            raise ValueError("parse_build_key must not be blank")
        return value.strip()

    @field_validator("sync_deleted_files")
    @classmethod
    def disable_deletion(cls, value):
        if value:
            raise ValueError("Policy history cannot be deleted by source disappearance")
        return value


class ShanghaiMedicalConnector(LoadConnector, PollConnector):
    """Discover URLs independently of dates, archive captures before yielding assets."""

    def __init__(self, config: ShanghaiMedicalConnectorConfig):
        self.config = config
        self.capture_sink: Callable[[dict], dict] | None = None
        self.should_stop: Callable[[], bool] = lambda: False
        self._cancelled = threading.Event()
        self.discovery_complete = False
        self.observation_complete = False
        self.scan_complete = False
        self.scan_health = self._empty_scan_health()
        self.retryable_errors: list[str] = []
        self.errors: list[str] = []
        self.backlog: list[str] = []
        self._discovered_provenance: dict[str, dict] = {}
        self._cache: dict[str, tuple[bytes, dict, float]] = {}
        self._session = requests.Session()
        self._session.trust_env = False
        if config.socks5_proxy_url:
            # socks5 keeps target DNS resolution local, inside the existing pin.
            self._session.proxies["https"] = config.socks5_proxy_url
        self._session.headers.update({"User-Agent": _USER_AGENT})
        self._started = time.monotonic()
        self._last_request: float | None = None
        self._downloaded = 0
        self._robots: RobotFileParser | None = None

    @classmethod
    def build_connector(cls, config: dict[str, Any]):
        # Credentials are consumed by the existing factory separately.
        values = dict(config)
        credentials = values.pop("credentials", {})
        connector = cls(ShanghaiMedicalConnectorConfig.model_validate(values))
        connector.load_credentials(credentials)
        return connector

    def load_credentials(self, credentials: dict[str, Any]):
        if credentials:
            raise ValueError("ShanghaiMedicalConnector accepts only empty credentials")
        return None

    @staticmethod
    def _empty_scan_health():
        return dict(
            discovery_complete=False,
            observation_complete=False,
            scan_complete=False,
            discovered_count=0,
            observed_count=0,
            sealed_count=0,
            failed_count=0,
            needs_review_count=0,
            deferred_count=0,
            unchanged_count=0,
            ingestion_policy_count=0,
            ingestion_asset_count=0,
            retryable_error_count=0,
            out_of_scope_count=0,
        )

    def _retryable(self, error):
        if not any(previous in error for previous in self.retryable_errors):
            self.retryable_errors.append(error)
        self.scan_health["retryable_error_count"] = len(self.retryable_errors)

    def _begin(self):
        self.discovery_complete = False
        self.observation_complete = False
        self.scan_complete = False
        self.scan_health = self._empty_scan_health()
        self.retryable_errors = []
        self.errors = []
        self.backlog = []
        self._discovered_provenance = {}
        self._started = time.monotonic()
        self._downloaded = 0
        self._robots = None

    def validate_connector_settings(self):
        self._begin()
        self._read_robots()
        for url in self.config.list_urls:
            column = urlsplit(url).path.split("/")[1]
            visited, hashes = set(), set()
            _, next_pages = self._read_list_page(url, column, visited, hashes)
            if next_pages:
                self._read_list_page(next_pages[0], column, visited, hashes)

    def load_from_state(self):
        yield from self._scan()

    def poll_source(self, start, end):
        # Official publication dates cannot exclude newly found or amended URLs.
        del start, end
        yield from self._scan()

    def cancel(self):
        self._cancelled.set()
        self._session.close()

    def _check(self):
        if self._cancelled.is_set() or self.should_stop():
            raise _FatalScanError("Policy scan cancelled")
        if time.monotonic() - self._started >= self.config.task_timeout_seconds:
            raise _FatalScanError("Policy scan task timeout")

    def _wait(self, seconds):
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            self._check()
            time.sleep(min(0.25, end - time.monotonic()))

    def _read_robots(self):
        blob, headers = self._fetch(_BASE + "/robots.txt", "robots")
        if headers.get("Content-Type", "").split(";")[0].lower() not in {"text/plain", "text/html"}:
            raise ValueError("Unreadable robots response")
        self._robots = RobotFileParser()
        self._robots.parse(blob.decode("utf-8-sig").splitlines())

    def _fetch(self, url, kind):
        current = _official_url(url, kind)
        cached = self._cache.get(current)
        headers = {}
        if kind != "robots" and cached and time.time() - cached[2] < 7 * 86400:
            for source, target in (("ETag", "If-None-Match"), ("Last-Modified", "If-Modified-Since")):
                if cached[1].get(source):
                    headers[target] = cached[1][source]
        for redirect in range(6):
            self._check()
            if self._robots and not self._robots.can_fetch(_USER_AGENT, current):
                raise _FatalScanError("robots disallows policy source")
            for attempt in range(3):
                self._check()
                if self._last_request is not None:
                    interval = max(self.config.request_interval_seconds, self._robots.crawl_delay(_USER_AGENT) or 0) if self._robots else self.config.request_interval_seconds
                    self._wait(max(0, interval - (time.monotonic() - self._last_request)))
                try:
                    hostname, ip = assert_url_is_safe(current, allowed_schemes=frozenset({"https"}))
                except ValueError as exc:
                    raise _FatalScanError(str(exc)) from exc
                response = None
                try:
                    self._last_request = time.monotonic()
                    with pin_dns(hostname, ip):
                        response = self._session.get(current, timeout=self.config.request_timeout_seconds, headers=headers, stream=True, allow_redirects=False)
                        if (response.status_code == 429 or 500 <= response.status_code < 600) and attempt < 2:
                            retry_after = response.headers.get("Retry-After", "")
                            delay = 2**attempt
                            if retry_after:
                                try:
                                    delay = max(delay, float(retry_after))
                                except ValueError:
                                    delay = max(delay, (parsedate_to_datetime(retry_after) - datetime.now(timezone.utc)).total_seconds())
                            response.close()
                            self._wait(delay)
                            continue
                        if response.status_code in {301, 302, 303, 307, 308}:
                            target = response.headers.get("Location")
                            if not target:
                                raise ValueError("Policy redirect has no location")
                            try:
                                current = _official_url(urljoin(current, target), kind)
                            except ValueError as exc:
                                raise _FatalScanError(str(exc)) from exc
                            headers = {}
                            break
                        if response.status_code == 304:
                            if not cached or current != url:
                                raise ValueError("304 without archived response")
                            return cached[0], cached[1]
                        if kind == "robots" and response.status_code == 404:
                            return b"", {"Content-Type": "text/plain"}
                        response.raise_for_status()
                        maximum = self.config.max_attachment_bytes if kind == "attachment" else self.config.max_html_bytes
                        if int(response.headers.get("Content-Length", 0)) > maximum:
                            raise ValueError("Policy source size limit exceeded")
                        parts = []
                        size = 0
                        for part in response.iter_content(chunk_size=65536):
                            self._check()
                            size += len(part)
                            self._downloaded += len(part)
                            if self._downloaded > self.config.max_archive_bytes:
                                raise _FatalScanError("Policy source archive size limit exceeded")
                            if size > maximum:
                                raise ValueError("Policy source size limit exceeded")
                            parts.append(part)
                        blob = b"".join(parts)
                        result_headers = dict(response.headers)
                        self._cache[current] = (blob, result_headers, time.time())
                        return blob, result_headers
                except (requests.Timeout, requests.ConnectionError) as exc:
                    if attempt == 2:
                        error = f"Policy source timed out: {current}" if isinstance(exc, requests.Timeout) else f"Policy source HTTP failure: {current}: {exc}"
                        self._retryable(error)
                        raise ValueError(error) from exc
                    self._wait(2**attempt)
                except requests.RequestException as exc:
                    error = f"Policy source HTTP failure: {current}: {exc}"
                    self._retryable(error)
                    raise ValueError(error) from exc
                finally:
                    if response is not None:
                        response.close()
            else:
                raise ValueError("Policy source retry exhausted")
        raise ValueError("Policy redirect limit exceeded")

    @staticmethod
    def _html(blob, headers):
        mime = headers.get("Content-Type", "").split(";")[0].strip().lower()
        if mime not in {"text/html", "application/xhtml+xml"}:
            raise ValueError("Policy page is not HTML")
        return bs4.BeautifulSoup(blob, "html.parser", from_encoding="utf-8")

    def _list_links(self, blob, url):
        soup = bs4.BeautifulSoup(blob, "html.parser", from_encoding="utf-8")
        links = {}
        for anchor in soup.select("a[href]"):
            target = urljoin(url, anchor["href"])
            try:
                target = _official_url(target, "detail")
            except ValueError:
                # Historical URLs are observed only where the official page
                # identifies a policy item; they never enter the fetch allowlist.
                structured = anchor.get("id") == "ivs_title" or anchor.find_parent(class_="policy-title") is not None
                parsed = urlsplit(target)
                try:
                    historical = (
                        structured
                        and parsed.scheme == "https"
                        and parsed.hostname == "ybj.sh.gov.cn"
                        and parsed.port in (None, 443)
                        and parsed.username is None
                        and parsed.password is None
                        and not parsed.query
                        and not parsed.fragment
                        and _HISTORICAL_PATH.fullmatch(parsed.path)
                        and unquote(parsed.path) == parsed.path
                        and posixpath.normpath(parsed.path) == parsed.path
                    )
                except ValueError:
                    historical = False
                if not historical:
                    continue
                target = urlunsplit(("https", "ybj.sh.gov.cn", parsed.path, "", ""))
                provenance = self._discovered_provenance.setdefault(target, {"source_kind": urlsplit(url).path.split("/")[1], "source_list_urls": []})
                if url not in provenance["source_list_urls"]:
                    provenance["source_list_urls"].append(url)
            links[target] = anchor.get("title") or anchor.get_text(" ", strip=True)
        next_pages = []
        for anchor in soup.select("a[href]"):
            if "next" in anchor.get("rel", []) or anchor.get_text(strip=True) in {"下一页", "下页", ">", "下一页 >"}:
                next_pages.append(_official_url(urljoin(url, anchor["href"]), "list"))
        # The live site supplies static index_N URLs through this fixed JS template.
        template = soup.find_all("script")
        for script in template:
            text = script.get_text()
            match = re.search(r"totalPage\s*:\s*(\d+)", text)
            if match and "index_" in text and ".html" in text:
                total = int(match[1])
                path = urlsplit(url).path
                current_match = re.search(r"index_(\d+)\.html$", path)
                current = int(current_match[1]) if current_match else 1
                if current < total:
                    next_pages.append(_official_url(urljoin(url, f"index_{current + 1}.html"), "list"))
        return links, list(dict.fromkeys(next_pages))

    def _read_list_page(self, url, column, visited, hashes):
        self._check()
        if url in visited:
            raise ValueError("Policy pagination loop")
        if urlsplit(url).path.split("/")[1] != column:
            raise ValueError("Policy pagination changed column")
        visited.add(url)
        blob, headers = self._fetch(url, "list")
        self._html(blob, headers)
        digest = hashlib.sha256(blob).hexdigest()
        if digest in hashes:
            raise ValueError("Policy pagination repeated page")
        hashes.add(digest)
        links, next_pages = self._list_links(blob, url)
        if not links:
            raise ValueError(f"Policy list contains no readable links: {url}")
        policy_signature = ("policy_urls", tuple(sorted(links)))
        if policy_signature in hashes:
            raise ValueError("Policy pagination repeated policy list")
        hashes.add(policy_signature)
        for following in next_pages:
            if following in visited:
                raise ValueError("Policy pagination loop")
            if urlsplit(following).path.split("/")[1] != column:
                raise ValueError("Policy pagination changed column")
        return links, next_pages

    def _matches_topic(self, title):
        return not self.config.topics or any(topic in title for topic in self.config.topics)

    def _discover(self):
        found = dict.fromkeys(self.config.covered_urls + self.config.seed_urls, "")
        for first in self.config.list_urls:
            pending = [first]
            visited, hashes = set(), set()
            column = urlsplit(first).path.split("/")[1]
            while pending:
                if len(visited) >= self.config.max_pages_per_column:
                    raise ValueError("Policy pagination page limit exceeded")
                links, next_pages = self._read_list_page(pending.pop(0), column, visited, hashes)
                found.update((link, title) for link, title in links.items() if link in self._discovered_provenance or self._matches_topic(title) or _CHANGE_NOTICE.search(title))
                pending.extend(next_pages)
        return found

    def _associate_change_notice(self, capture):
        title = capture["title"]
        if self._matches_topic(title) or not _CHANGE_NOTICE.search(title):
            return
        associations = []
        for number in self.config.covered_document_numbers:
            pattern = r"\s*".join(re.escape(character) for character in number)
            for line in capture["text"].splitlines():
                match = re.search(pattern, line)
                if match:
                    associations.append(
                        {
                            "kind": "association_reference",
                            "document_number": number,
                            "source_span": line[max(0, match.start() - 256) : match.end() + 256],
                            "status": "needs_review",
                        }
                    )
                    break
        capture["covered_document_numbers_matched"] = [association["document_number"] for association in associations]
        capture["relations"] = associations
        if not associations:
            capture["errors"].append("needs_review: generic policy change notice has no confirmed covered-document reference")

    @staticmethod
    def _date(value):
        match = re.search(_DATE, value or "")
        if not match:
            return None
        try:
            return datetime(*(int(part) for part in match.groups())).date().isoformat()
        except ValueError:
            return None

    def _capture_record(self, url, observed_at):
        return {
            "policy_key": hashlib.sha256(url.encode()).hexdigest(),
            "source_url": url,
            "source_kind": urlsplit(url).path.split("/")[1],
            "title": "",
            "document_number": None,
            "published_at": None,
            "signed_at": None,
            "observed_at": observed_at,
            "raw_html": b"",
            "text": "",
            "assets": [],
            "errors": [],
            "revision_hash": None,
            "parse_build_key": self.config.parse_build_key,
        }

    def _capture(self, url, observed_at):
        capture = self._capture_record(url, observed_at)
        try:
            raw, headers = self._fetch(url, "detail")
            capture["raw_html"] = raw
            capture["raw_html_sha256"] = hashlib.sha256(raw).hexdigest()
            soup = self._html(raw, headers)
            meta = {tag.get("name", "").lower(): tag.get("content", "") for tag in soup.select("meta[name]")}
            title = meta.get("articletitle") or meta.get("dc.title")
            if not title:
                node = soup.select_one(".article-title, .policy-title, h1, h2")
                title = node.get_text(" ", strip=True) if node else ""
            capture["title"] = title
            capture["published_at"] = self._date(meta.get("pubdate") or meta.get("dc.date") or meta.get("publishdate"))
            page_text = soup.get_text(" ", strip=True)
            number = re.search(r"(?:文号|发文字号)[：:\s]*([^\s]+〔\d{4}〕\s*\d+\s*号)", page_text)
            agency = soup.select_one(".content-agency")
            agency_number = re.search(r"([^\s]+〔\d{4}〕\s*\d+\s*号)", agency.get_text(" ", strip=True)) if agency else None
            found_number = number or agency_number
            capture["document_number"] = re.sub(r"\s+", "", found_number.group(1)) if found_number else None
            signed = re.search(r"成文日期[：:\s]*" + _DATE, page_text)
            capture["signed_at"] = self._date(signed.group(0)) if signed else None
            if not capture["published_at"]:
                published = re.search(r"(?:发布时间|发布日期)[：:\s]*" + _DATE, page_text)
                capture["published_at"] = self._date(published.group(0)) if published else None
            bodies = soup.select(".content-detail, .article-content, .ArticleContent, #articleContent, #ivs_content, .TRS_Editor, .policy-content")
            if not bodies or not title:
                raise ValueError("Policy title/body structure needs_review")
            # Prefer the containing body: mobile duplicates never enter cleaned text.
            body = bodies[0]
            seen_bodies = set()
            for nested in body.select(".content-detail, .article-content, .ArticleContent, #articleContent, #ivs_content, .TRS_Editor, .policy-content"):
                signature = str(nested.get_text("\n", strip=True))
                if signature in seen_bodies:
                    nested.decompose()
                else:
                    seen_bodies.add(signature)
            for node in body.select("script, style, nav, footer, aside"):
                node.decompose()
            text = body.get_text("\n", strip=True).strip()
            if not text:
                raise ValueError("Empty policy body needs_review")
            if not capture["signed_at"]:
                signatures = []
                for paragraph in body.select("p[style]"):
                    if "right" not in paragraph.get("style", ""):
                        continue
                    previous = paragraph.find_previous_sibling("p")
                    if previous and "上海市医疗保障局" in previous.get_text():
                        date = self._date(paragraph.get_text())
                        if date:
                            signatures.append((date, paragraph.get_text(strip=True)))
                if len(signatures) == 1:
                    capture["signed_at"], capture["signed_at_source"] = signatures[0]
            capture["text"] = text
            capture["text_sha256"] = hashlib.sha256(text.encode()).hexdigest()
            body_blob = (title + "\n\n" + text).encode()
            capture["assets"].append({"role": "body", "url": url, "extension": ".txt", "blob": body_blob, "sha256": hashlib.sha256(body_blob).hexdigest()})
            targets = {}
            relations = []
            anchors = list(body.select("a[href]"))
            for container in body.parent.select(".content-attaches, .content-links, #enclosure, #xgljDiv"):
                anchors.extend(container.select("a[href]"))
            for anchor in anchors:
                target = urljoin(url, anchor["href"])
                suffix = posixpath.splitext(urlsplit(target).path)[1].lower()
                if suffix in {".pdf", ".docx", ".doc", ".xls", ".xlsx", ".zip"} or "/cmsres/" in target or "附件" in anchor.get_text():
                    targets[target] = suffix
                elif _DETAIL_PATH.fullmatch(urlsplit(target).path):
                    try:
                        relations.append(_official_url(target, "detail"))
                    except ValueError:
                        pass
            capture["related_urls"] = list(dict.fromkeys(relations))
            for control in body.select("button, a:not([href]), input[type=button]"):
                label = control.get_text(" ", strip=True) or control.get("value", "")
                if "附件" in label or "下载" in label:
                    capture["errors"].append("unsupported_attachment: JavaScript-only download")
            if body.select("img"):
                capture["errors"].append("needs_review: policy body includes images")
            for target, suffix in targets.items():
                asset = {"role": "attachment:" + hashlib.sha256(target.encode()).hexdigest()[:16], "url": target, "extension": suffix, "blob": None, "sha256": None}
                capture["assets"].append(asset)
                try:
                    _official_url(target, "attachment")
                    if suffix not in {".pdf", ".docx"}:
                        raise ValueError("unsupported_attachment")
                    blob, attachment_headers = self._fetch(target, "attachment")
                    asset.update(blob=blob, sha256=hashlib.sha256(blob).hexdigest())
                    mime = attachment_headers.get("Content-Type", "").split(";")[0].lower()
                    if suffix == ".pdf":
                        if mime not in {"application/pdf", "application/octet-stream"} or not blob.startswith(b"%PDF-"):
                            raise ValueError("Invalid PDF content type/signature")
                    else:
                        if mime not in {"application/vnd.openxmlformats-officedocument.wordprocessingml.document", "application/octet-stream"} or not blob.startswith(b"PK\x03\x04"):
                            raise ValueError("Invalid DOCX content type/signature")
                        with zipfile.ZipFile(io.BytesIO(blob)) as archive:
                            if not {"[Content_Types].xml", "word/document.xml"}.issubset(archive.namelist()):
                                raise ValueError("Invalid DOCX office archive")
                    asset.update(blob=blob, sha256=hashlib.sha256(blob).hexdigest())
                except _FatalScanError:
                    raise
                except (ValueError, zipfile.BadZipFile) as exc:
                    capture["errors"].append(f"attachment {target}: {exc}")
            self._associate_change_notice(capture)
            if not capture["errors"]:
                revision = {key: capture[key] for key in ("text", "source_url", "source_kind", "title", "document_number", "published_at", "signed_at", "related_urls")}
                revision["attachments"] = sorted((asset["url"], asset["sha256"]) for asset in capture["assets"] if asset["role"] != "body")
                capture["revision_hash"] = hashlib.sha256(json.dumps(revision, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
        except _FatalScanError:
            raise
        except ValueError as exc:
            capture["errors"].append(str(exc))
        return capture

    def _scan(self):
        self._begin()
        self._read_robots()
        discovered = self._discover()
        self.discovery_complete = True
        self.scan_health.update(discovery_complete=True, discovered_count=len(discovered))
        fatal_errors = []
        new_count = 0
        asset_count = 0
        covered = set(self.config.covered_urls)
        batch = []
        for url in discovered:
            self._check()
            observed = datetime.now(timezone.utc)
            provenance = self._discovered_provenance.get(url)
            if provenance:
                capture = self._capture_record(url, observed.isoformat())
                capture.update(
                    title=discovered[url],
                    source_kind=provenance["source_kind"],
                    source_list_url=provenance["source_list_urls"][0],
                    relations=[{"kind": "discovered_on", "source_list_url": listed, "status": "needs_review"} for listed in provenance["source_list_urls"]],
                )
                capture["errors"].append("needs_review: historical source path outside configured scope")
                self.scan_health["out_of_scope_count"] += 1
            elif url not in covered and (new_count >= self.config.max_new_policies or len(covered) + new_count >= self.config.max_policies):
                capture = self._capture_record(url, observed.isoformat())
                capture.update(title=discovered[url], errors=["policy_limit: deferred"])
                self.backlog.append(url)
            else:
                if url not in covered:
                    new_count += 1
                capture = self._capture(url, observed.isoformat())
                if not capture["title"]:
                    capture["title"] = discovered[url] or "待读取政策：" + url
            capture["_asset_budget_remaining"] = self.config.max_assets - asset_count
            if self.capture_sink:
                # The catalog proves unchanged published assets can be reused, or
                # persists an over-budget candidate before returning, once only.
                extra = self.capture_sink(capture)
            else:
                extra = {}
                if not capture["errors"] and len(capture["assets"]) > capture["_asset_budget_remaining"]:
                    capture["errors"].append("asset_limit: deferred")
                    capture["revision_hash"] = None
            if not isinstance(extra, dict) or any(not isinstance(value, str) for value in extra.values()):
                raise _FatalScanError("capture_sink must return string metadata")
            self.scan_health["observed_count"] += 1
            if capture["errors"]:
                self.errors.extend(capture["errors"])
                if any("policy_limit" in error or "asset_limit" in error for error in capture["errors"]):
                    self.scan_health["deferred_count"] += 1
                    if url not in self.backlog:
                        self.backlog.append(url)
                elif any("needs_review" in error or "unsupported_attachment" in error or "Invalid PDF" in error or "Invalid DOCX" in error for error in capture["errors"]):
                    self.scan_health["needs_review_count"] += 1
                else:
                    self.scan_health["failed_count"] += 1
                for error in capture["errors"]:
                    if "HTTP failure" in error or "timed out" in error or "archive_failed" in error:
                        self._retryable(error)
                    if "archive_failed" in error:
                        fatal_errors.append(error)
                continue
            self.scan_health["sealed_count"] += 1
            required = extra.get("policy_ingestion_required", "1")
            if required not in {"0", "1"}:
                raise _FatalScanError("capture_sink returned invalid policy_ingestion_required")
            if required == "0":
                self.scan_health["unchanged_count"] += 1
                continue
            if len(capture["assets"]) > capture["_asset_budget_remaining"]:
                error = "asset_limit: deferred; capture_sink did not persist budget deferral"
                self.errors.append(error)
                fatal_errors.append(error)
                self.backlog.append(url)
                continue
            asset_count += len(capture["assets"])
            self.scan_health["ingestion_policy_count"] += 1
            self.scan_health["ingestion_asset_count"] = asset_count
            for asset in capture["assets"]:
                metadata = {
                    **(extra or {}),
                    **{
                        key: capture[key]
                        for key in ("policy_key", "source_url", "source_kind", "title", "document_number", "published_at", "signed_at", "observed_at", "revision_hash", "parse_build_key")
                    },
                }
                metadata = {key: value for key, value in metadata.items() if value is not None}
                metadata.update(asset_role=asset["role"], asset_sha256=asset["sha256"], asset_url=asset["url"])
                identity = ":".join((capture["policy_key"], capture["revision_hash"], self.config.parse_build_key, asset["role"], asset["sha256"]))
                batch.append(
                    Document(
                        id="shanghai_medical:" + identity,
                        source=DocumentSource.SHANGHAI_MEDICAL,
                        semantic_identifier=_asset_filename(capture["title"], capture["policy_key"], capture["revision_hash"], self.config.parse_build_key, asset["role"], asset["extension"]),
                        extension=asset["extension"],
                        blob=asset["blob"],
                        doc_updated_at=observed,
                        size_bytes=len(asset["blob"]),
                        metadata=metadata,
                        fingerprint=xxhash.xxh128(asset["blob"]).hexdigest(),
                    )
                )
                if len(batch) >= self.config.batch_size:
                    yield batch
                    batch = []
        if batch:
            yield batch
        if fatal_errors:
            raise _FatalScanError("Policy observation failed: " + "; ".join(fatal_errors[:5]))
        self.observation_complete = True
        self.scan_complete = not self.errors
        self.scan_health.update(observation_complete=True, scan_complete=self.scan_complete)
