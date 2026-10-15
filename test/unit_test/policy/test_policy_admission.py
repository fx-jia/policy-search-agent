"""Execute native upload admission with offline file, metadata and queue boundaries."""

import ast
import logging
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from typing import Optional
from unittest.mock import patch

from pydantic import BaseModel
import pytest
from peewee import CharField, Model, SqliteDatabase


def native_duplicate(metadata, write_result=True, readback=None):
    path = Path(__file__).resolve().parents[3] / "api/db/services/connector_service.py"
    tree = ast.parse(path.read_text())
    owner = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "SyncLogsService")
    method = next(node for node in owner.body if isinstance(node, ast.FunctionDef) and node.name == "duplicate_and_parse")
    cls = ast.ClassDef(name="NativeSync", bases=[], keywords=[], body=[method], decorator_list=[])
    calls, stored = [], {}

    def write(doc_id, values, **kwargs):
        calls.append("write")
        if write_result:
            stored.update(values)
        return write_result

    def read(doc_id, *, strict=False):
        if callable(readback):
            return readback(doc_id, strict=strict)
        return dict(stored) if readback is None else readback

    namespace = dict(BaseModel=BaseModel, Optional=Optional,
                     DocMetadataService=SimpleNamespace(update_document_metadata=write,
                         get_document_metadata=read,
                         refresh_tenant_index=lambda tenant: calls.append("refresh")),
                     DocumentService=SimpleNamespace(run=lambda *args: calls.append("run")))
    exec(compile(ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[])), str(path), "exec"), namespace)
    files = ModuleType("api.db.services.file_service")
    files.FileService = SimpleNamespace(_is_sync_cancelled=lambda *args: False,
        upload_document=lambda *args, **kwargs: ([], [({"id": "doc", "name": "policy.txt"}, b"body")]))
    docs = [{"id": "doc", "semantic_identifier": "policy", "extension": ".txt", "blob": b"body", "metadata": metadata}]
    return namespace["NativeSync"], files, docs, calls


@pytest.mark.parametrize("write_result,readback,metadata", [
    (False, None, {"scope_id": "scope", "revision_id": "rev", "parse_build_key": "build"}),
    (True, {}, {"scope_id": "scope", "revision_id": "rev", "parse_build_key": "build"}),
    (True, {"scope_id": "other", "revision_id": "rev", "parse_build_key": "build"}, {"scope_id": "scope", "revision_id": "rev", "parse_build_key": "build"}),
    (True, None, {}),
])
def test_policy_upload_requires_persisted_admission_before_parse(write_result, readback, metadata):
    service, files, docs, calls = native_duplicate(metadata, write_result, readback)
    with patch.dict(sys.modules, {"api.db.services.file_service": files}), pytest.raises(ValueError, match="metadata"):
        service.duplicate_and_parse({}, docs, "tenant", "shanghai_medical/connector")
    assert "run" not in calls


def test_policy_upload_parses_after_successful_admission_readback():
    service, files, docs, calls = native_duplicate({"scope_id": "scope", "revision_id": "rev", "parse_build_key": "build"})
    with patch.dict(sys.modules, {"api.db.services.file_service": files}):
        assert service.duplicate_and_parse({}, docs, "tenant", "shanghai_medical/connector") == ([], ["doc"])
    assert calls == ["write", "run", "refresh"]


def test_other_source_keeps_existing_optional_metadata_behavior():
    service, files, docs, calls = native_duplicate({"source": "mysql"}, False)
    with patch.dict(sys.modules, {"api.db.services.file_service": files}):
        service.duplicate_and_parse({}, docs, "tenant", "mysql/connector")
    assert calls == ["write", "run", "refresh"]


def test_policy_upload_strict_read_error_blocks_parse():
    metadata = {"scope_id": "scope", "revision_id": "rev", "parse_build_key": "build"}

    def read(doc_id, *, strict=False):
        if strict:
            raise OSError("metadata read unavailable")
        return metadata

    service, files, docs, calls = native_duplicate(metadata, readback=read)
    with patch.dict(sys.modules, {"api.db.services.file_service": files}), pytest.raises(OSError, match="read unavailable"):
        service.duplicate_and_parse({}, docs, "tenant", "shanghai_medical/connector")
    assert "run" not in calls


@pytest.fixture
def native_metadata_reader():
    """Use the real Peewee query, metadata extraction, and ES get method."""
    database = SqliteDatabase(":memory:")

    class Knowledgebase(Model):
        id = CharField(primary_key=True)
        tenant_id = CharField()

    class Document(Model):
        id = CharField(primary_key=True)
        kb_id = CharField()

    database.bind([Knowledgebase, Document])
    database.create_tables([Knowledgebase, Document])
    Knowledgebase.create(id="kb", tenant_id="tenant")
    Document.create(id="doc", kb_id="kb")
    root = Path(__file__).resolve().parents[3]
    namespace = dict(Document=Document, Knowledgebase=Knowledgebase, logging=logging)
    path = root / "api/db/services/doc_metadata_service.py"
    owner = next(node for node in ast.parse(path.read_text()).body if isinstance(node, ast.ClassDef) and node.name == "DocMetadataService")
    methods = [node for node in owner.body if isinstance(node, ast.FunctionDef) and node.name in {"get_document_metadata", "_get_doc_meta_index_name", "_extract_metadata"}]
    for method in methods:
        method.decorator_list = [decorator for decorator in method.decorator_list if isinstance(decorator, ast.Name)]
    cls = ast.ClassDef(name="NativeMetadata", bases=[], keywords=[], body=methods, decorator_list=[])
    exec(compile(ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[])), str(path), "exec"), namespace)

    class NotFoundError(Exception):
        pass

    path = root / "common/doc_store/es_conn_base.py"
    owner = next(node for node in ast.parse(path.read_text()).body if isinstance(node, ast.ClassDef) and node.name == "ESConnectionBase")
    method = next(node for node in owner.body if isinstance(node, ast.FunctionDef) and node.name == "get")
    es_namespace = dict(ATTEMPT_TIME=2, NotFoundError=NotFoundError)
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), es_namespace)
    remote = SimpleNamespace(get=lambda **kwargs: {"_source": {"meta_fields": {"category": "medical"}}})
    connection = SimpleNamespace(es=remote, logger=logging.getLogger(__name__))
    connection.get = lambda *args: es_namespace["get"](connection, *args)
    remote.doc_store = connection
    namespace["settings"] = SimpleNamespace(docStoreConn=connection)
    try:
        yield namespace["NativeMetadata"].get_document_metadata, remote, NotFoundError
    finally:
        database.close()


def test_metadata_strict_read_preserves_success_and_actual_absence(native_metadata_reader):
    read, remote, not_found = native_metadata_reader
    assert read("doc", strict=True) == {"category": "medical"}

    def missing(**kwargs):
        raise not_found()

    remote.get = missing
    assert read("doc", strict=True) == {}
    assert read("doc") == {}


@pytest.mark.parametrize("failure", [ConnectionError("connection failed"), TimeoutError("timed out")])
def test_metadata_strict_read_raises_es_failure_but_default_retains_empty_result(native_metadata_reader, failure):
    read, remote, _ = native_metadata_reader

    def unavailable(**kwargs):
        raise failure

    remote.get = unavailable
    assert read("doc") == {}
    with pytest.raises(type(failure), match=str(failure)):
        read("doc", strict=True)


@pytest.mark.parametrize("raw", [
    {"meta_fields": "{damaged"}, {"meta_fields": "[]"}, {"meta_fields": "null"},
    {"meta_fields": []}, {"meta_fields": ["unknown"]}, {"meta_fields": None},
    {"meta_fields": 42}, {"meta_fields": ""}, {},
])
def test_strict_metadata_rejects_existing_unknown_shape(native_metadata_reader, raw):
    read, remote, _ = native_metadata_reader
    remote.get = lambda **kwargs: {"_source": raw}
    with pytest.raises(ValueError):
        read("doc", strict=True)


@pytest.mark.parametrize("raw", [{"meta_fields": {}}, {"meta_fields": "{}"}, {"meta_fields": '{"category":"medical"}'}])
def test_strict_metadata_accepts_empty_or_encoded_valid_object(native_metadata_reader, raw):
    read, remote, _ = native_metadata_reader
    remote.get = lambda **kwargs: {"_source": raw}
    expected = {"category": "medical"} if "category" in str(raw) else {}
    assert read("doc", strict=True) == expected
    assert read("doc") == expected


@pytest.mark.parametrize("raw", [{}, [], "", 0, False])
def test_strict_metadata_rejects_falsey_non_none_store_result(native_metadata_reader, raw):
    read, remote, _ = native_metadata_reader
    remote.doc_store.get = lambda *args: raw
    assert read("doc") == {}
    with pytest.raises(ValueError):
        read("doc", strict=True)
