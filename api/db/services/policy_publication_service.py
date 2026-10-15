"""Validate indexed policy assets and atomically publish immutable release manifests."""

import hashlib
import json
import math
import re
import time
import uuid
from datetime import date, datetime
from numbers import Real
from zoneinfo import ZoneInfo

import xxhash

from api.db.services.policy_catalog_service import _now
from api.db.services.policy_sync_service import _embedding_config, canonical_embedding_config, embedding_config_sha256


class PolicyPublicationService:
    def __init__(self, catalog, document_reader=None, chunk_reader=None, object_store=None):
        self.catalog = catalog
        self.document_reader = document_reader or self._document
        self.chunk_reader = chunk_reader
        self._object_store = object_store

    @property
    def object_store(self):
        return self._object_store if self._object_store is not None else self.catalog.object_store

    @staticmethod
    def _document(doc_id):
        from api.db.services.document_service import DocumentService

        exists, doc = DocumentService.get_by_id(doc_id)
        if not exists:
            return None
        from api.db.services.knowledgebase_service import KnowledgebaseService

        ok, kb = KnowledgebaseService.get_by_id(doc.kb_id)
        if not ok:
            return None
        return dict(doc.to_dict(), tenant_id=DocumentService.get_tenant_id(doc_id), embedding_config=_embedding_config(kb.to_dict()))

    def _chunks(self, scope, doc_id):
        if self.chunk_reader is not None:
            return self.chunk_reader(doc_id)
        from common import settings

        from common.doc_store.doc_store_base import OrderByExpr
        from rag.nlp.search import index_name

        fields = ["doc_id", "kb_id", "docnm_kwd", "content_with_weight", "img_id", "position_int", "page_num_int", "top_int"]
        rows = settings.retriever.chunk_list(doc_id, scope["tenant_id"], [scope["kb_id"]], fields=fields, retrieve_all=True, sort_by_position=True)
        if not rows:
            return rows
        # Discover the actual vector column. Model metadata can omit dimensions.
        # Native get returns complete fields for Infinity/OpenSearch/ES <=8.
        first = settings.docStoreConn.get(rows[0]["id"], index_name(scope["tenant_id"]), [scope["kb_id"]]) or {}
        vector_fields = [key for key in first if re.fullmatch(r"q_[1-9][0-9]*_vec", key)]
        if not vector_fields:
            # ES 9 excludes vectors from _source; native search explicitly asks
            # for *_vec in hit.fields. Expand names before native get_fields.
            result = settings.docStoreConn.search(["q_*_vec"], [], {"doc_id": doc_id}, [], OrderByExpr(), 0, 1, index_name(scope["tenant_id"]), [scope["kb_id"]])
            result = getattr(result, "body", result)
            if callable(getattr(result, "get", None)):
                vector_fields = sorted({key for hit in result.get("hits", {}).get("hits", []) for key in (*hit.get("_source", {}), *hit.get("fields", {})) if re.fullmatch(r"q_[1-9][0-9]*_vec", key)})
        if len(vector_fields) != 1:
            raise ValueError("index embedding vector column is missing or ambiguous")
        return settings.retriever.chunk_list(doc_id, scope["tenant_id"], [scope["kb_id"]], fields=fields + vector_fields, retrieve_all=True, sort_by_position=True)

    @staticmethod
    def _embedding_dimension(indexed):
        dimensions = set()
        for chunk in indexed:
            vectors = [(key, value) for key, value in chunk.items() if re.fullmatch(r"q_[0-9]+_vec", key)]
            if len(vectors) != 1:
                raise ValueError("index embedding vector is missing or ambiguous")
            key, vector = vectors[0]
            dimension = int(key.split("_")[1])
            if callable(getattr(vector, "tolist", None)):
                vector = vector.tolist()
            if dimension < 1 or not isinstance(vector, (list, tuple)) or len(vector) != dimension or any(isinstance(v, bool) or not isinstance(v, Real) or not math.isfinite(v) for v in vector):
                raise ValueError("index embedding dimension or vector values are invalid")
            chunk[key] = [float(v) for v in vector]
            dimensions.add(dimension)
        if len(dimensions) != 1:
            raise ValueError("index embedding dimensions are inconsistent")
        return dimensions.pop()

    def _get(self, scope, key):
        return self.object_store.get(scope["kb_id"], key, tenant_id=scope["tenant_id"])

    def _manifest(self, scope, release):
        raw = self._get(scope, release.chunk_manifest_object_key)
        if hashlib.sha256(raw).hexdigest() != release.chunk_manifest_sha256:
            raise ValueError("release chunk manifest SHA256 mismatch")
        return json.loads(raw)

    @staticmethod
    def _position(chunk):
        # A bounding-box list may contain several columns/regions. Do not guess
        # their reading order; only a single explicit position is accepted.
        value = chunk.get("position_int")
        if isinstance(value, list):
            if len(value) != 1:
                return None
            value = value[0]
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            page = chunk.get("page_num_int", 0)
            if isinstance(page, list):
                if len(page) != 1:
                    return None
                page = page[0]
            if isinstance(page, (int, float)):
                return page, value
        return None

    @staticmethod
    def _metadata(revision, asset, build):
        fields = ("policy_key", "source_url", "source_kind", "title", "document_number", "published_at", "signed_at", "observed_at", "revision_hash")
        value = {field: getattr(revision, field) for field in fields}
        temporal = revision.temporal_metadata or {"status": "unknown"}
        value.update(
            revision_id=revision.id,
            asset_role=asset["role"],
            asset_sha256=asset["sha256"],
            asset_url=asset.get("url"),
            parse_build_key=build,
            temporal_metadata=temporal,
            effective_from=temporal.get("effective_from"),
            effective_to=temporal.get("effective_to"),
            confirmation_status=temporal.get("status", "unknown"),
            relations=revision.relations,
        )
        return value

    def _validate(self, scope, revision, build):
        if not revision.revision_hash or revision.scope_id != scope["id"]:
            raise ValueError("unsealed or foreign revision")
        for kind in ("html", "text"):
            archive = revision.raw_object_keys.get(kind)
            if not archive or hashlib.sha256(self._get(scope, archive["key"])).hexdigest() != archive["sha256"]:
                raise ValueError("raw capture archive missing or SHA256 mismatch")
        bound = revision.parsing_builds.get(build, [])
        expected = {(a["role"], a["sha256"]) for a in revision.assets}
        actual = {(a["role"], a["sha256"]) for a in bound}
        if not expected or expected != actual or len(bound) != len(expected):
            raise ValueError("required body/attachment document bindings are incomplete")
        if len({a["doc_id"] for a in bound}) != len(bound):
            raise ValueError("multiple assets share a document ID")
        sources, chunks = {}, {}
        group_embedding = None
        for asset in revision.assets:
            binding = next(a for a in bound if (a["role"], a["sha256"]) == (asset["role"], asset["sha256"]))
            doc_id = binding["doc_id"]
            build_config = binding.get("parse_build_config")
            if not isinstance(build_config, dict) or hashlib.sha256(json.dumps(build_config, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest() != build:
                raise ValueError("embedding/parser binding does not match parse build identity")
            embedding_config = canonical_embedding_config(binding.get("embedding_model"))
            if embedding_config != build_config.get("embedding_config") or any(binding.get(k) != build_config.get(k) for k in ("tenant_embd_id", "embd_id")):
                raise ValueError("embedding binding differs from parse build snapshot")
            if binding.get("parser_config", {}) != build_config.get("parser_config"):
                raise ValueError("parser binding differs from parse build snapshot")
            raw = self._get(scope, asset["object_key"])
            if hashlib.sha256(raw).hexdigest() != asset["sha256"]:
                raise ValueError("source archive SHA256 mismatch")
            doc = self.document_reader(doc_id)
            if not doc or doc.get("id") != doc_id or doc.get("kb_id") != scope["kb_id"] or doc.get("tenant_id") != scope["tenant_id"]:
                raise ValueError("document missing or tenant/knowledge base mismatch")
            if canonical_embedding_config(doc.get("embedding_config")) != embedding_config:
                raise ValueError("document embedding configuration differs from the parse build snapshot")
            if str(doc.get("run")) != "3" or float(doc.get("progress", 0)) < 1 or int(doc.get("chunk_num", 0)) < 1 or doc.get("error") or doc.get("errors"):
                raise ValueError("document parse is incomplete or failed")
            for field in ("parser_id", "parser_config"):
                if field in binding and doc.get(field) != binding[field]:
                    raise ValueError(f"parsed document {field} differs from the parse build snapshot")
            fingerprint = binding.get("fingerprint") or xxhash.xxh128(raw).hexdigest()
            if doc.get("content_hash") != fingerprint:
                raise ValueError("parsed document fingerprint differs from archived asset")
            indexed = self._chunks(scope, doc_id)
            if not indexed or len(indexed) != int(doc["chunk_num"]):
                raise ValueError("document index is empty or incomplete")
            dimension = self._embedding_dimension(indexed)
            for field in ("dimension", "dimensions", "embedding_dimension"):
                declared = embedding_config.get(field)
                if declared is not None and (type(declared) is not int or declared != dimension):
                    raise ValueError("index embedding dimension differs from model configuration")
            identity = (embedding_config_sha256(embedding_config), dimension)
            if group_embedding is not None and group_embedding != identity:
                raise ValueError("policy assets have inconsistent embedding identities or dimensions")
            group_embedding = identity
            positions = [self._position(c) for c in indexed]
            ordered = all(p is not None for p in positions) and len(set(positions)) == len(positions)
            if ordered:
                indexed = [c for _, c in sorted(zip(positions, indexed), key=lambda pair: pair[0])]
            for ordinal, chunk in enumerate(indexed):
                cid, text = chunk.get("id"), chunk.get("content_with_weight")
                if not cid or cid in chunks or not isinstance(text, str) or not text.strip():
                    raise ValueError("index has missing/duplicate chunk ID or empty text")
                if chunk.get("doc_id", doc_id) != doc_id:
                    raise ValueError("index returned a foreign document chunk")
                if chunk.get("kb_id", scope["kb_id"]) not in (scope["kb_id"], [scope["kb_id"]]):
                    raise ValueError("index returned a foreign knowledge base chunk")
                chunks[cid] = {
                    "id": cid,
                    "document_id": doc_id,
                    "title": chunk.get("docnm_kwd") or revision.title,
                    "text": text,
                    "order": ordinal if ordered else None,
                    "sha256": hashlib.sha256(text.encode()).hexdigest(),
                    "native": {key: value for key, value in chunk.items() if not re.fullmatch(r"q_[0-9]+_vec", key)},
                    "ordered": ordered,
                }
            sources[doc_id] = dict(
                self._metadata(revision, asset, build),
                tenant_id=scope["tenant_id"],
                kb_id=scope["kb_id"],
                archive_key=asset["object_key"],
                fingerprint=fingerprint,
                embedding_config=embedding_config,
                embedding_config_sha256=identity[0],
                embedding_dimension=dimension,
            )
        return sources, chunks

    def _lease(self, scope_id):
        owner = uuid.uuid4().hex
        Scope = self.catalog.Scope
        now = time.time()
        changed = Scope.update(lease_owner=owner, lease_until=now + 300).where((Scope.id == scope_id) & (Scope.lease_until <= now)).execute()
        return owner if changed else None

    def _release_lease(self, scope_id, owner):
        Scope = self.catalog.Scope
        Scope.update(lease_owner=None, lease_until=0).where((Scope.id == scope_id) & (Scope.lease_owner == owner)).execute()

    def reconcile(self, scope_id):
        owner = self._lease(scope_id)
        if owner is None:
            return {"published": False, "reason": "lease_busy"}
        try:
            return self._reconcile(scope_id, owner)
        finally:
            self._release_lease(scope_id, owner)

    def _mark_validated_published(self, scope_id, revision_ids):
        if not revision_ids:
            return
        Revision = self.catalog.Revision
        changed = [
            row.id
            for row in Revision.select(Revision.id, Revision.state, Revision.errors).where((Revision.scope_id == scope_id) & Revision.id.in_(revision_ids))
            if row.state != "published" or row.errors != []
        ]
        if changed:
            Revision.update(state="published", errors=[]).where((Revision.scope_id == scope_id) & Revision.id.in_(changed)).execute()

    def _reconcile(self, scope_id, owner):
        scope = self.catalog.get_scope(scope_id)
        Release, Revision, Scope = self.catalog.Release, self.catalog.Revision, self.catalog.Scope
        prior = Release.get_by_id(scope["active_release_id"]) if scope["active_release_id"] else None
        sources = dict(prior.document_manifest) if prior else {}
        chunks = self._manifest(scope, prior) if prior else {}
        revisions = dict(prior.revision_ids) if prior else {}
        errors, accepted, validated_published = {}, [], []
        for key, head in scope["source_heads"].items():
            if not head.get("revision_id"):
                continue
            revision = Revision.get_by_id(head["revision_id"])
            try:
                new_sources, new_chunks = self._validate(scope, revision, head["parse_build_key"])
                if revisions.get(key) == {"revision_id": revision.id, "parse_build_key": head["parse_build_key"]}:
                    previous_chunks = {cid: chunk for cid, chunk in chunks.items() if chunk["document_id"] in new_sources}
                    if previous_chunks != new_chunks:
                        raise ValueError("published chunks changed; a new parse build is required")
            except Exception as exc:
                errors[key] = str(exc)
                # Runtime validation errors remain observable and retryable.
                state, failure = "failed" if revision.state != "published" else "published", [str(exc)]
                if revision.state != state or revision.errors != failure:
                    Revision.update(errors=failure, state=state).where(Revision.id == revision.id).execute()
                continue
            removed = {d for d, source in sources.items() if source["policy_key"] == key}
            kept_sources = {d: s for d, s in sources.items() if d not in removed}
            kept_chunks = {c: v for c, v in chunks.items() if v["document_id"] not in removed}
            if set(kept_chunks) & set(new_chunks) or set(kept_sources) & set(new_sources):
                errors[key] = "document/chunk IDs collide across policy groups"
                continue
            kept_sources.update(new_sources)
            kept_chunks.update(new_chunks)
            sources, chunks = kept_sources, kept_chunks
            if revision.state == "published" and revisions.get(key) == {"revision_id": revision.id, "parse_build_key": head["parse_build_key"]}:
                validated_published.append(revision.id)
            revisions[key] = {"revision_id": revision.id, "parse_build_key": head["parse_build_key"]}
            accepted.append(revision.id)
        pending = self.catalog.pending(scope_id)
        published_targets = {(k, v["revision_id"], v["parse_build_key"]) for k, v in revisions.items()}
        pending = [p for p in pending if (p["policy_key"], p["revision_id"], p["parse_build_key"]) not in published_targets]
        coverage = {"coverage": scope["coverage_manifest"], "pending": pending, "scanned_at": scope["last_scan_success_at"]}
        if not accepted or (
            prior and prior.document_manifest == sources and self._manifest(scope, prior) == chunks and prior.coverage_status == coverage and prior.source_heads == scope["source_heads"]
        ):
            self._mark_validated_published(scope_id, validated_published)
            return {"published": False, "release_id": scope["active_release_id"], "errors": errors, "pending": pending}
        release_id = uuid.uuid4().hex
        manifest = json.dumps(chunks, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
        digest = hashlib.sha256(manifest).hexdigest()
        object_key = f"policy/{scope_id}/releases/{release_id}/{digest}.json"
        self.object_store.put(scope["kb_id"], object_key, manifest, tenant_id=scope["tenant_id"])
        if self._get(scope, object_key) != manifest:
            raise ValueError("release manifest readback mismatch")

        class Conflict(Exception):
            pass

        try:
            with self.catalog.db.atomic():
                # CAS precedes insertion inside this same transaction. Any failed
                # insert rolls back the pointer and generation together.
                changed = (
                    Scope.update(active_release_id=release_id, generation=Scope.generation + 1, last_published_at=_now())
                    .where((Scope.id == scope_id) & (Scope.generation == scope["generation"]) & (Scope.lease_owner == owner) & (Scope.lease_until > time.time()))
                    .execute()
                )
                if not changed:
                    raise Conflict()
                Release.create(
                    id=release_id,
                    release_id=release_id,
                    scope_id=scope_id,
                    parent_release_id=scope["active_release_id"],
                    revision_ids=revisions,
                    document_manifest=sources,
                    chunk_manifest_object_key=object_key,
                    chunk_manifest_sha256=digest,
                    coverage_status=coverage,
                    source_heads=scope["source_heads"],
                    created_at=_now(),
                )
                self._mark_validated_published(scope_id, accepted)
        except Conflict:
            return {"published": False, "reason": "generation_conflict", "errors": errors}
        return {"published": True, "release_id": release_id, "errors": errors, "pending": pending}

    def acquire_snapshot(self, tenant_id, kb_id, as_of=None):
        target = date.fromisoformat(as_of) if as_of else datetime.now(ZoneInfo("Asia/Shanghai")).date()
        scopes = list(self.catalog.Scope.select().where((self.catalog.Scope.tenant_id == tenant_id) & (self.catalog.Scope.kb_id == kb_id)))
        empty = {"release_id": None, "doc_ids": [], "sources": {}, "chunks": {}, "pending": [], "coverage": {}, "scanned_at": None}
        if not scopes:
            return empty
        if len(scopes) != 1:
            raise ValueError("knowledge base has multiple policy scopes")
        scope = self.catalog.get_scope(scopes[0].id)
        pending = self.catalog.pending(scope["id"])
        if not scope["active_release_id"]:
            return dict(empty, pending=pending, coverage=scope["coverage_manifest"], scanned_at=scope["last_scan_success_at"])
        release = self.catalog.Release.get_by_id(scope["active_release_id"])
        if release.scope_id != scope["id"]:
            raise ValueError("release scope mismatch")
        chunks = self._manifest(scope, release)
        sources = {}
        for doc_id, source in release.document_manifest.items():
            temporal = source.get("temporal_metadata", {})
            if temporal.get("status") == "confirmed":
                start, end = temporal.get("effective_from"), temporal.get("effective_to")
                if start and target < date.fromisoformat(start):
                    continue
                if end and target > date.fromisoformat(end):
                    continue
                if temporal.get("repealed_at") and target >= date.fromisoformat(temporal["repealed_at"]):
                    continue
            sources[doc_id] = source
        # Copy current observed backlog once per request; sources and chunk hashes
        # always come solely from the immutable release, never mutable revisions.
        return {
            "release_id": release.id,
            "doc_ids": sorted(sources),
            "sources": sources,
            "chunks": {k: v for k, v in chunks.items() if v["document_id"] in sources},
            "pending": pending,
            "coverage": release.coverage_status["coverage"],
            "scanned_at": scope["last_scan_success_at"],
            "source_heads": scope["source_heads"],
            "as_of": target.isoformat(),
        }

    def rollback(self, scope_id, release_id):
        owner = self._lease(scope_id)
        if owner is None:
            raise ValueError("scope publication lease is busy")
        try:
            scope = self.catalog.get_scope(scope_id)
            release = self.catalog.Release.get_by_id(release_id)
            if release.scope_id != scope_id:
                raise ValueError("cannot rollback to a foreign scope release")
            expected = self._manifest(scope, release)
            for doc_id in release.document_manifest:
                current = self._chunks(scope, doc_id)
                hashes = {c["id"]: hashlib.sha256(c["content_with_weight"].encode()).hexdigest() for c in current}
                wanted = {cid: c["sha256"] for cid, c in expected.items() if c["document_id"] == doc_id}
                if not wanted or hashes != wanted:
                    raise ValueError("rollback release index is incomplete or changed")
                doc = self.document_reader(doc_id)
                source = release.document_manifest[doc_id]
                if self._embedding_dimension(current) != source["embedding_dimension"] or embedding_config_sha256(doc.get("embedding_config") if doc else None) != source["embedding_config_sha256"]:
                    raise ValueError("rollback embedding configuration or dimension changed")
                if not doc or doc.get("kb_id") != scope["kb_id"] or doc.get("tenant_id") != scope["tenant_id"]:
                    raise ValueError("rollback document scope mismatch")
                if str(doc.get("run")) != "3" or float(doc.get("progress", 0)) < 1 or int(doc.get("chunk_num", 0)) != len(wanted) or doc.get("content_hash") != source["fingerprint"]:
                    raise ValueError("rollback document parse is incomplete or changed")
                raw = self._get(scope, source["archive_key"])
                if hashlib.sha256(raw).hexdigest() != source["asset_sha256"]:
                    raise ValueError("rollback source archive changed")
            Scope = self.catalog.Scope
            with self.catalog.db.atomic():
                changed = (
                    Scope.update(active_release_id=release_id, generation=Scope.generation + 1, last_published_at=_now())
                    .where((Scope.id == scope_id) & (Scope.generation == scope["generation"]) & (Scope.lease_owner == owner) & (Scope.lease_until > time.time()))
                    .execute()
                )
                if not changed:
                    raise ValueError("rollback generation changed; retry")
            return {"release_id": release_id}
        finally:
            self._release_lease(scope_id, owner)
