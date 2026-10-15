"""Durable observations, immutable source archives, and desired policy builds."""

import hashlib
import json
from datetime import date, datetime, timedelta, timezone

from peewee import IntegrityError


def _now():
    return datetime.now(timezone.utc).isoformat()


def _identity(*parts):
    return hashlib.sha256(json.dumps(parts, ensure_ascii=False).encode()).hexdigest()[:32]


def _dict(row):
    return {key: value.isoformat() if isinstance(value, (datetime, date)) else value for key, value in row.__data__.items()}


class PolicyCatalogService:
    def __init__(self, models=None):
        if models is None:
            from api.db.db_models import PolicyRelease, PolicyRevision, PolicyScope

            models = PolicyScope, PolicyRevision, PolicyRelease
        self.Scope, self.Revision, self.Release = models
        self.db = self.Scope._meta.database
        self._object_store = None

    @property
    def object_store(self):
        if self._object_store is None:
            from common import settings

            if settings.STORAGE_IMPL is None:
                raise RuntimeError("RAGFlow object storage is not initialized")
            return settings.STORAGE_IMPL
        return self._object_store

    @object_store.setter
    def object_store(self, store):
        self._object_store = store

    def ensure_scope(self, tenant_id, kb_id, connector_id, coverage=None):
        if not all((tenant_id, kb_id, connector_id)):
            raise ValueError("tenant, knowledge base and connector are required")
        sid = _identity(tenant_id, kb_id, connector_id)
        row, _ = self.Scope.get_or_create(id=sid, defaults={"tenant_id": tenant_id, "kb_id": kb_id, "connector_id": connector_id, "coverage_manifest": coverage or {}})
        return _dict(row)

    def get_scope(self, scope_id):
        return _dict(self.Scope.get_by_id(scope_id))

    def covered_sources(self, scope_id):
        """Admitted historical sources, excluding discovered/deferred backlog."""
        self.get_scope(scope_id)
        urls, numbers = set(), set()
        for revision in self.Revision.select().where(self.Revision.scope_id == scope_id):
            if not revision.revision_hash and not any(asset.get("role") == "body" for asset in revision.assets):
                continue
            urls.add(revision.source_url)
            if revision.document_number:
                numbers.add(revision.document_number)
        if len(urls) > 200:
            raise ValueError("policy admitted source coverage exceeds 200 URLs")
        return {"covered_urls": sorted(urls), "covered_document_numbers": sorted(numbers)}

    def begin_scan(self, scope_id):
        with self.db.atomic():
            if not self.Scope.update(scan_sequence=self.Scope.scan_sequence + 1, generation=self.Scope.generation + 1).where(self.Scope.id == scope_id).execute():
                raise ValueError("unknown policy scope")
            return self.Scope.get_by_id(scope_id).scan_sequence

    def _archive(self, scope, blob):
        blob = blob.encode() if isinstance(blob, str) else bytes(blob)
        digest = hashlib.sha256(blob).hexdigest()
        key = f"policy/{scope['id']}/sha256/{digest}"
        self.object_store.put(scope["kb_id"], key, blob, tenant_id=scope["tenant_id"])
        if self.object_store.get(scope["kb_id"], key, tenant_id=scope["tenant_id"]) != blob:
            raise ValueError("archive readback does not match source bytes")
        return {"key": key, "sha256": digest}

    def _head(self, scope_id, sequence, key, observation_id, revision_id, build):
        for _ in range(8):
            scope = self.get_scope(scope_id)
            heads = scope["source_heads"]
            if heads.get(key, {}).get("capture_sequence", -1) > sequence:
                return
            heads[key] = {"capture_sequence": sequence, "observation_id": observation_id, "revision_id": revision_id, "parse_build_key": build}
            changed = self.Scope.update(source_heads=heads, generation=self.Scope.generation + 1).where((self.Scope.id == scope_id) & (self.Scope.generation == scope["generation"])).execute()
            if changed:
                return
        raise RuntimeError("policy source head changed concurrently; retry observation")

    def observe(self, scope_id, sequence, capture):
        scope = self.get_scope(scope_id)
        if sequence < 1 or sequence > scope["scan_sequence"]:
            raise ValueError("capture sequence must be allocated by begin_scan")
        for field in ("policy_key", "source_url", "source_kind", "title", "observed_at"):
            if not capture.get(field):
                raise ValueError(f"capture is missing {field}")
        key = capture["policy_key"]
        candidate_id = _identity(scope_id, key, sequence)
        build = str(capture.get("parse_build_key") or "default")
        relations = list(capture.get("relations") or []) + [{"type": "association", "url": url} for url in capture.get("related_urls") or []]
        relations = list({json.dumps(relation, sort_keys=True, ensure_ascii=False): relation for relation in relations}.values())
        defaults = {name: capture.get(name) for name in ("source_url", "source_kind", "title", "document_number", "published_at", "signed_at", "observed_at")}
        defaults.update(
            scope_id=scope_id,
            policy_key=key,
            capture_sequence=sequence,
            temporal_metadata=capture.get("temporal_metadata") or {"status": "unknown"},
            relations=relations,
            assets=[{k: v for k, v in a.items() if k != "blob"} for a in capture.get("assets", [])],
        )
        candidate, _ = self.Revision.get_or_create(id=candidate_id, defaults=defaults)
        # Commit the candidate and latest observation before network/object operations.
        desired = candidate.sealed_revision_id or (candidate.id if candidate.revision_hash else None)
        self._head(scope_id, sequence, key, candidate_id, desired, build)
        if candidate.revision_hash or candidate.sealed_revision_id:
            row = self.Revision.get_by_id(desired)
            return dict(_dict(row), revision_id=row.id, observation_id=candidate_id, parse_build_key=build)
        errors = [str(e) for e in capture.get("errors", [])]
        raw = {}
        assets = [{k: v for k, v in a.items() if k != "blob"} for a in capture.get("assets", [])]
        try:
            raw["html"] = self._archive(scope, capture.get("raw_html", b""))
            raw["text"] = self._archive(scope, capture.get("text", ""))
            for asset, archived in zip(capture.get("assets", []), assets):
                if asset.get("blob") is None:
                    errors.append(f"missing asset bytes: {asset.get('role')}")
                else:
                    record = self._archive(scope, asset["blob"])
                    if asset.get("sha256") != record["sha256"]:
                        raise ValueError(f"asset SHA256 mismatch: {asset.get('role')}")
                    archived.update(object_key=record["key"], sha256=record["sha256"])
        except Exception as exc:
            errors.append(f"archive_failed: {exc}")
        revision_hash = capture.get("revision_hash") if not errors else None
        if not revision_hash and not errors:
            errors.append("capture is not sealed")
        if not assets or not any(a.get("role") == "body" for a in assets):
            errors.append("missing required body asset")
            revision_hash = None
        values = dict(raw_object_keys=raw, assets=assets, errors=errors, state="archived" if revision_hash else "failed")
        self.Revision.update(**values).where(self.Revision.id == candidate_id).execute()
        row = self.Revision.get_by_id(candidate_id)
        if revision_hash:
            try:
                with self.db.atomic():
                    self.Revision.update(revision_hash=revision_hash).where(self.Revision.id == candidate_id).execute()
                row = self.Revision.get_by_id(candidate_id)
            except IntegrityError:
                row = self.Revision.get((self.Revision.scope_id == scope_id) & (self.Revision.policy_key == key) & (self.Revision.revision_hash == revision_hash))
                self.Revision.update(sealed_revision_id=row.id, state="superseded").where(self.Revision.id == candidate_id).execute()
            self._head(scope_id, sequence, key, candidate_id, row.id, build)
        return dict(_dict(row), revision_id=row.id, observation_id=candidate_id, parse_build_key=build)

    def bind_documents(self, scope_id, revision_id, assets):
        with self.db.atomic():
            if not self.Scope.update(generation=self.Scope.generation + 1).where(self.Scope.id == scope_id).execute():
                raise ValueError("unknown policy scope")
            row = self.Revision.get_by_id(revision_id)
            if row.scope_id != scope_id or not row.revision_hash:
                raise ValueError("document binding requires a sealed revision in this scope")
            builds = row.parsing_builds
            expected = {(a["role"], a["sha256"]) for a in row.assets}
            for value in assets:
                item = dict(value)
                digest = item.get("sha256") or item.get("hash")
                if not item.get("doc_id") or (item.get("role"), digest) not in expected:
                    raise ValueError("document asset role/hash does not match sealed source")
                build = str(item.get("parse_build_key") or self.get_scope(scope_id)["source_heads"].get(row.policy_key, {}).get("parse_build_key") or "default")
                item.update(sha256=digest, parse_build_key=build)
                current = list(builds.get(build, []))
                previous = next((a for a in current if (a["role"], a["sha256"]) == (item["role"], digest)), None)
                if previous and previous != item:
                    raise ValueError("parse build document/config binding is immutable")
                if not previous:
                    current.append(item)
                builds[build] = current
            self.Revision.update(parsing_builds=builds, state="queued" if row.state != "published" else "published").where(self.Revision.id == revision_id).execute()

    def is_published_capture(self, scope_id, capture, build):
        """Compare a complete observation with the active immutable asset set."""
        if not capture.get("revision_hash") or capture.get("errors"):
            return False
        revision = self.Revision.get_or_none((self.Revision.scope_id == scope_id) & (self.Revision.policy_key == capture["policy_key"]) & (self.Revision.revision_hash == capture["revision_hash"]))
        if not revision or not self.is_published_revision(scope_id, revision.id, build):
            return False
        assets = capture.get("assets", [])
        return len(assets) == len(revision.assets) and {(a.get("role"), a.get("sha256")) for a in assets} == {(a["role"], a["sha256"]) for a in revision.assets}

    def is_published_revision(self, scope_id, revision_id, build):
        """Skip ingestion only for a complete asset set in the active release."""
        scope = self.get_scope(scope_id)
        release = self.Release.get_or_none(self.Release.id == scope["active_release_id"]) if scope["active_release_id"] else None
        revision = self.Revision.get_or_none(self.Revision.id == revision_id)
        if not release or release.scope_id != scope_id or not revision or revision.scope_id != scope_id or not revision.revision_hash or revision.state != "published":
            return False
        if release.revision_ids.get(revision.policy_key) != {"revision_id": revision_id, "parse_build_key": build}:
            return False
        expected = {(asset["role"], asset["sha256"]) for asset in revision.assets}
        bindings = revision.parsing_builds.get(build, [])
        if not expected or len(bindings) != len(expected) or {(asset["role"], asset["sha256"]) for asset in bindings} != expected or len({asset["doc_id"] for asset in bindings}) != len(bindings):
            return False
        for asset in bindings:
            source = release.document_manifest.get(asset["doc_id"], {})
            if any(
                source.get(key) != value
                for key, value in {
                    "revision_id": revision_id,
                    "policy_key": revision.policy_key,
                    "parse_build_key": build,
                    "asset_role": asset["role"],
                    "asset_sha256": asset["sha256"],
                    "tenant_id": scope["tenant_id"],
                    "kb_id": scope["kb_id"],
                }.items()
            ):
                return False
        return True

    def review_revision(self, revision_id, temporal_metadata):
        """Persist an explicit, attributed date review; the publisher copies it later."""
        value = dict(temporal_metadata)
        allowed = {"status", "effective_from", "effective_to", "repealed_at", "evidence", "reviewed_by"}
        if set(value) - allowed or value.get("status") not in {"unknown", "confirmed"}:
            raise ValueError("invalid temporal review fields/status")
        if value["status"] == "confirmed" and not all(isinstance(value.get(k), str) and value[k].strip() for k in ("evidence", "reviewed_by")):
            raise ValueError("confirmed review requires source evidence and reviewer")
        for key in ("effective_from", "effective_to", "repealed_at"):
            if value.get(key) is not None:
                date.fromisoformat(value[key])
        if value.get("effective_from") and value.get("effective_to") and value["effective_from"] > value["effective_to"]:
            raise ValueError("effective_from must not be after effective_to")
        revision = self.Revision.get_by_id(revision_id)
        scope = self.get_scope(revision.scope_id)
        if value["status"] == "confirmed":
            archive = revision.raw_object_keys.get("text")
            if not archive:
                raise ValueError("review requires archived source text")
            raw = self.object_store.get(scope["kb_id"], archive["key"], tenant_id=scope["tenant_id"])
            if hashlib.sha256(raw).hexdigest() != archive["sha256"]:
                raise ValueError("review source archive SHA256 mismatch")
            if value["evidence"] not in raw.decode("utf-8"):
                raise ValueError("review evidence must be an exact quote from archived source text")
            value["evidence_archive_key"] = archive["key"]
            value["evidence_archive_sha256"] = archive["sha256"]
        value["reviewed_at"] = _now()
        with self.db.atomic():
            revision = self.Revision.get_by_id(revision_id)
            if not revision.revision_hash:
                raise ValueError("only sealed revisions can be reviewed")
            scope = self.get_scope(revision.scope_id)
            changed = self.Scope.update(generation=self.Scope.generation + 1).where((self.Scope.id == revision.scope_id) & (self.Scope.generation == scope["generation"])).execute()
            if not changed:
                raise RuntimeError("policy scope changed concurrently; retry review")
            self.Revision.update(temporal_metadata=value).where(self.Revision.id == revision_id).execute()
        return dict(_dict(self.Revision.get_by_id(revision_id)), revision_id=revision_id)

    def record_scan_result(self, scope_id, complete, error=None, *, scan_health=None, retryable=True):
        now = datetime.now(timezone.utc)
        scope = self.get_scope(scope_id)
        coverage = dict(scope["coverage_manifest"])
        health = dict(scan_health or {})
        coverage["scan_health"] = health
        for phase in ("discovery", "observation"):
            if health.get(f"{phase}_complete") is True:
                coverage[f"last_{phase}_success_at"] = now.isoformat()
        values = {
            "last_error": None if complete else str(error or "scan incomplete"),
            "next_scan_at": (now + timedelta(hours=24)).isoformat(),
            "retry_count": 0 if complete or not retryable else scope["retry_count"] + 1,
            "next_retry_at": None,
            "coverage_manifest": coverage,
            "generation": self.Scope.generation + 1,
        }
        if complete:
            values["last_scan_success_at"] = now.isoformat()
        self.Scope.update(**values).where(self.Scope.id == scope_id).execute()

    def pending(self, scope_id=None):
        scopes = self.Scope.select()
        if scope_id is not None:
            scopes = scopes.where(self.Scope.id == scope_id)
        result = []
        for scope in scopes:
            active = self.Release.get_or_none(self.Release.id == scope.active_release_id) if scope.active_release_id else None
            published = active.revision_ids if active else {}
            for key, head in scope.source_heads.items():
                revision = self.Revision.get_by_id(head["revision_id"] or head["observation_id"])
                current = published.get(key, {})
                if current.get("revision_id") == head["revision_id"] and current.get("parse_build_key") == head["parse_build_key"]:
                    continue
                result.append(
                    dict(
                        _dict(revision),
                        revision_id=revision.id,
                        observation_id=head["observation_id"],
                        parse_build_key=head["parse_build_key"],
                        pending_reasons=revision.errors or ["awaiting publication"],
                    )
                )
        return result

    def status(self, scope_id):
        scope = self.get_scope(scope_id)
        pending = self.pending(scope_id)
        last = scope["last_scan_success_at"]
        stale = not last or (datetime.now(timezone.utc) - datetime.fromisoformat(last)).total_seconds() > 48 * 3600
        covered = self.Revision.select(self.Revision.policy_key).where((self.Revision.scope_id == scope_id) & self.Revision.revision_hash.is_null(False)).distinct().count()
        release = self.Release.get_by_id(scope["active_release_id"]) if scope["active_release_id"] else None
        return dict(
            scope,
            covered_policy_count=covered,
            discovered_policy_count=len(scope["source_heads"]),
            published_policy_count=len(release.revision_ids) if release else 0,
            pending=pending,
            pending_count=len(pending),
            stale=stale,
        )
