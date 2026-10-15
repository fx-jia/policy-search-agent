"""Policy tables, defined once for the application database and isolated SQLite tests."""

import json

from peewee import CharField, FloatField, IntegerField, MySQLDatabase, TextField


class _JSONField(TextField):
    def db_value(self, value):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))

    def python_value(self, value):
        return json.loads(value) if isinstance(value, str) else value


def _make_policy_models(base):
    class JSON(_JSONField):
        field_type = "LONGTEXT" if isinstance(base._meta.database, MySQLDatabase) else "TEXT"

    class PolicyScope(base):
        id = CharField(max_length=32, primary_key=True)
        tenant_id = CharField(max_length=32)
        kb_id = CharField(max_length=32)
        connector_id = CharField(max_length=32)
        city = CharField(default="Shanghai")
        coverage_manifest = JSON(default=dict)
        active_release_id = CharField(max_length=32, null=True)
        last_scan_success_at = CharField(null=True)
        last_published_at = CharField(null=True)
        next_scan_at = CharField(null=True)
        retry_count = IntegerField(default=0)
        next_retry_at = CharField(null=True)
        last_error = TextField(null=True)
        lease_owner = CharField(null=True)
        lease_until = FloatField(default=0)
        generation = IntegerField(default=0)
        scan_sequence = IntegerField(default=0)
        source_heads = JSON(default=dict)

        class Meta:
            table_name = "policy_scope"
            indexes = ((("tenant_id", "kb_id", "connector_id"), True),)

    class PolicyRevision(base):
        id = CharField(max_length=32, primary_key=True)
        scope_id = CharField(max_length=32, index=True)
        policy_key = CharField(max_length=64)
        capture_sequence = IntegerField()
        revision_hash = CharField(max_length=64, null=True)
        sealed_revision_id = CharField(max_length=32, null=True)
        source_url = TextField()
        source_kind = CharField()
        title = TextField()
        document_number = TextField(null=True)
        published_at = CharField(null=True)
        signed_at = CharField(null=True)
        observed_at = CharField()
        raw_object_keys = JSON(default=dict)
        assets = JSON(default=list)
        parsing_builds = JSON(default=dict)
        temporal_metadata = JSON(default=lambda: {"status": "unknown"})
        relations = JSON(default=list)
        state = CharField(default="collecting")
        errors = JSON(default=list)
        retry_count = IntegerField(default=0)
        next_retry_at = CharField(null=True)

        class Meta:
            table_name = "policy_revision"
            indexes = ((("scope_id", "policy_key", "revision_hash"), True),)

    class PolicyRelease(base):
        id = CharField(max_length=32, primary_key=True)
        scope_id = CharField(max_length=32, index=True)
        release_id = CharField(max_length=32, unique=True)
        parent_release_id = CharField(max_length=32, null=True)
        revision_ids = JSON(default=dict)
        document_manifest = JSON(default=dict)
        chunk_manifest_object_key = TextField()
        chunk_manifest_sha256 = CharField(max_length=64)
        coverage_status = JSON(default=dict)
        source_heads = JSON(default=dict)
        created_at = CharField()

        class Meta:
            table_name = "policy_release"

    return PolicyScope, PolicyRevision, PolicyRelease
