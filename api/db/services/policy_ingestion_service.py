"""Check the native parser's actual model against its admitted policy build."""

from contextlib import nullcontext
from functools import partial


def _task_parser_config(config):
    if not isinstance(config, dict):
        raise ValueError("policy parser configuration is missing from its build binding")
    # TaskService.get_task omits these independently queued post-processing stages.
    return {key: value for key, value in config.items() if key not in ("graphrag", "raptor")}


def validate_policy_parse_task(task, model_config, *, catalog=None, metadata_reader=None):
    source_name, _, connector_id = str(task.get("source_type") or "").partition("/")
    if source_name != "shanghai_medical":
        return
    from api.db.services.policy_sync_service import embedding_config_sha256

    owns_connection = catalog is None
    if catalog is None:
        from api.db.services.policy_catalog_service import PolicyCatalogService

        catalog = PolicyCatalogService()
    if metadata_reader is None:
        from api.db.services.doc_metadata_service import DocMetadataService

        metadata_reader = partial(DocMetadataService.get_document_metadata, strict=True)
    with catalog.db.connection_context() if owns_connection else nullcontext():
        metadata = metadata_reader(task["doc_id"]) or {}
        if not all(metadata.get(key) for key in ("scope_id", "revision_id", "parse_build_key")):
            raise ValueError("policy parse task is missing its admitted build binding")
        scope = catalog.get_scope(metadata["scope_id"])
        if not connector_id or connector_id != scope["connector_id"]:
            raise ValueError("policy parse task connector does not match its admitted scope")
        if any(scope[key] != task.get(key) for key in ("tenant_id", "kb_id")):
            raise ValueError("policy parse task belongs to another scope")
        revision = catalog.Revision.get_by_id(metadata["revision_id"])
        if revision.scope_id != scope["id"] or not revision.revision_hash:
            raise ValueError("policy parse task has an unsealed or foreign revision")
        assets = revision.parsing_builds.get(metadata["parse_build_key"], [])
        bindings = [asset for asset in assets if asset["doc_id"] == task["doc_id"]]
        if len(bindings) != 1:
            raise ValueError("policy parse task document has no unique admitted build binding")
        binding = bindings[0]
        if not binding.get("parser_id") or binding["parser_id"] != task.get("parser_id"):
            raise ValueError("policy parser identity changed after admission")
        if _task_parser_config(binding.get("parser_config")) != _task_parser_config(task.get("parser_config")):
            raise ValueError("policy parser configuration changed after admission")
        if any(binding.get(key) != task.get(key) for key in ("embd_id", "tenant_embd_id")):
            raise ValueError("policy embedding reference changed after admission")
        if not binding.get("embedding_model") or embedding_config_sha256(model_config) != embedding_config_sha256(binding["embedding_model"]):
            raise ValueError("policy embedding configuration changed after admission")
