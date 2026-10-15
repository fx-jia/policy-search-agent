"""Operate Shanghai policy sync and the live, bounded Search Agent."""

import argparse
import asyncio
from dataclasses import asdict, is_dataclass
import json
from pathlib import Path
import sys
import uuid


def _print(value):
    print(json.dumps(asdict(value) if is_dataclass(value) else value, ensure_ascii=False, indent=2, default=str))


def _config(path):
    import yaml

    values = yaml.safe_load(Path(path).read_text())
    if not isinstance(values, dict):
        raise ValueError("policy source YAML must contain an object")
    return values


def _runtime():
    from common import settings

    settings.init_settings()
    from api.db.services.policy_catalog_service import PolicyCatalogService
    from api.db.services.policy_publication_service import PolicyPublicationService

    catalog = PolicyCatalogService()
    return catalog, PolicyPublicationService(catalog)


def _authorize(tenant_id, kb_id):
    from api.db.services.knowledgebase_service import KnowledgebaseService

    ok, kb = KnowledgebaseService.get_by_id(kb_id)
    if not ok or kb.tenant_id != tenant_id:
        raise PermissionError("knowledge base does not belong to this tenant")
    return kb


def _scope(catalog, tenant_id, kb_id):
    _authorize(tenant_id, kb_id)
    rows = list(catalog.Scope.select().where((catalog.Scope.tenant_id == tenant_id) & (catalog.Scope.kb_id == kb_id)))
    if len(rows) != 1:
        raise ValueError("expected exactly one initialized Shanghai policy scope")
    return rows[0].id


def check_source(args):
    from common import settings  # noqa: F401 - native registry imports require settings bootstrap first
    from common.data_source.shanghai_medical_connector import ShanghaiMedicalConnector

    config = _config(args.config)
    config.update(parse_build_key="source-preflight")
    connector = ShanghaiMedicalConnector.build_connector(config)
    captures = []
    try:
        connector.validate_connector_settings()
        for url in connector.config.seed_urls[:3]:
            from datetime import datetime, timezone

            row = connector._capture(url, datetime.now(timezone.utc).isoformat())
            captures.append({key: row.get(key) for key in ("source_url", "title", "document_number", "published_at", "signed_at", "revision_hash", "errors")})
            captures[-1]["assets"] = [{key: asset.get(key) for key in ("role", "url", "sha256")} for asset in row["assets"]]
        _print({"preflight_complete": all(not row["errors"] for row in captures), "captures": captures, "list_urls": connector.config.list_urls})
        if any(row["errors"] for row in captures):
            raise RuntimeError("official source preflight has incomplete policy captures")
    finally:
        connector.cancel()


def initialize(args):
    catalog, _ = _runtime()
    from api.db.services.connector_service import ConnectorService, Connector2KbService
    from api.db.services.policy_sync_service import parse_build_key, validate_policy_config
    from common.constants import TaskStatus
    from api.db import InputType

    kb = _authorize(args.tenant_id, args.kb_id)
    config = _config(args.config)
    config["parse_generation"] = args.parse_generation
    config["parse_build_key"] = parse_build_key(kb.to_dict(), parse_generation=args.parse_generation)
    validate_policy_config("shanghai_medical", config)
    catalog.db.create_tables([catalog.Scope, catalog.Revision, catalog.Release], safe=True)
    scopes = list(catalog.Scope.select().where((catalog.Scope.tenant_id == args.tenant_id) & (catalog.Scope.kb_id == args.kb_id)))
    links = Connector2KbService.list_connectors(args.kb_id)
    if scopes:
        if len(scopes) != 1 or not any(link["id"] == scopes[0].connector_id for link in links):
            raise ValueError("policy scope is ambiguous or its connector is unlinked")
        connector_id = scopes[0].connector_id
        ok, connector = ConnectorService.get_by_id(connector_id)
        if not ok or connector.tenant_id != args.tenant_id or connector.source != "shanghai_medical":
            raise PermissionError("policy connector ownership/source mismatch")
        ConnectorService.update_by_id(connector_id, {"config": config, "refresh_freq": 1440, "timeout_secs": config.get("task_timeout_seconds", 3600)})
    else:
        if links:
            raise ValueError("initialize a dedicated policy knowledge base without other connectors")
        connector_id = uuid.uuid4().hex
        ConnectorService.save(
            id=connector_id,
            tenant_id=args.tenant_id,
            name="上海医保政策",
            source="shanghai_medical",
            input_type=InputType.POLL,
            config=config,
            refresh_freq=1440,
            prune_freq=1440,
            timeout_secs=config.get("task_timeout_seconds", 3600),
            status=TaskStatus.UNSTART,
        )
        Connector2KbService.link_connectors(args.kb_id, [{"id": connector_id, "auto_parse": "1"}], args.tenant_id)
    scope = catalog.ensure_scope(args.tenant_id, args.kb_id, connector_id, coverage={"city": "Shanghai", "sources": config["list_urls"], "foundational_complete": False})
    _print({"scope_id": scope["id"], "connector_id": connector_id, "kb_id": args.kb_id, "parse_build_key": config["parse_build_key"], "refresh_minutes": 1440})


def operate(args):
    catalog, publisher = _runtime()
    scope_id = _scope(catalog, args.tenant_id, args.kb_id)
    if args.command == "status":
        _print(catalog.status(scope_id))
    elif args.command == "sync-now":
        from api.db.services.connector_service import SyncLogsService
        from common.constants import TaskStatus

        scope = catalog.get_scope(scope_id)
        rows = SyncLogsService.query(connector_id=scope["connector_id"], kb_id=args.kb_id, status=TaskStatus.RUNNING)
        if rows:
            raise ValueError("policy scan already running")
        SyncLogsService.schedule(scope["connector_id"], args.kb_id, reindex=True, run_immediately=True)
        _print({"scheduled": True, "scope_id": scope_id})
    elif args.command == "review":
        revision = catalog.Revision.get_by_id(args.revision_id)
        if revision.scope_id != scope_id:
            raise PermissionError("revision does not belong to this policy scope")
        meta = json.loads(Path(args.metadata).read_text())
        meta["reviewed_by"] = args.tenant_id
        _print(catalog.review_revision(args.revision_id, meta))
    elif args.command == "publish":
        _print(publisher.reconcile(scope_id))


async def ask(args):
    _runtime()
    from api.db.services.policy_search_service import PolicySearchService

    service = PolicySearchService(args.tenant_id, args.kb_id, uuid.uuid4().hex, args.output, budget=args.budget)
    if args.question:
        _print(await service.ask(args.question, as_of=args.as_of))
        return
    print("上海医保 Search Agent。/clear 清空会话；/exit 退出。")
    while True:
        question = await asyncio.to_thread(input, "问题> ")
        if question.strip() == "/exit":
            return
        if question.strip() == "/clear":
            await service.clear()
            continue
        if question.strip():
            record = await service.ask(question, as_of=args.as_of)
            _print(record)
            if record.get("ledger", {}).get("pending_calls") or record.get("ledger", {}).get("fatal_reason"):
                return


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    default = str(Path(__file__).resolve().parents[2] / "conf/policy_sources/shanghai.yaml")
    source = subs.add_parser("check-source", help="Bounded official-site preflight; no model calls")
    source.add_argument("--config", default=default)
    for name in ("init", "sync-now", "status", "review", "publish", "ask"):
        sub = subs.add_parser(name)
        sub.add_argument("--tenant-id", required=True)
        sub.add_argument("--kb-id", required=True)
        if name == "init":
            sub.add_argument("--config", default=default)
            sub.add_argument("--parse-generation", type=int, default=0, help="Increase to rebuild a published parser/index without overwriting prior assets")
        elif name == "review":
            sub.add_argument("--revision-id", required=True)
            sub.add_argument("--metadata", required=True, help="JSON with status, ISO effective dates, and exact source evidence")
        elif name == "ask":
            sub.add_argument("--question")
            sub.add_argument("--as-of", help="YYYY-MM-DD, defaults to current Shanghai date")
            sub.add_argument("--output", required=True, type=Path)
            sub.add_argument("--budget", type=int, default=60000)
    args = parser.parse_args(argv)
    try:
        if args.command == "check-source":
            check_source(args)
        elif args.command == "init":
            initialize(args)
        elif args.command == "ask":
            asyncio.run(ask(args))
        else:
            operate(args)
    except (Exception, KeyboardInterrupt) as exc:
        print(type(exc).__name__ + ": " + str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
