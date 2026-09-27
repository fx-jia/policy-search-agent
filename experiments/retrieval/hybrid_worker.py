"""Use the existing local retrieval API; its token stays inside container RAM."""

import json
import os
from pathlib import Path
import sys


def main():
    import pymysql
    import yaml

    init = json.loads(sys.stdin.readline())
    try:
        config = init["config"]
        if config["base_url"] != "http://127.0.0.1:9380" or config["method"] != "hybrid" or config.get("rerank_model"):
            raise ValueError("Expected frozen local Hybrid without rerank")
        settings = yaml.safe_load(Path("/ragflow/conf/service_conf.yaml").read_text())["mysql"]
        db = pymysql.connect(host=settings["host"], port=int(settings["port"]), user=settings["user"], password=settings["password"], database=settings["name"])
        with db.cursor() as cursor:
            cursor.execute("SELECT tenant_id FROM knowledgebase WHERE id=%s", (config["dataset_ids"][0],))
            tenant = cursor.fetchone()[0]
            cursor.execute("SELECT token FROM api_token WHERE tenant_id=%s AND (dialog_id IS NULL OR dialog_id='') ORDER BY create_time DESC LIMIT 1", (tenant,))
            row = cursor.fetchone()
        db.close()
        if not row:
            raise ValueError("Configured RAGFlow API token missing")
        namespace = {"__name__": "heldout_ragflow_adapter"}
        exec(compile(init["adapter_source"], "ragflow_adapter.py", "exec"), namespace)
        os.environ["RAGFLOW_API_KEY"] = row[0]
        client = namespace["RAGFlowClient"](config, init["mapping"]["documents"])
        del os.environ["RAGFLOW_API_KEY"]
        print(json.dumps({"status": "ready"}), flush=True)
    except Exception as exc:
        print(json.dumps({"status": "init_error", "error_type": type(exc).__name__}), flush=True)
        return
    count = 0
    for line in sys.stdin:
        item = json.loads(line)
        if count >= 99:
            print(json.dumps({"status": "error", "error": "call_limit", "request_id": item["request_id"]}), flush=True)
            continue
        count += 1
        try:
            chunks = client.retrieve(item["question"], 64)
            result = {"status": "ok", "chunks": chunks}
        except namespace["RetrievalError"] as exc:
            result = {"status": "error", "error": str(exc), "chunks": []}
        except Exception as exc:
            result = {"status": "error", "error": type(exc).__name__, "chunks": []}
        result["request_id"] = item["request_id"]
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
