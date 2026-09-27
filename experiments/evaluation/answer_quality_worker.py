"""One isolated, durably accounted Judge request. No retrieval or embeddings."""
import asyncio
import hashlib
import json
from pathlib import Path
import sys
import time


async def run(data):
    import httpx
    import pymysql
    import yaml
    from common.token_utils import num_tokens_from_string

    ns = {}
    exec(compile(data["ledger_source"], "<quality-ledger>", "exec"), ns)
    job = data["job"]
    ledger = ns["CallLedger"](data["ledger_path"], run_limit=60000, question_limit=60000, call_limit=3)
    ledger.start(job["id"])
    cfg = yaml.safe_load(Path("/ragflow/conf/service_conf.yaml").read_text())["mysql"]
    db = pymysql.connect(host=cfg["host"], port=int(cfg["port"]), user=cfg["user"], password=cfg["password"], database=cfg["name"])
    try:
        with db.cursor() as cur:
            cur.execute("SELECT i.api_key,i.extra FROM tenant_model m JOIN tenant_model_instance i ON i.id=m.instance_id JOIN tenant_model_provider p ON p.id=m.provider_id WHERE p.tenant_id=(SELECT tenant_id FROM dialog WHERE id=%s) AND m.model_name=%s AND m.status='active' LIMIT 1", (data["chat_id"], data["credential_model"]))
            model = cur.fetchone()
        if not model:
            raise ValueError("configured_judge_missing")
        key, extra = model
        endpoint = json.loads(extra or "{}").get("base_url", "").rstrip("/")
        if endpoint != "https://dashscope.aliyuncs.com/compatible-mode/v1":
            raise ValueError("unexpected_provider_endpoint")
        body = {"model": data["model"], "messages": [{"role": "system", "content": data["system"]},
                {"role": "user", "content": json.dumps(job["packet"], ensure_ascii=False)}],
                "temperature": 0, "max_tokens": 1200, "enable_thinking": False,
                "response_format": {"type": "json_object"}, "stream": False}
        tokens = sum(num_tokens_from_string(m["content"]) for m in body["messages"])
        if tokens + 1200 + 100 > data.get("max_context_tokens", 8192):
            raise ValueError("judge_context_budget")
        reserve = len(json.dumps(body, ensure_ascii=False).encode()) + 4096 + 1200
        async with httpx.AsyncClient(timeout=httpx.Timeout(45, connect=10), trust_env=False) as client:
            for attempt in range(3):
                cid = f'{job["id"]}:{attempt}'
                ledger.reserve(cid, reserve, body)
                try:
                    response = await client.post(endpoint + "/chat/completions", json=body, headers={"Authorization": "Bearer " + key})
                except httpx.ConnectError:
                    ledger.finish_not_sent(cid)
                    if attempt < 2:
                        await asyncio.sleep(0.25 * (2 ** attempt))
                        continue
                    raise ValueError("provider_connect_failed") from None
                raw = response.json()
                usage = raw.get("usage")
                if not usage:
                    ledger.abort("provider_usage_unknown")
                    code = raw.get("error", {}).get("code")
                    return {"status": "provider_error", "error_code": str(code)[:100], "http_status": response.status_code,
                            "ledger": ledger.summary(), "calls": list(ledger.calls.values())}
                ledger.finish(cid, usage)
                response.raise_for_status()
                choice = raw["choices"][0]
                return {"status": "ok", "output": choice["message"]["content"], "finish_reason": choice.get("finish_reason"),
                        "input_tokens_estimate": tokens, "provider_usage": usage,
                        "request_sha256": hashlib.sha256(json.dumps(body, ensure_ascii=False, sort_keys=True).encode()).hexdigest(),
                        "ledger": ledger.summary(), "calls": list(ledger.calls.values())}
    except Exception as exc:
        return {"status": "technical_error", "error_type": type(exc).__name__,
                "reason": str(exc) if isinstance(exc, ValueError) and str(exc).replace("_", "").isalnum() else "request_failed",
                "ledger": ledger.summary(), "calls": list(ledger.calls.values())}
    finally:
        db.close()


if __name__ == "__main__":
    config = json.load(sys.stdin)
    started = time.monotonic()
    try:
        result = asyncio.run(run(config))
    except Exception as exc:
        result = {"status": "worker_error", "error_type": type(exc).__name__}
    result.update(id=config["job"]["id"], latency_ms=round((time.monotonic() - started) * 1000))
    print("QUALITY_RESULT:" + json.dumps(result, ensure_ascii=False), flush=True)
