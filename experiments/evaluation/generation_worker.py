"""Bounded provider worker executed in the existing container; keys stay in RAM."""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
import sys
import time


def validate_provider(config, provider):
    if not provider:
        raise ValueError("Configured provider connection missing")
    extra = json.loads(provider["extra"]) if isinstance(provider["extra"], str) else provider["extra"]
    if extra["base_url"].rstrip("/") != config["base_url"] or config["base_url"] != "https://dashscope.aliyuncs.com/compatible-mode/v1":
        raise ValueError("Configured endpoint differs from frozen request")
    if config["model_name"] != "qwen3.7-flash" or not 0 < config["max_calls"] <= 20 or not 0 < config["parameters"]["max_tokens"] <= 1200:
        raise ValueError("Unbounded model request")


def completion_payload(config, messages):
    parameters = dict(config["parameters"])
    extra = parameters.pop("extra_body", {})
    if set(extra) & set(parameters):
        raise ValueError("Conflicting provider parameters")
    return {"model": config["model_name"], "messages": messages, "stream": False, **parameters, **extra}


def parse_response(body, status):
    if status != 200:
        error = body.get("error", body)
        code = error.get("code")
        category = "auth" if status in (401, 403) else "configuration" if status in (400, 404) else "quota" if any(word in str(code).lower() for word in ["quota", "balance", "arrearage"]) else "transport_or_service"
        return {"status": "error", "http_status": status, "error_type": error.get("type", "ProviderError"), "error_code": code, "error_category": category}
    choice = body["choices"][0]
    return {"status": "output_truncated" if choice["finish_reason"] == "length" else "ok", "text": choice["message"]["content"], "finish_reason": choice["finish_reason"], "model": body["model"], "provider_request_id": body["id"], "provider_created": body.get("created"), "usage": body.get("usage") or {}}


def main():
    logging.disable(logging.CRITICAL)
    config = json.loads(sys.stdin.readline())
    try:
        import pymysql
        import yaml
        import requests

        settings = yaml.safe_load(Path("/ragflow/conf/service_conf.yaml").read_text())["mysql"]
        db = pymysql.connect(host=settings["host"], port=int(settings["port"]), user=settings["user"], password=settings["password"], database=settings["name"], cursorclass=pymysql.cursors.DictCursor)
        with db.cursor() as cursor:
            cursor.execute("SELECT api_key,extra FROM tenant_model_instance WHERE id=%s", (config["provider_instance_id"],))
            provider = cursor.fetchone()
        db.close()
        validate_provider(config, provider)
        client = requests.Session()
        client.headers.update({"Authorization": "Bearer " + provider["api_key"]})
        print(json.dumps({"status": "ready", "model": config["model_name"]}), flush=True)
    except Exception as exc:
        print(json.dumps({"status": "init_error", "error_type": type(exc).__name__}), flush=True)
        return
    calls = 0
    for line in sys.stdin:
        started = time.perf_counter()
        requested_at = datetime.now(timezone.utc).isoformat()
        request = json.loads(line)
        if calls >= config["max_calls"]:
            print(json.dumps({"status": "error", "error_type": "CallLimit"}), flush=True)
            continue
        calls += 1
        try:
            response = client.post(config["base_url"] + "/chat/completions", json=completion_payload(config, request["messages"]), timeout=(10, 45))
            result = parse_response(response.json(), response.status_code)
        except Exception as exc:
            message = str(exc).lower()
            status = getattr(exc, "status_code", None)
            category = "auth" if status in (401, 403) else "configuration" if status in (400, 404) else "quota" if any(word in message for word in ["quota", "arrearage", "insufficient balance"]) else "transport_or_service"
            result = {"status": "error", "error_type": type(exc).__name__, "http_status": status, "error_category": category, "tls_eof": "EOF" in str(exc)}
        result.update({"requested_at": requested_at, "latency_ms": (time.perf_counter() - started) * 1000})
        print(json.dumps(result, ensure_ascii=False), flush=True)
    client.close()


if __name__ == "__main__":
    main()
