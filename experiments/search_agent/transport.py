"""Bounded accounting; reconnect only when transport proves no request was sent."""
import asyncio
import json
import zlib
import aiohttp
import httpx

from experiments.evaluation.native_agent_transport import UsageCapture, _Tee, _exception_chain


def monitored_send(original, ledger, config, before_send, event):
    retries = config.get("connect_retries", 0)
    if type(retries) is not int or not 0 <= retries <= 2:
        raise ValueError("invalid_connect_retries")
    async def send(client, request):
        ledger.assert_open()
        if request.url.host not in config["provider_hosts"] or request.method != "POST" or not request.url.path.endswith("/chat/completions"):
            raise ValueError("unexpected_provider_endpoint")
        body = json.loads(request.content)
        if body.get("model") != config["model"] or body.get("enable_thinking") is not False or not 0 < body.get("max_tokens", 0) <= 1200:
            raise ValueError("unexpected_provider_configuration")
        before_send(body)
        reserve = len(json.dumps(body, ensure_ascii=False).encode()) + 4096 + body["max_tokens"]
        for attempt in range(retries + 1):
            cid = f"{ledger.question_id}:{len(ledger.calls) + 1}"
            ledger.reserve(cid, reserve, body)
            event({"event": "provider_reserved", "call_id": cid, "reserve_tokens": reserve})
            try:
                response = await original(client, request)
                break
            except BaseException as exc:
                unsent = isinstance(exc, httpx.ConnectError) and isinstance(exc.__cause__, aiohttp.ClientConnectorError)
                event({"event": "provider_send_failure", "call_id": cid, "not_sent": unsent,
                       "exception_type": type(exc).__name__, "exception_chain": _exception_chain(exc)})
                if unsent:
                    ledger.finish_not_sent(cid)
                    event({"event": "provider_finished", "call_id": cid, "transport_status": "not_sent", "usage": {"total_tokens": 0}})
                    if attempt < retries:
                        await asyncio.sleep(.25 * (attempt + 1))
                        continue
                    ledger.abort("provider_connect_failed")
                else:
                    ledger.abort("provider_usage_unknown")
                raise

        def fail(reason="stream_incomplete"):
            event({"event": "provider_usage_failure", "call_id": cid,
                   "http_status": getattr(response, "status_code", None),
                   "error_code": capture.error_code(), "response_bytes": len(capture.body), "reason": reason})
            ledger.abort("provider_usage_unknown")

        def complete(capture):
            try:
                usage = capture.finish()
                ledger.finish(cid, usage)
                event({"event": "provider_finished", "call_id": cid, "usage": usage})
            except (ValueError, KeyError, UnicodeError, OSError, EOFError, zlib.error) as exc:
                fail(type(exc).__name__)

        if response.is_stream_consumed:
            capture = UsageCapture(response.headers.get("content-type", ""), "")
            capture.feed(response.content)
            complete(capture)
        else:
            capture = UsageCapture(response.headers.get("content-type", ""), response.headers.get("content-encoding", ""))
            response.stream = _Tee(response.stream, capture, complete, fail)
        return response
    return send
