"""Count embedding attempts; reconnect only after a proven TLS handshake EOF."""
from urllib.parse import urlparse
import ssl
import time


def _handshake_eof(error):
    pending, seen = [error], set()
    while pending and len(seen) < 20:
        exc = pending.pop()
        if not isinstance(exc, BaseException) or id(exc) in seen:
            continue
        seen.add(id(exc))
        if isinstance(exc, ssl.SSLEOFError):
            tb = exc.__traceback__
            while tb:
                if tb.tb_frame.f_globals.get("__name__") == "ssl" and tb.tb_frame.f_code.co_name == "do_handshake":
                    return True
                tb = tb.tb_next
        pending.extend([exc.__cause__, exc.__context__, getattr(exc, "reason", None), *exc.args])
    return False


def single_request(client, inputs, text_type, sdk_call, api_scope, stats, connect_retries=0):
    if type(connect_retries) is not int or not 0 <= connect_retries <= 2:
        raise ValueError("invalid_connect_retries")
    if text_type not in {"query", "document"}:
        raise ValueError("invalid_embedding_purpose")
    base = client._dashscope_http_api_url or "https://dashscope.aliyuncs.com/api/v1"
    host = urlparse(base).hostname
    key = "query_physical_calls" if text_type == "query" else "citation_physical_calls"
    limit = 3 if text_type == "query" else 16
    with client._session_lock, api_scope(client._dashscope_http_api_url):
        session = client._session
        if session is None:
            raise ValueError("closed_embedding_client")
        original = session.send
        attempts = 0

        def send(request, **kwargs):
            nonlocal attempts
            if attempts or stats.get(key, 0) >= limit:
                raise ValueError("embedding_physical_call_limit_or_retry")
            url = urlparse(request.url)
            if request.method != "POST" or url.scheme != "https" or url.hostname != host or not url.path.endswith("/text-embedding/text-embedding"):
                raise ValueError("unexpected_embedding_endpoint")
            attempts += 1
            stats[key] = stats.get(key, 0) + 1
            kwargs.update(timeout=15, allow_redirects=False, verify=True)
            # requests' adapter must not retry below Session.send either.
            if hasattr(session, "get_adapter"):
                from urllib3.util.retry import Retry
                session.get_adapter(request.url).max_retries = Retry(total=0, connect=0, read=0, redirect=0, status=0)
            return original(request, **kwargs)

        session.send = send
        try:
            for attempt in range(connect_retries + 1):
                attempts = 0
                try:
                    response = sdk_call(model=client.model_name, input=inputs, api_key=client.key, text_type=text_type, session=session)
                    if response.status_code != 200:
                        raise ValueError("embedding_http_error:" + str(response.status_code))
                    return response
                except BaseException as exc:
                    if attempts and _handshake_eof(exc):
                        unsent_key = "query_not_sent_calls" if text_type == "query" else "citation_not_sent_calls"
                        stats[unsent_key] = stats.get(unsent_key, 0) + 1
                        if attempt < connect_retries and stats.get(key, 0) < limit:
                            time.sleep(.25 * (attempt + 1))
                            continue
                    elif attempts:
                        stats["unknown_embedding_usage"] = True
                    raise
        finally:
            session.send = original
