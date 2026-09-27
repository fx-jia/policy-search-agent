"""Bounded gte-rerank-v2 HTTP worker; credentials stay in container memory."""

from datetime import datetime, timezone
import json
import logging
import math
from pathlib import Path
import re
import sys
import time


ENDPOINT = 'https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank'


def make_payload(query, documents):
    if not isinstance(query, str) or not 0 < len(query) <= 200 or not 1 <= len(documents) <= 32 or any(not isinstance(t, str) or not 0 < len(t) <= 1800 for t in documents) or sum(len(t) + len(query) for t in documents) > 18000:
        raise ValueError('Bounded rerank request limits exceeded')
    return {'model': 'gte-rerank-v2', 'input': {'query': query, 'documents': documents}, 'parameters': {'top_n': len(documents), 'return_documents': False}}


def parse_result(body, http_status, count):
    if http_status != 200:
        code = body.get('code', '') if isinstance(body, dict) else ''
        code = code if isinstance(code, str) and re.fullmatch(r'[a-zA-Z0-9_.-]{0,100}', code) else 'Unclassified'
        category = 'quota' if any(s in code.lower() for s in ['quota', 'balance', 'arrearage', 'freetier']) else 'auth' if http_status in (401,403) else 'configuration' if http_status in (400,404) else 'transport_or_service'
        return {'status': 'error', 'http_status': http_status, 'error_code': code, 'error_category': category}
    try:
        rows = body['output']['results']
        if len(rows) != count or {r['index'] for r in rows} != set(range(count)) or any(type(r['index']) is not int or type(r['relevance_score']) not in (int,float) or not math.isfinite(r['relevance_score']) or not 0 <= r['relevance_score'] <= 1 for r in rows):
            raise ValueError('Incomplete or invalid scores')
        return {'status': 'ok', 'scores': [{'index': r['index'], 'relevance_score': r['relevance_score']} for r in rows], 'usage': body.get('usage') or {}, 'provider_request_id': body.get('request_id')}
    except (ValueError, KeyError, TypeError):
        return {'status': 'error', 'http_status': http_status, 'error_category': 'invalid_response', 'error_type': 'IncompleteScores'}


def main():
    logging.disable(logging.CRITICAL)
    config = json.loads(sys.stdin.readline())
    try:
        import pymysql
        import requests
        import yaml
        import dashscope
        if config['endpoint'] != ENDPOINT or config['model_id'] != '7ff1b702e59c52ad520d8c0b6720cdce' or not 0 < config['max_calls'] <= 60 or dashscope.base_http_api_url.rstrip('/') != ENDPOINT.split('/services/')[0]:
            raise ValueError('Provider or budget mismatch')
        mysql = yaml.safe_load(Path('/ragflow/conf/service_conf.yaml').read_text())['mysql']
        db = pymysql.connect(host=mysql['host'],port=int(mysql['port']),user=mysql['user'],password=mysql['password'],database=mysql['name'],cursorclass=pymysql.cursors.DictCursor)
        with db.cursor() as cursor:
            cursor.execute('SELECT m.model_name,m.model_type,i.api_key FROM tenant_model m JOIN tenant_model_instance i ON m.instance_id=i.id WHERE m.id=%s', (config['model_id'],))
            provider = cursor.fetchone()
        db.close()
        if not provider or provider['model_name'] != 'gte-rerank-v2' or not int(provider['model_type']) & 16:
            raise ValueError('Configured reranker differs')
        client = requests.Session()
        client.headers.update({'Authorization': 'Bearer ' + provider['api_key']})
        print(json.dumps({'status':'ready','model':'gte-rerank-v2'}),flush=True)
    except Exception as exc:
        print(json.dumps({'status':'init_error','error_type':type(exc).__name__}),flush=True)
        return
    calls=0
    for line in sys.stdin:
        request=json.loads(line)
        started=time.perf_counter()
        attempted=False
        try:
            payload=make_payload(request['query'],request['documents'])
            if calls>=config['max_calls']:
                raise ValueError('Call limit exceeded')
            calls+=1;attempted=True
            response=client.post(ENDPOINT,json=payload,timeout=(10,45))
            try:
                body=response.json()
            except ValueError:
                body={}
            result=parse_result(body,response.status_code,len(request['documents']))
        except Exception as exc:
            result={'status':'error','error_category':'transport_or_service' if attempted else 'configuration','error_type':type(exc).__name__,'tls_eof':'EOF' in str(exc)}
        result.update(request_id=request['request_id'],attempted=attempted,latency_ms=(time.perf_counter()-started)*1000,recorded_at=datetime.now(timezone.utc).isoformat())
        print(json.dumps(result),flush=True)
    client.close()


if __name__=='__main__':
    main()
