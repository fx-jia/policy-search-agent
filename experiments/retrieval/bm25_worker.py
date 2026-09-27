"""Read-only container worker: local BM25, no model wrappers or API tokens."""

import contextlib
import hashlib
import json
import logging
from pathlib import Path
import sys
import time


def bm25_body(tokenized, dataset_ids, document_ids, limit=64):
    if not tokenized.strip() or not dataset_ids or not document_ids or limit < 1:
        raise ValueError('Missing query or frozen scope')
    return {
        'size': limit, '_source': ['id', 'doc_id', 'kb_id'],
        'sort': [{'_score': 'desc'}, {'id': 'asc'}],
        'query': {'bool': {
            'filter': [{'terms': {'kb_id': dataset_ids}}, {'terms': {'doc_id': document_ids}}],
            'must_not': [{'range': {'available_int': {'lt': 1}}}],
            'must': [{'match': {'content_ltks': {'query': tokenized, 'operator': 'or', 'minimum_should_match': 1}}}],
        }},
    }


def _run(request):
    import pymysql
    import yaml
    from elasticsearch import Elasticsearch
    from rag.nlp import rag_tokenizer

    mapping = request['mapping']
    settings = yaml.safe_load(Path('/ragflow/conf/service_conf.yaml').read_text())
    mysql = settings['mysql']
    db = pymysql.connect(host=mysql['host'], port=int(mysql['port']), user=mysql['user'], password=mysql['password'], database=mysql['name'])
    with db.cursor() as cursor:
        placeholders = ','.join(['%s'] * len(mapping['dataset_ids']))
        cursor.execute(f'SELECT DISTINCT tenant_id FROM knowledgebase WHERE id IN ({placeholders})', mapping['dataset_ids'])
        tenants = cursor.fetchall()
    db.close()
    if len(tenants) != 1:
        raise ValueError('Expected one tenant index')
    config = settings['es']
    es = Elasticsearch(config['hosts'].split(','), basic_auth=(config['username'], config['password']), verify_certs=True, request_timeout=30, max_retries=0)
    index = 'ragflow_' + tenants[0][0]

    def index_state():
        stats = es.indices.stats(index=index, level='shards')['indices'][index]
        return {'index_uuid': stats['uuid'], 'tenant_index_documents': stats['primaries']['docs']['count'], 'primary_sequence_numbers': {k: v['seq_no'] for k, vs in stats['shards'].items() for v in vs if v['routing']['primary']}}

    before = index_state()
    if before != request['expected_index_state']:
        raise ValueError('Index changed since frozen chunk snapshot')
    properties = es.indices.get_mapping(index=index)[index]['mappings']['properties']
    index_config = es.indices.get_settings(index=index)[index]['settings']['index']
    field = properties['content_ltks']
    if field.get('type') != 'text' or field.get('analyzer') != 'whitespace' or field.get('similarity', 'BM25') != 'BM25' or 'default' in index_config.get('similarity', {}):
        raise ValueError('Expected standard BM25 whitespace content field')
    scope = bm25_body('scope-check', mapping['dataset_ids'], list(mapping['documents']))['query']['bool']
    del scope['must']
    if es.count(index=index, query={'bool': scope})['count'] != mapping['indexed_chunks']:
        raise ValueError('Frozen scope count changed')
    if len(request['queries']) != 51 or len({q['question_id'] for q in request['queries']}) != 51:
        raise ValueError('Only the approved 51-query batch is allowed')
    rows = []
    for query in request['queries']:
        started = time.perf_counter()
        row = {'question_id': query['question_id'], 'status': 'error', 'hits': []}
        try:
            tokenized = rag_tokenizer.tokenize(query['question'])
            body = bm25_body(tokenized, mapping['dataset_ids'], list(mapping['documents']))
            result = es.search(index=index, body=body, search_type='dfs_query_then_fetch', preference='rrf-fusion-v1')
            if result.get('timed_out') or result.get('_shards', {}).get('failed', 0):
                raise RuntimeError('Incomplete search response')
            hits = []
            for hit in result['hits']['hits']:
                source = hit['_source']
                if source['id'] != hit['_id'] or source['doc_id'] not in mapping['documents']:
                    raise ValueError('Chunk identity or scope mismatch')
                kbids = source['kb_id'] if isinstance(source['kb_id'], list) else [source['kb_id']]
                if mapping['document_datasets'][source['doc_id']] not in kbids:
                    raise ValueError('Dataset identity mismatch')
                hits.append({'id': hit['_id'], 'document_id': source['doc_id'], 'score': hit['_score']})
            row.update(status='ok', hits=hits, tokenized_query=tokenized, total_hits=result['hits']['total'])
        except Exception as exc:
            row['error_type'] = type(exc).__name__  # Never emit raw service errors.
        row['local_bm25_ms'] = (time.perf_counter() - started) * 1000
        rows.append(row)
    after = index_state()
    if after != before:
        raise ValueError('Index changed during retrieval')
    metadata = {'before': before, 'after': after, 'field_mapping': field, 'bm25_parameters': {'k1': 1.2, 'b': .75}, 'search_type': 'dfs_query_then_fetch', 'elasticsearch_version': es.info()['version']['number'], 'tokenizer_source_sha256': hashlib.sha256(Path(rag_tokenizer.__file__).read_bytes()).hexdigest(), 'model_calls': 0, 'index_writes': 0}
    es.close()
    return {'status': 'ok', 'metadata': metadata, 'queries': rows}


def main():
    logging.disable(logging.CRITICAL)
    try:
        request = json.loads(sys.stdin.read())
        with contextlib.redirect_stdout(sys.stderr):
            result = _run(request)
    except Exception as exc:
        result = {'status': 'initialization_or_integrity_error', 'error_type': type(exc).__name__}
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
