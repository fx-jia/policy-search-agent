"""Read existing local ES chunks and count cached cl100k tokens. No model calls."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess


# Deliberately imports no RAGFlow model wrappers; only SELECT, ES read and local BPE.
SCRIPT = r'''
import hashlib,json,os,sys,yaml,pymysql,tiktoken
from pathlib import Path
from elasticsearch import Elasticsearch
from elasticsearch.helpers import scan
mapping=json.loads(sys.stdin.read())
c=yaml.safe_load(Path('/ragflow/conf/service_conf.yaml').read_text()); m=c['mysql']
db=pymysql.connect(host=m['host'],port=int(m['port']),user=m['user'],password=m['password'],database=m['name'])
with db.cursor() as cursor:
 cursor.execute('SELECT tenant_id FROM knowledgebase WHERE id=%s',(mapping['dataset_ids'][0],))
 tenant=cursor.fetchone()[0]
db.close()
s=c['es']; es=Elasticsearch(s['hosts'].split(','),basic_auth=(s['username'],s['password']),verify_certs=True,request_timeout=30,max_retries=0)
index='ragflow_'+tenant
def snapshot():
 stats=es.indices.stats(index=index,level='shards')['indices'][index]
 return {'index_uuid':stats['uuid'],'tenant_index_documents':stats['primaries']['docs']['count'],'primary_sequence_numbers':{k:v['seq_no'] for k,vs in stats['shards'].items() for v in vs if v['routing']['primary']}}
before=snapshot()
# Fail before tiktoken can try a network download if its cache is absent.
cache=Path(os.environ.get('TIKTOKEN_CACHE_DIR',os.environ.get('DATA_GYM_CACHE_DIR','/tmp/data-gym-cache')))
cache_file=cache/hashlib.sha1(b'https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken').hexdigest()
if not cache_file.is_file(): raise ValueError('Tokenizer cache missing; network download prohibited')
enc=tiktoken.get_encoding('cl100k_base')
query={'query':{'bool':{'filter':[{'terms':{'doc_id':list(mapping['documents'])}},{'terms':{'kb_id':mapping['dataset_ids']}}],'must_not':[{'range':{'available_int':{'lt':1}}}]}},'_source':['doc_id','kb_id','content_with_weight','top_int','page_num_int','position_int']}
chunks=[]
for hit in scan(es,index=index,query=query):
 row=hit['_source']; did=row['doc_id']; sid=mapping['documents'][did]; cid=hit['_id']
 if mapping['document_datasets'][did] not in (row['kb_id'] if isinstance(row['kb_id'],list) else [row['kb_id']]): raise ValueError('Dataset mapping mismatch')
 top=row.get('top_int'); position=row.get('position_int')
 if not isinstance(top,list) or len(top)!=1 or not isinstance(top[0],int): raise ValueError('Ambiguous chunk order')
 order=top[0]
 if position != [[order+1,order,order,order,order]]: raise ValueError('Expected markdown sequential position metadata')
 content=row['content_with_weight']
 block=f'[source={sid}; chunk={cid}]\n{content}\n\n'
 chunks.append({'id':cid,'document_id':did,'source_document_id':sid,'order':order,'content':content,'content_sha256':hashlib.sha256(content.encode()).hexdigest(),'tokens':len(enc.encode(block,disallowed_special=()))})
if len(chunks)!=mapping['indexed_chunks']: raise ValueError('Frozen chunk count mismatch')
for did in mapping['documents']:
 orders=sorted(r['order'] for r in chunks if r['document_id']==did)
 if orders != list(range(len(orders))) or not orders: raise ValueError('Non-contiguous document chunk order')
after=snapshot()
if before!=after: raise ValueError('Index changed during read')
es.close()
print(json.dumps({'metadata':{'before':before,'after':after,'tokenizer':'cl100k_base','tiktoken_version':tiktoken.__version__,'tokenizer_cache_sha256':hashlib.sha256(cache_file.read_bytes()).hexdigest(),'token_accounting':'sum independently encoded [source=ID; chunk=ID] + newline + content + two newlines','model_calls':0},'chunks':sorted(chunks,key=lambda r:(r['document_id'],r['order']))},ensure_ascii=False))
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--document-map", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise ValueError("Refusing to overwrite snapshot")
    mapping = Path(args.document_map).read_bytes()
    result = subprocess.run(["docker", "exec", "-i", "docker-ragflow-cpu-1", "/ragflow/.venv/bin/python", "-c", SCRIPT], input=mapping.decode(), text=True, capture_output=True)
    if result.returncode:
        # A service traceback could contain config values; do not echo it.
        raise RuntimeError("Local chunk export failed; no model request was made")
    snapshot = json.loads(result.stdout)
    snapshot["metadata"]["document_map_sha256"] = hashlib.sha256(mapping).hexdigest()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(snapshot, stream, ensure_ascii=False)
        stream.write("\n")
    print(json.dumps({"chunks": len(snapshot["chunks"]), "metadata": snapshot["metadata"], "output": str(output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
