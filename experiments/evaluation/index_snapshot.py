"""Record only non-secret index counters to detect writes during comparisons."""

import argparse
import json
from pathlib import Path
import subprocess


SCRIPT = """
import json,sys,yaml,pymysql
from pathlib import Path
from elasticsearch import Elasticsearch
c=yaml.safe_load(Path('/ragflow/conf/service_conf.yaml').read_text())
m=c['mysql']
d=pymysql.connect(host=m['host'],port=int(m['port']),user=m['user'],password=m['password'],database=m['name'])
with d.cursor() as q:
 q.execute('SELECT tenant_id FROM knowledgebase WHERE id=%s',(sys.argv[1],))
 tenant=q.fetchone()[0]
d.close()
s=c['es']
es=Elasticsearch(s['hosts'].split(','),basic_auth=(s['username'],s['password']),verify_certs=True,request_timeout=30)
index='ragflow_'+tenant
stats=es.indices.stats(index=index,level='shards')['indices'][index]
primary={shard:entry['seq_no'] for shard,entries in stats['shards'].items() for entry in entries if entry['routing']['primary']}
count=es.count(index=index,query={'term':{'kb_id':sys.argv[1]}})['count']
print(json.dumps({'index_uuid':stats['uuid'],'tenant_index_documents':stats['primaries']['docs']['count'],'knowledge_base_chunks':count,'primary_sequence_numbers':primary}))
es.close()
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = subprocess.run(["docker", "exec", "docker-ragflow-cpu-1", "/ragflow/.venv/bin/python", "-c", SCRIPT, args.dataset_id], text=True, capture_output=True, check=True)
    snapshot = json.loads(result.stdout)
    with Path(args.output).open("x") as stream:
        json.dump(snapshot, stream, indent=2)
        stream.write("\n")
    print(json.dumps(snapshot))


if __name__ == "__main__":
    main()
