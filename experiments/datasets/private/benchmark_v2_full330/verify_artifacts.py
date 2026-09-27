"""Independent local release checks; character alignment is not semantic approval."""
import collections
import hashlib
import json
import re
import shutil
import subprocess
import unicodedata
from html.parser import HTMLParser
from pathlib import Path

out=Path(__file__).resolve().parent
root=out.parents[3]
def readj(name):return json.loads((out/name).read_text())
def readl(name):return [json.loads(s) for s in (out/name).read_text().splitlines()]
def sha(s):return hashlib.sha256(s if isinstance(s,bytes) else s.encode()).hexdigest()
def norm(s):return ''.join(c for c in unicodedata.normalize('NFKC',s) if c.isalnum())
checks={}
for name,digest in readj('input_lock.json').items():assert sha((root/name).read_bytes())==digest,name
checks['protected_inputs_unchanged']=True
core=readl('benchmark_v2_core300.jsonl'); new=readl('benchmark_v2_new240.jsonl'); bound=readl('benchmark_v2_boundary30.jsonl')
assert len(core)==300 and len(new)==240 and len(bound)==30
assert core[60:]==new
assert core[:60]==[json.loads(s) for s in (out.parent/'benchmark_v2_pilot60/benchmark_v2_pilot60.jsonl').read_text().splitlines()]
assert [q['question_id'] for q in core]==[f'mb2-{n:03d}' for n in range(1,301)]
assert [q['question_id'] for q in bound]==[f'mb2-b{n:03d}' for n in range(1,31)]
checks['counts_ids_pilot_preservation']=True
allq=core+bound
assert len({norm(q['question']) for q in allq})==330
old=[json.loads(s) for s in (root/'experiments/datasets/private/expansion_v1/benchmark_150_reviewed_v3.jsonl').read_text().splitlines()]
assert not ({norm(q['question']) for q in allq}&{norm(q['question']) for q in old})
checks['no_exact_normalized_question_duplicates']=True
records={str(r['系统ID']):r for r in (json.loads(s) for s in (root/'data/documents.jiangsu_and_2city.jsonl').read_text().splitlines())}
frozen=json.loads((root/'experiments/datasets/private/ragflow_corpus1000_v1/frozen_corpus.json').read_text())
fmap={str(d['source_document_id']):d for d in frozen['documents']}
config=json.loads((root/'experiments/results/private/native-agent-high-compose-fix-v1-full-review/config.json').read_text())
entries=config['review']['entries']; ev_count=0
for q in allq:
    evs=q['evidence']; assert len({e['evidence_id'] for e in evs})==len(evs)
    assert q['review_status']=='pending_human_review' and q['eligible_for_unseen_test'] is False
    assert q['reference_answer'] and q['difficulty_reason']
    assert q['difficulty'] in ['Easy','Medium','Hard']
    for e in evs:
        ev_count+=1; sid=e['source_document_id']; assert sid in fmap
        body=records[sid]['政策正文']
        assert body[e['start']:e['end']]==e['text']
        assert sha(body)==e['body_sha256'] and sha(e['text'])==e['quote_sha256']
        assert e['ragflow_document_id']==fmap[sid]['document_id']
        needle=norm(e['text']); covered=set()
        for link in e['indexed_safe_chunk_links']:
            entry=entries[link['chunk_id']]
            assert str(entry['source_document_id'])==sid
            assert sha(entry['sanitized_content'])==link['safe_sha256']==entry['safe_sha256']
            assert link['raw_sha256']==entry['raw_sha256']
            hay=norm(entry['sanitized_content'])
            for a,b in link['normalized_quote_spans']:
                assert 0<=a<b<=len(needle)
                assert needle[a:b] in hay,(q['question_id'],e['evidence_id'],a,b)
                covered.update(range(a,b))
        assert covered==set(range(len(needle))),q['question_id']
        assert e['safe_text_alignment']['coverage']==1
    assert sorted({e['source_document_id'] for e in evs})==q['known_relevant_document_ids']
    if q in core:
        assert q['answerable'] is True
        for p in q['required_points']:
            assert p['text'] and p['support_sets']
            assert all(bundle and set(bundle)<={e['evidence_id'] for e in evs} for bundle in p['support_sets'])
    else:
        assert q['answerable'] is None
        assert q['missing_conditions'] and q['prohibited_inferences']
        assert set(q['parent_question_ids'])<={c['question_id'] for c in core}
checks['original_quotes_offsets_hashes_frozen_sources_safe_chunk_visibility']=True
checks['support_ids_boundary_links_status']=True
manifest=readj('manifest.json')
assert manifest['difficulty_counts']==dict(collections.Counter(q['difficulty'] for q in core))
assert manifest['selected_source_documents']==len({e['source_document_id'] for q in core for e in q['evidence']})
assert manifest['core_evidence_count']==sum(len(q['evidence']) for q in core)
assert manifest['core_question_families']==len({q['question_family_id'] for q in core})
assert all(manifest[k]==0 for k in ['embedding_api_calls','generation_api_calls','retrieval_api_calls','system_evaluation_calls','unseen_test_count','human_approved_in_this_release'])
assert len(readj('review_template.json'))==330
assert {q['question_id'] for q in readj('review_template.json')}=={q['question_id'] for q in allq}
checks['manifest_template_consistency']=True
for first in range(0,300,60):
    md=(out/f'report_{first+1:03d}_{first+60:03d}.md').read_text()
    assert re.findall(r'^## (mb2-\d{3}) ·',md,re.M)==[q['question_id'] for q in core[first:first+60]]
    for q in core[first:first+60]:assert q['question'] in md and q['reference_answer'] in md
md=(out/'boundary_review_report.md').read_text()
assert re.findall(r'^## (mb2-b\d{3}) ·',md,re.M)==[q['question_id'] for q in bound]
class CheckHTML(HTMLParser):
    def __init__(self):super().__init__();self.articles=[];self.active=[];self.in_script=False;self.script=''
    def handle_starttag(self,tag,attrs):
        if tag=='article':self.articles.append(dict(attrs))
        if tag=='script':self.in_script=True
    def handle_endtag(self,tag):
        if tag=='script':self.in_script=False
    def handle_data(self,s):
        if self.in_script:self.script+=s
h=CheckHTML();page=(out/'benchmark_review_report.html').read_text();h.feed(page)
assert len(h.articles)==330
assert collections.Counter(a['data-kind'] for a in h.articles)=={'主集':300,'边界':30}
for q in allq:assert f"<h2>{q['question_id']} ·" in page
node=shutil.which('node'); assert node,'Node required for stated JS syntax check'
r=subprocess.run([node,'--check'],input=h.script,text=True,capture_output=True)
assert r.returncode==0,r.stderr
checks['markdown_coverage_html_card_counts_javascript_syntax']=True
checks['browser_visual_interaction_test']='not_run; browser policy restricted preview; no workaround attempted'
result={'passed':True,'core_count':300,'new_core_count':240,'boundary_count':30,'evidence_occurrences_checked_including_boundary':ev_count,'checks':checks,
    'limitations':['No automated semantic proof; human reviews remain pending.','No exhaustive alternative-source pooling or semantic deduplication.','No held-out split or system outcome measured.','Static HTML checks only; not browser visual or interaction testing.']}
(out/'validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(result,ensure_ascii=False,indent=2))
