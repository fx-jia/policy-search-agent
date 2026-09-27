"""Check frozen paired-run integrity without changing or rerunning predictions."""
import hashlib
import json
from collections import Counter
from pathlib import Path
import re

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[3]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def readl(path):
    return [json.loads(s) for s in path.read_text().splitlines()] if path.exists() else []


def main():
    manifest = json.loads((OUT / 'manifest.json').read_text())
    for name, value in manifest['protected_files'].items():
        assert sha(ROOT / name) == value, name
    selection = manifest['selection']
    test = {q['question_id'] for q in readl(OUT.parent / 'benchmark-v2-baselines-v2-nonthinking/test.jsonl')}
    assert not set(selection['extension']) & (set(selection['prior150']) | test)
    assert len(test) == 95 and len(selection['extension']) == 55
    summary = {}
    for cohort, arms in manifest['arms'].items():
        for arm in arms:
            folder = OUT / cohort / arm
            assert sha(folder / 'config.json') == manifest['config_sha256'][f'{cohort}/{arm}']
            config = json.loads((folder / 'config.json').read_text())
            assert all(set(q) == {'question_id', 'question'} for q in config['questions'])
            assert config['question_ids'] == selection[cohort]
            rows = readl(folder / 'records.jsonl')
            ids = [r['question_id'] for r in rows]
            assert len(ids) == len(set(ids)) and set(ids) == set(selection[cohort]), (cohort, arm, 'incomplete')
            attempts = readl(folder / 'attempts.jsonl')
            assert len(attempts) == len(ids), (cohort, arm, 'duplicate attempt')
            final_calls = []; judge_calls = []; wrong_citations = []; unsettled = []
            for r in rows:
                calls = r.get('calls', [])
                assert len(calls) <= config['limits']['question_calls'], (arm, r['question_id'], 'calls')
                assert all(c['request']['model'] == config['model'] for c in calls)
                assert all(c['request'].get('enable_thinking') is False for c in calls)
                if arm not in ('rag', 'low'):
                    assert r['embedding_physical']['query_physical_calls'] <= 3
                    assert sum(t['event'] == 'tool' for t in r['trace']) <= 2
                    assert len(r['selected']) <= 8
                    assert len({c['chunk_id'] for c in r['selected']}) == len(r['selected'])
                    assert all(c['chunk_id'] in r['seen_chunks'] for c in r['selected'])
                    final = [c for c in calls if not c['request'].get('response_format') and c.get('transport_status') != 'not_sent']
                    assert len(final) <= 1, (arm, r['question_id'], 'extra answer')
                    final_calls.append(len(final))
                    judge_calls.append(sum(bool(c['request'].get('response_format')) for c in calls))
                for c in calls:
                    if 'usage' not in c:
                        unsettled.append({'question_id': r['question_id'], 'call_id': c['call_id']})
                refs = r.get('reference', {}).get('chunks', [])
                bad = [int(i) for i in re.findall(r'\[ID:(\d+)\]', r.get('answer', '')) if int(i) >= len(refs)]
                if bad:
                    wrong_citations.append({'question_id': r['question_id'], 'ids': bad})
            summary[f'{cohort}/{arm}'] = {
                'records': len(rows), 'attempts': len(attempts), 'records_sha256': sha(folder / 'records.jsonl'),
                'maximum_physical_generation_calls': max(len(r.get('calls', [])) for r in rows),
                'maximum_final_sent_calls': max(final_calls) if final_calls else None,
                'maximum_judge_attempts': max(judge_calls) if judge_calls else None,
                'request_temperatures': dict(Counter(str(c['request'].get('temperature')) for r in rows for c in r.get('calls', []))),
                'invalid_citation_numbers': wrong_citations, 'unsettled_calls': unsettled,
            }
    native = {r['question_id']: r for r in readl(OUT / 'extension/rag/records.jsonl')}
    initial = {}
    for arm in ('v7', 'a', 'ab'):
        compared = []; order_diff = []; set_diff = []
        for row in readl(OUT / 'extension' / arm / 'records.jsonl'):
            event = next((t for t in row.get('trace', []) if t['event'] == 'initial_search'), None)
            base = [c.get('chunk_id', c.get('id')) for c in native[row['question_id']].get('ordinary_context', {}).get('chunks', [])]
            if not event or not base:
                continue
            q = row['question_id']; compared.append(q)
            if event['chunk_ids'] != base:
                order_diff.append(q)
            if set(event['chunk_ids']) != set(base):
                set_diff.append(q)
        initial[arm] = {'compared': len(compared), 'different_order': order_diff, 'different_candidates': set_diff}
    result = {'protected_inputs_unchanged': True, 'heldout95_untouched': True,
              'no_reference_answers_in_runtime_questions': True, 'arms': summary,
              'initial_retrieval_vs_rag': initial,
              'scope': 'Frozen hashes, record completeness, request identity, call/action caps and citation numbering. Content and semantic citation support are assessed separately.'}
    (OUT / 'integrity.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
