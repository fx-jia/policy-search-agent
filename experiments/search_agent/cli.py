"""Run bounded, audited conversations in one isolated Search Agent worker."""

import argparse
import copy
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time
import uuid

from experiments.evaluation.benchmark_v2_baselines import STOP_WORKER, receive_line, validate_ready_ledger
from experiments.evaluation.native_agent_guard import CallLedger
from experiments.evaluation.native_baseline import DOCKER
from experiments.evaluation.search_agent_eval import MODULES, ROOT, bootstrap_source, verify_frozen_inputs
from experiments.search_agent.core import AgentLimits


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _save(path, value):
    with path.open('w', encoding='utf-8') as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


def _append(path, value):
    with path.open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(value, ensure_ascii=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


def prepare_config(config_path, output, max_tokens, *, root=ROOT):
    """Validate the prior freeze and create a separate conversation configuration."""
    config_path, output, root = Path(config_path), Path(output), Path(root)
    if type(max_tokens) is not int or max_tokens <= 0:
        raise ValueError('max_tokens_must_be_positive_integer')
    if output.exists():
        raise FileExistsError('conversation_output_already_exists:' + str(output))
    source = json.loads(config_path.read_text())
    if not source.get('module_sources') or not source.get('input_hashes'):
        raise ValueError('reviewed_search_agent_config_required')
    manifest_path = config_path.parent / 'manifest.json'
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest.get('config_sha256') != _digest(config_path):
            raise ValueError('frozen_config_changed')
    verify_frozen_inputs(source, root)
    required = ('model', 'chat_id', 'chat_config', 'provider_config', 'provider_hosts',
                'runtime_code_hashes', 'scope_document_ids', 'document_map', 'review',
                'document_title_review', 'index_snapshot', 'snapshot_rows', 'embedding_model')
    config = {key: copy.deepcopy(source[key]) for key in required}
    for key in ('connect_retries', 'runtime_model_override'):
        if key in source:
            config[key] = copy.deepcopy(source[key])
    token = uuid.uuid4().hex
    config.update(reasoning=0, model_thinking=False, native_patch=None,
                  limits={'run_tokens': max_tokens, 'question_tokens': 60000,
                          'question_calls': 4, 'query_embedding_calls': 3},
                  agent_limits=asdict(AgentLimits()), question_timeout_seconds=60,
                  conversation={'max_rounds': 6, 'history_tokens': 3000, 'resolver_seconds': 10},
                  worker_token=token, container_ledger=f'/tmp/search-conversation-{token}/calls.jsonl')
    modules = list(MODULES)
    position = modules.index('experiments.search_agent.runtime')
    modules[position:position] = ['experiments.search_agent.resolution', 'experiments.search_agent.conversation']
    config['module_sources'] = {
        name: (root / (name.replace('.', '/') + '.py')).read_text() for name in modules
    }
    hashes = {name.replace('.', '/') + '.py': hashlib.sha256(text.encode()).hexdigest()
              for name, text in config['module_sources'].items()}
    config['input_hashes'] = hashes
    output.mkdir(parents=True, exist_ok=False)
    _save(output / 'config.json', config)
    _save(output / 'manifest.json', {
        'config_sha256': _digest(output / 'config.json'), 'code_hashes': hashes,
        'source_config': str(config_path.resolve()), 'source_config_sha256': _digest(config_path),
        'model': config['model'], 'max_tokens': max_tokens, 'mode': 'conversation',
    })
    return config


def _stop_worker(config):
    result = subprocess.run(DOCKER + [STOP_WORKER],
                            input=json.dumps({'token': config['worker_token'], 'ledger': config['container_ledger']}),
                            text=True, capture_output=True, timeout=15, check=True)
    return json.loads(result.stdout)


def _observed_reservations(active):
    calls = {}
    for event in active['events'] if active is not None else ():
        if event.get('event') != 'provider_reserved' or not event.get('call_id'):
            continue
        reserve = event.get('reserve_tokens')
        calls[event['call_id']] = {
            **event, 'question_id': event.get('question_id') or active.get('question_id') or active['attempt_id'],
            'reserve_tokens': reserve if type(reserve) is int and reserve > 0 else None,
            'accounting_status': 'unknown', 'accounting_source': 'observed_worker_event',
        }
    return calls


def _save_incomplete_record(output, active, failure, summary, calls):
    _append(output / 'records.jsonl', {
        'attempt_id': active['attempt_id'], 'question_id': active.get('question_id') or active['attempt_id'],
        'session_id': active['session_id'], 'question': active['question'], 'effective_question': '',
        'answer': '', 'reference': {}, 'status': 'host_timeout' if isinstance(failure, TimeoutError) else 'host_error',
        'stop_reason': str(failure), 'usage': {}, 'ledger': summary, 'calls': calls,
        'events': active['events'], 'control_trace_complete': False,
    })


def _write_snapshot(output, config, snapshot, records, active):
    path = output / 'ledger.jsonl'
    with path.open('w', encoding='utf-8') as stream:
        for row in snapshot['ledger_events']:
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())
    limits = config['limits']
    ledger = CallLedger(path, run_limit=limits['run_tokens'], question_limit=60000, call_limit=4)
    summary = ledger.summary()
    start_unknown = active is not None and (
        not active.get('question_id') or active['question_id'] not in ledger.started
    )
    observed = _observed_reservations(active)
    missing = [call for cid, call in observed.items() if cid not in ledger.calls]
    changed = any(call['reserve_tokens'] is not None and call['reserve_tokens'] != ledger.calls[cid]['reserve_tokens']
                  for cid, call in observed.items() if cid in ledger.calls)
    if start_unknown or missing or changed or any(ledger.calls.get(call['call_id']) != call for record in records for call in record['calls']):
        known_reserved = sum(call['reserve_tokens'] for call in missing if call['reserve_tokens'] is not None)
        remote_reserved = sum(call['reserve_tokens'] for call in ledger.calls.values() if 'usage' not in call)
        _save(output / 'ledger_summary.json', {
            'status': 'unreconciled', 'reported': records[-1]['ledger'] if records else None, 'remote': summary,
            'provider_calls': None, 'provider_total_tokens': None, 'committed_tokens': None,
            'pending_calls': None if start_unknown and not observed else list(dict.fromkeys(
                summary['pending_calls'] + [call['call_id'] for call in missing])),
            'reserved_tokens': None if (start_unknown and not observed) or any(
                call['reserve_tokens'] is None for call in missing) else known_reserved + remote_reserved,
            'active_question_id': active.get('question_id') if active is not None else None,
            'active_start_verified': not start_unknown if active is not None else None,
            'observed_reservations': list(observed.values()),
            'error': 'ledger_reconciliation_failed',
        })
        raise ValueError('ledger_reconciliation_failed; observed usage and reservations retained in audit files')
    summary['reserved_tokens'] = sum(call['reserve_tokens'] for call in ledger.calls.values() if 'usage' not in call)
    summary['committed_tokens'] = summary['provider_total_tokens'] + summary['reserved_tokens']
    _save(output / 'ledger_summary.json', summary)
    return ledger


def _validate_record(record, active, records):
    if record.get('session_id') != active['session_id'] or record.get('question') != active['question']:
        raise ValueError('completed_question_identity_mismatch')
    qid = record.get('question_id')
    if not isinstance(qid, str) or not qid or any(row['question_id'] == qid for row in records):
        raise ValueError('duplicate_or_empty_question_id')
    if active.get('question_id') and active['question_id'] != qid:
        raise ValueError('completed_question_identity_mismatch')
    required = ('answer', 'reference', 'status', 'stop_reason', 'usage', 'ledger', 'calls', 'events', 'effective_question')
    if any(key not in record for key in required):
        raise ValueError('incomplete_conversation_record')
    calls = [call for row in records + [record] for call in row['calls']]
    ledger = record['ledger']
    expected = {
        'provider_calls': len(calls),
        'provider_not_sent_calls': sum(call.get('transport_status') == 'not_sent' for call in calls),
        'provider_total_tokens': sum(call.get('usage', {}).get('total_tokens', 0) for call in calls),
        'pending_calls': [call['call_id'] for call in calls if 'usage' not in call],
    }
    if any(ledger.get(key) != value for key, value in expected.items()):
        raise ValueError('record_ledger_does_not_match_calls')
    if any(call['question_id'] != qid for call in record['calls']):
        raise ValueError('call_question_identity_mismatch')


def _blocked(ledger, max_tokens):
    return bool(ledger.get('pending_calls') or ledger.get('fatal_reason')
                or ledger['provider_total_tokens'] >= max_tokens)


def _render(record, stream):
    print('展开问题: ' + record['effective_question'], file=stream)
    print('回答: ' + record['answer'], file=stream)
    print('引用: ' + json.dumps(record['reference'], ensure_ascii=False), file=stream)
    print(f"状态: {record['status']}；停止原因: {record['stop_reason']}", file=stream, flush=True)


def run_interactive(config, output, *, input_stream=None, output_stream=None,
                    process_factory=None, stop_worker=None, turn_seconds=60, startup_seconds=120):
    """Supervise one worker; user input waiting does not consume a turn deadline."""
    if not 0 < turn_seconds <= 60 or not 0 < startup_seconds <= 120:
        raise ValueError('invalid_host_deadline')
    output = Path(output)
    input_stream = sys.stdin if input_stream is None else input_stream
    output_stream = sys.stdout if output_stream is None else output_stream
    process_factory = subprocess.Popen if process_factory is None else process_factory
    stop_worker = _stop_worker if stop_worker is None else stop_worker
    token = config['worker_token']
    command = DOCKER[:3] + ['-e', 'BENCHMARK_RUN_TOKEN=' + token] + DOCKER[3:] + [bootstrap_source()]
    lines = queue.Queue()
    records, active, failure = [], None, None
    session_id = uuid.uuid4().hex
    with (output / 'stderr.log').open('a') as stderr, (output / 'noise.log').open('a') as noise:
        worker = process_factory(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                 stderr=stderr, text=True, bufsize=1)

        def read_stdout():
            try:
                for line in worker.stdout:
                    lines.put(line)
            finally:
                lines.put(None)

        reader = threading.Thread(target=read_stdout, daemon=True)
        reader.start()

        def send(value):
            worker.stdin.write(json.dumps(value, ensure_ascii=False) + '\n')
            worker.stdin.flush()

        def receive(expected, deadline):
            while True:
                line = receive_line(lines, deadline)
                if line is None:
                    raise RuntimeError('worker_exited; inspect stderr.log')
                if not line.startswith('AGENT_RESULT:'):
                    noise.write(line)
                    noise.flush()
                    continue
                row = json.loads(line[len('AGENT_RESULT:'):])
                _append(output / 'events.jsonl', row)
                if active is not None:
                    active['events'].append(row)
                    if row.get('question_id'):
                        if active.get('question_id') and active['question_id'] != row['question_id']:
                            raise ValueError('worker_event_question_identity_mismatch')
                        active['question_id'] = row['question_id']
                if row.get('event') == expected:
                    return row
                if row.get('event') in {'ready', 'question_complete', 'session_cleared'}:
                    raise ValueError('unexpected_worker_protocol_event')

        try:
            send(config)
            ready = receive('ready', time.monotonic() + startup_seconds)
            if ready.get('worker_token') != token:
                raise ValueError('worker_identity_mismatch')
            validate_ready_ledger(ready['ledger'], [])
            if ready['ledger'].get('fatal_reason'):
                raise ValueError('worker_ready_with_fatal_ledger')
            print('会话已就绪。/new 新会话；/clear 清空当前会话；/exit 退出。', file=output_stream, flush=True)
            while True:
                print('问题> ', end='', file=output_stream, flush=True)
                line = input_stream.readline()
                if not line:
                    break
                question = line.strip()
                if not question:
                    continue
                if question == '/exit':
                    break
                if question in {'/clear', '/new'}:
                    send({'command': 'clear', 'session_id': session_id})
                    cleared = receive('session_cleared', time.monotonic() + turn_seconds)
                    if cleared.get('session_id') != session_id:
                        raise ValueError('cleared_session_identity_mismatch')
                    if question == '/new':
                        session_id = uuid.uuid4().hex
                    print('会话已清空。', file=output_stream, flush=True)
                    continue
                if question.startswith('/'):
                    print('可用命令: /new /clear /exit', file=output_stream, flush=True)
                    continue
                active = {'attempt_id': uuid.uuid4().hex, 'session_id': session_id,
                          'question': question, 'started': time.monotonic(), 'events': []}
                _append(output / 'attempts.jsonl', {key: value for key, value in active.items() if key != 'events'})
                send({'session_id': session_id, 'question': question})
                done = receive('question_complete', active['started'] + turn_seconds)['record']
                _validate_record(done, active, records)
                done['attempt_id'] = active['attempt_id']
                _append(output / 'records.jsonl', done)
                records.append(done)
                active = None
                _render(done, output_stream)
                if _blocked(done['ledger'], config['limits']['run_tokens']):
                    reason = done['ledger'].get('fatal_reason') or (
                        'unresolved_provider_usage' if done['ledger'].get('pending_calls') else 'run_token_budget'
                    )
                    failure = ValueError(reason)
                    break
        except (Exception, KeyboardInterrupt) as exc:
            failure = exc
            _append(output / 'host_errors.jsonl', {'error': str(exc), 'type': type(exc).__name__})
        finally:
            if worker.stdin and not worker.stdin.closed:
                try:
                    worker.stdin.close()
                except BrokenPipeError:
                    pass
            try:
                snapshot = stop_worker(config)
                _append(output / 'shutdown_terminations.jsonl', snapshot)
                ledger = _write_snapshot(output, config, snapshot, records, active)
                if active is not None:
                    qid = active.get('question_id') or active['attempt_id']
                    active['question_id'] = qid
                    _save_incomplete_record(output, active, failure, ledger.summary(),
                                            [dict(c) for c in ledger.calls.values() if c['question_id'] == qid])
                    active = None
                if ledger.summary()['pending_calls'] or ledger.fatal_reason:
                    failure = failure or ValueError('conversation_unknown_or_fatal_usage')
            except Exception as exc:
                _append(output / 'host_errors.jsonl', {'error': str(exc), 'type': type(exc).__name__, 'stage': 'shutdown'})
                if active is not None:
                    summary_path = output / 'ledger_summary.json'
                    if summary_path.exists():
                        summary = json.loads(summary_path.read_text())
                    else:
                        summary = {'status': 'unknown', 'provider_calls': None, 'provider_total_tokens': None,
                                   'committed_tokens': None, 'error': str(exc),
                                   'observed_reservations': list(_observed_reservations(active).values())}
                        _save(summary_path, summary)
                    _save_incomplete_record(output, active, failure or exc, summary,
                                            list(_observed_reservations(active).values()))
                raise
            finally:
                if worker.poll() is None:
                    worker.kill()
                worker.wait(timeout=5)
                reader.join(timeout=1)
                worker.stdout.close()
    if failure is not None:
        print('会话已停止: ' + str(failure) + '；审计保存在 ' + str(output), file=output_stream, flush=True)
        return 1
    return 0


def main(argv=None, *, input_stream=None, output_stream=None, process_factory=None, stop_worker=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True, help='Existing reviewed Search Agent config.json')
    parser.add_argument('--output', type=Path, required=True, help='New audit directory; must not exist')
    parser.add_argument('--max-tokens', type=int, required=True, help='Total generation token limit for this conversation run')
    args = parser.parse_args(argv)
    config = prepare_config(args.config, args.output, args.max_tokens)
    return run_interactive(config, args.output, input_stream=input_stream, output_stream=output_stream,
                           process_factory=process_factory, stop_worker=stop_worker)


if __name__ == '__main__':
    raise SystemExit(main())
