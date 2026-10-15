import hashlib
import importlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


WORKER = r'''
import json, os, sys, time, uuid
from pathlib import Path
from experiments.evaluation.native_agent_guard import CallLedger
config = json.loads(sys.stdin.readline())
ledger = CallLedger(os.environ['LOCAL_LEDGER'], run_limit=config['limits']['run_tokens'],
                    question_limit=60000, call_limit=4)
def emit(row):
    print('AGENT_RESULT:' + json.dumps(row, ensure_ascii=False), flush=True)
emit({'event': 'ready', 'worker_token': config['worker_token'], 'ledger': ledger.summary()})
for line in sys.stdin:
    item = json.loads(line)
    with open(os.environ['PROTOCOL'], 'a') as stream:
        stream.write(json.dumps(item, ensure_ascii=False) + '\n')
    if item.get('command') == 'clear':
        emit({'event': 'session_cleared', 'session_id': item['session_id']})
        continue
    qid = 'turn-' + uuid.uuid4().hex
    ledger.start(qid)
    if os.environ.get('WORKER_MODE') == 'timeout_start_no_call':
        emit({'event': 'turn_started', 'question_id': qid, 'session_id': item['session_id']})
        time.sleep(60)
        continue
    call_id = uuid.uuid4().hex
    ledger.reserve(call_id, min(100, config['limits']['run_tokens']), {'stage': 'resolve'})
    mode = os.environ.get('WORKER_MODE', '')
    if item['question'] == 'completed':
        mode = ''
    reserved = {'event': 'provider_reserved', 'question_id': qid, 'call_id': call_id}
    if mode == 'timeout_with_tokens':
        reserved['reserve_tokens'] = 100
    if mode == 'timeout_start_only':
        emit({'event': 'turn_started', 'question_id': qid, 'session_id': item['session_id']})
    elif mode != 'timeout_no_event':
        emit(reserved)
    if mode in {'timeout', 'timeout_with_tokens', 'timeout_start_only', 'timeout_no_event'}:
        time.sleep(60)
        continue
    if mode == 'unknown':
        ledger.abort('provider_usage_unknown')
    else:
        ledger.finish(call_id, {'prompt_tokens': 6, 'completion_tokens': 4, 'total_tokens': 10})
    status = 'guard_stopped' if mode == 'unknown' else 'error' if item['question'] == 'fail' else 'ok'
    row = {'session_id': item['session_id'], 'question_id': qid, 'question': item['question'],
           'effective_question': '完整问题：' + item['question'], 'answer': '已回答：' + item['question'],
           'reference': {'chunks': [{'id': 'e1', 'document_id': 'd', 'title': '政策原文'}]},
           'status': status, 'stop_reason': ledger.fatal_reason or ('invalid_json' if status == 'error' else 'supported'),
           'usage': {'total_tokens': 0 if mode == 'unknown' else 10}, 'ledger': ledger.summary(),
           'calls': [dict(ledger.calls[call_id])], 'events': []}
    print('worker noise', flush=True)
    emit({'event': 'question_complete', 'record': row})
'''


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


class CLITests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.output = self.root / 'run'
        self.config_path = self.root / 'trusted.json'
        self.protocol = self.root / 'protocol.jsonl'
        self.ledger_path = self.root / 'remote-ledger.jsonl'
        self.processes = []
        self.stops = []

    def cli(self):
        try:
            return importlib.import_module('experiments.search_agent.cli')
        except ModuleNotFoundError:
            self.fail('interactive CLI is not implemented')

    def source_config(self):
        from experiments.evaluation.search_agent_eval import MODULES
        names = list(MODULES)
        names[-1:-1] = ['experiments.search_agent.resolution', 'experiments.search_agent.conversation']
        for name in names:
            path = self.root / (name.replace('.', '/') + '.py')
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('# current module\n')
        evidence = self.root / 'review.json'
        evidence.write_text('{"reviewed": true}\n')
        self.source = {
            'model': 'reviewed-model', 'chat_id': 'chat', 'chat_config': {'dataset_ids': ['kb']},
            'provider_config': {'api_base': 'https://provider.invalid'}, 'provider_hosts': ['provider.invalid'],
            'runtime_code_hashes': {'rag/model.py': 'runtime-digest'}, 'scope_document_ids': ['d'],
            'document_map': {'documents': {'d': 'source-d'}}, 'review': {'entries': {'e1': {'safe': '原文'}}},
            'document_title_review': {'entries': {'d': {'safe_title': '政策原文'}}},
            'index_snapshot': {'compiled_rows': 0, 'chunks': {}}, 'snapshot_rows': [],
            'embedding_model': 'reviewed-embedding', 'connect_retries': 0,
            'module_sources': {MODULES[-1]: '# frozen old runtime\n'},
            'input_hashes': {
                MODULES[-1].replace('.', '/') + '.py': hashlib.sha256(b'# frozen old runtime\n').hexdigest(),
                'review.json': hashlib.sha256(evidence.read_bytes()).hexdigest(),
            },
            'limits': {'run_tokens': 450000, 'question_tokens': 999999, 'question_calls': 99},
            'reasoning': 2, 'questions': [{'question': 'old gold'}], 'question_ids': ['old'],
            'replay': {'old': {'expected_request': {}}}, 'sources': {'old': 'code'},
            'stage_note': 'old evaluation', 'historical_native_report': 'old/results.json',
            'input_sha256': {'old': 'digest'},
        }
        self.config_path.write_text(json.dumps(self.source, ensure_ascii=False))
        return self.source

    def prepare(self, max_tokens=60000):
        self.source_config()
        return self.cli().prepare_config(self.config_path, self.output, max_tokens, root=self.root)

    def factory(self, mode=''):
        def spawn(command, **kwargs):
            env = dict(os.environ, LOCAL_LEDGER=str(self.ledger_path), PROTOCOL=str(self.protocol), WORKER_MODE=mode)
            process = subprocess.Popen([sys.executable, '-u', '-c', WORKER], env=env, **kwargs)
            self.processes.append(process)
            return process
        return spawn

    def stop(self, config):
        self.stops.append(config['worker_token'])
        for process in self.processes:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=3)
        return {'terminated_pids': [], 'ledger_events': read_rows(self.ledger_path)}

    def run_cli(self, text, mode='', **kwargs):
        config = self.prepare()
        stream = io.StringIO()
        result = self.cli().run_interactive(config, self.output, input_stream=io.StringIO(text),
                                            output_stream=stream, process_factory=self.factory(mode),
                                            stop_worker=self.stop, **kwargs)
        return result, stream.getvalue()

    def test_freezes_current_modules_and_removes_old_evaluation_budget(self):
        config = self.prepare()
        self.assertEqual(config['limits'], {'run_tokens': 60000, 'question_tokens': 60000,
                                           'question_calls': 4, 'query_embedding_calls': 3})
        self.assertEqual(config['reasoning'], 0)
        self.assertEqual(config['conversation'], {'max_rounds': 6, 'history_tokens': 3000, 'resolver_seconds': 10})
        for key in ('replay', 'questions', 'question_ids', 'sources', 'input_sha256', 'stage_note', 'historical_native_report'):
            self.assertNotIn(key, config)
        self.assertEqual(config['review'], self.source['review'])
        names = list(config['module_sources'])
        self.assertLess(names.index('experiments.search_agent.resolution'), names.index('experiments.search_agent.conversation'))
        self.assertLess(names.index('experiments.search_agent.conversation'), names.index('experiments.search_agent.runtime'))
        self.assertEqual(config['module_sources']['experiments.search_agent.runtime'], '# current module\n')
        from experiments.evaluation.search_agent_eval import verify_frozen_inputs
        verify_frozen_inputs(config, self.root)
        manifest = json.loads((self.output / 'manifest.json').read_text())
        self.assertEqual(manifest['config_sha256'], hashlib.sha256((self.output / 'config.json').read_bytes()).hexdigest())
        self.assertEqual(json.loads(self.config_path.read_text()), self.source)

    def test_refuses_changed_source_or_existing_output_before_writing(self):
        self.source_config()
        (self.root / 'review.json').write_text('changed')
        with self.assertRaisesRegex(ValueError, 'frozen_input_changed'):
            self.cli().prepare_config(self.config_path, self.output, 60000, root=self.root)
        self.assertFalse(self.output.exists())
        self.source_config()
        self.output.mkdir()
        marker = self.output / 'keep.txt'
        marker.write_text('keep')
        with self.assertRaises(FileExistsError):
            self.cli().prepare_config(self.config_path, self.output, 60000, root=self.root)
        self.assertEqual(marker.read_text(), 'keep')

    def test_rejects_non_positive_or_unbounded_budget(self):
        self.source_config()
        for limit in (0, -1, True, 1.5):
            with self.subTest(limit=limit), self.assertRaises(ValueError):
                self.cli().prepare_config(self.config_path, self.output, limit, root=self.root)

    def test_continuous_answers_commands_and_durable_audit(self):
        result, text = self.run_cli('first\nsecond\n/clear\nthird\n/new\nfourth\n/exit\nignored\n')
        self.assertEqual(result, 0)
        commands = read_rows(self.protocol)
        self.assertEqual([x.get('question', x.get('command')) for x in commands],
                         ['first', 'second', 'clear', 'third', 'clear', 'fourth'])
        self.assertEqual(commands[0]['session_id'], commands[3]['session_id'])
        self.assertNotEqual(commands[0]['session_id'], commands[5]['session_id'])
        self.assertEqual(set(commands[0]), {'session_id', 'question'})
        records = read_rows(self.output / 'records.jsonl')
        self.assertEqual(len(records), 4)
        self.assertEqual(len({row['question_id'] for row in records}), 4)
        self.assertEqual(len(read_rows(self.output / 'attempts.jsonl')), 4)
        self.assertIn('完整问题：second', text)
        self.assertIn('已回答：fourth', text)
        self.assertIn('政策原文', text)
        self.assertEqual(json.loads((self.output / 'ledger_summary.json').read_text())['provider_total_tokens'], 40)
        self.assertEqual(len(self.stops), 1)
        self.assertTrue(all(p.poll() is not None for p in self.processes))

    def test_settled_technical_failure_allows_next_turn(self):
        result, _ = self.run_cli('fail\nafter\n')
        self.assertEqual(result, 0)
        self.assertEqual([r['status'] for r in read_rows(self.output / 'records.jsonl')], ['error', 'ok'])
        self.assertEqual(json.loads((self.output / 'ledger_summary.json').read_text())['provider_total_tokens'], 20)

    def test_unknown_usage_stops_and_keeps_reservation(self):
        result, _ = self.run_cli('first\nnever-send\n', 'unknown')
        self.assertEqual(result, 1)
        self.assertEqual(len(read_rows(self.protocol)), 1)
        ledger = json.loads((self.output / 'ledger_summary.json').read_text())
        self.assertEqual(len(ledger['pending_calls']), 1)
        self.assertEqual(ledger['reserved_tokens'], 100)
        self.assertEqual(read_rows(self.output / 'records.jsonl')[0]['calls'][0]['reserve_tokens'], 100)
        self.assertEqual(len(self.processes), 1)

    def test_timeout_stops_worker_and_keeps_attempt_and_unknown_usage(self):
        result, _ = self.run_cli('first\nnever-send\n', 'timeout', turn_seconds=0.1)
        self.assertEqual(result, 1)
        self.assertEqual(len(read_rows(self.output / 'attempts.jsonl')), 1)
        record = read_rows(self.output / 'records.jsonl')[0]
        self.assertEqual(record['status'], 'host_timeout')
        self.assertEqual(record['question'], 'first')
        self.assertEqual(record['calls'][0]['reserve_tokens'], 100)
        self.assertEqual(len(record['ledger']['pending_calls']), 1)
        self.assertEqual(len(self.stops), 1)
        self.assertTrue(all(p.poll() is not None for p in self.processes))

    def test_shutdown_failure_is_saved_and_propagated(self):
        config = self.prepare()
        def fail_stop(config):
            raise RuntimeError('cannot verify isolated worker termination')
        with self.assertRaisesRegex(RuntimeError, 'cannot verify'):
            self.cli().run_interactive(config, self.output, input_stream=io.StringIO('/exit\n'),
                                       output_stream=io.StringIO(), process_factory=self.factory(), stop_worker=fail_stop)
        self.assertIn('cannot verify', (self.output / 'host_errors.jsonl').read_text())
        self.assertTrue(all(p.poll() is not None for p in self.processes))

    def test_shutdown_missing_ledger_does_not_reset_reported_usage_to_zero(self):
        config = self.prepare()
        def empty_ledger(config):
            self.stop(config)
            return {'terminated_pids': [], 'ledger_events': []}
        with self.assertRaisesRegex(ValueError, 'ledger_reconciliation_failed'):
            self.cli().run_interactive(config, self.output, input_stream=io.StringIO('first\n/exit\n'),
                                       output_stream=io.StringIO(), process_factory=self.factory(), stop_worker=empty_ledger)
        summary = json.loads((self.output / 'ledger_summary.json').read_text())
        self.assertEqual(summary['status'], 'unreconciled')
        self.assertEqual(summary['reported']['provider_total_tokens'], 10)
        self.assertIn('provider_total_tokens', summary)
        self.assertIsNone(summary['provider_total_tokens'])
        self.assertIsNone(summary['committed_tokens'])
        self.assertEqual(read_rows(self.output / 'records.jsonl')[0]['calls'][0]['usage']['total_tokens'], 10)

    def check_active_timeout_missing_snapshot(self, mode, reserve_tokens):
        config = self.prepare()
        def empty_ledger(config):
            self.stop(config)
            return {'terminated_pids': [], 'ledger_events': []}
        with self.assertRaisesRegex(ValueError, 'ledger_reconciliation_failed'):
            self.cli().run_interactive(config, self.output, input_stream=io.StringIO('first\nnever-send\n'),
                                       output_stream=io.StringIO(), process_factory=self.factory(mode),
                                       stop_worker=empty_ledger, turn_seconds=0.1)
        summary = json.loads((self.output / 'ledger_summary.json').read_text())
        self.assertEqual(summary['status'], 'unreconciled')
        self.assertIsNone(summary['provider_total_tokens'])
        self.assertIsNone(summary['committed_tokens'])
        self.assertEqual(summary['reserved_tokens'], reserve_tokens)
        observed = [event for event in read_rows(self.output / 'events.jsonl') if event['event'] == 'provider_reserved']
        self.assertEqual(summary['pending_calls'], [observed[0]['call_id']])
        records = read_rows(self.output / 'records.jsonl')
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]['status'], 'host_timeout')
        self.assertEqual(records[0]['ledger']['status'], 'unreconciled')
        self.assertEqual(records[0]['calls'][0]['call_id'], observed[0]['call_id'])
        self.assertEqual(records[0]['calls'][0]['reserve_tokens'], reserve_tokens)
        self.assertEqual(records[0]['calls'][0]['accounting_status'], 'unknown')
        self.assertNotIn('usage', records[0]['calls'][0])
        self.assertEqual(len(read_rows(self.output / 'attempts.jsonl')), 1)
        self.assertEqual(len(read_rows(self.protocol)), 1)
        self.assertTrue(all(process.poll() is not None for process in self.processes))

    def test_active_timeout_missing_snapshot_preserves_known_reservation(self):
        self.check_active_timeout_missing_snapshot('timeout_with_tokens', 100)

    def test_active_timeout_missing_snapshot_marks_reservation_amount_unknown(self):
        self.check_active_timeout_missing_snapshot('timeout', None)

    def check_active_timeout_without_observed_reservation(self, mode):
        config = self.prepare()
        def empty_ledger(config):
            self.stop(config)
            return {'terminated_pids': [], 'ledger_events': []}
        with self.assertRaisesRegex(ValueError, 'ledger_reconciliation_failed'):
            self.cli().run_interactive(config, self.output, input_stream=io.StringIO('first\nnever-send\n'),
                                       output_stream=io.StringIO(), process_factory=self.factory(mode),
                                       stop_worker=empty_ledger, turn_seconds=0.1)
        summary = json.loads((self.output / 'ledger_summary.json').read_text())
        self.assertEqual(summary['status'], 'unreconciled')
        for field in ('provider_calls', 'provider_total_tokens', 'committed_tokens', 'reserved_tokens'):
            self.assertIsNone(summary[field])
        self.assertFalse(any(row['event'] == 'provider_reserved' for row in read_rows(self.output / 'events.jsonl')))
        record = read_rows(self.output / 'records.jsonl')[0]
        self.assertEqual(record['status'], 'host_timeout')
        self.assertEqual(record['ledger']['status'], 'unreconciled')
        self.assertEqual(len(read_rows(self.output / 'attempts.jsonl')), 1)
        self.assertTrue(all(process.poll() is not None for process in self.processes))

    def test_active_timeout_with_start_event_missing_start_snapshot_is_unknown(self):
        self.check_active_timeout_without_observed_reservation('timeout_start_only')

    def test_active_timeout_without_any_event_is_unknown(self):
        self.check_active_timeout_without_observed_reservation('timeout_no_event')

    def test_unknown_active_question_never_uses_previous_ledger_question_id(self):
        config = self.prepare()
        def previous_ledger(config):
            snapshot = self.stop(config)
            completed = read_rows(self.output / 'records.jsonl')[0]
            call_id = completed['calls'][0]['call_id']
            snapshot['ledger_events'] = [row for row in snapshot['ledger_events']
                                         if row.get('question_id') == completed['question_id'] or row.get('call_id') == call_id]
            return snapshot
        with self.assertRaisesRegex(ValueError, 'ledger_reconciliation_failed'):
            self.cli().run_interactive(config, self.output, input_stream=io.StringIO('completed\nactive\n'),
                                       output_stream=io.StringIO(), process_factory=self.factory('timeout_no_event'),
                                       stop_worker=previous_ledger, turn_seconds=0.1)
        records = read_rows(self.output / 'records.jsonl')
        self.assertEqual(len(records), 2)
        self.assertNotEqual(records[1]['question_id'], records[0]['question_id'])
        self.assertEqual(records[1]['question_id'], records[1]['attempt_id'])
        self.assertEqual(records[1]['ledger']['status'], 'unreconciled')

    def test_verified_active_start_without_calls_may_report_zero_usage(self):
        result, _ = self.run_cli('first\nnever-send\n', 'timeout_start_no_call', turn_seconds=0.1)
        self.assertEqual(result, 1)
        summary = json.loads((self.output / 'ledger_summary.json').read_text())
        self.assertEqual(summary['provider_calls'], 0)
        self.assertEqual(summary['provider_total_tokens'], 0)
        self.assertEqual(summary['committed_tokens'], 0)
        record = read_rows(self.output / 'records.jsonl')[0]
        self.assertTrue(record['question_id'].startswith('turn-'))
        self.assertEqual(record['calls'], [])

    def test_run_budget_stops_before_sending_another_turn(self):
        config = self.prepare(10)
        text = io.StringIO()
        result = self.cli().run_interactive(config, self.output, input_stream=io.StringIO('first\nnever-send\n'),
                                            output_stream=text, process_factory=self.factory(), stop_worker=self.stop)
        self.assertEqual(result, 1)
        self.assertEqual(len(read_rows(self.protocol)), 1)
        self.assertIn('run_token_budget', text.getvalue())

    def test_user_input_wait_does_not_consume_turn_deadline(self):
        import time
        class SlowInput(io.StringIO):
            def readline(self):
                time.sleep(0.15)
                return super().readline()
        config = self.prepare()
        result = self.cli().run_interactive(config, self.output, input_stream=SlowInput('first\n/exit\n'),
                                            output_stream=io.StringIO(), process_factory=self.factory(),
                                            stop_worker=self.stop, turn_seconds=0.1)
        self.assertEqual(result, 0)
        self.assertEqual(read_rows(self.output / 'records.jsonl')[0]['status'], 'ok')

    def test_main_uses_real_arguments_and_writes_reviewable_output(self):
        self.source_config()
        self.source['input_hashes'].pop('review.json')
        self.config_path.write_text(json.dumps(self.source))
        text = io.StringIO()
        result = self.cli().main(['--config', str(self.config_path), '--output', str(self.output), '--max-tokens', '60000'],
                                 input_stream=io.StringIO('first\n/exit\n'), output_stream=text,
                                 process_factory=self.factory(), stop_worker=self.stop)
        self.assertEqual(result, 0)
        self.assertEqual(read_rows(self.output / 'records.jsonl')[0]['question'], 'first')
        self.assertIn('已回答：first', text.getvalue())

    def test_help_does_not_require_config_or_services(self):
        self.cli()
        result = subprocess.run([sys.executable, '-m', 'experiments.search_agent.cli', '--help'],
                                text=True, capture_output=True, timeout=5)
        self.assertEqual(result.returncode, 0)
        self.assertIn('--max-tokens', result.stdout)
        self.assertIn('--config', result.stdout)


if __name__ == '__main__':
    unittest.main()
