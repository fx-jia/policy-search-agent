"""Protect evaluation isolation, grouping and stop semantics, without services."""
import importlib
import pytest

@pytest.fixture
def api():
    import importlib.util
    assert importlib.util.find_spec('experiments.evaluation.benchmark_v2_baselines'), 'v2 baseline adapter missing'
    return importlib.import_module('experiments.evaluation.benchmark_v2_baselines')

def test_pilot_source_and_policy_links_cannot_leak_to_test(api):
    rows=[{'question_id':'mb2-001','source_group_seed':['s1'],'question_family_id':'f1'},
          {'question_id':'mb2-061','source_group_seed':['s2'],'question_family_id':'f2'},
          {'question_id':'mb2-062','source_group_seed':['s3'],'question_family_id':'f3'}]
    grouped=api.policy_groups(rows,[['s1','s2']])
    assert sorted(sorted(g) for g in grouped)==[['mb2-001','mb2-061'],['mb2-062']]
    split=api.split_groups(grouped,rows,test_target=1)
    assert split['mb2-001']==split['mb2-061']=='dev'
    assert split['mb2-062']=='test'

def test_runtime_payload_never_includes_gold_or_test(api):
    rows=[{'question_id':'q','question':'问题','reference_answer':'秘密答案','difficulty':'Hard','split':'dev'},
          {'question_id':'t','question':'测试题','reference_answer':'秘密','split':'test'}]
    assert api.runtime_questions(rows,['q'])==[{'question_id':'q','question':'问题'}]
    with pytest.raises(ValueError,match='test'):api.runtime_questions(rows,['t'])
    with pytest.raises(ValueError,match='missing'):api.runtime_questions(rows,['absent'])

def test_partial_attempts_block_resume_and_known_budget_failures_are_not_retried(api):
    with pytest.raises(ValueError,match='incomplete'):api.remaining_ids(['a','b'],['a'],[])
    assert api.remaining_ids(['a','b'],['a'],[{'question_id':'a','status':'budget_censored','ledger':{'pending_calls':[]}}])==['b']
    with pytest.raises(ValueError,match='unknown'):api.remaining_ids(['a','b'],['a'],[{'question_id':'a','ledger':{'pending_calls':['call1']}}])

def test_deleted_remote_ledger_cannot_reset_local_historical_budget(api):
    assert hasattr(api,'validate_ready_ledger'), 'ready ledger reconciliation missing'
    previous={'ledger':{'provider_calls':4,'provider_total_tokens':90,'provider_not_sent_calls':0,'pending_calls':[],'fatal_reason':None}}
    empty={'provider_calls':0,'provider_total_tokens':0,'provider_not_sent_calls':0,'pending_calls':[],'fatal_reason':None}
    with pytest.raises(ValueError,match='ledger'):api.validate_ready_ledger(empty,[previous])
    api.validate_ready_ledger(previous['ledger'],[previous])

def test_stdout_silence_has_an_independent_deadline(api):
    import queue,time
    assert hasattr(api,'receive_line'), 'host deadline missing'
    started=time.monotonic()
    with pytest.raises(TimeoutError):api.receive_line(queue.Queue(),started+0.03)
    assert time.monotonic()-started<0.3

def test_completed_record_must_match_current_question(api):
    assert hasattr(api,'validate_result'), 'response identity check missing'
    with pytest.raises(ValueError,match='question'):api.validate_result('a',{'question_id':'b','ledger':{'pending_calls':[]}})
def test_high_is_disabled_after_user_scope_change(monkeypatch):
    import pytest
    from pathlib import Path
    from experiments.evaluation.benchmark_v2_baselines import run
    def forbidden(*args, **kwargs):
        raise AssertionError('disabled arm must not read configs or launch a worker')
    monkeypatch.setattr(Path, 'read_text', forbidden)
    with pytest.raises(ValueError, match='high_disabled_by_user'):
        run('high')
