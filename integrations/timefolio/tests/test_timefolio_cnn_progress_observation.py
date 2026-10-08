import json

from quant.timefolio_cnn_remote import observe_progress


def test_empty_or_partial_ssh_observation_keeps_last_good_progress(tmp_path):
    path=tmp_path/'progress.json';before='previous snapshot';path.write_text(before)
    for raw in ['', '{', 'null', '[]', '{"completed":[]}',
                '{"completed":[],"active":["a"],"pending":1,"at":NaN}']:
        assert observe_progress(raw, ['a','b'], path) is None
        assert path.read_text()==before


def test_rejects_unknown_duplicate_and_incomplete_job_partitions(tmp_path):
    path=tmp_path/'progress.json'
    for completed,active,pending in [(['x'],['b'],0),(['a'],['a'],0),(['a'],[],0),([],['a'],True)]:
        raw=json.dumps(dict(completed=[dict(name=n,exit_code=0,seconds=1.) for n in completed],active=active,pending=pending))
        assert observe_progress(raw,['a','b'],path) is None
        assert not path.exists()


def test_later_valid_observation_after_corruption_advances_the_same_queue(tmp_path):
    path=tmp_path/'progress.json'
    first=dict(completed=[],active=['a'],pending=1)
    assert observe_progress(json.dumps(first),['a','b'],path)==first
    before=path.read_bytes();assert observe_progress('', ['a','b'],path) is None
    assert path.read_bytes()==before
    later=dict(completed=[dict(name='a',exit_code=0,seconds=2.)],active=['b'],pending=0)
    assert observe_progress(json.dumps(later),['a','b'],path)==later
    assert json.loads(path.read_text())==later and not path.with_suffix('.tmp').exists()


def test_completed_failed_worker_is_retained_for_recovery_and_validation(tmp_path):
    path=tmp_path/'progress.json'
    result=dict(completed=[dict(name='a',exit_code=1,seconds=3.)],active=[],pending=0)
    assert observe_progress(json.dumps(result),['a'],path)==result
