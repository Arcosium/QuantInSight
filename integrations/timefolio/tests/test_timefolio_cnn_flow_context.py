import numpy as np
import pytest

from quant.timefolio_cnn_flow_context import compose_images, flow_rows, read_completed_reference


def fixture():
    values = np.tile([1.,2.,3.],(50,1))
    return values, -values, np.ones(values.shape,bool)


def test_cumulative_ranks_end_before_the_signal_session():
    institution, foreign, members = fixture()
    context, available = flow_rows(institution,foreign,members)
    np.testing.assert_array_equal(context[20,:,:4], [[-1,1,-1,1],[0,0,0,0],[1,-1,1,-1]])
    assert not available[:5,:,0].any() and available[5:,:,0].all()
    assert not available[:20,:,1].any() and available[20:,:,1].all()
    institution[20] *= -1000
    changed,_ = flow_rows(institution,foreign,members)
    np.testing.assert_array_equal(changed[:21],context[:21])
    assert not np.array_equal(changed[21],context[21])


def test_future_values_and_future_membership_cannot_rewrite_history():
    institution, foreign, members = fixture()
    before,_ = flow_rows(institution,foreign,members)
    institution[35:] *= -100
    foreign[35:] = np.nan
    members[36:] = False
    after,_ = flow_rows(institution,foreign,members)
    np.testing.assert_array_equal(before[:36],after[:36])


def test_missing_is_distinct_from_observed_zero_and_not_forward_filled():
    institution, foreign, members = fixture()
    institution[:]=foreign[:]=0
    foreign[10,0]=np.nan
    members[:,2]=False
    context,available=flow_rows(institution,foreign,members)
    assert context[15,0,4]==-1 and context[16,0,4]==1
    assert context[20,0,5]==-1 and context[31,0,5]==1
    assert (context[20,1,:4]==0).all() and (context[20,1,4:]==1).all()
    assert not context[:,2].any() and not available[:,2].any()


def test_lag_two_delays_information_one_extra_session():
    institution,foreign,members=fixture()
    institution[22,0]=100
    first,a=flow_rows(institution,foreign,members,lag=1)
    second,b=flow_rows(institution,foreign,members,lag=2)
    np.testing.assert_array_equal(first[:-1],second[1:])
    np.testing.assert_array_equal(a[:-1],b[1:])
    for lag in [0,-1,True,1.5]:
        with pytest.raises(ValueError):flow_rows(institution,foreign,members,lag=lag)


def test_equal_height_controls_preserve_chart_and_separate_missingness():
    context,_=flow_rows(*fixture())
    images=np.arange(2*16*8,dtype=np.float32).reshape(2,1,16,8)/300
    signal=np.array([25,26]);security=np.array([0,1])
    result={mode:compose_images(images,signal,security,context,mode=mode)
            for mode in ['padded','missingness','flow']}
    for value in result.values():
        assert value.shape==(2,1,22,8)
        np.testing.assert_array_equal(value[:,:,:16],images)
    assert not result['padded'][:,:,16:].any()
    assert not result['missingness'][:,:,16:20].any()
    np.testing.assert_array_equal(result['flow'][:,:,20:],result['missingness'][:,:,20:])
    assert result['flow'][:,:,16:20].any()
    context[27:]*=-1
    np.testing.assert_array_equal(compose_images(images,signal,security,context,mode='flow'),result['flow'])


def test_invalid_identity_history_and_infinite_flow_are_rejected():
    institution,foreign,members=fixture()
    institution[0,0]=np.inf
    with pytest.raises(ValueError):flow_rows(institution,foreign,members)
    context=np.zeros((50,3,6),np.float32);images=np.zeros((1,1,16,8),np.float32)
    for day,key,mode in [(5,0,'flow'),(50,0,'flow'),(20,3,'flow'),(20,0,'unknown')]:
        with pytest.raises(ValueError):
            compose_images(images,np.array([day]),np.array([key]),context,mode=mode)


def test_active_reference_cannot_be_consumed_as_complete(tmp_path):
    (tmp_path/'merged').mkdir()
    (tmp_path/'merged'/'005930.json').write_text('{}')
    with pytest.raises(FileNotFoundError):
        read_completed_reference(tmp_path,np.array(['20260923']),np.array(['005930']))
