import json

import numpy as np
import pytest

from quant.timefolio_heatmap_seed_evaluation import merge_months


def write_forecasts(folder):
    dates=['20251231','20260102','20260105','20260202','20260203']
    di=np.repeat(np.arange(len(dates)),2)
    for month,first,start,end,last in [('202601','20260102',0,2,'20251230'),
                                      ('202602','20260202',2,4,'20260130')]:
        p=folder/(month+'.json');p.write_text(json.dumps({'last_refit_label':last,'first_execution':first}))
        pred=np.full(len(di),np.nan);mask=(di>=start)&(di<end);pred[mask]=di[mask]
        np.save(p.with_suffix('.pred.npy'),pred)
    return di,dates


def test_month_end_forecasts_belong_to_next_origin(tmp_path):
    di,dates=write_forecasts(tmp_path);out=merge_months(tmp_path,di,dates)
    np.testing.assert_array_equal(out[di<4],di[di<4]);assert np.isnan(out[di==4]).all()


def test_replication_rejects_future_labels(tmp_path):
    di,dates=write_forecasts(tmp_path)
    (tmp_path/'202602.json').write_text(json.dumps({'last_refit_label':'20260202','first_execution':'20260202'}))
    with pytest.raises(AssertionError,match='Future label'):merge_months(tmp_path,di,dates)


def test_replication_rejects_missing_prediction_dates(tmp_path):
    di,dates=write_forecasts(tmp_path);path=tmp_path/'202601.pred.npy';a=np.load(path);a[0]=np.nan;np.save(path,a)
    with pytest.raises(AssertionError,match='coverage'):merge_months(tmp_path,di,dates)
