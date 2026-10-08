"""Merge portable GPU folds without rewriting their frozen metadata."""
import json
from pathlib import Path
import numpy as np


def merge_months(folder, di, dates):
    dates=np.asarray(dates);di=np.asarray(di);result=np.full(len(di),np.nan,np.float32)
    seen=np.zeros(len(di),bool);months=[]
    for path in sorted(Path(folder).glob('*.json')):
        row=json.loads(path.read_text());fold=row['fold'];bounds=row['boundaries']
        start,end=int(fold['start']),int(fold['end'])
        if fold['month']!=path.stem or dates[start][:6]!=path.stem:raise AssertionError('Monthly origin changed')
        if bounds['first_execution']!=dates[start] or bounds['last_execution']!=dates[end-1]:raise AssertionError('Execution boundaries changed')
        if bounds['last_refit_label']>=bounds['first_execution']:raise AssertionError('Future refit label')
        if bounds['last_inner_train_label']>=bounds['first_inner_validation_signal']:raise AssertionError('Unpurged validation')
        expected=(di>=start-1)&(di<end-1)
        if np.any(seen & expected):raise AssertionError('Overlapping monthly predictions')
        values=np.load(path.with_suffix('.pred.npy'),allow_pickle=False)
        if values.shape!=result.shape or not np.array_equal(np.isfinite(values),expected):raise AssertionError('Prediction coverage changed')
        result[expected]=values[expected];seen|=expected;months.append(path.stem)
    expected_months=sorted({date[:6] for date in dates if '20260101'<=date<='20260923'})
    if months!=expected_months:raise AssertionError('Incomplete monthly GPU cohort')
    return result
