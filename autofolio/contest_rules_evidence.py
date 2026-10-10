"""Read-only, pinned public RFM rule snapshots already held by this workspace."""
import hashlib
import json
from functools import lru_cache
from pathlib import Path

RULE_ROOT = Path.home()/'vault/ArcTrade/timefolio_cnn_4y/20261002_v1/rules'
HASHES = {
 '743.json':'35a75e9473a7fe4302c42b4946329ec5afedd9c5c36754ed6df73766ad55561f',
 '104.json':'9fe7bc213abec8ba857f9e2bd57ade816ae8623e8fa0ab9cf9cfb896e86be82d',
 '305.json':'2861e3d316f4b8378be04e7b82051779e5d2615182f0920908d694ae3770fc05',
 '708.json':'5a2f0cfd759278cf8922cf7f28a40f7d0ac540d2001dc456a3d527013c71313b',
 '8.json':'c38c4be7dab078909876ee2458d41998f35f336b68661ebf6a95e7b7a9d188a1',
}


def profile():
    sources=[]
    for name,digest in HASHES.items():
        try:
            raw=(RULE_ROOT/name).read_bytes()
            if hashlib.sha256(raw).hexdigest()!=digest:return None
            doc=json.loads(raw)
            sources.append(dict(id=int(name[:-5]),sha256=digest,url=doc.get('url'),retrieved_at=doc.get('retrieved_at')))
        except (OSError,ValueError):return None
    return dict(contest=13,start='20261001',end='20261130',initial_cash=1e9,
                weekly_min=.05,max_violations=3,position_limits={'default':.15,'005930':.4,'000660':.3},
                sources=sources,scope='13회 규정을 과거 가격에 적용한 연구 검사; 실제 대회 계좌 인증 아님')


@lru_cache(maxsize=2)
def _prices(path,mtime,size,digest):
    import pandas as pd
    from .model_recipe import file_hash
    if not digest or file_hash(path)!=digest:raise ValueError('원본 시세 해시 불일치')
    frame=pd.read_parquet(path,columns=['date','symbol','open','close'])
    frame['day']=pd.to_datetime(frame.date).dt.strftime('%Y%m%d')
    return {day:{str(r.symbol):(float(r.open),float(r.close)) for r in part.itertuples()} for day,part in frame.groupby('day')}


def price_book(report):
    path=(report.get('recipe') or {}).get('input',{}).get('path')
    if not path:return None
    try:
        file=Path(path);stat=file.stat()
        return _prices(str(file),stat.st_mtime_ns,stat.st_size,report['recipe']['input'].get('sha256'))
    except (OSError,ValueError,KeyError):return None
