"""One audited, immutable developmental stock panel per evaluation window.

Current collector universes introduce survivor selection. No claim of a complete
point-in-time exchange universe is made, and original academic files stay intact.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

from .config import HOME, RUNS
from .period import expected_dates, window

VERSION = 1
MIN_SYMBOLS = 20
OHLC = ['open', 'high', 'low', 'close']
ROOTS = {'kr': HOME/'vault/CryptoBars/data/KRX/daily_cnn',
         'us': HOME/'vault/CryptoBars/data/USA/daily_policy'}


def paths(market):
    if market not in ROOTS:
        raise ValueError('Unsupported stock market')
    start, end = window()
    folder = RUNS/'research_inputs'/f'{start}-{end}'
    return folder/f'{market}.parquet', folder/f'{market}.audit.json'


def metadata(market):
    _, audit = paths(market)
    try:
        return json.loads(audit.read_text())
    except (OSError, ValueError):
        return {}


def status(market):
    data = metadata(market)
    snapshot, _ = paths(market)
    ready = bool(data.get('ready') and data.get('version') == VERSION
                 and data.get('window') == list(window()) and snapshot.exists())
    return dict(ready=ready, start=data.get('start'), end=data.get('end'),
                source_start=data.get('start'), source_end=data.get('end'),
                symbols=data.get('symbols', 0), message=data.get('message', '최근 36개월 연구 입력 준비 중'))


def validate_frame(frame):
    import numpy as np
    import pandas as pd
    d = frame.copy()
    if 'date' not in d.columns:
        d = d.reset_index()
    d['date'] = pd.to_datetime(d.date).dt.tz_localize(None)
    if d.date.isna().any() or d.duplicated('date').any():
        raise ValueError('duplicate_or_invalid_date')
    values = d[OHLC+['volume']].to_numpy(float)
    if (not np.isfinite(values).all() or (values[:, :4] <= 0).any()
            or (values[:, 4] < 0).any()
            or (d.high < d[OHLC].max(axis=1)-1e-7).any()
            or (d.low > d[OHLC].min(axis=1)+1e-7).any()):
        raise ValueError('invalid_ohlcv')
    return d.sort_values('date').reset_index(drop=True)


def bridge(current, academic, min_overlap=120):
    """Only accept one stable OHLC and volume basis over a long overlap.

    0.5% max relative OHLC error permits rounding but rejects splits, dividend
    drift and material provider differences. Volume must also be stable within
    0.5%; matching prices alone cannot validate image volume channels.
    """
    import numpy as np
    c, a = validate_frame(current), validate_frame(academic)
    overlap = c.merge(a, on='date', suffixes=('_c', '_a'))
    if len(overlap) < min_overlap:
        raise ValueError('insufficient_bridge_overlap')
    ratios = np.column_stack([overlap[k+'_c']/overlap[k+'_a'] for k in OHLC])
    factor = float(np.median(ratios))
    if not np.isfinite(factor) or np.max(np.abs(ratios/factor-1)) > .005:
        raise ValueError('ambiguous_price_basis')
    good = (overlap.volume_c > 0) & (overlap.volume_a > 0)
    if good.sum() < min_overlap or ((overlap.volume_c == 0) != (overlap.volume_a == 0)).any():
        raise ValueError('ambiguous_volume_basis')
    volumes = overlap.loc[good, 'volume_c']/overlap.loc[good, 'volume_a']
    volume_factor = float(np.median(volumes))
    if np.max(np.abs(volumes/volume_factor-1)) > .005:
        raise ValueError('ambiguous_volume_basis')
    prefix = a[a.date < c.date.min()].copy()
    prefix[OHLC] *= factor
    prefix['volume'] *= volume_factor
    return pd_concat(prefix, c), dict(overlap=len(overlap), price_factor=factor,
                                     volume_factor=volume_factor)


def pd_concat(*frames):
    import pandas as pd
    return pd.concat(frames, ignore_index=True)


def coverage(frame, market, minimum=MIN_SYMBOLS):
    import pandas as pd
    expected = expected_dates(market, *window())
    counts = frame[frame.eligible].groupby(frame.date.dt.strftime('%Y%m%d')).symbol.nunique()
    missing = [d for d in expected if counts.get(d, 0) < minimum]
    return dict(ready=not missing, insufficient_dates=missing,
                minimum_symbols=min((int(counts.get(d, 0)) for d in expected), default=0),
                sessions=len(expected))


def fingerprint(path):
    s = path.stat()
    return dict(path=str(path), size=s.st_size, mtime_ns=s.st_mtime_ns)


def prepare(market):
    import numpy as np
    import pandas as pd
    import pyarrow.parquet as pq
    from .research import SOURCES
    snapshot, audit_path = paths(market)
    if status(market)['ready']:
        return pq.ParquetFile(snapshot).read(use_threads=False).to_pandas()
    files = sorted(ROOTS[market].glob('*.parquet'))
    extended = RUNS/'research_inputs/us_extended_daily'
    if market == 'us' and extended.exists():
        files = [extended/f.name if (extended/f.name).exists() else f for f in files]
    symbols = [f.stem for f in files]
    common = ['date','symbol',*OHLC,'volume','tradable_buy','tradable_sell',
              'equity_scope_category','additional_equity_audit_pending']
    extra = ['known_reference_date','common_stock_control_eligible','sector'] if market == 'us' else ['native_volume_unit_verified']
    # ParquetFile avoids dataset scanner background-thread lifetime failures.
    academic = pq.ParquetFile(SOURCES[market]).read(columns=common+extra, use_threads=False).to_pandas()
    academic['symbol'] = academic.symbol.astype(str)
    academic = academic[academic.symbol.isin([s+'@0' if market == 'us' else s for s in symbols])].copy()
    academic['date'] = pd.to_datetime(academic.date)
    academic['eligible'] = ~academic.additional_equity_audit_pending.fillna(True).astype(bool)
    if market == 'kr':
        academic['eligible'] &= academic.equity_scope_category.isin(['real_equity_candidate','dated_native_listed_KONEX_equity_candidate'])
    else:
        known = pd.to_datetime(academic.known_reference_date, errors='coerce')
        academic['eligible'] &= academic.equity_scope_category.eq('common_equity_candidate') & academic.common_stock_control_eligible.fillna(False).astype(bool) & known.notna() & (known <= academic.date)
    if 'sector' not in academic:
        academic['sector'] = 'UNKNOWN'
    grouped = dict(tuple(academic.groupby('symbol', sort=False)))
    frames, dropped, bridges = [], {}, {}
    for file in files:
        symbol = file.stem+'@0' if market == 'us' else file.stem
        a = grouped.get(symbol)
        if a is None or not a.eligible.any():
            dropped[symbol] = 'not_verified_common_equity'; continue
        try:
            c = validate_frame(pq.ParquetFile(file).read(use_threads=False).to_pandas())
            if market == 'us' and file.parent != extended:
                c, bridges[symbol] = bridge(c, a[['date',*OHLC,'volume']])
            c['symbol'] = symbol
            # Same-date historical evidence takes priority. Later rows use only
            # previously published identity; no future eligibility is backfilled.
            evidence = a[['date','eligible','sector','tradable_buy','tradable_sell']].sort_values('date')
            c = pd.merge_asof(c.sort_values('date'), evidence, on='date', direction='backward')
            c['eligible'] = c.eligible.fillna(False).astype(bool)
            c['sector'] = c.sector.fillna('UNKNOWN')
            later = c.date > a.date.max()
            c.loc[later, ['tradable_buy','tradable_sell']] = (c.loc[later, 'volume'] > 0).to_numpy()[:, None].repeat(2, axis=1)
            for col in ['tradable_buy','tradable_sell']:
                c[col] = c[col].fillna(False).astype(bool) & (c.volume > 0)
            c = c[c.date <= pd.Timestamp(window()[1])]
            frames.append(c[['date','symbol',*OHLC,'volume','tradable_buy','tradable_sell','sector','eligible']])
        except (ValueError, KeyError) as exc:
            dropped[symbol] = str(exc)
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=['date','symbol',*OHLC,'volume','tradable_buy','tradable_sell','sector','eligible'])
    out['date'] = pd.to_datetime(out.date)
    out = out.sort_values(['symbol','date']).reset_index(drop=True)
    out['_row'] = np.arange(len(out), dtype=np.int64)
    check = coverage(out, market)
    info = dict(version=VERSION, window=list(window()), market=market, **check,
                start=str(out.date.min().date()) if len(out) else None,
                end=str(out.date.max().date()) if len(out) else None,
                symbols=int(out.symbol.nunique()), dropped_symbols=dropped, bridges=bridges,
                source_files=[fingerprint(SOURCES[market])]+[fingerprint(f) for f in files],
                price_volume_basis=('NAVER daily chart OHLCV provider basis; volume units not independently exchange-audited. Academic native-volume flags apply to the older source and do not certify these NAVER rows.' if market == 'kr' else 'Single-provider adjusted OHLCV history where extended; strict constant-basis bridge otherwise.'),
                limitations=['Current collector universe is survivorship-biased developmental universe, not a full PIT universe.',
                             'Eligibility uses only dated past academic identity; rows after academic end carry provisional identity.',
                             'Corporate-action and trading-status validation after academic end is provisional.',
                             'No missing OHLCV bars are filled.'],
                feature_source_start=str(out.date.min()) if len(out) else None,
                feature_source_end=str(out.date.max()) if len(out) else None)
    info['message'] = ('최근 36개월 개발 백테스트 입력 준비 완료' if check['ready'] else f"36개월 입력 부족: {len(check['insufficient_dates'])}거래일, 최소 {check['minimum_symbols']}종목")
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    # Failed attempts produce only audit; do not present partial data as ready.
    if check['ready']:
        temporary = snapshot.with_suffix('.parquet.tmp')
        out.to_parquet(temporary, index=False)
        info['snapshot_hash'] = hashlib.sha256(temporary.read_bytes()).hexdigest()
        os.replace(temporary, snapshot)
    temp_audit = audit_path.with_suffix('.json.tmp')
    temp_audit.write_text(json.dumps(info, ensure_ascii=False, indent=2))
    os.replace(temp_audit, audit_path)
    if not check['ready']:
        raise ValueError(info['message'])
    return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--market', choices=['kr','us'], required=True)
    args = parser.parse_args()
    try:
        prepare(args.market)
    except ValueError as error:
        print(str(error))
        raise SystemExit(1)
    print(json.dumps(status(args.market), ensure_ascii=False))
