"""KOSPI 200 + KOSDAQ 150 시총 상위 유니버스 생성 → data/universe.csv.

시장별 수집(사장 지시 2026-07-03: 코스닥 150은 KOSDAQ150 지수급 우량주 — 시총 상위 근사):
1순위: pykrx (공식 KRX 데이터 — 단, 2026 현재 KRX 로그인 필수라
       KRX_ID/KRX_PW 환경변수가 있을 때만 사용 가능).
2순위: 네이버 시가총액 페이지(sise_market_sum, sosok=0/1) 스크래핑 (로그인 불필요, ~2초/시장).
우선주(코드 끝자리 != '0')와 ETF/ETN/리츠/스팩은 제외한다.
사용: python3 tools/build_universe.py  (또는 crawler 가 CSV 부재 시 자동 호출)
"""
import csv
import os
import re

import requests

from . import bars_config as config

URL = "https://finance.naver.com/sise/sise_market_sum.naver?sosok={sosok}&page={page}"
HEADERS = {"User-Agent": "Mozilla/5.0 (X11; Linux aarch64) AppleWebKit/537.36"}
ROW_RE = re.compile(
    r'href="/item/main\.naver\?code=(\d{6})"[^>]*class="tltle">([^<]+)</a>')
# ETF/ETN/리츠/스팩 제외 — 스펙 유니버스는 개별 종목
ETF_RE = re.compile(
    r"KODEX|TIGER|ARIRANG|KBSTAR|HANARO|KOSEF|SOL |ACE |PLUS |RISE |WON |"
    r"TIME |KIWOOM|UNICORN|마이다스|히어로즈|FOCUS|TREX|파워 |KTOP|"
    r"레버리지|인버스|ETN|선물|채권|리츠|액티브|나스닥|S&P|TOP\s?\d|"
    r"배당주|고배당|밸류업|\bETF\b|스팩")

# (시장명, 네이버 sosok, 종목 수) — 수는 config/_env 로 조정
def _market_specs():
    return (("KOSPI", 0, config.UNIVERSE_KOSPI), ("KOSDAQ", 1, config.UNIVERSE_KOSDAQ))


def _build_pykrx(market, size):
    """pykrx 로 해당 시장 시총 상위 종목 수집. KRX 로그인(KRX_ID/KRX_PW) 필요 —
    미설정/실패 시 None 반환하고 네이버 폴백을 탄다."""
    if not (os.environ.get("KRX_ID") and os.environ.get("KRX_PW")):
        return None
    try:
        from datetime import datetime, timedelta
        from pykrx import stock
        d = datetime.now()
        for _ in range(7):                       # 최근 영업일 탐색
            ds = d.strftime("%Y%m%d")
            caps = stock.get_market_cap(ds, market=market)
            if len(caps):
                break
            d -= timedelta(days=1)
        else:
            return None
        caps = caps.sort_values("시가총액", ascending=False)
        rows = []
        for code in caps.index:
            if code[-1] != "0":                  # 우선주 제외
                continue
            name = stock.get_market_ticker_name(code)
            if not name or ETF_RE.search(str(name)):
                continue
            rows.append((code, str(name)))
            if len(rows) >= size:
                break
        return rows if len(rows) >= size * 0.8 else None
    except Exception:
        return None


def _build_naver(session, sosok, size):
    seen, rows = set(), []
    for page in range(1, 25):
        r = session.get(URL.format(sosok=sosok, page=page), headers=HEADERS,
                        timeout=config.REQUEST_TIMEOUT)
        r.encoding = "euc-kr"
        found = ROW_RE.findall(r.text)
        if not found:
            break
        for code, name in found:
            if code in seen or code[-1] != "0":   # 우선주 제외
                continue
            if ETF_RE.search(name):               # ETF/ETN/리츠/스팩 제외
                continue
            seen.add(code)
            rows.append((code, name.strip()))
            if len(rows) >= size:
                break
        if len(rows) >= size:
            break
    return rows


def build(out_path=None):
    out_path = out_path or config.UNIVERSE_CSV
    session = requests.Session()
    all_rows, seen = [], set()
    for market, sosok, size in _market_specs():
        if size <= 0:
            continue
        rows = _build_pykrx(market, size) or _build_naver(session, sosok, size)
        if len(rows) < size * 0.8:
            raise RuntimeError(f"{market} 유니버스 수집 부족: {len(rows)}/{size}")
        for code, name in rows:
            if code not in seen:
                seen.add(code)
                all_rows.append((code, name, market))   # 시장 라벨 보존 (2팩터 중립화용)
        print(f"{market}: {len(rows)} 종목")
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["code", "name", "market"])
        w.writerows(all_rows)
    print(f"universe.csv 생성: {len(all_rows)} 종목 → {out_path}")
    return all_rows


if __name__ == "__main__":
    build()
