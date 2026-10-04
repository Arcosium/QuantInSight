"""
Arquant v1.0 - Market Data Collector
Adapted from KRX Quant Simulator CrawlerUtil:
  - 글로벌 지수 실시간 크롤링 (네이버 금융)
  - 종목 3년치 일봉 + 수급 크롤링 (네이버 금융 HTML)
  - 분봉 데이터는 KIS API 사용 (kis_broker.kr_minute_chart)
"""
import csv, re, json, logging, time
from datetime import datetime, timedelta, timezone

KST = timezone(timedelta(hours=9))
from pathlib import Path
from typing import List, Dict, Optional
import requests
from bs4 import BeautifulSoup
import pandas as pd

try:  # 공용 시세(EOD) 레이어 — pykrx/yfinance 우선 (사장 지시 2026-07-02), 실패 시 네이버 크롤 폴백
    import arcmarket
except ImportError:
    arcmarket = None

logger = logging.getLogger("MARKET_DATA")
DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)
HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}

# ═══════════════════ 글로벌 지수 워치리스트 ═══════════════════
# Robust sources (JSON APIs where possible). Each entry: {name, kind, sym}.
#   kind="domestic" → polling.finance.naver.com/api/realtime/domestic/index/{sym}
#   kind="world"    → api.stock.naver.com/index/{sym}/basic
#   kind="fx"/"oil" → api.stock.naver.com/marketindex/{exchange|energy}/{sym}
#     (2026-09-25: finance.naver.com/marketindex 가 stock.naver.com 으로 이관돼 HTML 스크랩이
#      9/16부터 조용히 ok=False — 환율 캐시가 1,500원에 9일 고착됐다.)
INDEX_WATCHLIST = {
    "KOSPI":  {"name":"코스피",       "kind":"domestic", "sym":"KOSPI"},
    "KOSDAQ": {"name":"코스닥",       "kind":"domestic", "sym":"KOSDAQ"},
    "KPI200": {"name":"코스피200",    "kind":"domestic", "sym":"KPI200"},
    "DJI":    {"name":"다우존스",     "kind":"world",    "sym":".DJI"},
    "SPX":    {"name":"S&P500",      "kind":"world",    "sym":".INX"},
    "IXIC":   {"name":"나스닥",       "kind":"world",    "sym":".IXIC"},
    "SHS":    {"name":"상해종합",     "kind":"world",    "sym":".SSEC"},
    "NKY":    {"name":"니케이225",    "kind":"world",    "sym":".N225"},
    "USDKRW": {"name":"원/달러 환율", "kind":"fx",       "sym":"FX_USDKRW"},
    "WTI":    {"name":"WTI 원유",     "kind":"oil",      "sym":"CLcv1"},
}

_NUM_RE = None
def _to_float(s):
    try: return float(str(s).replace(",", "").replace("원", "").replace("달러", "").strip())
    except (TypeError, ValueError): return None

# ── per-source fetchers (return {"value":float, "change":float|None, "rate":float|None, "ok":bool}) ──
def _fetch_domestic(sym: str) -> Dict:
    r = requests.get(f"https://polling.finance.naver.com/api/realtime/domestic/index/{sym}",
                     headers=HEADERS, timeout=8)
    d = (r.json().get("datas") or [{}])[0]
    return {"value": _to_float(d.get("closePriceRaw") or d.get("closePrice")),
            "change": _to_float(d.get("compareToPreviousClosePriceRaw") or d.get("compareToPreviousClosePrice")),
            "rate": _to_float(d.get("fluctuationsRatioRaw") or d.get("fluctuationsRatio")),
            "ok": True}

def _fetch_world(sym: str) -> Dict:
    r = requests.get(f"https://api.stock.naver.com/index/{sym}/basic", headers=HEADERS, timeout=8)
    d = r.json()
    return {"value": _to_float(d.get("closePrice")),
            "change": _to_float(d.get("compareToPreviousClosePrice")),
            "rate": _to_float(d.get("fluctuationsRatio")), "ok": True}

def _fetch_marketindex(category: str, sym: str) -> Dict:
    r = requests.get(f"https://api.stock.naver.com/marketindex/{category}/{sym}", headers=HEADERS, timeout=8)
    d = r.json()
    d = d.get("exchangeInfo") or d          # 환율은 exchangeInfo 아래, 유가는 최상위
    val = _to_float(d.get("closePrice"))
    return {"value": val, "change": _to_float(d.get("fluctuations")),
            "rate": _to_float(d.get("fluctuationsRatio")), "ok": val is not None}

def _fetch_index(key: str) -> Dict:
    info = INDEX_WATCHLIST[key]
    try:
        if info["kind"] == "domestic":  return _fetch_domestic(info["sym"])
        if info["kind"] == "world":     return _fetch_world(info["sym"])
        if info["kind"] == "fx":        return _fetch_marketindex("exchange", info["sym"])
        if info["kind"] == "oil":       return _fetch_marketindex("energy", info["sym"])
    except Exception as e:
        logger.warning(f"[지수] {key} 조회 실패: {e}")
    return {"value": None, "change": None, "rate": None, "ok": False}


# ── 라이브 USD/KRW 환율 (사장 지시 2026-05-22) ──────────────────────────────
# 환율은 계속 변하므로 하드코딩하지 않는다. 5분 지수 크롤(get_index_data)이 USDKRW 를
# 가져올 때마다 캐시를 갱신하고, 모든 USD↔KRW 환산(예산·리스크 사이징)이 이 값을 읽는다.
# 디스크에도 영속해 재시작 직후 첫 사이클도 직전 환율을 쓰게 한다.
_FX_CACHE_PATH = Path(__file__).resolve().parent.parent / "data" / "usdkrw_fx.json"
_LAST_FX = 0.0

def set_usdkrw(rate) -> None:
    """라이브 환율 갱신 (sanity 500~5000 범위만 채택)."""
    global _LAST_FX
    try:
        r = float(rate)
    except (TypeError, ValueError):
        return
    if 500.0 < r < 5000.0:
        _LAST_FX = r
        try:
            _FX_CACHE_PATH.parent.mkdir(exist_ok=True)
            _FX_CACHE_PATH.write_text(json.dumps({"rate": r, "ts": time.time()}), encoding="utf-8")
        except Exception:
            pass

def get_usdkrw(default: float = 1500.0) -> float:
    """최신 라이브 USD/KRW 환율. 미수집 시 디스크 캐시 → 그래도 없으면 default."""
    global _LAST_FX
    if _LAST_FX and _LAST_FX > 0:
        return _LAST_FX
    try:
        d = json.loads(_FX_CACHE_PATH.read_text(encoding="utf-8"))
        r = float(d.get("rate") or 0.0)
        if 500.0 < r < 5000.0:
            _LAST_FX = r
            return r
    except Exception:
        pass
    return float(default)


def get_index_data() -> Dict[str, Dict]:
    """Structured snapshot of every watchlist index — only validated numbers, never garbage.
    Returns {key: {"name","value","change","rate","ok"}}; ok=False ⇒ value/change/rate are None."""
    out: Dict[str, Dict] = {}
    for key, info in INDEX_WATCHLIST.items():
        d = _fetch_index(key)
        # sanity: an index value of 0/None or an absurdly small one for an equity index is "not ok"
        v = d.get("value")
        if v is None or v <= 0:
            d = {"value": None, "change": None, "rate": None, "ok": False}
        out[key] = {"name": info["name"], **d}
    # 사장 지시 2026-05-22: 크롤한 원/달러 환율을 라이브 캐시에 반영(예산·리스크 환산이 읽음)
    _fx = out.get("USDKRW") or {}
    if _fx.get("ok") and _fx.get("value"):
        set_usdkrw(_fx["value"])
    return out


def kr_price_naver(code: str) -> float:
    """KIS API가 0/실패를 반환할 때 폴백 — 네이버 polling API에서 KR 종목 현재가(원) 조회.
    (사장 지시 2026-05-14 — 028670 등 트랜지언트 0원 문제 해결).
    Returns 0.0 if 네이버도 실패."""
    code = str(code).strip()
    if not (code.isdigit() and len(code) == 6):
        return 0.0
    try:
        # Primary: polling JSON API (가장 가벼움, 실시간 가까움)
        r = requests.get(f"https://polling.finance.naver.com/api/realtime/domestic/stock/{code}",
                         headers=HEADERS, timeout=6)
        d = (r.json().get("datas") or [{}])[0]
        for k in ("closePriceRaw", "closePrice", "tradePriceRaw", "tradePrice"):
            v = _to_float(d.get(k))
            if v and v > 0:
                return v
    except Exception:
        pass
    try:
        # Fallback within Naver: scrape item/main page (slower but more robust)
        r = requests.get(f"https://finance.naver.com/item/main.naver?code={code}", headers=HEADERS, timeout=6)
        r.encoding = "euc-kr"
        soup = BeautifulSoup(r.text, "html.parser")
        # `.no_today .blind` holds the price as raw digits (e.g. "117400")
        el = soup.select_one("p.no_today .blind") or soup.select_one(".today .blind")
        if el:
            return _to_float(el.get_text(strip=True)) or 0.0
    except Exception:
        pass
    return 0.0


_NAME_CACHE: Dict[str, str] = {}
_NAME_MISS: set = set()          # 조회 실패 심볼(프로세스 수명) — 반복 404 억제, 재시작 시 재검증
def get_stock_name(code: str) -> str:
    """종목코드 → 종목명(캐시). KR=6자리 코드(네이버), US=영문 티커(yfinance). '' on failure.

    US 분기 추가 2026-08-04: 후보 검증이 KR 만 있고 US 는 무검증이라 'Rocket Lab(RVLV)'
    (실제 RVLV=Revolve Group) 같은 티커 오배정이 그대로 통과하던 문제."""
    code = str(code).strip()
    if code in _NAME_CACHE:
        return _NAME_CACHE[code]
    if not re.fullmatch(r"\d{6}", code):
        return _us_stock_name(code)
    name = ""
    try:
        r = requests.get(f"https://polling.finance.naver.com/api/realtime/domestic/stock/{code}",
                         headers=HEADERS, timeout=6)
        d = (r.json().get("datas") or [{}])[0]
        name = (d.get("stockName") or "").strip()
    except Exception:
        try:
            r = requests.get(f"https://finance.naver.com/item/main.naver?code={code}", headers=HEADERS, timeout=6)
            r.encoding = "euc-kr"
            t = BeautifulSoup(r.text, "html.parser").select_one("div.wrap_company h2 a")
            name = t.get_text(strip=True) if t else ""
        except Exception:
            name = ""
    if name:
        _NAME_CACHE[code] = name
    return name


def _us_stock_name(ticker: str) -> str:
    """US 티커 → 회사명(yfinance, 캐시). 실패 시 '' — **호출부는 fail-open** 이어야 한다
    (네트워크 장애로 US 후보가 통째로 사라지면 안 됨; 없는 티커는 하류 일봉 게이트가 거른다)."""
    t = str(ticker).strip().upper()
    if not re.fullmatch(r"[A-Z][A-Z.\-]{0,5}", t) or t in _NAME_MISS:
        return ""
    name = ""
    try:
        import yfinance as yf
        info = yf.Ticker(t).info or {}
        name = (info.get("longName") or info.get("shortName") or "").strip()
    except Exception:
        name = ""
    if name:
        _NAME_CACHE[t] = name
    else:
        _NAME_MISS.add(t)
    return name


def canonical_name(code: str, fallback: str = "", resolver=get_stock_name) -> str:
    """코드의 **권위(정본) 종목명**. 버그 2026-06-12: 뉴스/센티(LLM)와 계량(데이터제공자)이
    코드→이름을 따로 만들어 불일치(241520=PKC vs DSC인베스트먼트). 결정·thesis 기록엔
    코드 기준 정본 이름을 쓴다. resolver 실패('')면 fallback(보통 LLM 이름)으로 폴백.
    resolver 는 테스트 주입용(기본 get_stock_name)."""
    try:
        nm = (resolver(code) or "").strip()
    except Exception:
        nm = ""
    return nm or (fallback or "").strip()


_CODE_CACHE: Dict[str, str] = {}
def resolve_kr_stock_code(name: str) -> str:
    """종목명 → 6자리 KR 코드 (네이버 금융 검색). 실패 시 ''.

    주식운용실장(LLM)이 종목명만 알고 코드를 환각(123456 등)할 때, 이름으로 정확한 코드를
    채우는 폴백 (사장 지시 2026-05-22). 네이버 검색은 EUC-KR 인코딩을 요구한다.
    첫 번째 매칭(가장 관련도 높은 종목)의 코드를 돌려준다."""
    name = (name or "").strip()
    if not name or re.fullmatch(r"\d{6}", name):
        return name if re.fullmatch(r"\d{6}", name) else ""
    if name in _CODE_CACHE:
        return _CODE_CACHE[name]
    code = ""
    try:
        from urllib.parse import quote
        q = quote(name.encode("euc-kr"))
        r = requests.get(f"https://finance.naver.com/search/search.naver?query={q}", headers=HEADERS, timeout=6)
        r.encoding = "euc-kr"
        m = re.findall(r"/item/main\.naver\?code=(\d{6})", r.text)
        code = m[0] if m else ""
    except Exception:
        code = ""
    if code:
        _CODE_CACHE[name] = code
    return code


def crawl_index_snapshot(data: Optional[Dict[str, Dict]] = None) -> str:
    """Human-readable snapshot string (used in the dashboard log)."""
    data = data if data is not None else get_index_data()
    lines = ["[글로벌 지수 현황]\n"]
    for key, d in data.items():
        if d["ok"]:
            chg = (f" ({d['change']:+,.2f}" + (f", {d['rate']:+.2f}%)" if d['rate'] is not None else ")")) if d["change"] is not None else ""
            lines.append(f"  📈 {d['name']} ({key}): {d['value']:,.2f}{chg}")
        else:
            lines.append(f"  ⚠ {d['name']} ({key}): N/A (조회 실패 — 분석에서 제외)")
    return "\n".join(lines)


def format_indices_for_macro(data: Optional[Dict[str, Dict]] = None) -> str:
    """Strict facts block for the macro LLM: ONLY the verified numbers, with an instruction
    not to invent any. Skips indices whose fetch failed."""
    data = data or get_index_data()
    rows, missing = [], []
    for key, d in data.items():
        if d["ok"]:
            # 등락률이 없을 때(WTI 등 일부 소스) 전일대비·현재가로 역산 — 'n/a' 를 LLM이
            # '+0.70.00' 처럼 깨뜨려 인용하던 문제 방지(사장 지시 2026-07-21).
            _rate = d["rate"]
            if _rate is None and d.get("change") is not None and d.get("value"):
                _prev = d["value"] - d["change"]
                if _prev:
                    _rate = d["change"] / _prev * 100.0
            r = f"{_rate:+.2f}%" if _rate is not None else "n/a"
            c = f"{d['change']:+,.2f}" if d["change"] is not None else "n/a"
            rows.append(f"- {d['name']} ({key}): 현재가={d['value']:,.2f} | 전일대비={c} | 등락률={r}")
        else:
            missing.append(d["name"])
    body = "\n".join(rows) if rows else "(수집된 지수 없음)"
    note = ("\n[주의] 위 숫자만 사용하십시오. 표에 없는 수치(예: 특정 지수의 절대 레벨)를 추정하거나 지어내지 마십시오. "
            "조회 실패 지수: " + (", ".join(missing) if missing else "없음"))
    return f"[검증된 글로벌 지수 — {datetime.now().strftime('%Y-%m-%d %H:%M')}]\n{body}{note}"


# ═══════════════════ 종목별 3년치 일봉 크롤링 (KRX 퀀트 시뮬레이터 로직) ═══════════════════

def fetch_stock_daily(code: str, years: int = 3) -> pd.DataFrame:
    """네이버 금융에서 종목 일봉 OHLCV 크롤링 → CSV 누적
    Adapted from KRX Quant Simulator CrawlerUtil.fetch_naver_stock_html()
    """
    csv_path = DATA_DIR / f"daily_{code}.csv"
    stop_date = _get_csv_latest_date(csv_path)

    # 1순위 arcmarket(pykrx/yfinance) — 성공하면 크롤 없이 같은 CSV 규격으로 누적
    if arcmarket is not None:
        try:
            df = arcmarket.kr_daily(code, days=int(years * 365))
            if df is not None and not df.empty:
                if stop_date is not None:
                    df = df[df.index >= pd.to_datetime(stop_date)]  # 당일 봉 upsert 갱신 유지
                df = df.fillna(0)
                rows = [{'date': d.strftime('%Y-%m-%d'), 'open': int(x['open']),
                         'high': int(x['high']), 'low': int(x['low']),
                         'close': int(x['close']), 'volume': int(x['volume'])}
                        for d, x in df.iterrows()]
                if rows:
                    _append_csv(csv_path, rows, ['date', 'open', 'high', 'low', 'close', 'volume'])
                    logger.info(f"[일봉] {code}: arcmarket {len(rows)}건 (pykrx/yfinance)")
                    return pd.DataFrame(rows)
        except Exception as e:
            logger.warning(f"[일봉] {code} arcmarket 실패 — 네이버 폴백: {e}")

    result = []
    max_pages = int(years * 26) + 10
    target_limit = datetime.today() - timedelta(days=years * 365)

    logger.info(f"[일봉] {code} 크롤링 시작 (stop_date={stop_date})")

    try:
        for page in range(1, min(max_pages, 400) + 1):
            url = f"https://finance.naver.com/item/sise_day.nhn?code={code}&page={page}"
            res = requests.get(url, headers=HEADERS, timeout=8)
            soup = BeautifulSoup(res.text, 'lxml')
            rows = soup.select('table.type2 tr')
            valid = 0
            for row in rows:
                cols = row.find_all('td')
                if len(cols) != 7:
                    continue
                try:
                    date_text = cols[0].text.strip()
                    if not date_text:
                        continue
                    date = pd.to_datetime(date_text)
                    # '<=' → '<' : 최신 날짜(당일 포함)는 다시 수집해 _append_csv upsert 로 갱신한다
                    # (당일 봉이 첫 조회값에 고정되던 버그 2026-06-15 수정).
                    if stop_date and date < stop_date:
                        break
                    if date < target_limit:
                        break
                    close = int(cols[1].text.replace(',', ''))
                    open_ = int(cols[3].text.replace(',', ''))
                    high = int(cols[4].text.replace(',', ''))
                    low = int(cols[5].text.replace(',', ''))
                    volume = int(cols[6].text.replace(',', ''))
                    result.append({'date': date.strftime('%Y-%m-%d'), 'open': open_,
                                   'high': high, 'low': low, 'close': close, 'volume': volume})
                    valid += 1
                except:
                    continue
            if (stop_date and valid == 0 and page > 1) or (valid == 0 and page > 3):
                break
            time.sleep(0.15)  # Rate limit
    except Exception as e:
        logger.error(f"[일봉] {code} 크롤링 오류: {e}")

    if result:
        _append_csv(csv_path, result, ['date', 'open', 'high', 'low', 'close', 'volume'])
        logger.info(f"[일봉] {code}: {len(result)}건 누적 완료")

    df = pd.DataFrame(result)
    return df


def fetch_investor_data(code: str, years: int = 3) -> pd.DataFrame:
    """네이버 금융에서 종목 수급 데이터(기관/외인 순매수) 크롤링 → CSV 누적
    Adapted from KRX Quant Simulator CrawlerUtil.fetch_investor_data()
    """
    csv_path = DATA_DIR / f"investor_{code}.csv"
    stop_date = _get_csv_latest_date(csv_path)

    result = []
    target_limit = datetime.today() - timedelta(days=years * 365)

    logger.info(f"[수급] {code} 크롤링 시작")

    # 2026-09-25: finance.naver.com/item/frgn.nhn 이 stock.naver.com 으로 이관돼 9/10 부터 전 종목 수급이
    # 조용히 멈췄다. 모바일 JSON(최근 약 60거래일, page 인자 없음)으로 바꾼다. 값은 옛 표와 같은 순매매
    # 수량(주)이다(005930 9/8·9/9 대조 일치). 60거래일보다 오래 끊긴 종목은 그 사이가 빈다.
    try:
        res = requests.get(f"https://m.stock.naver.com/api/stock/{code}/trend?pageSize=60",
                           headers=HEADERS, timeout=8)
        for x in res.json():
            date = pd.to_datetime(str(x.get("bizdate")))
            if (stop_date and date <= stop_date) or date < target_limit:
                continue
            _n = lambda v: int(str(v).replace(',', '').replace('+', ''))
            result.append({'date': date.strftime('%Y-%m-%d'),
                           'inst_net': _n(x["organPureBuyQuant"]), 'foreign_net': _n(x["foreignerPureBuyQuant"])})
    except Exception as e:
        logger.error(f"[수급] {code} 조회 오류: {e}")

    if result:
        _append_csv(csv_path, result, ['date', 'inst_net', 'foreign_net'])
        logger.info(f"[수급] {code}: {len(result)}건 누적 완료")

    return pd.DataFrame(result)


def _csv_row_count(path: Path) -> int:
    """Count data rows (excluding header) in a CSV, 0 if missing/empty."""
    if not path.exists():
        return 0
    try:
        df = pd.read_csv(path)
        return len(df)
    except Exception:
        return 0


def crawl_company_full(code: str, kis_broker=None) -> str:
    """종목 코드에 대해 3년 일봉 + 수급을 한번에 크롤링하고 요약 반환.

    Fallback chain (사장 지시 2026-05-14):
      1) 네이버 금융 일봉/수급 크롤링 (primary)
      2) 누적 CSV가 여전히 0행이면 KIS API (`kr_daily_chart`) 로 일봉 폴백
    이 함수는 동기(blocking)라 KIS 폴백은 broker가 명시적으로 전달된 경우에만 수행한다.
    `summary`는 사장님 요청에 따라 '신규 N건 / 누적 M건' 형식으로 보고."""
    new_daily = 0; new_inv = 0
    try:
        daily_df = fetch_stock_daily(code, years=3)
        new_daily = len(daily_df) if daily_df is not None else 0
    except Exception as e:
        logger.warning(f"[일봉] 네이버 크롤링 예외 ({code}): {e}")
    try:
        investor_df = fetch_investor_data(code, years=3)
        new_inv = len(investor_df) if investor_df is not None else 0
    except Exception as e:
        logger.warning(f"[수급] 네이버 크롤링 예외 ({code}): {e}")

    csv_daily = DATA_DIR / f"daily_{code}.csv"
    csv_inv = DATA_DIR / f"investor_{code}.csv"
    total_daily = _csv_row_count(csv_daily)
    total_inv = _csv_row_count(csv_inv)

    # KIS 폴백은 비동기라서 동기 함수 안에서 직접 호출할 수 없다 — 호출처(_collect_company_data)
    # 에서 누적이 0인 경우에 별도로 await broker.kr_daily_chart()를 실행한다.
    needs_kis_fallback = (total_daily == 0)

    summary = (f"[{code}] 일봉: 신규 +{new_daily} / 누적 {total_daily}행, "
               f"수급: 신규 +{new_inv} / 누적 {total_inv}행"
               + (" | ⚠ 일봉 0행 → 호출처에서 KIS 폴백 요청" if needs_kis_fallback else ""))

    # Build quant summary if data exists
    if total_daily > 0:
        try:
            df = pd.read_csv(csv_daily).sort_values('date').reset_index(drop=True)
            df = _strip_trailing_zero_volume(df)   # 프리마켓 진행 중(거래량 0) 행 제거 → '장 마감' 오인 방지
            df = _strip_provisional_today(df, datetime.now(KST).strftime('%Y-%m-%d'))  # 당일 미완성봉 제외
            latest = df.iloc[-1]
            summary += (f"\n  최근(완성봉 {latest['date']}): 종가 {latest.get('close','-')} | "
                        f"거래량 {latest.get('volume','-')} · ⓘ 당일봉은 장중 미완성이라 비교 제외")
        except Exception:
            pass
    else:
        summary += "\n  ⚠ 일봉 데이터 없음 (네이버+KIS 모두 실패) — 퀀트 분석 제한"

    summary += f"\n  CSV: {csv_daily.name}, {csv_inv.name}"
    return summary


def _strip_trailing_zero_volume(df: pd.DataFrame) -> pd.DataFrame:
    """말미의 거래량 0 행(프리마켓 등 '장 시작 전' 진행 중 placeholder)을 제거한다.

    프리마켓 사이클이 당일 거래량 0 행을 CSV 에 누적하면 퀀트가 '장 마감'으로 오인하고
    지표가 하루 stale 해진다(2026-06-15). 말미(trailing)만 떼고 중간 거래정지(0거래량) 행은
    보존하며, 전부 0이면 최소 1행을 남긴다(fail-open)."""
    if df is None or df.empty or 'volume' not in df.columns:
        return df
    vol = pd.to_numeric(df['volume'], errors='coerce').fillna(0)
    keep = len(df)
    while keep > 1 and vol.iloc[keep - 1] <= 0:
        keep -= 1
    return df.iloc[:keep].reset_index(drop=True) if keep < len(df) else df


def _strip_provisional_today(df: pd.DataFrame, today_str: str) -> pd.DataFrame:
    """당일(미완성) 봉을 시계열에서 제외 — 장중 부분 거래량을 전일 '전체'와 비교해 '거래량 98%
    급감'으로 오독하는 것을 막는다(2026-06-15). 현재가 사이징은 라이브 호가(kr_last_price)를
    쓰므로 영향 없음. 말미 기준 당일(또는 그 이후) 행만 떼고 전부 당일이면 최소 1행 유지."""
    if df is None or df.empty or 'date' not in df.columns:
        return df
    d = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
    keep = len(df)
    while keep > 1 and d.iloc[keep - 1] >= today_str:
        keep -= 1
    return df.iloc[:keep].reset_index(drop=True) if keep < len(df) else df


def load_daily_csv(code: str) -> Optional[pd.DataFrame]:
    """CSV에서 일봉 데이터 로드 (퀀트 전략 에이전트용).
    KR 6자리 코드 → daily_{code}.csv / US 티커 → daily_US_{ticker}.csv.
    Returns None when the file doesn't exist or is empty."""
    code = str(code).strip()
    is_kr = code.isdigit() and len(code) == 6
    path = DATA_DIR / (f"daily_{code}.csv" if is_kr else f"daily_US_{code.upper()}.csv")
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
        if df.empty:
            return None
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
        df = _strip_trailing_zero_volume(df)
        # 당일 미완성 봉 제외 — 완성된 봉으로만 지표·거래량 비교(2026-06-15).
        return _strip_provisional_today(df, datetime.now(KST).strftime('%Y-%m-%d'))
    except Exception:
        return None


def forward_return_after(code: str, signal_ts: str, window_days: int = 30) -> Optional[float]:
    """성과귀인(IC)용 — signal_ts(='YYYY-MM-DD ...') 시점 종가 대비 window_days(영업일) 경과 종가 수익률.
    아직 미래 데이터가 없으면(최근 신호) 또는 데이터 부족이면 None(표본에서 제외 — 무음 안전)."""
    try:
        df = load_daily_csv(code)
        if df is None or len(df) < 2:
            return None
        base = pd.to_datetime(str(signal_ts)[:10])
        mask = df['date'] <= base
        if not mask.any():
            return None
        start_pos = int(df.index[mask][-1])
        end_pos = min(start_pos + int(window_days), len(df) - 1)
        if end_pos <= start_pos:
            return None
        p0 = float(df['close'].iloc[start_pos]); p1 = float(df['close'].iloc[end_pos])
        return (p1 / p0 - 1.0) if p0 > 0 else None
    except Exception:
        return None


def load_investor_csv(code: str) -> Optional[pd.DataFrame]:
    """CSV에서 수급 데이터 로드"""
    path = DATA_DIR / f"investor_{code}.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df['date'] = pd.to_datetime(df['date'])
    return df.sort_values('date').reset_index(drop=True)


def _rsi(close, period: int = 14):
    """Wilder-smoothed RSI on a pd.Series; returns the latest value, or None if not enough data."""
    if close is None or len(close) <= period:
        return None
    try:
        delta = close.diff()
        gain = delta.where(delta > 0, 0.0).ewm(alpha=1/period, adjust=False).mean()
        loss = (-delta.where(delta < 0, 0.0)).ewm(alpha=1/period, adjust=False).mean()
        rs = gain / loss.replace(0, 1e-9)
        return float((100 - 100 / (1 + rs)).iloc[-1])
    except Exception:
        return None


def _macd_hist(close, fast: int = 12, slow: int = 26, signal: int = 9):
    """Return latest MACD histogram value (signal-line crossover indicator)."""
    if close is None or len(close) < slow + signal:
        return None
    try:
        ema_f = close.ewm(span=fast, adjust=False).mean()
        ema_s = close.ewm(span=slow, adjust=False).mean()
        macd = ema_f - ema_s
        sig = macd.ewm(span=signal, adjust=False).mean()
        return float((macd - sig).iloc[-1])
    except Exception:
        return None


def _vwap_20d(daily_df):
    """20-day rolling VWAP using (H+L+C)/3 as the typical price.
    Returns the latest VWAP value, or None if not enough data."""
    if daily_df is None or len(daily_df) < 20:
        return None
    try:
        tp = (daily_df['high'].astype(float) + daily_df['low'].astype(float) + daily_df['close'].astype(float)) / 3.0
        vol = daily_df['volume'].astype(float)
        pv = (tp * vol).rolling(20).sum()
        vv = vol.rolling(20).sum().replace(0, 1e-9)
        return float((pv / vv).iloc[-1])
    except Exception:
        return None


def _adx(daily_df, period: int = 14):
    """Average Directional Index — Wilder's smoothing. Returns latest ADX or None.
    ADX > 25 → 강한 추세 / 20 미만 → 횡보. Direction is signed by +DI vs -DI."""
    if daily_df is None or len(daily_df) < period * 2 + 2:
        return None
    try:
        h = daily_df['high'].astype(float); l = daily_df['low'].astype(float); c = daily_df['close'].astype(float)
        up = h.diff(); dn = -l.diff()
        plus_dm = ((up > dn) & (up > 0)) * up
        minus_dm = ((dn > up) & (dn > 0)) * dn
        tr1 = (h - l).abs()
        tr2 = (h - c.shift(1)).abs()
        tr3 = (l - c.shift(1)).abs()
        import pandas as _pd
        tr = _pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        # Wilder smoothing
        atr = tr.ewm(alpha=1/period, adjust=False).mean().replace(0, 1e-9)
        plus_di = 100 * plus_dm.ewm(alpha=1/period, adjust=False).mean() / atr
        minus_di = 100 * minus_dm.ewm(alpha=1/period, adjust=False).mean() / atr
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, 1e-9)
        adx = dx.ewm(alpha=1/period, adjust=False).mean()
        return {"adx": float(adx.iloc[-1]), "plus_di": float(plus_di.iloc[-1]), "minus_di": float(minus_di.iloc[-1])}
    except Exception:
        return None


def _cmf(daily_df, period: int = 20):
    """Chaikin Money Flow(20). 매집(종가 고가근처)→+, 분산(저가근처)→−. 범위 -1..1, None if 데이터 부족.
    (사장 지시 2026-06-04: 결정론 점수 엔진 신규 지표)"""
    if daily_df is None or len(daily_df) < period:
        return None
    try:
        h = daily_df['high'].astype(float); l = daily_df['low'].astype(float)
        c = daily_df['close'].astype(float); v = daily_df['volume'].astype(float)
        rng = (h - l).replace(0, 1e-9)
        mfv = (((c - l) - (h - c)) / rng) * v
        denom = v.rolling(period).sum().replace(0, 1e-9)
        return float((mfv.rolling(period).sum() / denom).iloc[-1])
    except Exception:
        return None


def compute_quant_indicators(code: str, daily=None, investor=None) -> dict:
    """결정론 점수 엔진(tools/quant_score) 입력용 구조화 지표 dict (사장 지시 2026-06-04).
    format_quant_data_for_agent 와 동일 계산을 '숫자'로 반환. 값 없는 지표는 키 자체를 생략한다
    (indicator_signals 가 결손을 분모에서 제외). daily/investor 미주입 시 CSV 로드(테스트는 주입)."""
    if daily is None:
        daily = load_daily_csv(code)
    if investor is None:
        investor = load_investor_csv(code)
    out: dict = {}
    if daily is None or len(daily) < 20:
        return out
    close = daily['close'].astype(float)
    last = float(close.iloc[-1])
    rsi = _rsi(close, 14)
    if rsi is not None:
        out["rsi14"] = rsi
    mh = _macd_hist(close)
    if mh is not None and last:
        out["macd_hist_pct"] = mh / last * 100.0
    adx_dict = _adx(daily, 14)
    if adx_dict is not None:
        out["adx"] = adx_dict["adx"]
        out["adx_dir"] = 1.0 if adx_dict["plus_di"] >= adx_dict["minus_di"] else -1.0
    vwap = _vwap_20d(daily)
    if vwap is not None and vwap > 0:
        out["vwap_dev"] = (last / vwap - 1.0) * 100.0
    if len(close) >= 21:
        try:
            out["sigma20"] = float(close.pct_change().tail(20).std() * (252 ** 0.5) * 100)
        except Exception:
            pass
        try:
            out["mom_1m"] = (last / float(close.iloc[-21]) - 1.0) * 100.0
        except Exception:
            pass
    if len(close) >= 63:
        try:
            out["mom_3m"] = (last / float(close.iloc[-63]) - 1.0) * 100.0
        except Exception:
            pass
    if len(daily) >= 252:
        try:
            high52 = float(daily['high'].astype(float).tail(252).max())
            if high52 > 0:
                out["high52_prox"] = last / high52
        except Exception:
            pass
    cmf = _cmf(daily, 20)
    if cmf is not None:
        out["cmf"] = cmf
    # 거래량 급증(사장 지시 2026-07-21) — 당일 거래량 / 직전 20일 평균 - 1 (0=평균, +1=2배)
    try:
        _vol = daily['volume'].astype(float)
        if len(_vol) >= 21:
            _avg20 = float(_vol.iloc[-21:-1].mean())
            if _avg20 > 0:
                out["vol_surge"] = float(_vol.iloc[-1]) / _avg20 - 1.0
    except Exception:
        pass
    # 갭업(사장 지시 2026-07-21) — 당일 시가 대비 전일 종가 %(추격 회피 필터용, 신호 아님)
    try:
        if len(daily) >= 2 and 'open' in daily.columns:
            _pc = float(close.iloc[-2]); _to = float(daily['open'].astype(float).iloc[-1])
            if _pc > 0:
                out["gap_up_pct"] = (_to / _pc - 1.0) * 100.0
    except Exception:
        pass
    # flow — 외인+기관 순매수를 거래량 대비 비율로 정규화([-1,1] 사전정규화 신호)
    if investor is not None and len(investor) > 0:
        try:
            net5 = float(investor['inst_net'].tail(5).sum() + investor['foreign_net'].tail(5).sum())
            vol5 = float(daily['volume'].astype(float).tail(5).sum()) or 1.0
            ratio = net5 / vol5
            if len(investor) >= 20:
                net20 = float(investor['inst_net'].tail(20).sum() + investor['foreign_net'].tail(20).sum())
                vol20 = float(daily['volume'].astype(float).tail(20).sum()) or 1.0
                ratio = 0.5 * ratio + 0.5 * (net20 / vol20)
            out["flow"] = max(-1.0, min(1.0, ratio * 4.0))
        except Exception:
            pass
    return out


def format_quant_data_for_agent(code: str) -> str:
    """퀀트 에이전트에게 전달할 종목 데이터 포맷팅 (사장 지시 2026-05-14 — 추가 지표: RSI/MACD/실현변동성/수급추세)"""
    daily = load_daily_csv(code)
    investor = load_investor_csv(code)
    lines = [f"[{code} 퀀트 데이터]\n"]

    if daily is not None and len(daily) > 0:
        recent = daily.tail(5)
        first_date = daily.iloc[0]['date'].strftime('%Y-%m-%d')
        last_date = daily.iloc[-1]['date'].strftime('%Y-%m-%d')
        lines.append(f"  일봉 데이터: {len(daily)}일 ({first_date} ~ {last_date})")
        lines.append(f"  최근 5일:")
        for _, r in recent.iterrows():
            lines.append(f"    {r['date'].strftime('%Y-%m-%d')}: O={r['open']} H={r['high']} L={r['low']} C={r['close']} V={r['volume']}")

        # Indicators
        if len(daily) >= 20:
            close = daily['close'].astype(float)
            sma5 = close.rolling(5).mean().iloc[-1]
            sma20 = close.rolling(20).mean().iloc[-1]
            ind = [f"SMA5: {sma5:.0f}", f"SMA20: {sma20:.0f}"]
            if len(daily) >= 60:
                sma60 = close.rolling(60).mean().iloc[-1]
                ind.append(f"SMA60: {sma60:.0f}")
            # RSI(14) — momentum/mean-reversion signal
            rsi = _rsi(close, 14)
            if rsi is not None:
                tag = " 과매수" if rsi >= 70 else (" 과매도" if rsi <= 30 else "")
                ind.append(f"RSI14: {rsi:.1f}{tag}")
            # MACD histogram — trend-change momentum
            mh = _macd_hist(close)
            if mh is not None:
                ind.append(f"MACD_hist: {mh:+.1f}")
            # 20일 실현변동성 (annualized)
            if len(close) >= 21:
                try:
                    rv = float(close.pct_change().tail(20).std() * (252 ** 0.5) * 100)
                    ind.append(f"σ20(연환산): {rv:.1f}%")
                except Exception: pass
            # 52주 신고가 근접도
            if len(daily) >= 252:
                high52 = float(daily['high'].astype(float).tail(252).max())
                cur = float(close.iloc[-1])
                if high52 > 0:
                    ind.append(f"52w고가 근접도: {cur/high52*100:.1f}%")
            # 1개월 / 3개월 모멘텀
            try:
                if len(close) >= 21:
                    ind.append(f"1M수익률: {(close.iloc[-1]/close.iloc[-21]-1)*100:+.1f}%")
                if len(close) >= 63:
                    ind.append(f"3M수익률: {(close.iloc[-1]/close.iloc[-63]-1)*100:+.1f}%")
            except Exception: pass
            # VWAP (20일 거래량 가중 평균가) — 현재가 vs 기관성 평균가 비교용
            vwap = _vwap_20d(daily)
            if vwap is not None and vwap > 0:
                cur = float(close.iloc[-1]); dev = (cur/vwap - 1) * 100
                ind.append(f"VWAP20: {vwap:.0f}원 (현재가 {dev:+.1f}%)")
            # ADX(14) — 추세 강도
            adx_dict = _adx(daily, 14)
            if adx_dict is not None:
                a = adx_dict["adx"]; pdi = adx_dict["plus_di"]; mdi = adx_dict["minus_di"]
                strength = "강한 추세" if a >= 25 else ("약한 추세" if a >= 20 else "횡보")
                direction = "상승우위" if pdi > mdi else "하락우위"
                ind.append(f"ADX14: {a:.1f}({strength}/{direction})")
            lines.append("  " + " | ".join(ind))
    else:
        lines.append("  ⚠ 일봉 데이터 없음 — 분석 제한")

    if investor is not None and len(investor) > 0:
        recent_inv = investor.tail(5)
        lines.append(f"\n  수급 데이터: {len(investor)}일")
        for _, r in recent_inv.iterrows():
            lines.append(f"    {r['date'].strftime('%Y-%m-%d')}: 기관 {r['inst_net']:+,} | 외인 {r['foreign_net']:+,}")
        # 누적 추세 (최근 5일/20일)
        try:
            inst5 = int(investor['inst_net'].tail(5).sum()); frgn5 = int(investor['foreign_net'].tail(5).sum())
            lines.append(f"  최근 5일 누적: 기관 {inst5:+,} | 외인 {frgn5:+,}")
            if len(investor) >= 20:
                inst20 = int(investor['inst_net'].tail(20).sum()); frgn20 = int(investor['foreign_net'].tail(20).sum())
                lines.append(f"  최근 20일 누적: 기관 {inst20:+,} | 외인 {frgn20:+,}")
        except Exception: pass
    else:
        lines.append("\n  ⚠ 수급 데이터 없음")

    return "\n".join(lines)


# ═══════════════════ Utilities ═══════════════════

def _get_csv_latest_date(path: Path) -> Optional[pd.Timestamp]:
    """CSV에서 가장 최근 날짜 가져오기 (중복 방지용)"""
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
        if df.empty:
            return None
        return pd.to_datetime(df['date']).max()
    except:
        return None


def _append_csv(path: Path, rows: List[Dict], columns: List[str]):
    """CSV upsert — 키(columns[0], 보통 date/datetime)가 같은 행은 **덮어쓴다**.
    당일 봉이 첫 조회값에 고정되던 버그(2026-06-15) 수정: 같은 날짜를 스킵하지 않고 갱신해
    장중 거래량·종가가 최신화되고, 장 마감 후 완성봉으로 확정된다. 겹침이 없으면 기존처럼 append."""
    if not rows:
        return
    key = columns[0]
    new_keys = {str(r.get(key, '')) for r in rows if str(r.get(key, '')) != ''}
    existing = None
    seen = set()
    if path.exists():
        try:
            existing = pd.read_csv(path)
            seen = set(existing[key].astype(str))
        except Exception:
            existing = None
    overlap = new_keys & seen
    if not overlap:
        # 빠른 경로 — 겹치는 날짜 없음 → 신규 행만 append
        fresh = [r for r in rows if str(r.get(key, '')) != '' and str(r.get(key, '')) not in seen]
        if not fresh:
            return
        exists = path.exists()
        with open(path, 'a', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=columns)
            if not exists:
                w.writeheader()
            w.writerows(fresh)
        return
    # 겹침 → 동일 키 덮어쓰기 (read-modify-write)
    new_df = pd.DataFrame(rows)[columns].drop_duplicates(subset=[key], keep='last')
    base = existing[~existing[key].astype(str).isin(new_keys)] if existing is not None else None
    merged = pd.concat([base, new_df], ignore_index=True) if base is not None else new_df
    merged = merged.sort_values(key).reset_index(drop=True)
    merged.to_csv(path, index=False, columns=columns)
