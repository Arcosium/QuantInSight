"""OpenDART 재무위험 캐시 — 일일 precompute, 핫패스는 dict 조회만(지연 0).

재무는 분기/연간 단위라 하루 1회 갱신으로 충분하다. LLM 은 느려서 못 쓰지만(사장 지시)
OpenDART API 는 종목당 재무제표 1콜이라, 유니버스 전체를 하루 한 번 백그라운드로 훑어
{종목: 위험플래그} 를 캐시하면 엔진은 즉시 조회로 '자본잠식·고부채·영업손실' 부실주를 매수
후보에서 뺄 수 있다. 잔차가 저평가여도 재무부실이면 반등이 아니라 지속 하락(낙하칼)이기 때문.

fail-soft: 키 없음·네트워크 실패·매핑 없음 → 빈 위험집합 → 게이트 무효(거래는 계속). 절대 죽지 않는다.
계정명 매핑 함정(2026-07: 부분일치가 완전일치를 이김) 회피 위해 account_nm **완전일치**만 쓴다.
"""
import json
import logging
import os
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import requests

import config
from utils import market_time as mt

log = logging.getLogger("lag.dart")

CACHE = config.DATA_DIR / "dart_risk.json"
CORP_XML = Path(os.environ.get("OPENDART_CORP_XML") or
                (config.PRIVATE_DATA_DIR / "opendart_corp_codes.xml"))
DART_URL = "https://opendart.fss.or.kr/api/fnlttSinglAcntAll.json"
_WANT = ("자본총계", "자본금", "부채총계", "자산총계", "영업이익")
_FIN_KW = ("은행", "증권", "보험", "금융", "지주", "캐피")   # 예금·보험부채로 부채비율이 구조적 高 → 부채비율 기준 제외

_mem = {"mtime": -1.0, "risky": set()}   # 파일 mtime 캐시 — compute_scores 매분 호출에도 재파싱 안 함


def _api_key():
    k = (os.environ.get("OPENDART_API_KEY") or os.environ.get("DART_API_KEY") or "").strip()
    if k:
        return k
    return ""


def _sector_map():
    """universe.csv → {6자리 티커: 섹터}. 금융사 부채비율 예외 판정용."""
    out = {}
    try:
        import csv
        with open(config.UNIVERSE_CSV, encoding="utf-8") as f:
            for row in csv.reader(f):
                if len(row) >= 4 and row[0] != "code":
                    out[row[0].zfill(6)] = row[3]
    except OSError:
        pass
    return out


def _is_financial(sector):
    return any(k in (sector or "") for k in _FIN_KW)


def _corp_map():
    """OpenDART corpCode.xml → {6자리 티커: 8자리 corp_code}."""
    out = {}
    try:
        for e in ET.parse(CORP_XML).getroot().iter("list"):
            sc = (e.findtext("stock_code") or "").strip()
            cc = (e.findtext("corp_code") or "").strip()
            if sc and cc and sc.strip("0"):
                out[sc.zfill(6)] = cc
    except Exception:                     # noqa: BLE001 — 매핑 실패 = 게이트 무효(fail-soft)
        log.warning("corp_code 매핑 로드 실패 — 재무위험 게이트 비활성")
    return out


def _fetch(key, corp_code, year):
    """fnlttSinglAcntAll(연간 사업보고서). CFS(연결) 우선, 없으면 OFS(별도)."""
    for fs in ("CFS", "OFS"):
        params = {"crtfc_key": key, "corp_code": corp_code, "bsns_year": str(year),
                  "reprt_code": "11011", "fs_div": fs}
        try:
            d = requests.get(DART_URL, params=params, timeout=12).json()
        except Exception:                 # noqa: BLE001
            return None
        if d.get("status") != "000":
            continue
        acc = {}
        for it in d.get("list") or []:
            nm = (it.get("account_nm") or "").strip()
            if nm in _WANT and nm not in acc:      # 완전일치만(부분일치 함정 회피)
                try:
                    acc[nm] = float(str(it.get("thstrm_amount") or "").replace(",", ""))
                except ValueError:
                    pass
        if acc:
            return acc
    return None


def _risk_of(acc, is_financial=False):
    eq, cap, li = acc.get("자본총계"), acc.get("자본금"), acc.get("부채총계")
    debt_ratio = None
    if li is not None and eq is not None:
        debt_ratio = (li / eq) if eq > 0 else 999.0    # 자본총계<=0(자본잠식) → 부채비율 무한대 취급
    neg_eq = eq is not None and eq <= 0                       # 완전 자본잠식
    impaired = eq is not None and cap is not None and eq < cap  # 부분 자본잠식(자본총계<자본금)
    op = acc.get("영업이익")
    op_loss = op is not None and op < 0
    # 금융사(은행·보험·증권·지주)는 부채비율이 구조적으로 높으므로 부채비율 기준을 적용하지 않는다.
    high_debt = (not is_financial) and (debt_ratio is not None and debt_ratio > config.DART_MAX_DEBT_RATIO)
    # 부실 판정: 완전 자본잠식 / (비금융) 극단 고부채 / (부분 자본잠식 & 영업손실). 명백한 부실만.
    risky = bool(neg_eq or high_debt or (impaired and op_loss))
    return {"risky": risky, "debt_ratio": round(debt_ratio, 2) if debt_ratio is not None else None,
            "neg_equity": bool(neg_eq), "impaired": bool(impaired), "op_loss": bool(op_loss),
            "financial": bool(is_financial)}


def refresh(tickers, year=None):
    """유니버스 재무위험 재계산 → 캐시. 느리다(종목당 1~2 API콜) — 백그라운드에서만 부른다."""
    key = _api_key()
    if not key:
        log.warning("OPENDART_API_KEY 없음 — 재무위험 게이트 비활성(fail-soft)")
        return {}
    cmap = _corp_map()
    smap = _sector_map()
    year = year or (mt.now_kst().year - 1)
    risk, n = {}, 0
    for t in tickers:
        tk = str(t).zfill(6)
        cc = cmap.get(tk)
        if not cc:
            continue
        try:                              # 종목 하나 실패가 배치를 죽이면 안 된다(fail-soft)
            acc = _fetch(key, cc, year) or _fetch(key, cc, year - 1)
            if acc:
                risk[tk] = _risk_of(acc, _is_financial(smap.get(tk)))
                n += 1
        except Exception:                 # noqa: BLE001
            log.debug("DART 재무 파싱 실패: %s", t, exc_info=True)
        time.sleep(0.03)                  # OpenDART 예의상 간격
    try:
        CACHE.write_text(json.dumps({"built": mt.now_kst().isoformat(timespec="seconds"),
                                     "year": year, "risk": risk}, ensure_ascii=False), encoding="utf-8")
    except OSError:
        pass
    log.info("DART 재무위험 캐시: %d/%d 종목 조회 · 부실 %d", n, len(tickers),
             sum(1 for v in risk.values() if v["risky"]))
    return risk


def risky_set():
    """부실주 티커 집합 — compute_scores 가 매분 부른다(파일 mtime 캐시라 재파싱 없음)."""
    if not config.DART_RISK_ENABLED:
        return set()
    try:
        mtime = CACHE.stat().st_mtime
    except OSError:
        return set()
    if mtime != _mem["mtime"]:
        try:
            d = json.loads(CACHE.read_text(encoding="utf-8"))
            _mem["risky"] = {t for t, v in (d.get("risk") or {}).items() if v.get("risky")}
            _mem["mtime"] = mtime
        except (OSError, ValueError):
            _mem["risky"] = set()
    return _mem["risky"]


def cache_age_hours():
    try:
        return (time.time() - CACHE.stat().st_mtime) / 3600.0
    except OSError:
        return 1e9


if __name__ == "__main__":                # 수동 캐시 빌드: python3.12 -m core.dart_risk
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from core import crawler
    codes = [c for c, _ in crawler.load_universe()]
    r = refresh(codes)
    print("부실주:", sorted(t for t, v in r.items() if v["risky"])[:20])
