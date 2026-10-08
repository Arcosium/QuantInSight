from datetime import datetime,timezone,timedelta
from typing import Optional
KST=timezone(timedelta(hours=9))

def _ts_to_kst(ts_str: str) -> Optional[datetime]:
    """Parse an ISO ('2026-05-13T17:13:57+09:00') or new-format ('2026-05-13 17:13:57') timestamp.
    Legacy ISO entries without tz are treated as UTC (the OCI server runs UTC, old format was naive UTC).
    New-format entries (space separator) are already KST (we strftime in KST). Returns KST-aware datetime."""
    s = (ts_str or "").strip()
    if not s:
        return None
    try:
        if "T" in s or "+" in s or "Z" in s:
            dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(KST)
        dt = datetime.strptime(s[:19], "%Y-%m-%d %H:%M:%S")
        return dt.replace(tzinfo=KST)
    except Exception:
        return None

def _equity_points(raw_equity, *, glitch_pct: float = 0.02):
    """정렬된 [(dt, adj_total, point)] — 입출금 보정(adj) 적용 + 결제 글리치 carry-forward.

    사장 지시 2026-06-11(수익률 환각 수정): 실거래 원장 평가(ledger_eval)가 기록된 포인트가
    하나라도 있으면 '그 시리즈만' 사용한다. KIS 집계 TR(총평가)은 3종 TR 자기불일치·USD 결제
    과도기·해외평가 증발(외화예수금 미포함)로 가짜 -43%류 수익률을 만들었다 — 원장 평가는
    우리 체결만으로 굴러가 결정론적이다. (원장 포인트가 없으면 기존 KIS 곡선 로직 유지.)

    글리치 판정(사장 보고 2026-09-25 '곡선이 계속 튄다'로 교체): 직전 채택값에서 glitch_pct 넘게
    벗어났다가 **다음 세 포인트(약 15분) 안에 되돌아오는 점**만 글리치로 보고 직전 값을 유지한다.
    모의 총평가는 국내 + 미국 주식 − USD 부채처럼 부호가 반대인 큰 덩어리의 합이라 한 덩어리가
    한 폴 빠지면 순액이 수~수십 % 튀고, 그 폴은 보유목록 변동과 자주 겹친다(uid2 9/8~9/25:
    3% 넘는 점프 90건). 그래서 종전의 '보유 변동이면 정상'·'10% 문턱'은 버린다. 종전의
    '보유 동일이면 즉시 유지'도 버린다 — 되돌림을 안 봐서 실제 가격 변동을 동결할 수 있었다.
    되돌아오지 않는 변동(실제 손익·장부 정정)은 그대로 보존하고, 마지막 점은 판단을 보류한다."""
    ledger_pts = []
    for p in (raw_equity or []):
        if not isinstance(p, dict):
            continue
        try:
            lv = float(p.get("ledger_eval") or 0.0)
        except (TypeError, ValueError):
            lv = 0.0
        if lv <= 0:
            continue
        dt = _ts_to_kst(p.get("ts", ""))
        if dt:
            ledger_pts.append((dt, lv, {**p, "adj_total_eval": lv}))
    if ledger_pts:
        ledger_pts.sort(key=lambda x: x[0])
        return ledger_pts
    enriched = []
    for p in (raw_equity or []):
        if not isinstance(p, dict) or not p.get("total_eval"):
            continue
        dt = _ts_to_kst(p.get("ts", ""))
        if not dt:
            continue
        try:
            ext = float(p.get("external_flow_cum", 0.0) or 0.0)
        except Exception:
            ext = 0.0
        try:
            adj = float(p["total_eval"]) - ext
        except Exception:
            continue
        enriched.append((dt, adj, p))
    enriched.sort(key=lambda x: x[0])
    out = []
    prev = None
    for i, (dt, adj, p) in enumerate(enriched):
        use = adj
        if prev and prev > 0 and abs(adj - prev) / prev > glitch_pct:
            ahead = enriched[i + 1:i + 4]
            if any(abs(a - prev) / prev <= glitch_pct for _dt, a, _p in ahead):
                use = prev  # 일시 스파이크 — 직전 값 유지
        out.append((dt, use, {**p, "adj_total_eval": use}))
        prev = use
    return out

def curve(raw):
    points=_equity_points(raw)
    if not points or points[0][1]<=0:return []
    base=points[0][1]
    initial_correction=float(points[0][2].get('reconcile_adj') or 0)
    corrections=0.
    daily={}
    for dt,value,row in points:
        corrections+=float(row.get('reconcile_adj') or 0)
        daily[dt.strftime('%Y%m%d')]=dict(date=dt.strftime('%Y%m%d'),net_return=(value-base-corrections+initial_correction)/base)
    return list(daily.values())
