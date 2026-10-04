"""결정론 백테스트 엔진 — 현재 전략 파라미터의 *리스크/청산 규칙*을 과거 일봉에 적용.

⚠️ 정직성 경계 (반드시 읽을 것)
────────────────────────────────────────────────────────────────────────
실제 ArQuant 의 종목 선정은 9개 LLM 에이전트의 2패스 협업이라 **오프라인
재현이 불가능**하다. 그러므로 이 백테스트는 종목 선정 능력을 평가하지
**않는다**. 대신:

  • 진입 신호는 **고정된 투명한 모멘텀 프록시**(SMA 상향 돌파)로 통일하고,
  • 변하는 것은 오직 전달된 **전략 파라미터의 결정론 규칙**
    (사이징·익절·손절·단일종목 한도·사이클 예산·MDD 차단)뿐이다.

→ 따라서 결과의 *절대 수익률* 이 아니라, **동일 신호 위에서 현재 설정의 규칙이
   리스크/MDD/회전율을 어떻게 만드는지**의 측정으로 해석해야 한다(주간 피드백에서
   운용지원실장이 파라미터를 조정하는 입력으로 쓰인다).
────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Dict, List

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


# ── 데이터 로딩 ───────────────────────────────────────────────────────────
def load_prices(data_dir: Path = DATA_DIR) -> Dict[str, List[dict]]:
    """data/daily_<code>.csv → {code: [{date,open,high,low,close,volume}, ...]} (날짜 오름차순).

    CSV 는 최신이 위(내림차순)이므로 시간순 워크포워드를 위해 **오름차순으로
    재정렬**한다 — 룩어헤드 편향 방지의 핵심.
    """
    out: Dict[str, List[dict]] = {}
    for fp in sorted(data_dir.glob("daily_*.csv")):
        code = fp.stem.replace("daily_", "")
        rows: List[dict] = []
        with fp.open(encoding="utf-8") as f:
            for r in csv.DictReader(f):
                try:
                    rows.append({"date": r["date"],
                                 "close": float(r["close"]),
                                 "high": float(r["high"]),
                                 "low": float(r["low"])})
                except (KeyError, ValueError):
                    continue
        rows.sort(key=lambda x: x["date"])  # 오름차순 — 과거 → 현재
        if len(rows) > 30:
            out[code] = rows
    return out


# ── 진입 신호 (고정 프록시 — LLM 픽 대체) ────────────────────────────────
def sma_breakout_signal(closes: List[float], i: int, lookback: int = 20) -> bool:
    """i 일자에 'SMA(lookback) 상향 신규 돌파'면 True. 결정론·룩어헤드 없음."""
    if i < lookback:
        return False
    window = closes[i - lookback:i]
    sma = sum(window) / lookback
    prev_sma = sum(closes[i - lookback - 1:i - 1]) / lookback if i > lookback else sma
    return closes[i - 1] <= prev_sma and closes[i] > sma


# ── 진입 적합 필터 (결정론 매수 규칙 — 라이브 퀀트 프롬프트와 동형) ──────────
# 라이브 계량분석은 아래 3개 하드필터로 추격·고변동·과매수 진입을 막는다(quant_report
# "매수 필터 위반: 변동성 149.6%" / "VWAP 대비 16% 이격" 사례). config 의 동명 키가
# 0(off) 이면 이 기본선을, 0보다 크면 그 값을 상한으로 쓴다 → ops 가 키를 조이면
# 백테스트도 즉시 반응(2026-06-15: 엔진이 키를 무시하던 피드백 단절 수정).
LIVE_BASELINE_VOL_PCT = 100.0   # 연환산 변동성 상한
LIVE_BASELINE_RSI     = 70.0    # RSI(14) 과매수 상한
LIVE_BASELINE_EXT_PCT = 10.0    # SMA(lookback) 대비 이격 상한(추격매수 회피)


def _annual_vol_pct(closes: List[float], i: int, lookback: int) -> float:
    """i 일까지 lookback 일 로그수익률의 표준편차 × √252 × 100 (연환산 변동성 %)."""
    if i < lookback:
        return 0.0
    rets = [math.log(closes[j] / closes[j - 1])
            for j in range(i - lookback + 1, i + 1)
            if closes[j - 1] > 0 and closes[j] > 0]
    if len(rets) < 2:
        return 0.0
    mean = sum(rets) / len(rets)
    var = sum((r - mean) ** 2 for r in rets) / len(rets)
    return math.sqrt(var) * math.sqrt(252) * 100.0


def _rsi(closes: List[float], i: int, period: int = 14) -> float:
    """단순 RSI(14). 데이터 부족 시 중립 50."""
    if i < period:
        return 50.0
    gains = losses = 0.0
    for j in range(i - period + 1, i + 1):
        ch = closes[j] - closes[j - 1]
        if ch >= 0:
            gains += ch
        else:
            losses -= ch
    if losses == 0:
        return 100.0
    rs = (gains / period) / (losses / period)
    return 100.0 - 100.0 / (1.0 + rs)


def _price_extension_pct(closes: List[float], i: int, lookback: int) -> float:
    """현재가가 SMA(lookback) 대비 몇 % 위인지(이격도). 음수면 0 으로 본다."""
    if i < lookback:
        return 0.0
    sma = sum(closes[i - lookback:i]) / lookback
    if sma <= 0:
        return 0.0
    return max(0.0, (closes[i] / sma - 1.0) * 100.0)


def passes_entry_filters(closes: List[float], i: int, lookback: int, params: dict) -> bool:
    """결정론 매수 적합 필터. 고변동·과매수·과이격이면 False(라이브 매수회피와 동형)."""
    vol_ceil = params.get("MAX_BUY_VOLATILITY_PCT") or LIVE_BASELINE_VOL_PCT
    rsi_ceil = params.get("RSI_OVERBOUGHT_SKIP") or LIVE_BASELINE_RSI
    ext_ceil = params.get("MAX_PRICE_EXTENSION_PCT") or LIVE_BASELINE_EXT_PCT
    if _annual_vol_pct(closes, i, lookback) > vol_ceil:
        return False
    if _rsi(closes, i) > rsi_ceil:
        return False
    if _price_extension_pct(closes, i, lookback) > ext_ceil:
        return False
    return True


# ── 백테스트 ──────────────────────────────────────────────────────────────
def run_backtest(params: dict,
                 prices: Dict[str, List[dict]],
                 start_cash: float = 10_000_000.0,
                 lookback: int = 20,
                 name: str = "현재 설정") -> dict:
    """전략 파라미터 1세트를 전 종목 공통 신호로 시뮬레이션. 반환: 성과 지표 dict.

    규칙 매핑(라이브 시맨틱 그대로):
      PER_ORDER_BUDGET_RATIO     → 1주문 = 가용현금 × 비율
      CONSERVATIVE_STOCK_RATIO   → 단일 종목 평가액 한도 (초과 시 진입 스킵)
      MAX_CYCLE_BUDGET_RATIO     → 하루(=1사이클) 총 매수 한도
      MIN_CASH_BUFFER            → 노티오날 × 버퍼 ≤ 현금이어야 진입
      CONSERVATIVE_MDD           → 계좌 평가손익 ≤ -MDD 면 신규 매수 전면 차단
      TAKE_PROFIT_PCT/STOP_LOSS_PCT → 보유 수익률 기준 전량 청산
    """
    p = params
    cash = start_cash
    holdings: Dict[str, dict] = {}        # code -> {qty, avg, peak_date}
    equity_curve: List[float] = []
    trades = wins = 0

    # 공통 날짜 축 (가장 긴 종목 기준).
    all_dates = sorted({d["date"] for rows in prices.values() for d in rows})
    closes_by_code = {c: [r["close"] for r in rows] for c, rows in prices.items()}
    date_idx = {c: {r["date"]: k for k, r in enumerate(rows)} for c, rows in prices.items()}

    for day in all_dates:
        # 1) 보유 평가 + 익절/손절.
        port_val = cash
        for code in list(holdings.keys()):
            k = date_idx[code].get(day)
            if k is None:
                port_val += holdings[code]["qty"] * holdings[code]["avg"]
                continue
            px = closes_by_code[code][k]
            h = holdings[code]
            pnl_pct = (px / h["avg"] - 1.0) * 100.0
            if pnl_pct >= p["TAKE_PROFIT_PCT"] or pnl_pct <= -p["STOP_LOSS_PCT"]:
                cash += h["qty"] * px
                trades += 1
                wins += 1 if pnl_pct > 0 else 0
                del holdings[code]
            else:
                port_val += h["qty"] * px
        total_eval = cash + sum(
            holdings[c]["qty"] * closes_by_code[c][date_idx[c][day]]
            for c in holdings if day in date_idx[c])

        equity_curve.append(total_eval)

        # 2) 계좌 MDD 차단 — 시작자본 대비 손실이 한도 초과면 신규 매수 금지.
        acct_pnl = total_eval / start_cash - 1.0
        cycle_spent = 0.0
        if acct_pnl <= -abs(p["CONSERVATIVE_MDD"]):
            continue

        # 3) 신규 진입 (신호 발생 + 한도 통과).
        for code, rows in prices.items():
            k = date_idx[code].get(day)
            if k is None or code in holdings:
                continue
            if not sma_breakout_signal(closes_by_code[code], k, lookback):
                continue
            if not passes_entry_filters(closes_by_code[code], k, lookback, p):
                continue
            px = closes_by_code[code][k]
            budget = cash * p["PER_ORDER_BUDGET_RATIO"]
            qty = int(budget // px)
            if qty <= 0:
                continue
            notional = qty * px
            if notional * p["MIN_CASH_BUFFER"] > cash:
                continue
            if notional > total_eval * p["CONSERVATIVE_STOCK_RATIO"]:
                continue
            if cycle_spent + notional > cash * p["MAX_CYCLE_BUDGET_RATIO"]:
                continue
            cash -= notional
            cycle_spent += notional
            holdings[code] = {"qty": qty, "avg": px}
            trades += 1

    # 청산가치로 마감.
    final = equity_curve[-1] if equity_curve else start_cash
    return _metrics(name, start_cash, final, equity_curve, trades, wins)


def _metrics(name, start, final, curve, trades, wins) -> dict:
    total_ret = (final / start - 1.0) * 100.0
    peak = -math.inf
    mdd = 0.0
    for v in curve:
        peak = max(peak, v)
        if peak > 0:
            mdd = min(mdd, v / peak - 1.0)
    # 샤프*는 **로그수익률** 기반 — 산술평균을 쓰면 고변동 구간에서 변동성 드래그로
    # '음수 복리수익 + 양수 샤프'라는 모순이 나와 돈 잃는 설정을 '위험조정 양호'로
    # 오인하게 만든다(2026-06-15 토요일 리뷰 -12.5%/+0.77 사례). 로그수익 평균의
    # 부호 = log(final/start)의 부호 = total_return 의 부호 → 항상 일치한다.
    rets = [math.log(curve[i] / curve[i - 1])
            for i in range(1, len(curve)) if curve[i - 1] > 0 and curve[i] > 0]
    mean = sum(rets) / len(rets) if rets else 0.0
    var = sum((r - mean) ** 2 for r in rets) / len(rets) if rets else 0.0
    vol = math.sqrt(var)
    sharpe_like = (mean / vol * math.sqrt(252)) if vol else 0.0
    return {"name": name,
            "total_return_pct": round(total_ret, 2),
            "max_drawdown_pct": round(mdd * 100.0, 2),
            "sharpe_like": round(sharpe_like, 2),
            "trades": trades,
            "win_rate_pct": round(wins / trades * 100.0, 1) if trades else 0.0,
            "final_eval": round(final)}
