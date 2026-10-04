"""Portfolio-wide entry budget, independent of committee confidence or ops tuning.

Cash and exposures are KRW equivalents. Risk is a sizing estimate, not a loss
guarantee: gaps and execution delays can exceed the assumed stop distance.
"""
import math


def _positive(value):
    try:
        value = float(value)
        return value if math.isfinite(value) and value > 0 else 0.0
    except (TypeError, ValueError):
        return 0.0


def constrain_entries(orders, *, prices, buying_power, holdings,
                      approved_stock_codes, sigmas=None, stop_loss_pct=5.0,
                      usdkrw=0.0, is_sleeve=lambda code: False):
    """Keep exits intact; shrink entries to concentration, risk and cash budgets.

    Fixed policy: stock <=10% NAV, risk <=0.5% NAV/name and <=3% NAV total,
    keep 10% NAV cash. Stops use at least 5% or two daily standard deviations,
    plus round-trip costs (KR .35%, US .70%, assumptions). Existing positions
    consume budgets; submitted sells do not release cash until actually filled.
    Sleeve ETFs retain their separate allocation limits but share the cash cap.
    """
    total = _positive(buying_power.get("total_eval"))
    cash = _positive(buying_power.get("cash"))
    remaining_cash = max(0.0, cash - total * .10)
    sigmas = sigmas or {}
    approved = set(approved_stock_codes or [])
    exposure, risk = {}, {}
    unknown_holding = False

    def rate(code):
        cost = .0035 if code.isdigit() and len(code) == 6 else .0070
        daily_sigma = _positive(sigmas.get(code)) / 100 / math.sqrt(252)
        return max(.05, _positive(stop_loss_pct) / 100, 2 * daily_sigma) + cost

    def fx(code):
        return 1.0 if code.isdigit() and len(code) == 6 else _positive(usdkrw)

    for h in holdings:
        code = str(h.get("code") or h.get("ticker") or "").strip()
        qty = _positive(h.get("qty"))
        if not qty:
            continue
        value = qty * _positive(h.get("cur_price")) * fx(code)
        if value <= 0:
            unknown_holding = True
            continue
        exposure[code] = exposure.get(code, 0) + value
        if not is_sleeve(code):
            risk[code] = exposure[code] * rate(code)
    remaining_risk = max(0.0, total * .03 - sum(risk.values()))
    output, notes = [], []
    for order in orders:
        if str(order.get("side", "buy")).lower() == "sell":
            output.append(dict(order))
            continue
        code = str(order.get("ticker") or "").strip()
        if order.get("entry_mode") == "watch":
            notes.append(f"{code}: 관망 지시 유지, 다음 사이클 재평가")
            continue
        sleeve = is_sleeve(code)
        if not sleeve and code not in approved:
            notes.append(f"{code}: 최종 심의 목록에 없어 매수 보류")
            continue
        quoted = _positive(prices.get(code))
        limit = _positive(order.get("entry_limit")) if order.get("entry_mode") == "limit" else 0.0
        if order.get("entry_mode") == "limit" and not limit:
            notes.append(f"{code}: 지정가를 검증할 수 없어 매수 보류")
            continue
        price = max(quoted, limit) * fx(code) if quoted else 0.0
        requested = int(_positive(order.get("qty")))
        if unknown_holding or not buying_power.get("ok") or total <= 0 or price <= 0:
            notes.append(f"{code}: 잔고·가격을 검증할 수 없어 매수 보류")
            continue
        # Cash includes estimated exit costs too, so repeated drafts cannot
        # consume the reserve or use pending sale proceeds.
        cost = .0035 if code.isdigit() and len(code) == 6 else .0070
        cap = remaining_cash / (1 + cost)
        if not sleeve:
            cap = min(cap, max(0.0, total * .10 - exposure.get(code, 0)),
                      max(0.0, total * .005 - risk.get(code, 0)) / rate(code),
                      remaining_risk / rate(code))
        qty = min(requested, max(0, int(cap / price)))
        if qty != requested:
            notes.append(f"{code}: 위험 예산·현금 한도로 {requested}주 → {qty}주")
        if not qty:
            continue
        value = qty * price
        remaining_cash -= value * (1 + cost)
        exposure[code] = exposure.get(code, 0) + value
        if not sleeve:
            risk[code] = risk.get(code, 0) + value * rate(code)
            remaining_risk -= value * rate(code)
        output.append(dict(order, qty=qty))
    return output, notes
