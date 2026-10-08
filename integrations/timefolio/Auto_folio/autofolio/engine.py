from __future__ import annotations

from pathlib import Path
from typing import Any

from . import ledger
from .models import BuyingPower, Holding
from .risk import validate_order_draft
from .sessions import current_session
from .sizing import assemble_orders


class AutoFolioEngine:
    def __init__(self, *, data_dir: Path | str = "data", uid: int = 1):
        self.data_dir = Path(data_dir)
        self.uid = uid

    def plan_cycle(
        self,
        *,
        target_codes: list[str],
        holdings: list[dict[str, Any]],
        buying_power: dict[str, Any],
        price_map: dict[str, float],
        sell_directives: dict[str, str] | None = None,
        session: str | None = None,
    ) -> dict[str, Any]:
        session = session or current_session()
        h = [Holding(**x) for x in holdings]
        bp = BuyingPower(**buying_power)
        order_obj, price_map, bp_dict = assemble_orders(
            target_codes=target_codes,
            holdings=h,
            buying_power=bp,
            price_map=price_map,
            session=session,
            sell_directives=sell_directives,
        )
        risk = validate_order_draft(order_obj, buying_power=bp_dict, price_map=price_map)
        approved_orders = [
            {**order_obj["orders"][i], **{"risk_status": r["status"]}}
            for i, r in enumerate(risk["results"])
            if r["status"] == "APPROVED" and i < len(order_obj["orders"])
        ]
        return {
            "session": session,
            "orders_planned": order_obj["orders"],
            "sizing_notes": order_obj.get("sizing_notes", []),
            "risk": risk,
            "approved_orders": approved_orders,
        }

    def seed_ledger(self, *, cash_krw: float, positions: dict[str, dict[str, Any]] | None = None) -> dict[str, Any]:
        return ledger.seed(self.data_dir, cash_krw=cash_krw, positions=positions, uid=self.uid)

    def apply_result(self, result: dict[str, Any]) -> bool:
        if not result.get("filled"):
            return False
        return ledger.apply_fill(
            self.data_dir,
            ticker=str(result.get("ticker") or ""),
            side=str(result.get("side") or ""),
            qty=int(result.get("qty") or 0),
            price=float(result.get("fill_price") or 0.0),
            uid=self.uid,
            note="timefolio_exec",
        )
