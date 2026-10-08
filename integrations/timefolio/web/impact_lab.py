"""HYFE market-impact research and isolated paper experiments."""
import json
import os
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, ConfigDict

from quant.market_impact import simulate

ROOT = Path(os.environ.get("ARCTRADE_IMPACT_DIR", str(Path.home()/"vault"/"ArcTrade"/"market_impact")))
STATIC = Path(__file__).resolve().parent/"static"
router = APIRouter(prefix="/api/impact", tags=["paper market impact"])


def load_study():
    try:
        data = json.loads((ROOT/"study.json").read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise HTTPException(503, "분석 결과가 아직 준비되지 않았습니다.")
    except (OSError, ValueError):
        raise HTTPException(503, "분석 결과를 읽을 수 없습니다.")
    if data.get("schema_version") != 1:
        raise HTTPException(503, "지원하지 않는 분석 자료 형식입니다.")
    return data


@router.get("/study")
def study():
    result = load_study()
    # The selected public snapshots are included; raw streams and paths are not.
    result.pop("source_manifest", None)
    return result


class Experiment(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    snapshot_id: str = Field(min_length=1, max_length=40)
    side: Literal["buy", "sell"] = "buy"
    quantity: float = Field(default=5, ge=.00001, le=1e9)
    slices: int = Field(default=5, ge=2, le=20)
    horizon: float = Field(default=5, ge=.5, le=10)
    refill: Literal["empirical", "none", "manual"] = "empirical"
    half_time: float | None = Field(default=None, ge=.01, le=300)
    fee_bps: float = Field(default=0, ge=0, le=100)


@router.post("/simulate")
def run_experiment(request: Experiment):
    data = load_study()
    snapshot = next((s for s in data["snapshots"] if s["id"] == request.snapshot_id), None)
    if snapshot is None:
        raise HTTPException(404, "해당 호가 표본이 없습니다.")
    if request.side != snapshot["event_side"]:
        raise HTTPException(422, "호가 상태 분류와 맞는 주문 방향의 표본을 선택하세요.")
    calibration = data["calibration"][snapshot["regime"]]
    if request.refill == "empirical":
        if not calibration["empirical"]:
            raise HTTPException(422, "이 상태의 회복 시간은 관찰창에서 추정하지 못했습니다. 회복 없음 또는 직접 가정을 선택하세요.")
        half_time = calibration["half_time_seconds"]
    elif request.refill == "manual":
        if request.half_time is None:
            raise HTTPException(422, "직접 가정할 잔량 부족분 반감기를 입력하세요.")
        half_time = request.half_time
    else:
        half_time = None
    try:
        result = simulate(snapshot, side=request.side, quantity=request.quantity,
                          slices=request.slices, horizon=request.horizon,
                          half_time=half_time, fee_bps=request.fee_bps)
    except ValueError as exc:
        raise HTTPException(422, str(exc))
    result.update(snapshot_id=snapshot["id"], snapshot_timestamp_ms=snapshot["timestamp_ms"],
                  calibration=calibration, refill_source=request.refill,
                  symbol=data["symbol"], regime=snapshot["regime"])
    return result


# Development preview deliberately has no trading workers or account imports.
preview_app = FastAPI(title="ArcTrade 시장충격 실험 미리보기", docs_url=None, redoc_url=None)
preview_app.include_router(router)


@preview_app.get("/")
def preview_index():
    return FileResponse(STATIC/"impact-lab.html")


preview_app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")
