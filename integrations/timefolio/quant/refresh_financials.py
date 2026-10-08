"""DART 재무제표 월간 갱신 — 5년치 재무 CSV를 재생성해 서버에 저장.

systemd user 타이머(arctrade-financials.timer)가 월 1회 실행한다. update_financial_data 가
전 상장사 재무를 DART 에서 배치 수집해 D-1y~D-5y_data.csv(엔진용) 로 쓴다.
추가로 5년치를 합친 financials_5y.csv(기준연도 컬럼 포함) 한 파일도 저장한다.

DART 키는 vault 에서만 온다(projects 안엔 비밀 금지). 순서: 환경변수 → vault .env.
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from quant import engine as qe

def _dart_key():
    return (os.environ.get("OPENDART_API_KEY") or os.environ.get("DART_API_KEY") or "").strip()


def _write_merged(base_path):
    """엔진용 D-1y~D-5y 를 합쳐 financials_5y.csv 한 파일로(기준연도 컬럼 추가)."""
    frames = []
    for year, key_name in qe.YEAR_MAPPING.items():
        fp = os.path.join(base_path, f"{key_name}.csv")
        if os.path.exists(fp):
            df = pd.read_csv(fp, encoding="utf-8-sig")
            df.insert(0, "기준연도", year)
            frames.append(df)
    if frames:
        out = os.path.join(base_path, "financials_5y.csv")
        pd.concat(frames, ignore_index=True).to_csv(out, index=False, encoding="utf-8-sig")
        print(f"[refresh] 통합 저장: {out} ({sum(len(f) for f in frames)}행)")


def main():
    key = _dart_key()
    if not key:
        print("[refresh] OPENDART_API_KEY 환경변수가 필요합니다", file=sys.stderr)
        sys.exit(1)
    base_path = os.path.dirname(os.path.abspath(qe.__file__))
    m = qe.QuantLogic()
    m.update_financial_data(key, progress_callback=lambda p, msg, *a: print(f"[refresh] {p}% {msg}"))
    _write_merged(base_path)
    print("[refresh] 완료")


if __name__ == "__main__":
    main()
