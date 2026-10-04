"""Export the research scatter figure; no dashboard or trading-service imports.

python3 -m research.plot_news_swarm --run data/swarm_research/<run>
Requires matplotlib in the research environment only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


def plot(run: Path) -> Path:
    frame = pd.read_csv(run / "evaluation.csv", dtype={"code": str})
    report = json.loads((run / "report.json").read_text())
    frame = frame[frame.horizon == 1]
    first_test = report["first_test_date"]
    train = frame[(frame.date < first_test) & (frame.label_end_date < first_test)]
    test = frame[frame.date >= first_test]
    if train.empty or test.empty:
        raise ValueError("Both training and chronological test samples are required")
    font_manager.fontManager.addfont("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
    font = font_manager.FontProperties(fname="/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc").get_name()
    plt.rcParams.update({"font.family": font, "font.size": 11, "axes.unicode_minus": False,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": "#b7b7b7", "axes.linewidth": .6,
                         "text.color": "#222222", "axes.labelcolor": "#333333"})
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.3), dpi=180)
    fig.subplots_adjust(left=.085, right=.97, top=.73, bottom=.22, wspace=.28)
    fig.text(.085, .91, "뉴스·공시 Swarm과 다음 날의 수급·수익률", size=19, weight="bold")
    fig.text(.085, .835, f"과거 탐색 실험  |  {frame.code.nunique()}종목  |  학습 {len(train)}개 · 검증 {len(test)}개  |  검증 시작 {first_test}",
             size=10, color="#666666")
    for ax, score, target, ylabel in zip(axes, ["swarm_flow", "swarm_flow"],
                                       ["residual_ratio", "price_return"],
                                       ["역산 순매수 / 거래량 (%)", "다음 날 종가 수익률 (%)"]):
        ax.axhline(0, color="#bbbbbb", linewidth=.6)
        ax.axvline(0, color="#bbbbbb", linewidth=.6)
        ax.scatter(train[score], train[target] * 100, s=24, facecolors="none", edgecolors="#888888", linewidths=.7, label="학습")
        ax.scatter(test[score], test[target] * 100, s=27, color="#9a3838", alpha=.9, label="이후 기간 검증", zorder=3)
        matrix = np.column_stack([np.ones(len(train)), train[score]])
        beta = np.linalg.lstsq(matrix, train[target].to_numpy(), rcond=None)[0]
        x = np.linspace(frame[score].min(), frame[score].max(), 100)
        ax.plot(x, (beta[0] + beta[1] * x) * 100, color="#222222", linewidth=1, label="학습 회귀선")
        ax.set_xlabel("Swarm 매수 압력", labelpad=10)
        ax.set_ylabel(ylabel, labelpad=8)
        for tick in [*ax.get_xticklabels(), *ax.get_yticklabels()]:
            tick.set_fontfamily("DejaVu Sans Mono")
        ax.grid(axis="y", color="#dddddd", linewidth=.4)
        ax.set_axisbelow(True)
    axes[0].legend(loc="best", frameon=False, fontsize=9)
    fig.text(.085, .09, "입력: 뉴스 제목·요약 + 공시 제목. 역산값 = −(기관 순매수 + 외국인 순매수), 수량 기준.", size=9, color="#666666")
    fig.text(.085, .04, f"정답 기준 {frame.label_end_date.max()} · QuantInSight 뉴스·DART·네이버 시세/수급. 로컬 4인 모형, MiroFish 원본 아님. 과거정보 학습 가능성 있음.",
             size=8.5, color="#666666")
    path = run / "scatter.png"
    fig.savefig(path, facecolor="white")
    fig.savefig(run / "scatter.pdf", facecolor="white")
    plt.close(fig)
    with Image.open(path) as im:
        im.resize((im.width // 4, im.height // 4), Image.Resampling.LANCZOS).save(run / "scatter-quarter.png")
    # A responsive local preview lets the house checker verify image delivery at
    # desktop/mobile sizes. The exported scientific figure remains standalone.
    preview = '''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>뉴스·공시 Swarm 검증 도판</title><style>*{box-sizing:border-box}body{margin:0;background:white;color:#222;font-family:"Noto Sans CJK KR",sans-serif}
main{max-width:1280px;margin:auto;padding:32px 20px}h1{font-size:24px;margin:0 0 16px}p{font-size:15px;line-height:1.6;word-break:keep-all}img{display:block;width:100%;height:auto}</style>
<main><h1>뉴스·공시 Swarm 검증 도판</h1><p>학습에 사용한 기간과 이후 검증 기간을 구분했습니다. 과거 탐색 실험이며 실전 수익을 검증한 결과는 아닙니다.</p>
<img src="scatter.png" alt="Swarm 점수와 다음 날 역산 순매수 및 수익률의 산점도"></main></html>'''
    (run / "chart-preview.html").write_text(preview)
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    print(plot(Path(parser.parse_args().run)))
