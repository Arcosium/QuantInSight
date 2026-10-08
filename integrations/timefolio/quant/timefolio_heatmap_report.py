"""Research figures and a Korean report, generated only after the sealed test."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd

from quant.timefolio_heatmap_data import ROOT, atomic_json
from quant.timefolio_heatmap_study import BASE, transform


def bootstrap_active(a, b, seed=20260928, draws=3000, block=5):
    """Paired circular moving-block bootstrap; days, not stock rows, are resampled."""
    av, bv = np.array([r["nav"] for r in a]), np.array([r["nav"] for r in b])
    ar = av / np.r_[1e9, av[:-1]] - 1; br = bv / np.r_[1e9, bv[:-1]] - 1
    rng = np.random.default_rng(seed); n = len(ar)
    starts = rng.integers(n, size=(draws, int(np.ceil(n / block))))
    ix = ((starts[..., None] + np.arange(block)) % n).reshape(draws, -1)[:, :n]
    diff = np.prod(1 + ar[ix], axis=1) - np.prod(1 + br[ix], axis=1)
    return {"method": "paired circular moving block bootstrap", "block_sessions": block, "draws": draws,
            "active_return": float(av[-1] / 1e9 - bv[-1] / 1e9),
            "ci95": [float(x) for x in np.quantile(diff, [.025, .975])],
            "probability_active_positive": float(np.mean(diff > 0)),
            "caution": "Short, regime-specific holdout; CI excludes neither metadata error nor selection bias."}


def setup_style():
    paths = font_manager.findSystemFonts()
    font = next((p for p in paths if "NotoSansCJK-Regular" in p), None)
    if font:
        font_manager.fontManager.addfont(font)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=font).get_name()
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": "#BBBBBB", "axes.labelcolor": "#222222", "text.color": "#222222",
                         "xtick.color": "#666666", "ytick.color": "#666666", "axes.unicode_minus": False,
                         "figure.facecolor": "#FAFAF7", "axes.facecolor": "#FAFAF7", "savefig.facecolor": "#FAFAF7"})


def save_figure(fig, path):
    fig.savefig(path, dpi=180, bbox_inches="tight")
    fig.savefig(path.with_name(path.stem + "_thumb.png"), dpi=45, bbox_inches="tight")
    plt.close(fig)


def report(root=ROOT, out=None):
    root = Path(root)
    out = Path(out or Path(__file__).resolve().parents[1] / "_workspace/timefolio_heatmap_20260928")
    out.mkdir(parents=True, exist_ok=True)
    if (out / "REPORT.md").exists():
        raise RuntimeError("Existing report is preserved; render to a new output directory.")
    result = json.loads((root / "holdout.json").read_text())
    screen = json.loads((root / "screen_summary.json").read_text())
    selection = json.loads((root / "selection.json").read_text())
    audit = json.loads((root / "data_audit.json").read_text())
    weight_audit = json.loads((root / "independent_weight_audit.json").read_text())
    news = json.loads((root / "news_readiness.json").read_text()) if (root / "news_readiness.json").exists() else None
    if (root / "runpod_receipt.json").exists():
        receipt = json.loads((root / "runpod_receipt.json").read_text())
        compute_note = (f"Runpod {receipt['gpu']} / {receipt['cloud']}, 고지 단가 시간당 ${receipt['cost_per_hour']:.4f}. "
                        f"이번 대여의 시간 기반 추정 비용은 ${receipt['estimated_cost_usd']:.3f}다. 결과 회수 후 종료 및 삭제 확인: {receipt['deletion_verified']}.")
        cost = receipt["estimated_cost_usd"]
    else:
        receipt = json.loads((root / "compute_receipt.json").read_text())
        compute_note = ("Runpod의 저가 GPU 14개 조합에서 모두 재고 없음 응답을 받았다. 생성된 Pod는 없고 대여 비용은 $0이다. "
                        "기본 모델 한 회차는 CPU 4개 스레드에서 5.28초였다. 전체 실험은 별도 프로세스 그룹에서 CPU 3코어 상당, 메모리 상한 8GiB, 스왑 사용 금지, 낮은 실행 우선순위로 실행했다. 가용 메모리가 16GiB 아래로 떨어지면 학습만 일시 정지하는 감시도 두었다. 학습 횟수와 비교 조합은 줄이지 않았다.")
        cost = 0.
    cfg = result["selected_cnn"]; paths = result["paths"]
    uncertainty = {k: bootstrap_active(paths[k]["daily"], paths["momentum5"]["daily"]) for k in ["cnn", "gbm"]}
    atomic_json(out / "uncertainty.json", uncertainty)
    setup_style()
    # Quiet, aligned research plates: geometry carries the comparison; one accent marks CNN.
    fig, ax = plt.subplots(figsize=(10, 5.5))
    labels = {"cnn": "선택 CNN · 3개 시드", "gbm": "표 형태 GBM", "momentum5": "5일 모멘텀", "liquidity": "유동성 순위"}
    colors = {"cnn": "#2C6E65", "gbm": "#252525", "momentum5": "#7C7C7C", "liquidity": "#B6B6B6"}
    for k, path in paths.items():
        d = pd.DataFrame(path["daily"])
        ax.plot(pd.to_datetime(d.date), (d.nav / 1e9 - 1) * 100, label=labels[k], color=colors[k], lw=2 if k=="cnn" else 1.3)
    ax.axhline(0, color="#BBBBBB", lw=.7); ax.grid(axis="y", alpha=.18, lw=.5)
    fig.subplots_adjust(top=.76)
    ax.set_ylabel("누적 수익률 (%)"); ax.set_title("최종 평가 · 2026.07.01–09.23", loc="left", fontsize=16, pad=55)
    ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0, 1.01), ncol=2)
    fig.text(.125, .005, "KRX 분봉 + 당시 지정 이력·시총·거래대금 | 왕복 제비용 0.4% + 편도 슬리피지 0.05% | 주문장·권리 처리는 근사", fontsize=8, color="#666666")
    save_figure(fig, out / "holdout.png")
    # Same stock/date for every geometry. Only validation-window images are displayed.
    index = json.loads((root / "panel_index.json").read_text()); samples = np.load(root / "samples.npz")
    tags = np.repeat(np.arange(32, dtype=np.uint8)[None, :, None], 8, axis=2)
    mapped = transform(tags, dict(cfg, info="full"))[0, 0]
    permutation = mapped[0, :] if cfg["layout"] == "transpose" else mapped[:, 0]
    features = []
    for position, row in enumerate(permutation):
        row = int(row); use = "full"
        if ((cfg["info"] == "price_volume" and row >= 16) or
                (cfg["info"] == "no_rules" and row >= 24) or
                (cfg["info"] == "no_context" and 16 <= row < 24) or
                (cfg["info"] == "summary_only" and (row < 8 or row >= 28))):
            use = "masked"
        elif cfg["info"] == "summary_only" and 8 <= row < 16:
            use = "last_value_only"
        features.append({"display_position": position, "source_feature_row": row,
                         "feature": index["features"][row], "input_use": use,
                         "feature_axis": "columns" if cfg["layout"] == "transpose" else "rows"})
    pd.DataFrame(features).to_csv(out / "selected_feature_layout.csv", index=False)
    ci, di = samples["ci"], samples["di"]
    code = index["codes"].index("005930")
    available = np.where((ci == code) & (np.array(index["dates"])[di] < "20260701"))[0]
    sample = int(available[-1])
    choices = [("기본 배치", dict(BASE)), ("규칙 정보를 위로", dict(BASE, layout="rules_top")), ("검증에서 선택한 배치", cfg)]
    fig, axes = plt.subplots(3, 1, figsize=(10, 8.5), constrained_layout=True)
    for ax, (title, config) in zip(axes, choices):
        images = np.load(root / f"images_f{config['frequency']}_w{config['window']}.npy", mmap_mode="r")
        a = transform(images[sample:sample + 1], config)[0, 0]
        ax.imshow(a, aspect="auto", cmap="Greys", vmin=0, vmax=255, interpolation="nearest")
        ax.set_title(f"{title}  ·  {config['frequency']}분 × {config['window']}거래일", loc="left", fontsize=12)
        if config["layout"] == "transpose":
            ax.set_ylabel("완료된 과거 봉"); ax.set_xlabel("특성 행 →")
        else:
            ax.set_ylabel("특성 행"); ax.set_xlabel("완료된 과거 봉 →")
    fig.suptitle(f"열지도 입력 비교 · 삼성전자 · {index['dates'][di[sample]]}", x=.01, ha="left", fontsize=16)
    save_figure(fig, out / "heatmaps.png")
    # All attempts remain in a machine-readable table, including unsuccessful variants.
    table = []
    for r in screen:
        table.append({**r["config"], "selection_score": r["selection_score"], "validation_return": r["portfolio"]["return"],
                      "validation_sharpe": r["portfolio"]["sharpe"], "auc": r["prediction"]["auc"],
                      "weekly_violations": r["portfolio"]["low_turnover_weeks"], "seconds": r["seconds"]})
    pd.DataFrame(table).to_csv(out / "all_cnn_experiments.csv", index=False)
    pd.DataFrame([{**r["config"], "selection_score": r["selection_score"],
                   "validation_return": r["portfolio"]["return"], "best_iteration": r["best_iteration"]}
                  for r in selection["gbm_screen"]]).to_csv(out / "all_gbm_experiments.csv", index=False)
    selected_rows = [r for r in screen if r["config"]["id"] == "base" or r["config"]["id"].startswith("layout_")]
    fig, ax = plt.subplots(figsize=(10, 4))
    for i, r in enumerate(selected_rows):
        color = "#2C6E65" if r["config"]["id"] == cfg["id"] else "#7C7C7C"
        ax.scatter(r["selection_score"], i, color=color, s=45)
    ax.set_yticks(range(len(selected_rows)), [r["config"]["id"] for r in selected_rows]); ax.invert_yaxis()
    ax.set_xlabel("검증 구간 선택 점수 · 높을수록 양호"); ax.grid(axis="x", alpha=.18)
    ax.set_title("정보는 같고 위치만 바꾼 비교", loc="left", fontsize=16, pad=15)
    save_figure(fig, out / "layout_comparison.png")
    pct = lambda x: f"{x*100:.2f}%"
    cnn, gbm = paths["cnn"]["metrics"], paths["gbm"]["metrics"]
    lines = ["# 타임폴리오용 국장 열지도 — 1차 실험", "", "작성일: 2026-09-28", "",
             "현재 대회 규칙을 과거 데이터에 적용한 연구용 재생이다. 실제 모의계좌에는 주문하지 않았다.", "",
             "## 선택 결과", "",
             f"검증 구간에서 고른 CNN은 `{cfg['id']}`다. {cfg['frequency']}분봉 {cfg['window']}거래일, {cfg['horizon']}거래일 예측, `{cfg['info']}` 입력, `{cfg['layout']}` 배치, 폭 {cfg['width']}, 커널 {cfg['kernel']}을 쓴다. 최종 모델은 3개 시드의 예측 평균이다.", "",
             "모든 시도는 4~6월 검증 성과로 비교했다. 상위 3개 CNN을 3개 시드로 다시 확인하고 1개를 선택했다. 선택한 CNN과 GBM만 6월까지 재학습한 뒤 7~9월 최종 평가를 열었다. 이 결과를 보고 조합을 다시 고르지 않았다.", "",
             "비교 도중 KRX의 과거 지정 이력을 확보해 최종 평가를 잠근 채 매수 자격을 보완했다. 기존 열지도·학습 라벨·CNN 예측은 유지하고 모든 후보의 검증 거래를 새 규칙으로 다시 계산한 뒤 후보를 골랐다. 수정 전 결과와 변경 근거도 별도 보관했다.", "",
             "| 최종 평가 전략 | 수익률 | MDD | 일별 수익 샤프 | 평균 투자 비중 | 회전율 미달 주 |",
             "|---|---:|---:|---:|---:|---:|"]
    for k in ["cnn", "gbm", "momentum5", "liquidity"]:
        m = paths[k]["metrics"]
        lines.append(f"| {labels[k]} | {pct(m['return'])} | {pct(m['mdd'])} | {m['sharpe']:.2f} | {pct(m['mean_gross'])} | {m['low_turnover_weeks']}/{m['assessed_full_weeks']} |")
    lines += ["", "샤프는 짧은 구간의 일별 수익률을 연율화한 참고치다. 투자 비중이 서로 다르므로 수익률 차이 전체를 예측력으로 해석하면 안 된다. 현금 이자는 0이며, 평가 마지막 날 남은 포지션은 종가로 평가했다.", "",
              f"CNN은 종가 기준 종목·업종·소형주 비중 초과가 {cnn['closing_weight_breach_days']}거래일, 평가 종료 시 매도가 잠긴 권리 보유가 {cnn['unreleased_rights_positions']}종목이었다. GBM은 각각 {gbm['closing_weight_breach_days']}거래일, {gbm['unreleased_rights_positions']}종목이다. 종가 초과는 매수 시 한도 위반과 구분하며, 다음 거래일 조정 대상이다. CNN 잔여 보유분을 추가로 청산할 때의 기본 비용 추정치는 {cnn['terminal_liquidation_reserve']:,.0f}원이다.", "",
              "![최종 평가](holdout.png)", "", "## 불확실성과 체결 비용", ""]
    for k in ["cnn", "gbm"]:
        u = uncertainty[k]
        lines.append(f"- {labels[k]}의 모멘텀 대비 수익률 차이: {pct(u['active_return'])}p. 5거래일 블록 재표집 95% 구간: {pct(u['ci95'][0])}p~{pct(u['ci95'][1])}p.")
    lines += ["", "이 구간은 짧은 최종 평가의 통계적 불확실성만 나타낸다. 지정 이력의 누락 가능성, 생존편향, 업종 변경 및 체결 모형 오류를 포함하지 않는다.", "",
              "| 스트레스 조건 | CNN 수익률 | GBM 수익률 |", "|---|---:|---:|"]
    for key, title in [("slippage_10bp", "편도 슬리피지 0.10%"), ("slippage_25bp", "편도 슬리피지 0.25%"),
                       ("participation_1pct", "체결 창 거래량의 1%까지만 참여"), ("sector_floor_10pct", "모든 업종 한도를 10%로 제한")]:
        lines.append(f"| {title} | {pct(result['stress']['cnn'][key]['return'])} | {pct(result['stress']['gbm'][key]['return'])} |")
    lines += ["", "## 검증 구간에서 후보를 고른 근거", "",
              "아래는 4~6월 검증 결과이며, 위의 7~9월 최종 평가와 구분한다. 상위 후보의 3개 시드 예측을 평균한 뒤 과거 지정 이력을 포함한 같은 규칙으로 재생했다.", "",
              "| CNN 후보 | 검증 수익률 | 선택 점수 |", "|---|---:|---:|"]
    for row in sorted(selection["cnn_refinement"], key=lambda x: x["selection_score"], reverse=True):
        lines.append(f"| `{row['config']['id']}` | {pct(row['portfolio']['return'])} | {row['selection_score']:.3f} |")
    lines += ["", "선택 점수는 검증 구간을 둘로 나눴을 때 더 낮은 쪽의 연율화 일평균 초과수익에서 변동성 항을 뺀 값이다. 누적 수익률만 큰 후보를 고르지 않도록 사전에 정했다. 회전율 4회 미달에는 큰 감점을 적용한다. GBM 9개 결과는 [CSV](all_gbm_experiments.csv)에 남겼다."]
    lines += ["", "## 입력과 비교 범위", "",
              f"- 원자료: {audit['start']}~{audit['end']}, {audit['sessions']}거래일. KRX 분봉 집계 {audit['minute_security_days']:,} 종목·일, 현재 타임폴리오 업종과 연결된 {audit['codes']:,}종목.",
              f"- 같은 시점의 시총·거래대금으로 자격을 검사하고, 당시 20일 거래대금 상위 200종목까지 사용했다. 공통 표본은 {audit['eligible_signal_rows']:,}개다.",
              "- 최초 20거래일은 특성 계산을 위한 준비 구간이다. 학습·검증 경계에 미래 수익률이 걸치는 표본은 제거했다.",
              "- CNN 36개: 정보 제거, 규칙 행의 위·중간 배치, 교차 배치, 전치, 행·시간 순서 섞기, 15/30분, 3/5/10일 창, 3/5/10일 예측, 절대·상대 수익 라벨, 폭·커널·학습률·드롭아웃.",
              "- GBM 9개: 동일한 기본 열지도의 마지막 값·평균·표준편차를 사용하고, 리프 수와 예측 기간을 바꿨다.",
              "- 가격 8행, 거래량·유동성 8행, 상대강도·시장 맥락 8행, 시총·한도·체결 여력·세션 정보 8행. 32행의 고정 범위 정규화를 사용했다.", "",
              "![열지도 예시](heatmaps.png)", "", "![배치 비교](layout_comparison.png)", "",
              "전체 CNN 결과: [CSV](all_cnn_experiments.csv). 선택된 열지도의 특성과 표시 위치: [CSV](selected_feature_layout.csv). 열지도 그림은 최종 평가 이전의 동일 종목·날짜를 사용했다.", "",
              "## 규칙 반영 범위와 남은 한계", "",
              "- 10억원 현금 계좌, 공매도·레버리지 없음, 정수 주식 수. KRX 정규장만 사용했다.",
              "- 5일 평균 거래대금 30억원 초과, 시총 1,000억원 이상, 관측 이력이 충분한 종목만 매수 후보가 된다.",
              f"- KRX 관리·투자주의환기·투자주의·투자경고·투자위험 이력 {sum(v['records'] for v in audit['historical_designations'].values()):,}건을 연결했다. 기본 후보 중 다음 날 지정 때문에 매수할 수 없는 경우는 {audit['blocked_eligible_orders_next_day']:,} 종목·일이다. 주의는 지정일 1일, 나머지는 지정일부터 해제일 직전까지 매수를 막으며, 매도는 허용한다.",
              "- 일반 종목 15%, 삼성전자 40%, SK하이닉스 30%, 시총 1조원 미만 합계 30%를 적용한다. 전략 목표는 종목당 8%, 총 80% 이하다.",
              "- 업종 한도는 max(시장 비중×2, 10%)로 계산하고 목표 비중에는 5% 여유를 뒀다. 가격 변동으로 초과하면 다음 날 조정을 시도하며, 부분 체결로 남은 초과도 기록한다.",
              "- 전일 종가 이후 만든 신호를 다음 거래일 09:05~09:34에 실행한다. 체결가는 분봉의 대표가격을 거래량으로 가중한 근사치다. 해당 창 거래량의 5%까지만 체결하고, 상·하한가에 고정된 창은 해당 방향의 주문을 막는다.",
              "- 매수 0.1%, 매도 0.3%, 편도 슬리피지 0.05%를 기본으로 적용했다. 주문장 10호가와 단계별 대기 시간은 복원하지 못했다.",
              "- 주간 회전율은 (매수+매도 금액)/(주 평균 NAV)÷2로 계산한다. 양 끝의 불완전 주는 제외한다. 4회 미달 후보는 선택 점수에서 큰 감점을 적용한다. 사후 미달한 평가 경로는 연구용 성과이며 대회 존속 가능한 성과로 해석하지 않는다. 분산된 수익 기여 등을 보는 대회의 운용 점수는 재현하지 않았다.",
              f"- 공급자 수정주가가 들어간 {audit['vendor_adjusted_security_days']:,} 종목·일을 당시 KRX 종가 기준으로 환산했다. 기간별 5개 날짜의 KRX 시가·고가·저가도 대조했다.",
              f"- 가격 기준 변경에서 기업 이벤트 {audit['inferred_corporate_actions']:,}건을 추정했다. 늘어난 주수는 등록 상장주식 수가 추정 발행 수량에 도달할 때까지 잠근다. 정확한 이벤트 종류·배당금·권리 상장일을 복원한 것은 아니다.",
              "- GICS는 2026-09-28 타임폴리오 스냅샷이며 과거 업종 변경은 미복원이다. 지정 이력은 관리·환기·위험 2022년부터, 경고 2024년부터, 주의 2024-09-24부터 조회했다. 모든 KIND 공시와 전수 대조하지 않았고, 수집 종목 목록의 생존편향도 남는다. 따라서 어떤 후보도 대회 규칙을 완전히 통과했다고 인증하지 않는다.", ""]
    checks = [v for k, v in weight_audit.items() if k != "holdout"] + list(weight_audit.get("holdout", {}).values())
    lines += [f"저장된 체결을 별도로 재계산한 매수 검사 {sum(x['buy_checks'] for x in checks):,}건에서 사후 종목·업종·소형주·총 비중 및 지정 제한 위반은 {sum(len(x['post_buy_limit_violations']) for x in checks):,}건이었다. 원 계산과 NAV 재구성의 최대 차이는 {max(x['maximum_nav_reconstruction_error_krw'] for x in checks):.6f}원이다.", "",
              "## 원 논문과 달라진 점", "",
              "현재 AutoCrypto 논문은 4시간봉 60개와 요약 통계를 이미지로 만들고, 14일 수익률을 예측해 롱숏 포트폴리오를 구성한다. 이번 실험은 정규장 분봉, 다음 날 체결, 매수만 가능한 현금 계좌와 업종·종목 제한에 맞춘 별도 설계다. 원 논문의 수익률이나 샤프를 재현한 실험으로 보지 않는다. 논문에서도 요약 통계의 기여가 컸으므로 표 형태 GBM과 정보 제거 실험을 함께 비교했다.", ""]
    if news:
        lines += ["## 뉴스 레짐의 추가 가능성", "",
                  f"공용 뉴스 아카이브에는 확인 시점 기준 {news['articles']:,}건, {news['sources']}개 매체가 있다. 수집 기록의 시작은 `{news['first_collection_raw']}`로, 이번 모델의 학습·검증 구간과 겹치지 않는다. 따라서 뉴스로 1년 전체를 학습했다거나 추가 효과를 검증했다고 말할 수 없다.", "",
                  "기존 `QuantInSight/tools/regime.py`는 LLM 매크로 보고서의 주식·현금 권고를 risk_on/neutral/risk_off로 바꾸는 함수다. 이 출력이 당시 시각과 함께 보존돼 있다면 재사용 후보가 된다.", "",
                  "뉴스 아카이브는 같은 URL의 제목·요약을 갱신하면서 최초 수집 시각을 유지한다. 현재 텍스트를 과거 최초 수집 시점에 사용하면 수정 기사 내용이 새어 들어갈 수 있다. 시각의 시간대와 당시 원문 버전을 먼저 확보해야 한다.", "",
                  "다음 비교는 시장 위험도 1개와 업종별 뉴스 집중도 정도로 작게 시작하는 편이 낫다. 같은 사건의 중복 기사를 묶고, 위험도가 높을 때 목표 투자 비중을 제한된 범위에서 줄이는 방식이다. 기본 열지도, 가격만으로 판단한 국면 필터, 뉴스까지 더한 국면 필터를 같은 비중 범위와 비용 조건에서 비교해야 현금 확대 효과와 뉴스 효과를 구분할 수 있다. 업종·종목 상한은 항상 별도로 강제하고, 현금 비중 증가에 따른 주간 회전율 미달도 검사해야 한다. 이번 7~9월을 보고 뉴스 규칙을 고르면 이 구간은 뉴스 모델의 최종 평가로 다시 사용할 수 없다. 이번 작업에서는 뉴스 모델을 학습하거나 수집기를 수정하지 않았다.", ""]
    lines += ["## 재현과 비용", "",
              compute_note, "",
              "실행 소스: `quant/timefolio_heatmap_{data,features,replay,study,rule_amendment,audit,remote,report}.py`. 원본 DB에는 읽기 전용으로 접근했다. 고정 조건과 모델·예측·거래 내역은 아래 연구 폴더에 보관했다.", "",
              f"`{root}`", "",
              "원자료 집계와 특성 생성은 `timefolio_heatmap_data all`, `timefolio_heatmap_features all`로 수행했다. 저장된 예측과 지정 이력으로 같은 비교를 재현할 때는 기존 결과를 덮어쓰지 않도록 새 폴더를 지정한다. 아래의 `<새_연구_폴더>`와 `<새_보고서_폴더>`는 실제 경로로 바꿔야 한다.", "",
              "```bash", "python3 -m quant.timefolio_heatmap_rule_amendment prepare --dest <새_연구_폴더>",
              "python3 -m quant.timefolio_heatmap_rule_amendment rescore --dest <새_연구_폴더>",
              "# 별도 CPU·메모리 제한을 설정한 뒤 실행",
              "OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 python3 -m quant.timefolio_heatmap_study all --root <새_연구_폴더> --device cpu",
              "python3 -m quant.timefolio_heatmap_audit --root <새_연구_폴더>",
              "python3 -m quant.timefolio_heatmap_report --root <새_연구_폴더> --out <새_보고서_폴더>", "```", "",
              "기존 최종 평가 파일을 덮어쓰는 실행은 거부한다. 추가 튜닝은 새 연구 버전으로 기록하고, 이번 최종 평가 구간은 이후부터 검증 구간으로 취급해야 한다.", "",
              "공식 규칙: [타임폴리오 공지·매뉴얼](https://contest.timefolio.net/Notice), 제13회 공지 743, 운용규정 104·708, 주문 8, 회전율 FAQ 305, 기업 이벤트 10. 과거 지정 이력은 [KRX 이슈 통계](https://data.krx.co.kr/contents/MDC/STAT/issue/MDCSTAT233.jsp), 주의 지정의 적용 기간은 [KRX 공식 공시 예시](https://kind.krx.co.kr/external/2026/07/13/000543/20260713001222/70820.htm)를 확인했다. 원 논문은 현재 `HYFE_QTPA_논문_v1.docx`를 기준으로 읽었다."]
    (out / "REPORT.md").write_text("\n".join(lines) + "\n")
    atomic_json(out / "brief.json", {"selected": cfg, "metrics": {k: v["metrics"] for k, v in paths.items()},
                                     "uncertainty": uncertainty, "cost_usd_estimate": cost})
    print(str(out / "REPORT.md"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--root", type=Path, default=ROOT); ap.add_argument("--out", type=Path)
    a = ap.parse_args(); report(a.root, a.out)
