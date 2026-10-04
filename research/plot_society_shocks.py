"""Export paired shock response research plates from saved, completed results.

python3 -m research.plot_society_shocks --run data/society_shocks/<run>
"""
from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
from matplotlib import font_manager
import matplotlib.pyplot as plt
from PIL import Image

LABELS = {'pulse_up': '1일 후 원복', 'twenty_days_up': '20일 후 원복',
          'persistent_up': '60일 유지', 'persistent_down': '성향 −0.2',
          'persistent_small_up': '성향 +0.1', 'persistent_large_up': '성향 +0.4',
          'sham': '변화 없음'}
RED, INK, GRAY = '#923838', '#222222', '#777777'


def style():
    font_path = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
    font_manager.fontManager.addfont(font_path)
    font = font_manager.FontProperties(fname=font_path).get_name()
    plt.rcParams.update({'font.family': font, 'font.size': 11,
                         'axes.unicode_minus': False, 'axes.spines.top': False,
                         'axes.spines.right': False, 'axes.edgecolor': '#b7b7b7',
                         'axes.linewidth': .6, 'text.color': INK,
                         'axes.labelcolor': INK})


def finish_axes(axes):
    for ax in axes:
        ax.axhline(0, color='#aaaaaa', linewidth=.65, zorder=0)
        ax.grid(axis='y', color='#dddddd', linewidth=.4)
        ax.set_axisbelow(True)
        for tick in [*ax.get_xticklabels(), *ax.get_yticklabels()]:
            tick.set_fontfamily(['DejaVu Sans Mono', plt.rcParams['font.family'][0]])


def export(fig, run, name):
    fig.savefig(run / f'{name}.png', dpi=180, facecolor='white')
    fig.savefig(run / f'{name}.pdf', facecolor='white')
    plt.close(fig)
    with Image.open(run / f'{name}.png') as image:
        image.resize((image.width // 4, image.height // 4), Image.Resampling.LANCZOS).save(run / f'{name}-quarter.png')


def plot(run):
    report = json.loads((run / 'report.json').read_text())
    manifest = json.loads((run / 'manifest.json').read_text())
    results = json.loads((run / 'results.json').read_text())
    cases = report['cases']
    if len(results) != report['worlds'] * len(manifest['cases']):
        raise ValueError('Incomplete experiment')
    style()
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.3), gridspec_kw={'width_ratios': [1.45, 1]})
    fig.subplots_adjust(left=.085, right=.97, top=.72, bottom=.25, wspace=.29)
    fig.text(.085, .925, '군집 성향 변화가 남기는 가격 충격', size=21, weight='bold')
    fig.text(.085, .855,
             f"합성 사회 {report['worlds']}개 · 주민 {manifest['people']:,}명 · 동일한 생활 사건 · {manifest['warmup_days']}일 준비 후 {manifest['horizon_days']}일 비교",
             size=11, color=GRAY)
    left, right = axes
    for case, color, line in [('persistent_up', RED, '-'), ('twenty_days_up', INK, '--'), ('pulse_up', GRAY, ':')]:
        path = cases[case]['path']
        x = [0] + [p['offset'] for p in path]
        y = [0] + [p['price_gap_pct']['median'] for p in path]
        if case == 'persistent_up':
            left.fill_between(x, [0] + [p['price_gap_pct']['p10'] for p in path],
                              [0] + [p['price_gap_pct']['p90'] for p in path], color=RED, alpha=.08)
        left.plot(x, y, color=color, linestyle=line, linewidth=1.7, label=LABELS[case])
    left.set_title('A   충격 기간별 가격 차이', loc='left', fontsize=12, pad=15)
    left.set_xlabel('성향 변경 후 경과일', labelpad=10)
    left.set_ylabel('대조군 대비 지수 차이 (%)', labelpad=10)
    left.legend(frameon=False, fontsize=10, loc='best')
    dose_cases = ['persistent_down', 'persistent_small_up', 'persistent_up', 'persistent_large_up']
    for position, case in enumerate(dose_cases):
        values = [r['metrics']['terminal_price_gap_pct'] for r in results if r['case'] == case]
        offsets = [(i - (len(values) - 1) / 2) * .026 for i in range(len(values))]
        right.scatter([position + o for o in offsets], values, s=23, facecolors='none', edgecolors=GRAY, linewidths=.7)
        metric = cases[case]['metrics']['terminal_price_gap_pct']
        right.plot([position - .18, position + .18], [metric['median']] * 2, color=RED, linewidth=2)
    right.set_xticks(range(4), ['−0.2', '+0.1', '+0.2', '+0.4'])
    right.set_title(f"B   {manifest['horizon_days']}일 유지한 성향 변화", loc='left', fontsize=12, pad=15)
    right.set_xlabel('군집 성향 변화량 (0~1 척도)', labelpad=10)
    right.set_ylabel('마지막 날 지수 차이 (%)', labelpad=8)
    finish_axes(axes)
    fig.text(.085, .14, f"왼쪽: 주민 {round(manifest['people'] * manifest['target_fraction']):,}명에게 군집 성향 +0.2. 선은 중앙값, 음영은 사회 간 10~90 백분위.", size=10, color=GRAY)
    fig.text(.085, .085, '오른쪽 원은 각 사회, 붉은 가로선은 중앙값. 경계 0·1에서 성향 변화량은 잘릴 수 있음.', size=10, color=GRAY)
    fig.text(.085, .03, f"QuantInSight / {run.name} / 규칙 기반 합성 실험. 실제 시장의 신뢰구간·투자 수익 검증 아님.", size=9, color=GRAY)
    export(fig, run, 'shock-response')

    fig, axes = plt.subplots(1, 2, figsize=(13, 6.3))
    fig.subplots_adjust(left=.085, right=.97, top=.72, bottom=.25, wspace=.29)
    fig.text(.085, .925, '가격 충격과 거래·자산의 변화를 함께 읽기', size=20, weight='bold')
    fig.text(.085, .855, f"군집 성향 +0.2 · 대상 주민 {round(manifest['people'] * manifest['target_fraction']):,}명 · 각 원은 독립적으로 구성한 사회", size=11, color=GRAY)
    durations = ['pulse_up', 'twenty_days_up', 'persistent_up']
    for ax, key, title, ylabel in zip(axes,
                                     ['volume_change_pct', 'terminal_target_wealth_gap_pct'],
                                     ['A   기간 전체의 체결 거래량 변화', 'B   마지막 날 대상 집단의 순자산 변화'],
                                     ['대조군 대비 거래량 차이 (%)', '대조군 대비 대상 집단 순자산 차이 (%)']):
        for position, case in enumerate(durations):
            values = [r['metrics'][key] for r in results if r['case'] == case and r['metrics'][key] is not None]
            offsets = [(i - (len(values) - 1) / 2) * .026 for i in range(len(values))]
            ax.scatter([position + o for o in offsets], values, s=23, facecolors='none', edgecolors=GRAY, linewidths=.7)
            median = cases[case]['metrics'][key]['median']
            if median is not None:
                ax.plot([position - .18, position + .18], [median] * 2, color=RED, linewidth=2)
        ax.set_xticks(range(3), [LABELS[c] for c in durations])
        ax.set_title(title, loc='left', fontsize=12, pad=15)
        ax.set_ylabel(ylabel, labelpad=10)
    finish_axes(axes)
    fig.text(.085, .14, '순자산 = 현금 + 보유 주식의 평가액 − 부채·미지급 이자. 임금·소비·배당의 영향도 포함.', size=10, color=GRAY)
    fig.text(.085, .085, '전체 주민의 순매수 합계는 항상 0. 대상 집단의 순매수 변화는 나머지 집단이 반대로 흡수.', size=10, color=GRAY)
    fig.text(.085, .03, f"QuantInSight / {run.name} / 원: 사회별 결과 · 붉은 가로선: 중앙값 · 순자산 변화는 매매 알파가 아님.", size=9, color=GRAY)
    export(fig, run, 'shock-transmission')

    rows = []
    for case in durations + dose_cases[:1] + dose_cases[1:2] + dose_cases[3:]:
        v = cases[case]
        rows.append({'case': case, 'label': LABELS[case], 'worlds': v['worlds'],
                     'nominal_delta': next(c['delta'] for c in manifest['cases'] if c['name'] == case),
                     'mean_effective_delta': v['mean_effective_delta'],
                     **{f'{key}_{stat}': v['metrics'][key][stat]
                        for key in ['terminal_price_gap_pct', 'peak_abs_price_gap_pct', 'volume_change_pct', 'fill_rate_difference_pp', 'terminal_target_wealth_gap_pct']
                        for stat in ['median', 'p10', 'p90']}})
    with (run / 'summary.csv').open('w', newline='') as out:
        writer = csv.DictWriter(out, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    table = ''.join('<tr><td>' + html.escape(r['label']) + '</td><td>' + f"{r['terminal_price_gap_pct_median']:+.2f}%" + '</td><td>' + f"{r['volume_change_pct_median']:+.2f}%" + '</td></tr>' for r in rows[:3])
    preview = '''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>군집 성향 충격 실험</title><style>*{box-sizing:border-box}body{margin:0;background:white;color:#222;font-family:"Noto Sans CJK KR",sans-serif}
main{max-width:1280px;margin:auto;padding:32px 20px}h1{font-size:26px;margin:0 0 16px}p{max-width:60ch;font-size:16px;line-height:1.7;word-break:keep-all}img{display:block;width:100%;height:auto;margin-top:24px}table{border-collapse:collapse;width:100%;font-size:16px;margin:24px 0}th,td{text-align:left;border-bottom:1px solid #ccc;padding:12px 6px;word-break:keep-all}th{font-weight:500}td:nth-child(n+2){font-family:monospace}</style>
<main><h1>군집 성향 충격 실험</h1><p>같은 시점의 사회를 복제해 일부 주민의 군집 성향만 바꿨습니다. 두 사회는 같은 출근과 돌발 지출 사건을 겪습니다. 아래 수치는 합성 사회에서 얻은 결과이며 실제 시장의 투자 수익을 검증한 값은 아닙니다.</p>
<table><thead><tr><th>충격 기간</th><th>60일 가격</th><th>총 거래량</th></tr></thead><tbody>''' + table + '''</tbody></table>
<p>표는 대조군 대비 차이의 사회 간 중앙값입니다. 회색 원과 붉은 음영은 사회마다 결과가 얼마나 달랐는지 보여줍니다. 도표를 누르면 원본을 열 수 있습니다. PDF도 같은 폴더에 보관했습니다.</p>
<a href="shock-response.png"><img src="shock-response.png" alt="성향 충격 기간과 강도에 따른 가격 차이"></a><a href="shock-transmission.png"><img src="shock-transmission.png" alt="충격 기간별 체결 거래량과 대상 집단 순자산 차이"></a></main></html>'''
    (run / 'chart-preview.html').write_text(preview)
    return run / 'shock-response.png'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True)
    print(plot(Path(parser.parse_args().run)))
