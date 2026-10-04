"""Export research figures and a readable report from a completed frozen run.

python3 -m research.plot_korean_retail --run data/korean_retail/<run>
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
import numpy as np
from PIL import Image

INK, RED, GRAY = '#222222', '#923838', '#666666'
SERIF = font_manager.FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc')
PREDICTION_FIGURE = 'retail-prediction-v2'
STATISTICS_FIGURE = 'retail-price-statistics-v2'
ENGINE_LABELS = {
    'train_mean': '종목별 과거 평균', 'zero': '항상 0', 'persistence': '전일 값 유지',
    'linear': '기본 회귀', 'native_only': '행동 신호만',
    'native_no_bias_only': '편향 없는 행동 신호만',
    'linear_native': '기본 회귀 + 행동', 'linear_native_no_bias': '기본 회귀 + 편향 제거',
}
MOMENT_LABELS = {
    'return_std': '일간 수익률 표준편차', 'excess_kurtosis': '수익률 초과첨도',
    'return_acf1': '수익률 1일 자기상관', 'abs_return_acf1': '절대 수익률 1일 자기상관',
    'abs_return_acf5': '절대 수익률 5일 자기상관', 'log_volume_acf1': '로그 거래량 1일 자기상관',
    'return_log_volume_correlation': '수익률·로그 거래량 상관',
}


def style():
    font_path = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
    font_manager.fontManager.addfont(font_path)
    plt.rcParams.update({'font.family': font_manager.FontProperties(fname=font_path).get_name(),
                         'font.size': 12, 'axes.unicode_minus': False,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.linewidth': .6, 'axes.edgecolor': '#b7b7b7',
                         'text.color': INK, 'axes.labelcolor': INK})


def export(fig, run, name):
    fig.savefig(run / f'{name}.png', dpi=180, facecolor='white')
    fig.savefig(run / f'{name}.pdf', facecolor='white')
    plt.close(fig)
    with Image.open(run / f'{name}.png') as im:
        im.resize((im.width//4, im.height//4), Image.Resampling.LANCZOS).save(run / f'{name}-quarter.png')


def number(value, digits=3):
    return '자료 없음' if value is None else f'{value:,.{digits}f}'


def table(headers, rows):
    head = ''.join('<th scope="col">'+html.escape(h)+'</th>' for h in headers)
    body = ''.join('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in row)+'</tr>' for row in rows)
    return '<div class="table-wrap"><table><thead><tr>'+head+'</tr></thead><tbody>'+body+'</tbody></table></div>'


def plot(run, report_only=False):
    report = json.loads((run/'report.json').read_text())
    manifest = json.loads((run/'manifest.json').read_text())
    if report['smoke_only']:
        raise ValueError('Smoke results cannot be exported as a primary validation report')
    if not report_only and any((run/(name+suffix)).exists()
                              for name in [PREDICTION_FIGURE,STATISTICS_FIGURE]
                              for suffix in ['.png','.pdf']):
        raise ValueError('Preserve existing figures: use --report-only or a new output run')
    predictions = report['replay_probe_predictions']
    split = report['split']
    style()
    engines = ['train_mean','linear','native_only','linear_native','linear_native_no_bias']
    fig, axes = plt.subplots(1,2,figsize=(14,7))
    fig.subplots_adjust(left=.185,right=.965,top=.72,bottom=.31,wspace=.55)
    fig.text(.055,.925,'행동 규칙이 다음 날 예측에 더하는 정보',size=21,fontproperties=SERIF)
    fig.text(.055,.855,f"한국 주식 12종목 · 최종 검증 {split['dates']['test']}거래일 · 기준일 {split['test_label_start']} 이후",size=12,color=GRAY)
    for ax,target,title in zip(axes,['flow_proxy','entry_return'],['A   개인 수급 대용치','B   다음 날 시가 → 종가 수익률']):
        metrics = predictions[target]['metrics']
        values = [100*metrics[e]['oos_r2_vs_training_stock_mean'] for e in engines]
        positions = np.arange(len(engines))
        colors = [GRAY,INK,GRAY,RED,GRAY]
        ax.barh(positions,values,height=.46,color=colors)
        ax.set_yticks(positions,[ENGINE_LABELS[e] for e in engines],fontsize=11)
        ax.invert_yaxis()
        lo,hi = min(0,min(values)),max(0,max(values))
        width = max(hi-lo,1)
        ax.set_xlim(lo-.23*width,hi+.35*width)
        for y,value in zip(positions,values):
            ax.text(value + (.025*width if value>=0 else -.025*width),y,f'{value:+.2f}',
                    ha='left' if value>=0 else 'right',va='center',fontsize=11,
                    fontfamily='DejaVu Sans Mono')
        ax.axvline(0,color='#aaaaaa',linewidth=.7)
        ax.grid(axis='x',color='#dddddd',linewidth=.4)
        ax.set_axisbelow(True)
        ax.set_title(title,loc='left',size=12,pad=18)
        ax.set_xlabel('과거 평균 대비 검증 R² (%)',labelpad=12)
    fig.text(.055,.14,'0보다 크면 종목별 과거 평균보다 제곱오차가 작음. 붉은 막대는 기본 회귀에 행동 신호를 추가한 결과.',size=11,color=GRAY)
    fig.text(.055,.085,'기본 회귀: 과거 수익률·변동성·거래량·수급. 행동 신호: 지속 상태를 가진 1,000명의 주문 의향, 두 초기화 평균.',size=10,color=GRAY)
    fig.text(.055,.035,'실제 시세를 입력한 별도 재생 실험. 원래 합성 사회의 가격 생성·인과관계·실현 수익 검증은 별도 과제.',size=10,color=GRAY)
    if report_only:
        plt.close(fig)
    else:
        export(fig,run,PREDICTION_FIGURE)

    stats = report['original_price_statistics']
    fig,axes = plt.subplots(1,3,figsize=(14,6.4))
    fig.subplots_adjust(left=.075,right=.965,top=.71,bottom=.34,wspace=.43)
    fig.text(.055,.925,'원래 합성 사회의 가격 통계는 시장과 얼마나 비슷한가',size=19,fontproperties=SERIF)
    fig.text(.055,.855,'회색 구간: 실제 검증 기간 종목 간 10~90 백분위 · 점: 각 조건의 합성 가격 통계 중앙값',size=11,color=GRAY)
    for ax,key,title in zip(axes,['return_std','excess_kurtosis','abs_return_acf1'],
                            ['A   일간 변동성','B   극단 수익률의 꼬리','C   변동성의 지속성']):
        full = stats['regimes']['full']['moments'][key]
        mul = 100 if key=='return_std' else 1
        ax.plot([full['real_test_stock_p10']*mul,full['real_test_stock_p90']*mul],[0,0],color='#b0b0b0',lw=6,solid_capstyle='butt')
        values = [full['real_test_median']*mul,full['simulated_median']*mul,
                  stats['regimes']['no_bias']['moments'][key]['simulated_median']*mul]
        ax.scatter(values,[0,1,2],c=[INK,RED,GRAY],s=42,zorder=3)
        ax.set_yticks([0,1,2],['실제 시장','원래 모형','편향 제거'],fontsize=11)
        ax.set_ylim(2.65,-.8)
        for y,v in enumerate(values):
            ax.annotate(f'{v:.3f}',(v,y),xytext=(0,12),textcoords='offset points',ha='center',fontsize=10,fontfamily='DejaVu Sans Mono')
        ax.margins(x=.22)
        ax.set_title(title,loc='left',size=12,pad=16)
        ax.set_xlabel('일간 표준편차 (%)' if key=='return_std' else MOMENT_LABELS[key],labelpad=13,fontsize=11)
        ax.grid(axis='x',color='#dddddd',lw=.4)
    fig.text(.055,.16,'합성 사회는 주민끼리 거래하는 원래 모형. 4개 초기화 × 조건 2개 × 자산 3개, 각 240일을 측정.',size=11,color=GRAY)
    fig.text(.055,.105,'회색 구간은 종목 간 분산을 나타내며 신뢰구간이 아님. 일부 통계가 비슷해도 개인 행동이나 인과관계는 식별되지 않음.',size=10,color=GRAY)
    fig.text(.055,.045,f"학습 통계로 선택한 조건: {'원래 모형' if stats['training_selected_regime']=='full' else '편향 제거'}. 비교한 7개 통계의 전체 수치는 보고서에 수록.",size=10,color=GRAY)
    if report_only:
        plt.close(fig)
    else:
        export(fig,run,STATISTICS_FIGURE)

    rows=[]
    primary_tables=[]
    for target,label in [('flow_proxy','개인 수급 대용치'),('entry_return','다음 날 시가 → 종가 수익률')]:
        chosen = report['selection'][target]['engine']
        target_rows=[]
        for engine in manifest['engines']:
            m=predictions[target]['metrics'][engine]
            rows.append({'target':target,'engine':engine,**m})
            target_rows.append([ENGINE_LABELS[engine]+(' · 선택' if engine==chosen else ''),
                                number(m['oos_r2_vs_training_stock_mean']*100,2),
                                number(m['direction_accuracy']*100,1),number(m['correlation'])])
        primary_tables.append('<h3>'+label+'</h3>'+table(['모형','R² (%)','방향 (%)','상관'],target_rows))
    with (run/'prediction-summary.csv').open('w',newline='') as out:
        writer=csv.DictWriter(out,fieldnames=rows[0].keys());writer.writeheader();writer.writerows(rows)
    interval_rows=[]
    for target,label in [('flow_proxy','수급'),('entry_return','가격')]:
        for name,short in [('native_incremental','행동 추가'),('validation_selected','선택 모형')]:
            base=predictions[target]['metrics']['linear']['mse']
            for interval in predictions[target]['intervals_vs_linear'][name]:
                interval_rows.append([label+' · '+short,str(interval['block'])+'일',
                                      number(interval['mean_daily_mse_improvement']/base*100,2),
                                      number(interval['lower']/base*100,2)+' ~ '+number(interval['upper']/base*100,2)])
    moment_rows=[]
    for key,label in MOMENT_LABELS.items():
        a=stats['regimes']['full']['moments'][key]
        b=stats['regimes']['no_bias']['moments'][key]
        moment_rows.append([label,number(a['real_test_median']),number(a['simulated_median']),number(b['simulated_median'])])
    direct=predictions['direct_individual_secondary']
    full_support=[predictions[t]['native_incremental_supported'] for t in ['flow_proxy','entry_return']]
    verdict=('기본 회귀에 행동 신호를 추가한 개선이 두 목표 모두에서 확인됐습니다.' if all(full_support) else
             '기본 회귀에 행동 신호를 추가한 개선이 두 목표 모두에서 확인되지는 않았습니다.' if any(full_support) else
             '기본 회귀에 행동 신호를 추가했을 때의 개선은 확인되지 않았습니다.')
    native_count=stats['regimes']['full']['heldout_moments_inside_stock_p10_p90']
    price_metrics=predictions['entry_return']['metrics']
    chosen_price=report['selection']['entry_return']['engine']
    price_note='가격 목표에서 선택된 '+ENGINE_LABELS[chosen_price]+'의 제곱오차는 기본 회귀보다 '+('작았습니다.' if price_metrics[chosen_price]['mse']<price_metrics['linear']['mse'] else '작지 않았습니다.')
    price_note+=' 수익률을 항상 0으로 예측하는 기준보다 '+('오차가 작았습니다.' if price_metrics[chosen_price]['mse']<price_metrics['zero']['mse'] else '오차가 작지 않았습니다.')
    action_note=''
    if (run/'audit-final.json').exists():
        audit=json.loads((run/'audit-final.json').read_text())
        direction_counts=audit['selected_price_prediction_direction_values_per_stock']
        if all(count==1 for count in direction_counts.values()) and audit['selected_price_signal_sign_agreement_with_training_stock_mean']==1:
            action_note='<p>독립 재계산에서는 선택된 가격 모형의 예측 부호가 종목별로 검증 기간 내내 바뀌지 않았고, 종목별 과거 평균의 부호와 모두 같았습니다. 이 매매 시나리오에서는 행동 신호가 진입 시점을 바꾸는 역할을 하지 못했습니다.</p>'
    scenarios=predictions['illustrative_trading']['scenarios']
    trading_rows=[[ENGINE_LABELS.get(k,'매일 전체 매수'),number(v['total_return_pct'],2),v['trades']] for k,v in scenarios.items()]
    sources=''.join('<li><a href="'+html.escape(v,quote=True)+'">'+html.escape(k)+'</a></li>' for k,v in report['references'].items())
    preview='''<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>한국 개인투자자 모형 검증</title><style>
*{box-sizing:border-box}body{margin:0;background:#fff;color:#222;font-family:"Noto Sans CJK KR",sans-serif}
main{max-width:1140px;margin:auto;padding:44px 28px 70px}h1{font-family:"Noto Serif CJK KR",serif;font-size:34px;line-height:1.35;margin:16px 0 24px}
h2{font-size:23px;font-weight:500;margin:48px 0 16px}h3{font-size:19px;font-weight:500;margin:32px 0 12px}
p,li{font-size:17px;line-height:1.8;word-break:keep-all;overflow-wrap:break-word}p{max-width:70ch;margin:16px 0}
.meta{color:#555;font-size:15px}.lead{font-size:21px;line-height:1.65;font-weight:500}
a{color:#923838;text-decoration:underline;text-underline-offset:3px}figure{margin:32px 0}img{display:block;width:100%;height:auto}
figcaption{margin-top:12px;color:#555;font-size:15px;line-height:1.7;word-break:keep-all}
.table-wrap{overflow-x:auto;margin:20px 0}table{border-collapse:collapse;width:100%;font-size:16px}
th,td{text-align:left;border-bottom:1px solid #ccc;padding:13px 10px;word-break:keep-all;white-space:nowrap}th{font-weight:500;color:#555}
td:nth-child(n+2){font-family:"DejaVu Sans Mono",monospace;font-size:15px}
ul{padding-left:23px}footer{border-top:1px solid #ccc;margin-top:40px;padding-top:20px}
@media(max-width:600px){main{padding:24px 18px 48px}h1{font-size:27px}h2{font-size:21px}.lead{font-size:19px}th,td{padding:11px 7px}table{font-size:14px}td:nth-child(n+2){font-size:13px}}
</style></head><body><main><div class="meta">QuantInSight · 한국 개인투자자 · '''+html.escape(run.name)+'''</div>
<h1>한국 개인투자자 모형 검증</h1><p class="lead">'''+verdict+''' 기존 합성 사회가 실제 개인투자자의 행동과 가격 형성을 설명한다는 검증은 아직 성립하지 않습니다.</p>
<p>코스피 6종목과 코스닥 6종목을 대상으로 학습 301거래일, 모형 선택 101거래일, 최종 검증 101거래일을 분리했습니다. 최종 검증 기간은 '''+split['test_label_start']+'''부터 2026-10-01까지입니다. 과거 자료의 개정 시점을 알 수 없어 엄밀한 실시간 외부 검증은 아닙니다.</p>
<h2>1. 원래 사회의 수급 구조</h2><p>주민끼리만 거래하는 원래 사회는 주민 전체의 순매수가 항상 0입니다. 실제 개인 집단은 기관·외국인·기타 집단과 거래하므로 순매수가 발생합니다. 외부 집단을 도입해야 실제 개인 수급을 표현할 수 있습니다.</p>
<p>주 분석은 기관·외국인의 순매수 합계에 음수를 붙이고 거래량으로 나눈 대용치를 사용했습니다. 전체 거래량에서 순매수를 빼는 방식은 매수·매도 회계를 맞추지 못합니다. 이 대용치에도 기타 법인·기타 외국인 수급이 섞입니다.</p>
<h2>2. 다음 날 예측에 도움이 되는가</h2><p>실제 시세를 입력하고, 자금과 주식을 가진 수동적인 외부 투자자를 거래 상대로 넣은 별도 재생 실험입니다. 세 종목씩 네 묶음에 각각 1,000명의 상태와 습관을 유지했습니다. 두 초기화의 주문 의향을 평균했고, LLM과 Laya는 사용하지 않았습니다. 확인되지 않은 기업가치·배당은 입력에서 가렸습니다.</p>
<figure><a href="retail-prediction-v2.png"><img src="retail-prediction-v2.png" alt="개인 수급 대용치와 다음 날 수익률의 모형별 검증 R 제곱 비교"></a><figcaption>R²의 기준은 학습과 모형 선택 기간의 종목별 평균입니다. 그림을 누르면 원본을 볼 수 있습니다. PDF도 같은 폴더에 있습니다.</figcaption></figure>
'''+''.join(primary_tables)+'''<p>'''+price_note+''' 약한 기본 회귀를 이겼다는 결과만으로 유용한 가격 신호가 확인된 것은 아닙니다.</p>'''+action_note+'''
<p>방향 적중률은 모든 양수·음수·0 관측을 포함합니다. 높은 적중률만으로 투자 수익이나 균형 잡힌 분류 성능이 확인되지는 않습니다. 회귀 계수와 표준화는 최종 검증 기간 이전 자료로만 맞췄습니다.</p>
<h3>기본 회귀보다 제곱오차가 얼마나 줄었는가</h3>'''+table(['비교','블록','개선 (%)','98.75% 구간'],interval_rows)+'''<p>양수는 기본 회귀보다 오차가 작다는 뜻입니다. 날짜 단위로 전체 종목을 함께 재표집해 종목 간 상관을 보존했습니다. 네 주요 비교에 대한 다중 검정을 고려한 구간이며, 5일·10일 블록 모두의 하한이 0을 넘어야 개선 근거로 판단했습니다.</p>
<h3>직접 제공된 개인 순매수와의 대조</h3><p>최근 '''+str(direct['dates'])+'''거래일, '''+str(direct['rows'])+'''개 종목·날짜 관측에서 대용치와 개인 순매수의 부호가 '''+number(direct['proxy_direction_agreement']*100,1)+'''% 일치했습니다. 거래량 대비 순매수 비율의 차이는 절댓값 중앙값 '''+number(direct['median_abs_proxy_ratio_difference']*100,2)+'''%p입니다. 대용치는 개인 순매수와 같지 않습니다.</p>
<h2>3. 원래 가격의 통계적 특성</h2><p>원래 모형의 통계 7개 중 '''+str(native_count)+'''개가 실제 검증 기간의 종목 간 10~90 백분위에 들었습니다. 이 범위는 신뢰구간이나 합격 기준이 아닙니다. 주민 1,000명, 네 초기화, 60일 준비 후 240일을 측정했습니다.</p>
<figure><a href="retail-price-statistics-v2.png"><img src="retail-price-statistics-v2.png" alt="실제 한국 주식과 원래 합성 사회의 변동성 첨도 변동성 지속성 비교"></a><figcaption>변동성 그림은 % 단위입니다. 아래 표의 표준편차는 비율 단위이며 나머지는 첨도·상관계수입니다.</figcaption></figure>
'''+table(['통계','실제 중앙값','원래 모형','편향 제거'],moment_rows)+'''<h2>4. 수익과 개인 행동에서 남은 검증</h2>
<p>다음 날 시가에 매수해 종가에 매도하고 왕복 비용 20bp를 빼는 가상 매매를 계산했습니다. 12개 종목에 같은 비중을 배정하고 매수하지 않는 부분은 현금으로 남겼습니다. 실제 호가·체결 가능성·시장 충격을 검증한 수익이 아닙니다.</p>
'''+table(['가상 전략','누적 수익 (%)','매수 횟수'],trading_rows)+'''<p>계좌별 보유량·매입가·주문 이력·인구 특성 자료가 없어 처분 효과, 보유 기간, 개인별 거래 빈도와 성격 분포는 식별되지 않았습니다. 성향 변화가 실제 시장에 미치는 인과 효과도 확인되지 않았습니다. 익명 계좌 자료와 모형을 고정한 이후의 새 관측이 다음 검증에 필요합니다.</p>
<p>12개 종목은 기존 자료가 있는 편의 표본입니다. 당시 투자 가능 종목 집합·상장폐지·과거 정보 개정 시점을 복원하지 못했습니다. 과거와 새 시세의 보정 기준이 다른 두 종목은 본 적합 전에 자료 품질 기준으로 교체했고, 원자료와 이유를 보관했습니다. 투자자 배경과 자산 분포는 아직 실제 자료로 보정되지 않았습니다.</p>
<footer><p>원자료: 공개 시세·투자자별 수급의 고정 사본. 수치와 설정은 <a href="report.json">report.json</a>, <a href="manifest.json">manifest.json</a>, <a href="prediction-summary.csv">요약 CSV</a>, <a href="audit-final.json">독립 재계산 기록</a>에서 확인할 수 있습니다.</p><ul>'''+sources+'''</ul></footer></main></body></html>'''
    (run/'report-preview.html').write_text(preview)
    return run/'report-preview.html'


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',required=True)
    parser.add_argument('--report-only',action='store_true',help='Update HTML and CSV while preserving existing PNG/PDF figures')
    args=parser.parse_args()
    print(plot(Path(args.run),args.report_only))
