"""Research plates for all declared policies; no automatic candidate promotion."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

from quant.timefolio_heatmap_report import setup_style, save_figure
from quant.timefolio_heatmap_walkforward import DEST, SOURCE
from quant.timefolio_heatmap_study import context

OUT=Path(__file__).resolve().parents[1]/'_workspace/timefolio_heatmap_walkforward_v3'


def style():
    setup_style()
    plt.rcParams.update({'figure.facecolor':'none','axes.facecolor':'none','savefig.facecolor':'none'})


def inputs(root=DEST,out=OUT):
    root,out=Path(root),Path(out);out.mkdir(exist_ok=True,parents=True)
    if (out/'inputs.png').exists():raise RuntimeError('Existing plate is preserved; choose another output folder')
    p,ix,ci,di=context(SOURCE);date=np.asarray(ix['dates'])[di]
    samples=np.flatnonzero((ci==ix['codes'].index('005930'))&(date<'20260101'))
    sample=int(samples[-1]);style()
    fig,axes=plt.subplots(3,1,figsize=(11,8),layout='constrained')
    for ax,encoding,title in zip(axes,['raw','sector_center','vol_scaled'],
                                 ['원본 입력','같은 날 업종 중심값을 뺀 입력','가격 변화를 변동성으로 조정한 입력']):
        raw=np.load(root/f'images_{encoding}.npy',mmap_mode='r')
        ax.imshow(raw[sample],cmap='Greys',vmin=0,vmax=255,aspect='auto',interpolation='nearest')
        ax.set_title(title,loc='left',fontsize=13,pad=8)
        ax.set_yticks([3.5,11.5,19.5,27.5],['가격','거래량','요약','규칙'])
        ax.set_xticks([0,12,25,38,51,64]);ax.set_xlabel('완료된 과거 30분 봉 · 5거래일')
        for border in [7.5,15.5,23.5]:ax.axhline(border,color='white',lw=.5)
    fig.suptitle('입력 구성 비교 · 삼성전자 · '+date[sample],x=.01,ha='left',fontsize=19)
    save_figure(fig,out/'inputs.png')
    (out/'input_plate_metadata.json').write_text(json.dumps({'sample':sample,'code':'005930','date':str(date[sample]),
        'design':'research plate; monochrome; same sample and fixed 0..255 scale; no performance-dependent choice'},indent=2)+'\n')


def report(root=DEST,out=OUT):
    root,out=Path(root),Path(out);out.mkdir(exist_ok=True,parents=True)
    if (out/'REPORT.md').exists():raise RuntimeError('Existing report is preserved; choose another output folder')
    summary=json.loads((root/'evaluation_summary.json').read_text());audit=json.loads((root/'independent_audit.json').read_text())['summary']
    frame=pd.read_csv(root/'portfolio_summary.csv');statistics=pd.read_csv(root/'bootstrap.csv')
    if audit['portfolios']!=len(frame) or audit['post_buy_violations'] or audit['additional_errors']:raise RuntimeError('Full audit required')
    labels={'online_neural':'과거 검증으로 고른 열지도','online_neural3':'열지도 3개 평균','online_nonimage':'표 형태 모델','lowvol':'단순 저변동성'}
    colors={'online_neural':'#2C6E65','online_neural3':'#2C6E65','online_nonimage':'#252525','lowvol':'#999999'}
    policies={'sector20':'업종 상한 20%','sector15':'업종 상한 15%',
              'sector20_trend':'업종 20% + 가격 추세 비중','sector20_vol':'업종 20% + 변동성 비중'}
    style();fig,axes=plt.subplots(2,2,figsize=(13,8),sharex=True,sharey=True,layout='constrained')
    for ax,(policy,title) in zip(axes.flat,policies.items()):
        for model,label in labels.items():
            result=json.loads((root/'portfolios'/(model+'__'+policy+'.json')).read_text())
            daily=pd.DataFrame(result['daily'])
            ax.plot(pd.to_datetime(daily.date),100*(daily.nav/1e9-1),label=label,color=colors[model],
                    linestyle='--' if model=='online_neural3' else '-',lw=1.8 if model.startswith('online_neural') else 1.1)
        ax.axhline(0,color='#BBBBBB',lw=.6);ax.grid(axis='y',alpha=.18,lw=.5)
        ax.set_title(title,loc='left',fontsize=14);ax.set_ylabel('비용 차감 누적 수익률 (%)')
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2));ax.xaxis.set_major_formatter(mdates.DateFormatter('%m월'))
    axes[0,0].legend(frameon=False,fontsize=9,loc='best')
    fig.suptitle('월별 재학습 비교 · 2026.01–09 · 개발 구간',x=.01,ha='left',fontsize=20)
    save_figure(fig,out/'policies.png')
    for filename in ['portfolio_summary.csv','bootstrap.csv','online_choices.json']:
        shutil.copy2(root/filename,out/filename)
    pct=lambda x:f'{100*x:.2f}%'
    gate=summary['candidate_gate_passed']
    lines=['# 타임폴리오 열지도 월별 재학습 실험','',
           '이 기록은 이미 확인한 과거 자료를 다시 사용한 개발 실험이다. 독립 최종 평가나 실계좌 성과가 아니다.','',
           ('추가 재현 검사를 진행할 후보: '+', '.join('`'+x+'`' for x in gate)+'. 아직 독립 검증을 통과하지 않았다.') if gate else
           '이번 비교에서 사전에 정한 후보 기준을 모두 통과한 열지도는 없었다. 수익률 상위 결과를 곧바로 채택하지 않는다.','',
           '22개 학습 구성, 9개 월별 재학습 시점, 4개 운용 조건을 비교했다. 각 달의 모델은 그 이전에 관측된 수익률만 사용한다. 학습 반복 수는 직전 20거래일 검증으로 고르고, 월 시작 전에 관측된 라벨로 다시 학습했다. 업종 대비 수익률·절대 수익률·업종 내 순위, 업종 중심화·변동성 조정 입력, 2차원 CNN·분리형 모델·시간축 CNN을 비교했다.','',
           '열지도 선택은 매월 과거 검증에서의 종목 순위 예측력으로 한다. 표 형태 비교군도 같은 기준으로 MLP 2개와 GBM 4개 중 고른다. 현재 달 수익률로 모델을 바꾸지 않는다.','',
           '![같은 운용 조건의 비교](policies.png)','',
           '아래 표는 업종 상한 15% 조건을 고정한 비교다. 나머지 조건과 모든 개별 구성은 CSV에 남겼다.','',
           '| 전략 | 수익률 | MDD | 평균 투자 비중 | 회전율 미달 주 |',
           '|---|---:|---:|---:|---:|']
    for model,label in labels.items():
        row=frame[(frame.model==model)&(frame.policy=='sector15')].iloc[0]
        lines.append(f'| {label} | {pct(row["return"])} | {pct(row.mdd)} | {pct(row.mean_gross)} | {int(row.low_turnover_weeks)}/{int(row.assessed_full_weeks)} |')
    lines+=['','후보 기준은 비용 차감 수익률 양수, 세 분기 중 두 분기 이상 양수, MDD −20% 이내, 회전율 누적 미달에 따른 탈락 없음이다. 현금 및 같은 운용 조건의 표 형태 모델보다 우위가 남는지도 함께 검사한다.','',
            f'열지도 구성·운용 조건·비교 기준 {summary["family_hypotheses"]}개 가설을 한 묶음으로 보정했다. 종목 행이 아닌 거래일을 5일·10일 블록으로 묶어 4,000회 재표집하고, 최대 통계량으로 우연한 상위 결과를 반영했다. 이 보정이 앞선 모든 탐색이나 과거를 재사용한 문제까지 없애지는 않는다.','',
            f'별도 체결 검사는 {audit["portfolios"]}개 경로, 매수 {audit["buy_checks"]:,}건을 확인했다. 매수 후 비중 제한 위반 {audit["post_buy_violations"]}건, 주문 수·체결 가격·수수료·거래량 참여 한도 추가 오류 {audit["additional_errors"]}건이다. NAV 재계산의 최대 차이는 {audit["max_nav_error_krw"]:.6f}원이다.','',
            '종목 목표 비중은 4%·5%, 업종 상한은 대회 상한과 15%·20% 중 낮은 값이다. 총 투자 비중은 최대 80%이며 하루 체결 주문은 최대 20건으로 제한했다. 당일 처리하지 못한 주문은 다음 거래일의 최신 신호로 목표를 다시 계산한다. 호가 단계별 대기 시간의 정확한 복원은 아니다.','',
            '현재 GICS 분류와 수집 종목 목록의 생존편향, 기업 이벤트 추정, 분봉 체결 모형의 한계는 남아 있다. 주가 상승에 따른 장중·종가 비중 초과는 매수 당시 초과와 구분해 기록한다.','',
            '![같은 종목과 날짜의 입력](inputs.png)','',
            '뉴스·매크로 실험은 이 비교에 포함하지 않았다. 별도로 저장된 당시 매크로 자산배분 기록을 사용하며, 가격만으로 비중을 조절한 경우와 비교한다. 순수 뉴스의 효과라고 해석하지 않는다.','',
            '[전체 운용 결과](portfolio_summary.csv) · [통계 비교](bootstrap.csv) · [매월 선택 기록](online_choices.json)','',
            f'모델·조건·체결·감사 원본: `{root}`']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('command',choices=['inputs','report']);ap.add_argument('--root',type=Path,default=DEST);ap.add_argument('--out',type=Path,default=OUT)
    a=ap.parse_args();(inputs if a.command=='inputs' else report)(a.root,a.out)
