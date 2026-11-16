"""Draw matched held-out comparisons by replaying public §§95/99/103/104/111 records.

From BeeTrack root: uv run --offline --no-sync python path/to/trackers118.py
"""
import argparse
from pathlib import Path
from statistics import mean
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D


INK, GRAY, GRID = '#203745', '#536B79', '#DDE6E7'
COLORS = {33:'#007570', 35:'#80619D'}
METHODS = (('ByteTrack','預設 ByteTrack',9), ('calibrated ByteTrack','校正 ByteTrack',9),
           ('TOPICTrack selected','來源選參數的 TOPICTrack',3))


def scores(repo):
    sys.path.insert(0, str((repo/'tools/study111').resolve()))
    from summary111 import references
    rows = references(repo, repo/'docs/study111_tracker_graphs.json.gz')['cases']
    records = {(r['hive'],r['seed'],r['method']):r for r in rows}
    result = []
    for name, label, n in METHODS:
        for h in (33,35):
            p = [records[h,s,'P0'] for s in range(n)]
            other = [records[h,s,name] for s in range(n)]
            ws = [b['window']-a['window'] for a,b in zip(p,other)]
            f1 = None
            if all(b['events'] is not None for b in other):
                f1 = [a['events']['0.5']['f1']-b['events']['0.5']['f1'] for a,b in zip(p,other)]
            spread = max(max(r['window'] for r in model)-min(r['window'] for r in model) for model in (p,other))
            assert sum(w>0 for w in ws) == (7 if name=='ByteTrack' else n) and all(w!=0 for w in ws)
            assert f1 is None or all(v>0 for v in f1)
            if name!='calibrated ByteTrack':
                assert mean(ws)<spread
            result.append((label,h,ws,f1))
    return result


def draw(rows,path):
    fonts = {f.name for f in font_manager.fontManager.ttflist}
    plt.rcParams.update({'font.family':next(f for f in ('Microsoft JhengHei','Noto Sans CJK TC','Noto Sans TC','PingFang TC') if f in fonts), 'axes.unicode_minus':False})
    fig=plt.figure(figsize=(16,9),facecolor='white')
    def label(x,y,s,size=13,color=INK,bold=False,align='left'):
        fig.text(x,y,s,fontsize=size,color=color,va='center',ha=align,fontweight='bold' if bold else 'normal')
    label(.044,.94,'相同的偵測框交給不同追蹤器，哪個較能數準蜂群進出？',25,bold=True)
    label(.045,.874,'在留出 33 與 35 的完整前段，用相同偵測器 seed 逐一對照。每個點代表一個 seed，菱形是配對差的平均。',15,GRAY)
    axes=[fig.add_axes(box) for box in ((.310,.287,.290,.430),(.705,.287,.242,.430))]
    row_y=(5.7,4.9,3.5,2.7,1.3,.5)
    label(.045,.764,'比較對象',13,bold=True)
    label(.221,.764,'留出蜂箱',12.5,bold=True)
    label(.310,.764,'W 改善（百分點）',15,bold=True)
    label(.605,.764,'平均差',11.5,GRAY,bold=True)
    label(.705,.764,'官方事件 F1 改善',15,bold=True)
    label(.950,.764,'平均差',11.5,GRAY,bold=True)
    for ax, limits, ticks in zip(axes,((-40,40),(-.32,.32)),((-40,-20,0,20,40),(-.3,-.15,0,.15,.3))):
        ax.set(xlim=limits,ylim=(0,6.2),yticks=[],xticks=ticks)
        ax.set_axisbelow(True)
        ax.grid(axis='x',color=GRID,lw=.7)
        ax.axvline(0,color=GRAY,lw=1.1,zorder=1)
        for side in ('top','left','right'):
            ax.spines[side].set_visible(False)
        ax.spines['bottom'].set_color(GRID)
        ax.tick_params(axis='x',colors=GRAY,labelsize=10.6,length=2)
    for k,((method,h,ws,fs),y) in enumerate(zip(rows,row_y)):
        fy=.287+.430*y/6.2
        if k%2==0:
            label(.045,fy,method,13.2,bold=True)
            label(.045,fy-.042,f'seed 0–{len(ws)-1}，每片 {len(ws)} 組',10.5,GRAY)
        label(.232,fy,str(h),15,COLORS[h],True)
        for i,vs in enumerate((ws,fs)):
            ax=axes[i]
            if vs is None:
                ax.text(.017,y,'倍率只調計數',fontsize=11,color=GRAY,va='center')
                continue
            # Keep equal-valued seeds visible, with a stable vertical offset by seed.
            offsets=[(s-(len(vs)-1)/2)*(.21/max(len(vs)-1,1)) for s in range(len(vs))]
            ax.scatter(vs,[y+o for o in offsets],s=36,edgecolors=COLORS[h],facecolors='white',linewidths=1.25,zorder=2)
            ax.scatter([mean(vs)],[y],marker='D',s=52,facecolors=COLORS[h],edgecolors='white',linewidths=.75,zorder=3)
            label((.605,.950)[i],fy,format(mean(vs),('+.2f','+.3f')[i]),11.6,COLORS[h],True)
    fig.add_artist(Line2D((.045,.969),(.213,.213),transform=fig.transFigure,color=GRID,lw=1))
    label(.045,.172,'右側表示 P0 較好。W 取對照減 P0，F1 取 P0 減對照。每個蜂箱反覆使用同一支影片。',13,bold=True)
    label(.045,.124,'對預設 ByteTrack 與來源選 TOPICTrack，兩折的平均 W 差都小於兩方法較大的 seed 全距。校正 ByteTrack 僅輸出分向計數。',11.7,GRAY)
    label(.045,.082,'來源為 BeeTrack §§95、99 及 §111 公開參照重播。W 用 30 秒、30 個窗起點；F1 用官方 IoU 0.5。',11,GRAY)
    fig.savefig(path,dpi=160,facecolor='white')
    plt.close(fig)


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--repo',type=Path,default=Path.cwd())
    cli.add_argument('--output',type=Path,default=Path(__file__).with_suffix('.png'))
    args=cli.parse_args()
    draw(scores(args.repo),args.output)
