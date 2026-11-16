"""Plot source outcomes from a study117 public replay report. No footage is needed.

From BeeTrack root: python tools/study117/release117.py replay docs/study117_replay.json.gz --report report117.json
Then: python path/to/temporal117.py report117.json --output study117_temporal.png
"""
import argparse
import json
from math import isclose
from pathlib import Path
from statistics import mean

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle


ARMS = ('U', 'A', 'Z', 'Q', 'Zm', 'D')
NAMES = ('原生 640', '位移對齊', '同位置', '配對原分數平均', '固定接收格／同位置', '錯置供應格')
INK, GRAY, TEAL, RUST = '#1D3444', '#506571', '#006D66', '#9B481D'


def values(report):
    result = {}
    for hive in (33, 35):
        case = report['cases'][f'BEE24-{hive}-tail']
        fold = report['folds'][str(68-hive)]
        assert set(case) == {'0', '1', '2'}
        groups = {}
        for arm in ARMS:
            vals = [case[str(seed)][arm] for seed in range(3)]
            w = [v['w30_percent'] for v in vals]
            f1 = [v['iou']['0.5']['f1'] for v in vals]
            for v in vals:
                e = v['iou']['0.5']
                assert isclose(e['f1'], 2*e['tp']/(2*e['tp']+e['fp']+e['fn']), abs_tol=1e-10)
            assert isclose(fold['means'][arm]['w'], mean(w), abs_tol=1e-8)
            assert isclose(fold['means'][arm]['f1']['0.5'], mean(f1), abs_tol=1e-8)
            groups[arm] = (w+[mean(w)], f1+[mean(f1)])
        result[hive] = groups
    return result


def plot(report, path):
    data = values(report)
    fonts = {f.name for f in font_manager.fontManager.ttflist}
    plt.rcParams['font.family'] = next(f for f in ('Microsoft JhengHei', 'Noto Sans CJK TC', 'Noto Sans TC', 'PingFang TC') if f in fonts)
    fig = plt.figure(figsize=(16, 9), facecolor='white')
    def txt(x, y, s, size=13, color=INK, bold=False, align='left'):
        fig.text(x, y, s, fontsize=size, color=color, va='center', ha=align, fontweight='bold' if bold else 'normal')
    def line(x0, x1, y):
        fig.add_artist(Line2D((x0,x1),(y,y),transform=fig.transFigure,color='#D0DCDE',lw=.9))
    def rect(x,y,w,h,color):
        fig.add_artist(Rectangle((x,y),w,h,transform=fig.transFigure,facecolor=color,linewidth=0,zorder=-2))

    passing = [h for h in (33,35) if report['folds'][str(68-h)]['utility']['passed']]
    unchanged35 = all(data[35]['A'][m] == data[35]['U'][m] for m in (0,1))
    if not passing and unchanged35 and data[33]['A'][0][-1]>data[33]['U'][0][-1] and data[33]['A'][1][-1]>data[33]['U'][1][-1]:
        title = '借歷史特徵後，蜂箱 33 的 F1 微升、窗誤差也升，蜂箱 35 三組皆同分'
    elif not passing:
        title = '借用兩幀前的特徵，兩個來源蜂箱都未達整段進出計數門檻'
    elif len(passing) == 1:
        title = f'借用兩幀前的特徵，只有來源蜂箱 {passing[0]} 達整段計數門檻'
    else:
        title = '借用兩幀前的特徵，兩個來源蜂箱都達整段計數門檻'
    txt(.041,.942,title,25,bold=True)
    txt(.043,.881,'兩支已見來源尾段各 1,000 幀。每個 seed 用同一次 640 偵測前向產生六臂，各自走完整 P0 追蹤。',15,GRAY)
    txt(.044,.815,'每欄首列為原生 U 的實際成績，其餘為該臂減去同 seed 原生 U。',13)
    txt(.660,.815,'改善',13,TEAL,True)
    txt(.726,.815,'退步',13,RUST,True)
    txt(.792,.815,'0 表示與 U 相同',12,GRAY)
    row_y = [.60,.534,.477,.420,.363,.306]
    for idx,h in enumerate((33,35)):
        x0 = .045 + idx*.477
        x1 = x0+.438
        txt(x0,.748,f'來源蜂箱 {h}　·　no{68-h}',18,bold=True)
        txt(x0,.695,'處理',12,GRAY)
        for metric,x,subtitle in ((0,x0+.198,'W 減 U（百分點）'),(1,x0+.358,'F1 減 U')):
            txt(x,.699,subtitle,12.5,bold=True,align='center')
            for c, name in enumerate(('s0','s1','s2','平均')):
                xc = x+(c-1.5)*.0395
                txt(xc,.655,name,10.5,GRAY,align='center')
        line(x0,x1,.633)
        rect(x0-.003,.574,.448,.052,'#EFF3F4')
        for i,(arm,name,y) in enumerate(zip(ARMS,NAMES,row_y)):
            txt(x0+.001,y,f'{arm}  {name}',10.6 if arm=='Zm' else 11.7,INK,i in (0,1))
            for m,cx in enumerate((x0+.198,x0+.358)):
                for c in range(4):
                    xc = cx+(c-1.5)*.0395
                    raw = data[h][arm][m][c]
                    delta = raw-data[h]['U'][m][c]
                    if arm=='U':
                        val = f'{raw:.2f}' if m==0 else f'{raw:.3f}'
                        color = INK
                    elif abs(delta)<1e-9:
                        val,color='0',GRAY
                    else:
                        val = f'{delta:+.2f}' if m==0 else f'{delta:+.3f}'
                        if float(val)==0:
                            val = f'{delta:+.1e}'
                        color = TEAL if (delta<0 if m==0 else delta>0) else RUST
                    txt(xc,y,val,10.4 if m else 10.9,color,c==3,align='center')
        line(x0,x1,.272)
        gates = report['folds'][str(68-h)]
        u,c = (gates[k]['passed'] for k in ('utility','complexity'))
        mech=report['mechanism'][f'BEE24-{h}-tail']
        passed=len(mech['across_seeds']['passed_seeds'])
        txt(x0,.237,'全片效用'+('達標' if u else '未達')+'　·　複雜度'+('達標' if c else '未達')+f'　·　同蜂機制 {passed}/3 達標',13,TEAL if u and c else RUST,True)
        cost=report['cost_by_fold'][str(68-h)]
        timing=cost['medians_ms']
        txt(x0,.194,f'單臂偵測耗時估計每幀中位數　U {timing["u_component_sum"]:.1f} → A {timing["a_component_sum"]:.1f} ms',11.3,GRAY)
    txt(.044,.133,'整段計數須 W 平均降至少 2 點、至少兩個 seed 改善、官方 F1 平均持平以上。複雜度另要求 A 的 W 低於 Z 與 Q，且 F1 持平以上。',11.9)
    txt(.044,.087,'W 為 30 秒分向窗誤差，越低越好。F1 採官方事件 IoU 0.5，越高越好。成本用 CPU 偵測分項加總，各來源 3,000 幀次，不含 P0 追蹤。',11.1,GRAY)
    txt(.044,.047,'BeeTrack §117 來源重播。各來源三個 seed 共用影片與真蜂，留出蜂箱需另做驗證。',10.8,GRAY)
    fig.savefig(path, dpi=160, facecolor='white')
    plt.close(fig)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('report', type=Path)
    p.add_argument('--output', type=Path, default=Path(__file__).with_suffix('.png'))
    a = p.parse_args()
    plot(json.loads(a.report.read_bytes()), a.output)
