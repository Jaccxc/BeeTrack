"""Draw §114 from its public replay archive or a JSON copy of the archive's compact summary.

From the BeeTrack repository root, use uv run --no-sync python tools/study114/plot114.py --output docs/study114_return.png.
"""
import argparse
import gzip
import json
from math import isclose
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle


INK, MUTED, TEAL, ORANGE = "#183341", "#526671", "#006E70", "#A55323"
GRID, PALE = "#DBE4E5", "#EFF5F4"
METHODS = ("model", "distance", "native")


def load(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf8") as stream:
        bundle = json.load(stream)
    if "summary" in bundle:
        assert bundle["study"] == 114 and bundle["role"] == "source"
        summary = bundle["summary"]
    else:
        summary = bundle
    scores, gates = summary["scores"], summary["seed_gates"]
    assert set(scores) == {f"{f}_{s}_{p}" for f in (33,35) for s in range(3) for p in ("head","tail")}
    for hive in (33,35):
        fold = 68-hive
        per_seed = []
        for s in range(3):
            head, tail = (scores[f"{fold}_{s}_{p}"] for p in ("head","tail"))
            gate = gates[str(fold)][str(s)]
            for row in (head,tail):
                for method in METHODS:
                    val = row["summary"]["all"][method]
                    assert (val["bees"],val["groups"]) == (row["mixed_origin_bees"],row["mixed_groups"])
                    assert 0 <= val["equal_bee"] <= 1
                assert row["mixed_groups"] == row["counts"]["mixed"]
                assert row["mixed_origin_bees"] == row["counts"]["mixed_origin_bees"]
            m, d, _ = (tail["summary"]["all"][k]["equal_bee"] for k in METHODS)
            enough = min(head["mixed_origin_bees"],tail["mixed_origin_bees"]) >= 10
            assert (head["mixed_origin_bees"],tail["mixed_origin_bees"]) == (gate["head_bees"],gate["tail_bees"])
            assert gate["sufficient"] == enough and isclose(gate["gain"],m-d,abs_tol=1e-12)
            passed = enough and m >= .70 and m-d >= .10
            assert passed == gate["passes"]
            per_seed.append(passed)
        assert (sum(per_seed) >= 2) == summary["fold_gates"][str(fold)]["passes"]
    return summary


def label(fig, x, y, value, size=13, color=INK, align="left", bold=False):
    return fig.text(x,y,value,fontsize=size,color=color,ha=align,va="center",weight="bold" if bold else "normal")


def rule(fig,x0,x1,y):
    fig.add_artist(Line2D((x0,x1),(y,y),color=GRID,lw=.9,transform=fig.transFigure))


def render(data, path):
    available = {f.name for f in font_manager.fontManager.ttflist}
    family = next(f for f in ("Microsoft JhengHei","Noto Sans CJK TC","Noto Sans TC","PingFang TC") if f in available)
    plt.rcParams.update({"font.family":family})
    fig = plt.figure(figsize=(16,9),facecolor="white")
    rank, opp, gates = data["scores"], data["opportunity"], data["seed_gates"]
    gains33 = [100*gates["35"][str(s)]["gain"] for s in range(3)]
    tail35 = [gates["33"][str(s)]["tail_bees"] for s in range(3)]
    assert all(0 < d < 10 for d in gains33) and max(tail35) < 10
    label(fig,.043,.933,"辨認同蜂折返時，蜂箱 33 尾段的四特徵得分三次皆高於最近距離",25,bold=True)
    label(fig,.044,.865,f"蜂箱 33 尾段的優勢為 {min(gains33):.2f} 至 {max(gains33):.2f} 點，均低於預設的 10 點。蜂箱 35 尾段每個 seed 僅能比較 {min(tail35)} 至 {max(tail35)} 隻蜂。",15,MUTED)
    rule(fig,.043,.963,.811)
    label(fig,.043,.760,"有多少隻蜂可供比較？",18,bold=True)
    label(fig,.495,.760,"尾段的真折返能贏過得分最高的異蜂嗎？",18,bold=True)
    label(fig,.043,.711,"同一起點要同時找得到該蜂折返與另一隻蜂反向穿越。",11.7,MUTED)
    label(fig,.495,.711,"每隻原蜂等權，0 至 100 分。差值為四特徵減最近距離。",11.7,MUTED)

    # Left panel separates unique bees from distinct starting crossings. The three
    # detector columns revisit the same GT bees and have no pooled row.
    seed_x=(.308,.361,.415)
    label(fig,.046,.653,"影片時段",11.3,MUTED)
    label(fig,.183,.653,"只用標註",11.3,MUTED,"center")
    label(fig,.183,.614,"蜂數 / 起點數",10.5,MUTED,"center")
    label(fig,.361,.653,"偵測後可比較的蜂數",11.3,MUTED,"center")
    for s,x in enumerate(seed_x):
        label(fig,x,.614,f"seed {s}",10.5,MUTED,"center")
    rule(fig,.045,.448,.588)
    for h, ys in ((33,(.556,.496,.432)),(35,(.351,.291,.227))):
        ytitle,yhead,ytail=ys
        fold=68-h
        label(fig,.046,ytitle,f"來源蜂箱 {h}  ·  no{fold}",13.8,bold=True)
        for part,y,name in (("head",yhead,"前段"),("tail",ytail,"尾段")):
            o=opp[f"BEE24-{h}-{part}"]
            label(fig,.052,y,name,13)
            label(fig,.183,y,f"{o['mixed_bees']} / {o['mixed']}",14,INK,"center")
            for s,x in enumerate(seed_x):
                bee=rank[f"{fold}_{s}_{part}"]["mixed_origin_bees"]
                label(fig,x,y,str(bee),14,TEAL if bee>=10 else ORANGE,"center",True)
        if h == 33:
            rule(fig,.045,.448,.390)

    # Right panel uses a common zero-to-100 scale. Numerical columns disambiguate
    # co-located methods, particularly in the small hive-35 pool.
    col=(.752,.818,.882,.953)
    for x, title, color in zip(col,("四特徵","最近距離","P0 共 ID","差值"),(TEAL,ORANGE,INK,TEAL)):
        label(fig,x,.653,title,11.5,color,"center",True)
    for x,m,c,fill in zip(col,("o","D","+"),(TEAL,ORANGE,INK),(TEAL,"white",INK)):
        fig.add_artist(Line2D([x-.026],[.653],ls="",marker=m,mec=c,mfc=fill,ms=6.0,mew=1.4,transform=fig.transFigure))
    label(fig,.503,.653,"seed",11.5,MUTED,"center")
    label(fig,.545,.653,"蜂數",11.5,MUTED,"center")
    label(fig,.649,.653,"排序分數",11.5,MUTED,"center")
    ax=fig.add_axes((.575,.218,.144,.369))
    ax.set(xlim=(-3,103),ylim=(.218,.587),yticks=[],xticks=(0,25,50,75,100))
    ax.tick_params(axis="x",length=0,labelsize=9.5,colors=MUTED,pad=5)
    ax.grid(axis="x",color=GRID,lw=.7)
    ax.set_axisbelow(True)
    ax.spines[["top","left","right"]].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    for j,(h,ytitle,ys) in enumerate(((33,.609,(.561,.507,.453)),(35,.394,(.346,.292,.238)))):
        fold=68-h
        label(fig,.493,ytitle,f"蜂箱 {h}",12.1,INK,bold=True)
        if j:
            rule(fig,.493,.969,.421)
        for s,y in enumerate(ys):
            r=rank[f"{fold}_{s}_tail"]
            vals=[100*r["summary"]["all"][m]["equal_bee"] for m in METHODS]
            delta=100*gates[str(fold)][str(s)]["gain"]
            label(fig,.504,y,str(s),13,INK,"center")
            label(fig,.545,y,str(r["mixed_origin_bees"]),13,INK,"center")
            ax.plot((min(vals),max(vals)),(y,y),color="#B9C9CA",lw=1.5)
            for val,mkr,color,fill,ms in (
                (vals[1],"D",ORANGE,"white",7.5),(vals[2],"+",INK,INK,8.7),(vals[0],"o",TEAL,TEAL,4.8)):
                ax.plot(val,y,ls="",marker=mkr,mec=color,mfc=fill,ms=ms,mew=1.5)
            for x,v,c in zip(col[:3],vals,(TEAL,ORANGE,INK)):
                label(fig,x,y,f"{v:.2f}",13,c,"center")
            label(fig,col[3],y,f"{delta:+.2f}",13.4,TEAL,"center",True)

    fig.add_artist(Rectangle((.043,.119),.927,.051,facecolor=PALE,edgecolor="none",transform=fig.transFigure,zorder=-1))
    label(fig,.051,.144,"同一 seed 的前、尾段各需至少 10 隻，尾段四特徵至少得 70 分且比最近距離高至少 10 點。每折至少需 2/3 個 seed 過關，實測兩折皆 0/3。",12.2,bold=True)
    label(fig,.044,.086,"每個起點由標註確認。真返回勝過最強異蜂得 1 分，平手得半分。先在同蜂內平均，再比較各蜂。P0 共 ID 表示兩個候選共享原追蹤身分。",11.5,MUTED)
    label(fig,.044,.039,"資料取自 BeeTrack §114。四特徵為距離、時間、框面積比與等速外推殘差。前段 3,950 幀擬合，尾段 1,000 幀評估。三個 seed 共用同一段影片與標註蜂隻。",11.5,MUTED)
    path.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(path,dpi=160,facecolor="white")
    plt.close(fig)
    print(path.resolve())


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data",type=Path,default=Path("docs/study114_replay.json.gz"))
    p.add_argument("--output",type=Path,default=Path(__file__).with_suffix(".png"))
    args=p.parse_args()
    render(load(args.data),args.output)
