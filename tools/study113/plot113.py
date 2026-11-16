"""Draw §113 source seed 0 from public frozen coordinates, without image data.

From BeeTrack root: uv run --no-sync python tools/study113/plot113.py --output docs/study113_probe.png.
Use --bundle and --tools to point to a locally staged release instead.
"""
import argparse
import gzip
from hashlib import sha256
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon, Rectangle
import numpy as np


INK, MUTED = "#173442", "#516675"
ENTER, EXIT = "#00777A", "#7253A1"
EARLY, LATE = "#B95119", "#72301B"


def center(box):
    return (box[0] + box[2])/2, (box[1] + box[3])/2


def panel(ax, meas, half, zone, w, h):
    ref = 1-half
    events = [e for e in meas["events"] if e["reference_half"] == ref]
    assert all((e["f_minus"] <= 1975) == (e["f_plus"] <= 1975) == (half == 0) for e in events)
    ax.set(xlim=(0,w), ylim=(h,0), aspect="equal", facecolor="#FBFCFC")
    ax.add_patch(Polygon(zone, closed=True, facecolor="#E4ECEC", edgecolor="#456068", lw=1.25, zorder=2))
    for direction, color in (("in", ENTER), ("out", EXIT)):
        links = [(center(e["pre_xyxy"]), center(e["post_xyxy"])) for e in events if e["direction"] == direction]
        if links:
            ax.add_collection(LineCollection(links, colors=color, linewidths=.85, alpha=.75, zorder=3))
            xy = np.array(links)[:, 1]
            ax.scatter(*xy.T, s=8, c=color, lw=0, alpha=.85, zorder=4)
        assert len(links) == meas["counts_by_half"][half][direction]["assessable"]
    for mark in meas["landmarks"][ref]:
        for offset, (rank, color, style) in enumerate(zip(mark["ranks"], (EARLY,LATE), ("solid",(0,(2.8,1.4))))):
            x, y, x2, y2 = meas["quarters"][2*ref+offset][rank]["xyxy"]
            ax.add_patch(Rectangle((x,y),x2-x,y2-y, fill=False, edgecolor=color, linestyle=style,
                                   linewidth=1.9 if offset == 0 else .95, joinstyle="miter", zorder=5+offset))
    ax.set_xticks(range(0, w, 200))
    ax.set_yticks(range(0, h, 200))
    ax.set_axisbelow(True)
    ax.grid(color="#DDE6E8", linewidth=.45)
    ax.tick_params(labelsize=8.5, colors=MUTED, length=2, pad=2.2)
    for spine in ax.spines.values():
        spine.set_color("#B8C5C9")
    scores = [e["best_score"] for e in events if e["best_score"] is not None]
    return (sum(e["direction"] == "in" for e in events), sum(e["direction"] == "out" for e in events),
            len(meas["landmarks"][ref]), max(scores) if scores else None)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", type=Path, default=Path.cwd())
    p.add_argument("--bundle", type=Path)
    p.add_argument("--tools", type=Path)
    p.add_argument("--output", type=Path, default=Path(__file__).with_suffix(".png"))
    args = p.parse_args()
    sys.path.insert(0, str((args.tools or args.repo/"tools/study113").resolve()))
    import release113
    bundle = json.loads(gzip.decompress((args.bundle or args.repo/"docs/study113_replay.json.gz").read_bytes()))
    cases, report = release113.replay(bundle, args.repo/"docs/study108_source.json.gz")
    zone_data = (args.repo/"beetrack/zones.json").read_bytes()
    zones = json.loads(zone_data)
    family = next(name for name in ("Microsoft JhengHei","Noto Sans CJK TC","Noto Sans TC","PingFang TC")
                  if name in {f.name for f in font_manager.fontManager.ttflist})
    plt.rcParams.update({"font.family":family})
    fig = plt.figure(figsize=(14, 10), facecolor="white")
    fig.text(.042, .944, "兩段來源影片都有固定地標，沒有原生穿越符合觸發條件", size=25, color=INK, weight="bold")
    fig.text(.043, .895, "每格都畫完整畫面，兩折各用 seed 0 示範。用另一半片的兩張背景圖找地標，再核對當下的穿越。",
             size=13.8, color=MUTED)
    handles = [
        Rectangle((0,0),1,1,fc="#E4ECEC",ec="#456068",label="入口區域"),
        Line2D([],[],color=ENTER,marker="o",lw=1.1,ms=4,label="進入"),
        Line2D([],[],color=EXIT,marker="o",lw=1.1,ms=4,label="離開"),
        Rectangle((0,0),1,1,fill=False,ec=EARLY,lw=1.9,label="地標早圖框"),
        Rectangle((0,0),1,1,fill=False,ec=LATE,lw=1.2,ls="--",label="地標晚圖框"),
    ]
    fig.legend(handles=handles, ncol=5, loc="center left", bbox_to_anchor=(.035,.845), frameon=False,
               handlelength=1.75, columnspacing=1.25, labelcolor=INK, fontsize=12)
    fig.text(.650,.837,"線段連接跨界前後的框中心，圓點是抵達側。",size=11.4,color=MUTED)
    for hbox, cx in ((33,.399), (35,.785)):
        fold = 68-hbox
        key = f"{fold}_0"
        case, coverage = cases[key], report["hives"][str(hbox)]
        clip = case["clip"]
        assert case["seed"] == 0 and case["hive"] == hbox and case["fold"] == fold
        assert case["hashes"]["zones"] == sha256(zone_data).hexdigest()
        assert len(coverage["seeds"]) == 9 and all(r["m"] == 0 and r["cap_pp"] == 0 for r in coverage["seeds"])
        h,w,_ = bundle["medians"][clip]["shape"]
        fig.text(cx,.783,f"來源蜂箱 {hbox}  ·  no{fold}，seed 0",ha="center",color=INK,size=16,weight="bold")
        fig.text(cx,.753,f"原畫面 {w} × {h} 像素",ha="center",color=MUTED,size=11)
        for half, bottom in ((0,.464),(1,.137)):
            height = .265
            width = height * (10/14) * w/h
            ax = fig.add_axes((cx-width/2,bottom,width,height))
            entered, left, marks, score = panel(ax,case["measurement"],half,zones[clip],w,h)
            best = "無配對" if score is None else f"{score:.3f}"
            fig.text(cx,bottom-.040,f"進 {entered}、出 {left} 筆  ·  地標 {marks} 組  ·  最高分 {best}",
                     ha="center",color=INK,size=11.8)
    for half,y in ((0,.616),(1,.288)):
        fig.text(.044,y,("前半片","後半片")[half],size=18,color=INK,weight="bold")
        fig.text(.045,y-.034,("幀 1–1975","幀 1976–3950")[half],size=12.5,color=INK)
        fig.text(.045,y-.071,("地標取自後半片","地標取自前半片")[half],size=11.7,color=MUTED)
    fig.text(.043,.061,"碰觸分數取跨界前後兩個實框與地標兩框的四個 IoU 最小值，達 0.3 才觸發。只刪觸發事件，兩箱各九個 seed 的 30 秒窗誤差改善上限皆為 0 個百分點。",
             color=INK,size=12.2)
    observations = sum(row["counts"][d]["total"] for h in report["hives"].values() for row in h["seeds"] for d in ("in","out"))
    fig.text(.043,.026,f"資料取自 BeeTrack §113。兩段來源影片共 {observations:,} 筆跨 seed 的事件觀測，同一隻蜂可能出現在多份偵測結果中。",
             color=MUTED,size=11.3)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(args.output,dpi=160,facecolor="white")
    plt.close(fig)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
