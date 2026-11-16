"""Plot the six §115 held-out head cases.

From the public repository root, run uv run --no-sync python tools/study115/plot115.py --output docs/study115_integer.png.
--data also accepts the small target_scores.json; individual native P0 scores
then come from docs/study108_target.json.gz. --preview writes synthetic output only.
"""
import argparse
import gzip
from hashlib import sha256
import json
from math import ceil, isclose
from pathlib import Path
from statistics import mean
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle


INK, MUTED, TEAL, AMBER, ORACLE = "#203745", "#506674", "#007472", "#A65825", "#75578B"
LIGHT, GRID, GHOST = "#EFF5F4", "#D7E2E3", "#F3F0F6"
SEEDS = (0, 3, 6)


def read_json(path):
    data = path.read_bytes()
    return json.loads(gzip.decompress(data) if path.suffix == ".gz" else data)


def historical_native(archive, key):
    import numpy as np
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import maximum_bipartite_matching

    predicted = archive["masks"]["predictions"][key]["predictions"]
    old = archive["labels"][key]
    pred_index = {r[0]: i for i, r in enumerate(predicted)}
    gt_index = {r[0]: i for i, r in enumerate(old["gt"])}
    edges = old["edges05"]
    r = [pred_index[p] for p, g in edges]
    c = [gt_index[g] for p, g in edges]
    graph = csr_matrix((np.ones(len(edges)), (r,c)), shape=(len(pred_index),len(gt_index)))
    tp = sum(maximum_bipartite_matching(graph, perm_type="column") >= 0)
    return 2*tp/(len(pred_index)+len(gt_index))


def load(path, history_path, decoder):
    if path.name == "study115_replay.json.gz":
        sys.path.insert(0,str(decoder.resolve()))
        import release115
        record = release115.load_archive(path)
    else:
        record = read_json(path)
    if "roles" in record:
        assert record["study"] == 115
        payload = record["roles"]["target"]
        scores = payload["scores"]
        def native_values(h, s):
            full = payload["groups"][f"{h}_{s}_head"]["reported"]["native"]
            return [full[str(k)]["0.5"]["standalone"]["f1"] for k in range(s,s+3)]
    else:
        scores = record
        assert sha256(history_path.read_bytes()).hexdigest() == scores["archive108_sha256"]
        history = read_json(history_path)
        def native_values(h, s):
            return [historical_native(history,f"{h}_{k}_head") for k in range(s,s+3)]
    assert scores["role"] == "target"
    groups = scores["groups"]
    assert set(groups) == {f"{h}_{s}_head" for h in (33,35) for s in SEEDS}
    plot_data = {}
    for h in (33,35):
        cells = []
        for s in SEEDS:
            item = groups[f"{h}_{s}_head"]
            f, tp, w = item["0.5"]["f1"], item["0.5"]["tp"], item["count"]
            native = native_values(h,s)
            d = w["integer_minus_e3_pp"]
            assert isclose(w["integer_w_percent"]-w["e3_w_percent"],d,abs_tol=1e-9)
            assert abs(d) <= w["max_abs_delta_pp"]+1e-9
            assert isclose(mean(native),f["standalone_mean"],abs_tol=1e-10)
            assert f["selector_ceiling"]+1e-10 >= max(f["canonical"],f["geometry"])
            row = dict(e3=w["e3_w_percent"],integer=w["integer_w_percent"],delta=d,
                       geometric=f["geometry"],canonical=f["canonical"],best=f["selector_ceiling"],
                       native=native,beats_control=tp["geometry"]>tp["controls_median"])
            assert all(0<=x<=1 for x in [row["geometric"],row["canonical"],row["best"],*native])
            cells.append(row)
        fold = scores["equal_group_means"][f"no{h}_hive{h}_head"]
        assert isclose(mean(r["delta"] for r in cells),fold["count"]["integer_minus_e3_pp"],abs_tol=1e-10)
        plot_data[h] = cells
    return plot_data


def fake():
    data = {}
    for h in (33,35):
        cells=[]
        for i in range(3):
            e3 = (20 if h==33 else 23)+2.3*i
            delta = (-.64, .32, 2.16)[i] * (1 if h==33 else -1)
            f = (.36 if h==33 else .69)+.025*i
            cells.append(dict(e3=e3,integer=e3+delta,delta=delta,canonical=f,geometric=f+(.009,-.011,.016)[i],
                              native=[f-.048,f+.022,f+.054],best=f+.087,beats_control=i!=0))
        data[h]=cells
    return data


def text_at(fig,x,y,value,size=12,color=INK,bold=False,ha="left"):
    return fig.text(x,y,value,fontsize=size,color=color,weight="bold" if bold else "normal",ha=ha,va="center")


def rule(fig,x0,x1,y):
    fig.add_artist(Line2D((x0,x1),(y,y),lw=.8,color=GRID,transform=fig.transFigure))


def signed(x):
    return f"{x:+.2f}" if abs(x) >= .005 else "0.00"


def plot(data, path, preview=False):
    fonts = {font.name for font in font_manager.fontManager.ttflist}
    family = next(f for f in ("Microsoft JhengHei","Noto Sans CJK TC","Noto Sans TC","PingFang TC") if f in fonts)
    plt.rcParams.update({"font.family":family})
    fig=plt.figure(figsize=(16,9),facecolor="white")
    title = "把三個 P0 的平均計數改成整數事件，窗誤差與代表蜂的辨認有何變化？"
    costs = [mean(r["delta"] for r in rows) for rows in data.values()]
    if not preview and all(0 < w < .2 for w in costs):
        title = "把 E3 寫成整數事件，兩個留出蜂箱的平均窗誤差都增加不到 0.2 個百分點"
    text_at(fig,.041,.936,("合成排版 ｜ " if preview else "")+title,24,bold=True)
    txt = "E3 是三個 P0 的平均計數。同方向按時間分組後，固定選第 2 筆，幾何法依其他 seed 的位置支持選擇同秒代表，兩法的窗計數完全相同。"
    text_at(fig,.042,.871,txt,14.1,MUTED)
    rule(fig,.042,.963,.815)
    text_at(fig,.043,.765,"30 秒窗誤差相對 E3 的變化",17.5,bold=True)
    text_at(fig,.496,.765,"官方進出事件 F1（IoU 0.5）",17.5,bold=True)
    text_at(fig,.043,.716,"窗誤差 W 以百分比表示。差值為整數減 E3，負值代表誤差較小。",11.8,MUTED)
    text_at(fig,.496,.716,"幾何與固定代表共用三路全框。單 P0 各自在原框上評估。",11.8,MUTED)
    fig.add_artist(Rectangle((.883,.223),.085,.45,facecolor=GHOST,edgecolor="none",transform=fig.transFigure,zorder=-2))
    for pos, title in ((.046,"seed 組"),(.144,"E3 W"),(.212,"整數 W"),(.420,"差值")):
        text_at(fig,pos,.642,title,11.4,MUTED,ha="left" if pos==.046 else "center")
    text_at(fig,.312,.642,"差值（百分點）",11.4,MUTED,ha="center")
    fcol = (.662,.717,.809,.927)
    for x, label, c in zip(fcol,("幾何","固定","單 P0","最佳可達"),(TEAL,AMBER,INK,ORACLE)):
        text_at(fig,x,.649,label,12.5,c,bold=True,ha="center")
    for x,m,c,fill in zip(fcol,("o","D","+","^"),(TEAL,AMBER,INK,ORACLE),(TEAL,"white",INK,"white")):
        fig.add_artist(Line2D([x-(.031 if c==ORACLE else .022)],[.649],ls="",marker=m,mfc=fill,mec=c,ms=5.5,mew=1.35,transform=fig.transFigure))
    text_at(fig,.809,.617,"三個 seed 平均 / 全距",9.8,MUTED,ha="center")
    text_at(fig,.927,.617,"使用真實標註",9.8,ORACLE,ha="center")
    for x0,x1 in ((.044,.449),(.494,.968)):
        rule(fig,x0,x1,.590)

    deltas=[abs(r["delta"]) for rows in data.values() for r in rows]
    limit=max(2,ceil(max(deltas)))
    left=fig.add_axes((.258,.238,.110,.347))
    right=fig.add_axes((.493,.238,.129,.347))
    for ax,xlim,ticks in ((left,(-1.05*limit,1.05*limit),(-limit,0,limit)),
                          (right,(-.025,1.025),(0,.25,.5,.75,1))):
        ax.set(xlim=xlim,ylim=(.238,.585),xticks=ticks,yticks=[])
        ax.tick_params(axis="x",labelsize=9.6,colors=MUTED,length=0,pad=4)
        ax.spines[["left","right","top"]].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        ax.grid(axis="x",color=GRID,lw=.7)
        ax.set_axisbelow(True)
    left.axvline(0,color="#819BA0",lw=1.2)
    left.set_xticklabels([f"-{limit}","0",f"+{limit}"])
    right.set_xticklabels(["0",".25",".50",".75","1"])
    for hive,title_y,ys in ((33,.563,(.516,.467,.418)),(35,.369,(.326,.282,.244))):
        rows=data[hive]
        text_at(fig,.045,title_y,f"留出蜂箱 {hive}",13,INK,True)
        mw=mean(r["delta"] for r in rows)
        text_at(fig,.146,title_y,f"三組平均 {signed(mw)} 點",11.3,MUTED)
        for seed,y,r in zip(SEEDS,ys,rows):
            text_at(fig,.049,y,f"{seed}, {seed+1}, {seed+2}",12.6)
            for x,k in ((.144,"e3"),(.212,"integer")):
                text_at(fig,x,y,f"{r[k]:.2f}",12.8,INK,ha="center")
            d=r["delta"]
            left.plot([0,d],[y,y],lw=2.2,color="#8AA9A9")
            left.plot(d,y,marker="o",ms=5.2,mec=TEAL,mfc=TEAL)
            text_at(fig,.420,y,signed(d),13.3,TEAL,True,"center")
            native=r["native"]
            right.plot([min(native),max(native)],[y,y],color="#829397",lw=2.1)
            right.plot([min(native),max(native)],[y,y],ls="",marker="|",ms=7,color=INK)
            for val,marker,c,fill,ms in ((r["best"],"^",ORACLE,"white",7.5),(r["canonical"],"D",AMBER,"white",6.7),
                                        (mean(native),"+",INK,INK,7.3),(r["geometric"],"o",TEAL,TEAL,4.7)):
                right.plot(val,y,ls="",marker=marker,mec=c,mfc=fill,ms=ms,mew=1.25)
            for x,k,c in ((fcol[0],"geometric",TEAL),(fcol[1],"canonical",AMBER),(fcol[3],"best",ORACLE)):
                text_at(fig,x,y,f"{r[k]:.3f}",13.3,c,k=="geometric","center")
            text_at(fig,fcol[2],y+.007,f"{mean(native):.3f}",12,INK,False,"center")
            text_at(fig,fcol[2],y-.014,f"{min(native):.3f}–{max(native):.3f}",9.1,MUTED,False,"center")
    findings=[]
    for hive,rows in data.items():
        diffs=[r["geometric"]-r["canonical"] for r in rows]
        fixed=sum(d>1e-9 for d in diffs)
        controls=sum(r["beats_control"] for r in rows)
        findings.append(f"留 {hive} 為 {mean(diffs):+.3f}（{fixed}/3 勝固定，{controls}/3 勝任選）")
    text_at(fig,.043,.167,"幾何相對固定的 F1 平均差，"+"，".join(findings)+"。",13.2,TEAL,True)
    text_at(fig,.043,.116,"計數門檻為三組平均 W 增加至多 2 點。代表選擇門檻為平均 F1 至少提高 0.02、至少兩組提高，且至少兩組真陽性數高於 99 次同秒任選的中位數。",11.7,MUTED)
    text_at(fig,.043,.069,"最佳可達值用真實標註，在固定分組、同秒候選及三路全框內選代表。E3 的窗計數可有小數，沒有逐蜂位置可算 F1。資料取自 BeeTrack §115 兩個留出蜂箱。",11.7,MUTED)
    path.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(path,dpi=160,facecolor="white")
    plt.close(fig)
    print(path.resolve())


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data",type=Path,help="public §115 replay archive or target_scores.json")
    p.add_argument("--native108",type=Path,default=Path("docs/study108_target.json.gz"))
    p.add_argument("--tools",type=Path,default=Path("tools/study115"),help="public replay decoder")
    p.add_argument("--output",type=Path,default=Path(__file__).with_suffix(".png"))
    p.add_argument("--preview",type=Path,help="synthetic-only separate output path")
    args=p.parse_args()
    if args.preview:
        if args.data or args.preview.resolve()==args.output.resolve():
            p.error("Set a separate synthetic preview path without a real score input.")
        plot(fake(),args.preview,True)
    else:
        plot(load(args.data or Path("docs/study115_replay.json.gz"),args.native108,args.tools),args.output)
