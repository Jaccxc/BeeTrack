"""Draw study 116 from the public replay or its regenerated report and camera geometry.

From the BeeTrack root: uv run --no-sync python tools/study116/plot116.py --output docs/study116_sampling.png.
--preview writes a separate, visibly labeled synthetic layout without opening experiment results.
"""
import argparse
import gzip
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon, Rectangle
import numpy as np


INK, MUTED, TEAL, AMBER = "#1F3844", "#536977", "#007573", "#A15325"
GRAY, GRID = "#B9C9CC", "#DBE6E6"
COLORS = {"U": "#415E6D", "L": TEAL, "C": AMBER, "H": "#735C85"}


def load(path, decoder, geometry=None):
    if geometry is None:
        sys.path.insert(0, str(decoder.resolve()))
        import release116
        with gzip.open(path,"rt",encoding="utf8") as f:
            bundle=release116.scalar_codec(json.load(f),inverse=True)
        report=release116.replay(bundle)
        geometry=bundle["receipts"]["geometry"]
    else:
        report=json.loads(path.read_bytes())
        geometry=json.loads(geometry.read_bytes())
    data={}
    for h in (33,35):
        clip=f"BEE24-{h}-tail"
        r=report["visibility"]["clips"][clip]
        config=geometry["cases"][clip]
        assert r["fold"]==68-h and r["n"]==len(set(r["sample_ids"]))
        assert r["dose"]["main_rect"]==config["L"]["rect"] and r["dose"]["control_rect"]==config["C"]["rect"]
        assert r["dose"]["g"]==config["L"]["gain"] and r["dose"]["delta_j"]==config["dose_difference"]
        cells=[]
        for seed in range(3):
            entry=r["seeds"][str(seed)]
            v=entry["views"]["original"]["0.5"]
            tab=v["tables"]["all"]["both"]
            counts=tab["successes"]
            gain,loss=(tab["u_to_l"][k] for k in ("gain","loss"))
            high=tab["h_only"]
            assert tab["n"]==r["n"] and tab["known_four"]==tab["n"]
            assert counts["L"]-counts["U"]==gain-loss and high["l"]<=high["n"]
            a,b=(v["regions"][arm]["outside"] for arm in "UL")
            assert a["gt"]==b["gt"] and a["gt"]>0
            cells.append(dict(counts=counts,gain=gain,loss=loss,high=high["n"],rescued=high["l"],
                              gt=a["gt"],tp_u=a["tp"],tp_l=b["tp"],fp_u=a["fp"],fp_l=b["fp"],
                              passed=entry["gate"]["passes"]))
        assert [str(i) for i,c in enumerate(cells) if c["passed"]]==r["gate"]["passing_seeds"]
        assert len({c["gt"] for c in cells})==1
        data[h]=dict(n=r["n"],rows=cells,delta_j=r["dose"]["delta_j"],
                     g=r["dose"]["g"],advance=r["gate"]["passes"])
    return data,geometry["cases"]["BEE24-33-tail"]


def preview(zone_file):
    polygon=json.loads(zone_file.read_bytes())["BEE24-33-tail"]
    w,h=950,505
    vertices=np.asarray(polygon)
    pad=.05*min(w,h)
    lo=np.maximum(vertices.min(axis=0)-pad,0); hi=np.minimum(vertices.max(axis=0)+pad,[w,h])
    size=hi-lo
    dst=max((np.array([x,y]) for x in (0,w-size[0]) for y in (0,h-size[1])),
            key=lambda p:(sum((p-lo)**2),-p[0],-p[1]))
    gain=min(1.5,*((1-.5*(1-size/[w,h]))/(size/[w,h])))
    geo=dict(size=[w,h],polygon=polygon,L=dict(rect=[*lo,*hi],gain=gain),C=dict(rect=[*dst,*(dst+size)],gain=gain))
    data={}
    for h in (33,35):
        rows=[]
        for s in range(3):
            u=4+s if h==33 else 10+s
            gains,losses=(s+2,1) if h==33 else (s,2)
            rows.append(dict(counts=dict(U=u,L=u+gains-losses,C=u+1,H=min(16,u+3)),
                             gain=gains,loss=losses,high=3,rescued=min(2,gains),gt=197 if h==33 else 105,
                             tp_u=76+3*s,tp_l=72-2*s,fp_u=5+s,fp_l=3+2*s,passed=h==33 and s>0))
        data[h]=dict(n=16,rows=rows,delta_j=1.01,g=gain,advance=h==33)
    return data,geo


def label(fig,x,y,value,size=12,color=INK,bold=False,ha="left"):
    fig.text(x,y,value,fontsize=size,color=color,ha=ha,va="center",weight="bold" if bold else "normal")


def rule(fig,x0,x1,y):
    fig.add_artist(Line2D((x0,x1),(y,y),color=GRID,lw=.8,transform=fig.transFigure))


def grid_axis(length,start,end,gain,count):
    source=np.array((0.,start,end,float(length)))
    outside=(length-gain*(end-start))/(length-(end-start)) if end-start<length else 1.
    target=np.array((0.,start*outside,start*outside+(end-start)*gain,float(length)))
    keep=np.r_[True,np.diff(source)>0]
    return np.interp(np.linspace(0,length,count+1),target[keep],source[keep])[1:-1]


def diagrams(fig,geo):
    width,height=geo["size"]
    for arm,title,x in zip("ULCH",("整圖 640","放大入口 640","放大角落 640","整圖 960"),(.049,.235,.421,.607)):
        label(fig,x,.816,f"{arm}  {title}",13.5,COLORS[arm],True)
        aw=.137
        ah=aw*(16/9)*height/width
        ax=fig.add_axes((x,.643,aw,ah))
        ax.set(xlim=(0,width),ylim=(height,0),aspect="equal",xticks=[],yticks=[],facecolor="#F8FAFA")
        corners=(0.,0.,width,height) if arm in "UH" else geo[arm]["rect"]
        x0,y0,x1,y1=corners
        gain=1 if arm in "UH" else geo[arm]["gain"]
        nx,ny=(27,15) if arm=="H" else (18,10)
        gx=grid_axis(width,x0,x1,gain,nx)
        gy=grid_axis(height,y0,y1,gain,ny)
        ax.vlines(gx,0,height,colors=GRAY,lw=.43,zorder=2)
        ax.hlines(gy,0,width,colors=GRAY,lw=.43,zorder=2)
        if arm in "LC":
            ax.add_patch(Rectangle((x0,y0),x1-x0,y1-y0,facecolor=COLORS[arm],alpha=.13,lw=0))
            ax.add_patch(Rectangle((x0,y0),x1-x0,y1-y0,fill=False,edgecolor=COLORS[arm],lw=1.2,zorder=3))
        ax.add_patch(Polygon(geo["polygon"],closed=True,fill=False,edgecolor=INK,lw=1.35,zorder=4))
        for spine in ax.spines.values():
            spine.set_color(GRAY)
    label(fig,.778,.773,"以蜂箱 33 的入口為例",13,INK,True)
    label(fig,.778,.731,"深色輪廓為入口。",11.5,MUTED)
    label(fig,.778,.694,"格線越密，局部取樣越密。",11.5,MUTED)
    label(fig,.778,.657,"四種方式都保留完整視野。",11.5,MUTED)


def plot(data,geo,path,synthetic=False):
    fonts={f.name for f in font_manager.fontManager.ttflist}
    family=next(f for f in ("Microsoft JhengHei","Noto Sans CJK TC","Noto Sans TC","PingFang TC") if f in fonts)
    plt.rcParams.update({"font.family":family})
    fig=plt.figure(figsize=(16,9),facecolor="white")
    title="同一張 640 畫布把更多像素分給入口，能否同時看見穿越前後的蜂？"
    rows=[r for group in data.values() for r in group["rows"]]
    gains=[r["counts"]["L"]-r["counts"]["U"] for r in rows]
    if not synthetic and sorted(d for d in gains if d>0)==[1] and all(r["tp_l"]<r["tp_u"] for r in rows):
        title="放大入口只在一組多看全一隻蜂的穿越前後兩端，六組在矩形外都漏掉更多框"
    label(fig,.042,.94,("合成排版 ｜ " if synthetic else "")+title,24,bold=True)
    label(fig,.043,.878,"U、L、C 每幀各做一次相同尺寸的偵測，H 用較大的整圖作參照。格線畫回原相機座標，以相同疏化比例示意實際取樣位置。",14,MUTED)
    diagrams(fig,geo)
    rule(fig,.043,.966,.615)
    label(fig,.043,.580,"同一隻蜂的兩端都配對且在正確側",16.5,bold=True)
    label(fig,.572,.580,"入口矩形以外的所有標註框",16.5,bold=True)
    label(fig,.043,.536,"每個 seed 使用同一批標註蜂，四個數是雙端皆成功的蜂數。",11.4,MUTED)
    label(fig,.572,.536,"各幀只計一次，召回差為入口放大 L 減整圖 U 的百分點。",11.4,MUTED)
    left_cols=(.056,.111,.173,.235,.297,.390,.497)
    right_cols=(.603,.687,.776,.903)
    for x,h,c in zip(left_cols,("seed","U","L","C","H","新增 / 流失","L 救回 / H 獨有"),
                     (MUTED,*COLORS.values(),MUTED,MUTED)):
        label(fig,x,.487,h,11.2,c,True,"center")
    for x,h in zip(right_cols,("U 配對","L 配對","召回差（點）","未配預測 U → L")):
        label(fig,x,.487,h,11.2,MUTED,True,"center")
    for hive,yg,ys in ((33,.438,(.392,.35,.308)),(35,.253,(.209,.167,.125))):
        r=data[hive]; den=r["rows"][0]["gt"]
        dose=r["delta_j"]
        passed=sum(c["passed"] for c in r["rows"])
        name="較密" if hive==33 else "較疏"
        label(fig,.044,yg,f"來源 {hive}（{name}）· {r['n']} 隻",12.7,INK,True)
        label(fig,.220,yg,f"放大 {r['g']:.2f} 倍，入口取樣量差 {dose:+.2f}，達標 {passed}/3",11.2,MUTED)
        label(fig,.573,yg,f"矩形外 {den:,} 筆標註框",12.7,INK,True)
        for s,(y,c) in enumerate(zip(ys,r["rows"])):
            label(fig,left_cols[0],y,str(s),12.0,INK,False,"center")
            for a,x in zip("ULCH",left_cols[1:5]):
                value=c["counts"][a]
                fig.add_artist(Rectangle((x-.021,y-.014),.042,.004,color=GRID,linewidth=0,transform=fig.transFigure))
                if value:
                    fig.add_artist(Rectangle((x-.021,y-.014),.042*value/r["n"],.004,color=COLORS[a],linewidth=0,transform=fig.transFigure))
                label(fig,x,y+.002,str(value),13.5,COLORS[a],True,"center")
            label(fig,left_cols[5],y,f"{c['gain']} / {c['loss']}",12.7,INK,False,"center")
            recovery=f"{c['rescued']} / {c['high']}" if c["high"] else "無此例"
            label(fig,left_cols[6],y,recovery,12.7,INK,False,"center")
            diff=100*(c["tp_l"]-c["tp_u"])/den
            for x,val in zip(right_cols,(str(c["tp_u"]),str(c["tp_l"]),f"{diff:+.2f}",f"{c['fp_u']} → {c['fp_l']}")):
                label(fig,x,y,val,12.6,INK,False,"center")
        if hive==33:
            rule(fig,.043,.535,.280);rule(fig,.571,.964,.280)
    label(fig,.043,.072,"新增與流失比較同隻蜂的 U 與 L。H 獨有指 H 看全且 U 沒看全。入口取樣量差為 L 減 C 的局部面積倍率，須達 0.10，且至少兩個 seed 的 L 須勝 U、C 並救回一例。",11.3,MUTED)
    label(fig,.043,.028,"資料取自 BeeTrack §116 來源尾段。主分析在原圖座標做 NMS，再以 IoU 0.5 配對。右表分母是選定影格的標註框觀測數，包含同一隻蜂重複出現。",11.3,MUTED)
    path.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(path,dpi=160,facecolor="white")
    plt.close(fig)
    print(path.resolve())


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data",type=Path,help="study116_replay.json.gz, or report116.json with --geometry")
    p.add_argument("--geometry",type=Path,help="sealed geometry.json, for report input")
    p.add_argument("--tools",type=Path,default=Path("tools/study116"))
    p.add_argument("--output",type=Path,default=Path(__file__).with_suffix(".png"))
    p.add_argument("--preview",type=Path,help="synthetic-only separate output")
    args=p.parse_args()
    if args.preview:
        if args.data or args.geometry or args.preview.resolve()==args.output.resolve():
            p.error("Synthetic preview requires a separate filename and no experiment inputs.")
        plot(*preview(Path("beetrack/zones.json")),args.preview,True)
    else:
        plot(*load(args.data or Path("docs/study116_replay.json.gz"),args.tools,args.geometry),args.output)
