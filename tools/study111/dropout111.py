"""Draw the BeeTrack §111 two-phase deletion experiment from its scored case packages.

From the BeeTrack repository root, run uv run --no-sync python path/to/dropout111.py.
The default input is docs/study111_replay.json.gz.
Use --preview <temporary.png> for a clearly labeled synthetic layout.
"""
import argparse
from fractions import Fraction
import gzip
import json
import math
from pathlib import Path
from statistics import mean, median

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager


INK, MUTED, TEAL, ORANGE = "#173442", "#506875", "#006D70", "#AB4825"
GRAY, GRID = "#9DAEB3", "#DEE6E7"
TRUTH = {33: 184, 35: 129}
MASKS = ((1, 2, 3), (4, 5, 6), (1, 3, 5), (2, 4, 6))


def percent_bounds(case, kind, number):
    contrast = case["results"]["contrasts"].get(kind)
    if contrast is None:
        return None
    value = contrast["05"]
    assert value["n"] == number and value["denominator"] == 2 * number
    if not number:
        assert value["lower"] is None and value["upper"] is None
        return None
    bounds = tuple(100 * value[key + "_numerator"] / (2 * number) for key in ("lower", "upper"))
    assert bounds[0] <= bounds[1]
    assert all(math.isclose(bound / 100, value[key], abs_tol=1e-12)
               for key, bound in zip(("lower", "upper"), bounds))
    return bounds


def extract(case):
    evidence = case["evidence"]
    ids = evidence["target_ids"]
    near, paired = len(ids["all"]), len(ids["paired"])
    weak = case["results"]["arms"]["native"]["05"]["target_bounds"]["weak"]["always_unmatched_ids"]
    assert len(weak) == len(set(weak)) and set(weak).issubset(ids["weak"])
    row = dict(near=near, paired=paired, weak=len(weak),
               d=percent_bounds(case, "nearall", near),
               interaction=percent_bounds(case, "near_minus_far", paired))
    assert (row["d"] is None) == (near == 0)
    return row


def read_json(file):
    opener = gzip.open if file.suffix == ".gz" else open
    with opener(file, "rt", encoding="utf8") as handle:
        return json.load(handle)


def load_cases(freeze, packages):
    identity = {digest: tuple(map(int, key.split("_"))) for key, digest in freeze["cases"].items()}
    cases = {}
    for case in packages:
        hive, seed = identity[case["freeze_sha256"]]
        assert (hive, seed) not in cases and len(case["evidence"]["gt"]) == TRUTH[hive]
        row = extract(case)
        fold = freeze["folds"][str(hive)]
        assert (row["near"], row["paired"]) == (fold["nearall_counts"][seed], fold["paired_counts"][seed])
        cases[hive, seed] = row
    assert len(cases) == 18 and freeze["study"] == 111
    return {hive: [cases[hive, seed] for seed in range(9)] for hive in TRUTH}


def load_archive(path):
    archive = read_json(path)
    data = load_cases(archive["freeze"], (case["score"] for case in archive["cases"].values()))
    for hive, rows in data.items():
        gates = archive["fold_summary"][str(hive)]["gates"]
        assert median(r["weak"] for r in rows) == gates["weak_median_count"]
        for label, field, count in (("nearall", "d", "near"), ("near_minus_far", "interaction", "paired")):
            gate = gates[label]
            assert gate["qualified"] == [s for s, r in enumerate(rows) if r[count] >= 10]
            bounds = summary(rows, field, count)
            for index, key in enumerate(("mean_lower", "mean_upper")):
                if bounds is None:
                    assert gate[key] is None
                else:
                    assert math.isclose(bounds[index], 100 * float(Fraction(gate[key])), abs_tol=1e-10)
    return data


def preview_data():
    data = {}
    for hive in TRUTH:
        rows = []
        for seed in range(9):
            n = (0, 7, 9, 10, 12, 13, 10, 15, 17)[seed] + (5 if hive == 35 else 0)
            pairs = (min(n, 2 + seed) if hive == 33 else min(n, 9 + seed))
            lo = (0, -1, 1, 2, -1, 2, 1, 0, 3)[seed] * (1 if hive == 33 else -1)
            hi = lo + (1 if seed in (1, 2, 6) else 0)
            rows.append(dict(near=n, paired=pairs, weak=seed % 5,
                             d=(50*lo/n, 50*hi/n) if n else None,
                             interaction=(-5., 5.) if hive == 35 else None))
        data[hive] = rows
    return data


def number(value):
    return "0" if abs(value) < .00001 else f"{value:+.2f}"


def interval_text(bounds):
    if bounds is None:
        return "—"
    low, high = bounds
    return number(low) if math.isclose(low, high, abs_tol=1e-9) else f"{number(low)} ～ {number(high)}"


def summary(rows, field, count):
    bounds = [r[field] for r in rows if r[count] >= 10 and r[field] is not None]
    return (mean(b[0] for b in bounds), mean(b[1] for b in bounds)) if bounds else None


def methods(fig):
    fig.text(.045, .816, "刪框位置", size=15, color=INK, weight="bold")
    fig.text(.141, .817, "同一隻蜂跨界前或後的連續八幀，以及在入口另一側的一個框。",
             size=12.5, color=MUTED)
    ax = fig.add_axes((.045, .587, .66, .204))
    ax.set(xlim=(-3.85, 7.6), ylim=(-.55, 4.08))
    ax.set_axis_off()
    ax.axvline(-.51, color=GRAY, linewidth=1.15, linestyle="--")
    for col in range(-1, 8):
        ax.text(col, 3.72, "對側" if col == -1 else str(col), color=MUTED,
                fontsize=12, ha="center", va="center")
    for j, (label, deleted) in enumerate(zip(("連續 B0", "連續 B1", "分散 S0", "分散 S1"), MASKS)):
        y = 3 - j
        ax.text(-3.82, y, label, fontsize=14, color=INK, va="center")
        ax.plot([-1, 7], [y, y], color=GRID, lw=1.35, zorder=1)
        for col in range(-1, 8):
            if col in deleted:
                ax.scatter(col, y, s=84, marker="x", color=ORANGE, lw=2.2, zorder=3)
            else:
                anchor = col in (-1, 0, 7)
                ax.scatter(col, y, s=64 if anchor else 34,
                           facecolors="white" if anchor else GRAY,
                           edgecolors=TEAL if anchor else GRAY, lw=1.8, zorder=3)
    fig.text(.717, .784, "○  固定保留的觀測框", fontsize=14, color=TEAL, weight="bold")
    fig.text(.717, .737, "×  移除的實際偵測框", fontsize=14, color=ORANGE, weight="bold")
    fig.text(.717, .675, "0 最靠近跨界，7 最遠。", fontsize=13, color=INK)
    fig.text(.717, .620, "兩個相位合計，兩種刪法\n都把內側六個框各刪一次。", fontsize=12.5, color=MUTED, linespacing=1.28)


def plot(data, path, preview=False):
    present = {font.name for font in font_manager.fontManager.ttflist}
    family = next(f for f in ("Microsoft JhengHei", "Noto Sans CJK TC", "Noto Sans TC",
                              "PingFang TC") if f in present)
    plt.rcParams.update({"font.family": family, "axes.unicode_minus": False})
    fig = plt.figure(figsize=(16, 9), facecolor="white")
    title = "連續漏掉三個偵測框，比散開漏掉三個更傷進出辨認嗎？"
    if not preview and all(sum(r["near"] >= 10 for r in rows) < 7 for rows in data.values()):
        title = "跨界時連續漏掉三個框更傷辨認嗎？兩個蜂箱都未達預設樣本門檻"
    fig.text(.045, .934, ("合成排版預覽 ｜ " if preview else "") + title, color=INK, size=24, weight="bold")
    subtitle = "入口兩側的觀測都保留，每次僅移除同一側的三個框。"
    estimates = [(h, r) for h, rows in data.items() for r in rows]
    if all(r["d"] and r["d"][0] == r["d"][1] for _, r in estimates):
        nonzero = [(h, r) for h, r in estimates if r["d"][0] != 0]
        if len(nonzero) == 2:
            details = "，".join(f"蜂箱 {h} 有一組為 {number(r['d'][0])}（{r['near']} 例）" for h, r in nonzero)
            subtitle = (f"{len(estimates)} 組有 {len(estimates)-len(nonzero)} 組的命中率差為 0。"
                        f"其餘兩組以百分點計，{details}。")
    fig.text(.045, .875, subtitle, fontsize=14.5, color=MUTED)
    methods(fig)
    fig.text(.045, .540, "跨界附近的事件命中率差", fontsize=18, weight="bold", color=INK)
    fig.text(.326, .543, "分散刪除減連續刪除，單位為百分點。正值表示連續漏框時，命中的事件較少。",
             fontsize=13, color=MUTED)

    limits = [abs(v) for rows in data.values() for r in rows if r["d"] for v in r["d"]]
    span = max(10, max(limits, default=0))
    step = next(d for d in (5, 10, 20, 25, 50, 100) if span <= 4*d)
    top = step * math.ceil(span/step)
    ticks = list(range(-top, top+1, step))
    for hive, left in zip(TRUTH, (.046, .535)):
        rows = data[hive]
        assert len(rows) == 9
        nq = sum(r["near"] >= 10 for r in rows)
        pq = sum(r["paired"] >= 10 for r in rows)
        assert all((r["interaction"] is not None) == (pq >= 7 and r["paired"] > 0) for r in rows)
        fig.text(left, .486, f"蜂箱 {hive}  ·  {TRUTH[hive]} 個標註事件", color=INK, weight="bold", size=16)
        coverage = "涵蓋達標" if nq >= 7 else "涵蓋不足"
        fig.text(left+.225, .488, f"近處至少 10 例  {nq}/9 組（{coverage}）", color=MUTED, size=12.2)
        ax = fig.add_axes((left+.137, .170, .174, .275))
        ax.set(xlim=(-1.08*top, 1.08*top), ylim=(9.5, -.5), xticks=ticks, yticks=[])
        ax.set_xticklabels([f"{x:+d}" if x else "0" for x in ticks])
        ax.grid(axis="x", color=GRID, lw=.7)
        ax.axvline(0, color="#81999D", lw=1.3)
        ax.tick_params(axis="x", length=0, colors=MUTED, labelsize=10.5, pad=3)
        ax.spines[["top", "left", "right"]].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        for pos, label in ((.012, "seed"), (.059, "近處 N"), (.105, "配對 N"), (.36, "命中率差")):
            fig.text(left+pos, .456, label, ha="center", fontsize=11, color=MUTED)
        agg = summary(rows, "d", "near")
        for i, (r, bounds) in enumerate([(r, r["d"]) for r in rows] + [(None, agg)]):
            y = .170 + (9.5-i)/10 * .275
            color = TEAL if r is None or r["near"] >= 10 else MUTED
            fig.text(left+.012, y, str(i) if r else "平均", size=11, ha="center", va="center", color=color)
            if r:
                for x, n in ((.059, r["near"]), (.105, r["paired"])):
                    fig.text(left+x, y, str(n), size=11, ha="center", va="center", color=color)
            if bounds:
                low, high = bounds
                if low == high:
                    ax.plot(low, i, marker="D" if r is None else "o", ms=5.8,
                            mec=color, mfc=color if r is None or r["near"] >= 10 else "white", zorder=3)
                else:
                    ax.plot([low, high], [i, i], color=color, lw=2, zorder=3)
                    ax.plot([low, high], [i, i], ls="none", marker="|", ms=8, color=color)
            fig.text(left+.36, y, interval_text(bounds), color=color, ha="center", va="center",
                     size=10.8, weight="bold" if r is None else "normal")
        inter = summary(rows, "interaction", "paired")
        spatial = (f"近減遠的差為 {interval_text(inter)} 個百分點" if inter else "未估近遠差")
        positive = sum(r["d"][0] > 1e-7 for r in rows if r["near"] >= 10)
        if agg is None:
            decision = "近處各組均少於 10 例，無法判讀預設門檻。"
        else:
            decision = (f"近處下界為正的有 {positive} 組（門檻 7），"
                        f"平均下界 {number(agg[0])} 個百分點（門檻 +10）。")
        fig.text(left, .122, decision, size=11, color=INK)
        fig.text(left, .091, f"同蜂近遠配對有 {pq}/9 組達 10 例，{spatial}。", size=11, color=INK)
        weak = median(r["weak"] for r in rows)
        fig.text(left, .060, f"有弱框且原本必漏的事件中位數為 {weak:g}/{TRUTH[hive]}，門檻 {math.ceil(.03*TRUTH[hive])}。",
                 size=11, color=INK)
    fig.text(.046, .015,
             ("合成資料，僅供排版。 " if preview else "資料取自 BeeTrack §111。 ")
             + "九個 seed 共用同一段影片。線段為官方 IoU 0.5 最大配對的上下界。平均只採近處達 10 例的 seed，各組等權。",
             fontsize=11, color=MUTED)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, facecolor="white")
    plt.close(fig)
    print(path.resolve())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("packages", type=Path, nargs="*", help="18 scored JSON or JSON.gz cells")
    parser.add_argument("--freeze", type=Path, help="frozen case digest and coverage index, freeze111.json")
    parser.add_argument("--archive", type=Path, help="public study111 replay archive (default: docs/study111_replay.json.gz)")
    parser.add_argument("--output", type=Path, default=Path(__file__).with_suffix(".png"))
    parser.add_argument("--preview", type=Path, help="synthetic preview only; choose a separate destination")
    args = parser.parse_args()
    if args.preview:
        if args.packages or args.freeze or args.archive or args.preview.resolve() == args.output.resolve():
            parser.error("Synthetic preview requires a separate filename and no empirical inputs.")
        plot(preview_data(), args.preview, preview=True)
    elif args.packages or args.freeze:
        if len(args.packages) != 18 or not args.freeze or args.archive:
            parser.error("Supply the freeze index and one scored case per hive/seed.")
        plot(load_cases(read_json(args.freeze), map(read_json, args.packages)), args.output)
    else:
        plot(load_archive(args.archive or Path("docs/study111_replay.json.gz")), args.output)


if __name__ == "__main__":
    main()
