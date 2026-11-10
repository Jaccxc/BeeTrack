"""Trajectory and counting analytics from tracker output (Section 7A): figures for the report.
run: uv run python -m beetrack.analytics --split test --tracker ours --seq BEE24-36
writes runs/figs/<seq>_<tracker>_{trajectories,heatmap,counts,lengths}.png and <seq>_<tracker>_analytics.json
"""
import argparse
import json

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .common import RUNS, frame_path, split_specs  # noqa: E402
from .evaluate import BAND, load_tracks  # noqa: E402

FPS = 25


def crossing_events(rows, line_y, band):
    """(frame, +1 in / -1 out) for every band crossing, same rule as evaluate.crossings."""
    ev = []
    rows = rows[np.lexsort((rows[:, 0], rows[:, 1]))]
    for tid in np.unique(rows[:, 1]):
        r = rows[rows[:, 1] == tid]
        cy = r[:, 3] + r[:, 5] / 2
        side = 0
        for f, y in zip(r[:, 0], cy):
            s = 1 if y > line_y + band else -1 if y < line_y - band else side
            if side and s != side:
                ev.append((f, 1 if s < 0 else -1))
            side = s
    return np.array(ev).reshape(-1, 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--tracker", default="ours")
    ap.add_argument("--seq", default="BEE24-36")
    ap.add_argument("--bin", type=float, default=20, help="count bin in seconds")
    a = ap.parse_args()
    ev, src, s0, s1 = next(s for s in split_specs(a.split) if s[0] == a.seq)
    gt = load_tracks(RUNS / "gt" / a.split / ev / "gt" / "gt.txt")
    pr = load_tracks(RUNS / "trackers" / a.split / a.tracker / "data" / f"{ev}.txt")
    bg = cv2.cvtColor(cv2.imread(str(frame_path(src, s0))), cv2.COLOR_BGR2RGB)
    h, w = bg.shape[:2]
    out = RUNS / "figs"
    out.mkdir(parents=True, exist_ok=True)
    pre = out / f"{ev}_{a.tracker}"

    # 1. trajectories of the first 60 s, coloured by ID
    fig, ax = plt.subplots(figsize=(8, 8 * h / w))
    ax.imshow(bg, alpha=0.55)
    seg = pr[pr[:, 0] <= 60 * FPS]
    cmap = plt.get_cmap("tab20")
    for tid in np.unique(seg[:, 1]):
        r = seg[seg[:, 1] == tid]
        ax.plot(r[:, 2] + r[:, 4] / 2, r[:, 3] + r[:, 5] / 2, lw=1.2, color=cmap(int(tid) % 20))
    ax.axhline(h / 2, color="w", lw=1.5, ls="--")
    ax.axhspan(h / 2 - BAND * h, h / 2 + BAND * h, color="w", alpha=0.15)
    ax.set_axis_off()
    fig.savefig(f"{pre}_trajectories.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 2. occupancy heatmap over the whole sequence
    cx, cy = pr[:, 2] + pr[:, 4] / 2, pr[:, 3] + pr[:, 5] / 2
    H, _, _ = np.histogram2d(cy, cx, bins=(h // 10, w // 10), range=((0, h), (0, w)))
    fig, ax = plt.subplots(figsize=(8, 8 * h / w))
    ax.imshow(bg, alpha=0.5)
    ax.imshow(cv2.GaussianBlur(H, (0, 0), 1.5), cmap="inferno", alpha=0.6, extent=(0, w, h, 0))
    ax.set_axis_off()
    fig.savefig(f"{pre}_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 3. in / out counts per time bin, GT vs prediction
    n = s1 - s0 + 1
    edges = np.arange(0, n + a.bin * FPS, a.bin * FPS)
    fig, ax = plt.subplots(figsize=(8, 3))
    res = {}
    for name, rows, style in (("GT", gt, "-"), (a.tracker, pr, "--")):
        e = crossing_events(rows, h / 2, BAND * h)
        cin = np.histogram(e[e[:, 1] > 0, 0], edges)[0]
        cout = np.histogram(e[e[:, 1] < 0, 0], edges)[0]
        t = edges[:-1] / FPS
        ax.step(t, cin, where="post", ls=style, color="#d48a00", label=f"{name} in")
        ax.step(t, cout, where="post", ls=style, color="#3b5b92", label=f"{name} out")
        res[name] = dict(total_in=int(cin.sum()), total_out=int(cout.sum()), per_bin_in=cin.tolist(), per_bin_out=cout.tolist())
    ax.set_xlabel("time (s)")
    ax.set_ylabel(f"crossings / {a.bin:g} s")
    ax.legend(ncol=4, fontsize=8, frameon=False)
    fig.savefig(f"{pre}_counts.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 4. track length distribution (fragmentation shows up as many short predicted tracks)
    fig, ax = plt.subplots(figsize=(5, 3))
    for name, rows, c in (("GT", gt, "#888888"), (a.tracker, pr, "#d48a00")):
        lens = np.unique(rows[:, 1], return_counts=True)[1] / FPS
        ax.hist(lens, bins=np.logspace(-1.5, np.log10(n / FPS), 30), histtype="step", lw=1.5, color=c, label=name)
        res[name].update(n_tracks=int(len(lens)), median_track_s=float(np.median(lens)))
    ax.set_xscale("log")
    ax.set_xlabel("track length (s)")
    ax.set_ylabel("tracks")
    ax.legend(frameon=False)
    fig.savefig(f"{pre}_lengths.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # mean speed of predicted tracks (px / s), a simple activity statistic
    sp = []
    for tid in np.unique(pr[:, 1]):
        r = pr[pr[:, 1] == tid]
        if len(r) > 5:
            c = np.c_[r[:, 2] + r[:, 4] / 2, r[:, 3] + r[:, 5] / 2]
            sp.append(np.linalg.norm(np.diff(c, axis=0), axis=1).mean() * FPS)
    res[a.tracker]["median_speed_px_s"] = float(np.median(sp)) if sp else 0.0
    (out / f"{ev}_{a.tracker}_analytics.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: {kk: v for kk, v in d.items() if not kk.startswith("per_bin")} for k, d in res.items()}, indent=1))


if __name__ == "__main__":
    main()
