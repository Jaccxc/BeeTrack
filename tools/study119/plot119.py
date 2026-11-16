"""Plot the six paired P0 and GT-oracle results from study 119."""

import argparse
import json
from fractions import Fraction
from math import ceil, floor, isclose, isfinite
from pathlib import Path

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, StrMethodFormatter


CLIPS = {33: "BEE24-35-tail", 35: "BEE24-33-tail"}
INK, ACCENT = "#27333a", "#006b71"


def replay_rows(report):
    """Read the primary fixed-time, no-edge family emitted by replay_capacity.py --report."""
    records = {}
    for r in report["rows"]:
        if r["key"] in records:
            raise ValueError(f"Duplicate source cell: {r['key']}")
        records[r["key"]] = r
    expected = {f"{fold}_{seed}_tail" for fold in CLIPS for seed in range(3)}
    if set(records) != expected or set(report["gates"]) != {str(f) for f in CLIPS}:
        raise ValueError("Expected six registered source cells and both source decisions.")
    rows = []
    eps = Fraction(1, 10**9)
    for fold, clip in CLIPS.items():
        gains, f1_deltas, strict = [], [], 0
        for seed in range(3):
            r = records[f"{fold}_{seed}_tail"]
            family = r["families"]["canonical_no_edge"]
            base, best, den = (family[k] for k in ("base", "best", "den"))
            if (r["clip"] != clip or r["seed"] != seed or not all(type(x) is int for x in (base,best,den))
                    or not (0 <= best <= base and den == 30*(r["gt_in"]+r["gt_out"]) and den > 0)):
                raise ValueError(f"Inconsistent source or window numerator/denominator: {r['key']}")
            gain = Fraction(100*(base-best),den)
            if not isclose(family["gain"],float(gain),abs_tol=1e-12,rel_tol=0):
                raise ValueError(f"Oracle gain differs from its integer certificate: {r['key']}")
            rows.append(dict(fold=fold,seed=seed,clip=clip,baseline_w=100*base/den,oracle_w=100*best/den,
                             baseline_f1=r["base_f1"],oracle_f1=family["f1"]))
            gains.append(gain)
            f1_deltas.append(family["f1"] - r["base_f1"])
            strict += bool(family["strict"])
        mean_gain = sum(gains)/3
        positive = sum(g>eps for g in gains)
        delta = sum(f1_deltas)/3
        capacity = mean_gain>=2 and positive>=2
        physical = strict>=2 and delta>=-float(eps)
        gate = report["gates"][str(fold)]
        gate_fields = ("clip","positive_seeds","strict_seeds","capacity","physical","advance")
        if (tuple(gate[k] for k in gate_fields) != (clip,positive,strict,capacity,physical,capacity and physical)
                or not isclose(gate["mean_w_gain_pp"],float(mean_gain),abs_tol=1e-12,rel_tol=0)
                or not isclose(gate["delta_f1"],delta,abs_tol=1e-12,rel_tol=0)):
            raise ValueError(f"Source decision differs from the six-cell replay: {clip}")
    return rows


def plot_rows(summary):
    data = summary["rows"]
    if any("families" in r for r in data):
        data = replay_rows(summary)
    rows = sorted(data, key=lambda r: (r["fold"], r["seed"]))
    expected = [(fold, seed, clip) for fold, clip in CLIPS.items() for seed in range(3)]
    if [(r["fold"], r["seed"], r["clip"]) for r in rows] != expected:
        raise ValueError("Expected seeds 0, 1 and 2 for both registered source clips.")
    for r in rows:
        for key in ("baseline_w", "oracle_w", "baseline_f1", "oracle_f1"):
            value = r[key]
            if not isfinite(value) or value < 0 or (key.endswith("_f1") and value > 1):
                raise ValueError(f"Invalid {key} for fold {r['fold']}, seed {r['seed']}")
    return rows


def plot(summary, path):
    """Write a 300 dpi PNG from the public replay report or the publication summary."""
    rows = plot_rows(summary)

    fig = Figure(figsize=(8.2, 5.0), facecolor="white")
    FigureCanvasAgg(fig)
    grid = fig.add_gridspec(1, 3, width_ratios=[1.12, 2.2, 2.2], left=.055,
                            right=.97, top=.712, bottom=.165, wspace=.22)
    labels, w_ax, f_ax = [fig.add_subplot(grid[0, col]) for col in range(3)]
    for ax in (labels, w_ax, f_ax):
        ax.set_ylim(7.9, -.65)
    labels.set_axis_off()

    title = "Count error and event F1 on the source tails"
    fig.text(.055, .94, title, color=INK, fontsize=17, weight="bold", va="top")
    if summary.get("synthetic", False):
        fig.text(.97, .955, "SYNTHETIC EXAMPLE", color=INK, fontsize=7,
                 weight="bold", ha="right", va="bottom")
    p0 = Line2D([], [], linestyle="none", marker="o", markersize=8,
                markerfacecolor="white", markeredgecolor=INK, markeredgewidth=1.4)
    oracle = Line2D([], [], linestyle="none", marker="D", markersize=4.7,
                    markerfacecolor=ACCENT, markeredgecolor=ACCENT)
    fig.legend([p0, oracle], ["Native P0", "GT-selected oracle subset"],
               loc="upper left", bbox_to_anchor=(.051, .865), frameon=False,
               borderaxespad=0, ncols=2, handlelength=1.3, columnspacing=1.6,
               labelcolor=INK, fontsize=9)

    for ax, title, subtitle in [
        (w_ax, "Window count error W (%)", "Lower is better · certified minimum"),
        (f_ax, "Event F1 at IoU 0.5", "Higher is better · same W-optimal subset"),
    ]:
        ax.text(0, 1.07, title, transform=ax.transAxes, fontsize=11, weight="bold", color=INK)
        ax.text(0, 1.008, subtitle, transform=ax.transAxes, fontsize=8.4, color=INK)
        ax.set_yticks([])
        ax.spines[["left", "top", "right"]].set_visible(False)
        ax.spines["bottom"].set_color("#69757a")
        ax.tick_params(axis="x", colors=INK, labelsize=8, length=3)
        ax.grid(axis="x", color="#e3e8e9", linewidth=.65, zorder=0)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5, min_n_ticks=3))

    w_max = max(r[k] for r in rows for k in ("baseline_w", "oracle_w"))
    w_span = max(1.0, w_max * 1.06)
    w_ax.set_xlim(-.022 * w_span, w_span)
    w_ax.xaxis.set_major_locator(MaxNLocator(nbins=5, steps=[1, 2, 2.5, 5, 10]))
    # Round F1 limits outward and retain at least 0.1 of the full scale.
    f_values = [r[k] for r in rows for k in ("baseline_f1", "oracle_f1")]
    f_low = max(0, floor((min(f_values) - .025) * 20) / 20)
    f_high = min(1, ceil((max(f_values) + .025) * 20) / 20)
    if f_high - f_low < .10:
        f_low = max(0, f_high - .10)
        f_high = min(1, max(f_low + .10, f_high))
    f_ax.set_xlim(f_low, f_high)
    f_ax.xaxis.set_major_formatter(StrMethodFormatter("{x:.2f}"))

    for idx, r in enumerate(rows):
        group, seed = divmod(idx, 3)
        y = group * 4.2 + seed + 1.0
        if seed == 0:
            labels.text(0, y - 1.0, r["clip"], va="center", color=INK, weight="bold", fontsize=9.4)
        labels.text(.065, y, f"Seed {seed}", va="center", color=INK, fontsize=9)
        for ax, suffix in [(w_ax, "w"), (f_ax, "f1")]:
            a, b = r[f"baseline_{suffix}"], r[f"oracle_{suffix}"]
            ax.plot([a, b], [y, y], color="#8c999d", linewidth=1.3, zorder=2)
            for x, handle in [(a, p0), (b, oracle)]:
                ax.plot(x, y, marker=handle.get_marker(), markersize=handle.get_markersize(),
                        markerfacecolor=handle.get_markerfacecolor(), clip_on=False,
                        markeredgecolor=handle.get_markeredgecolor(),
                        markeredgewidth=handle.get_markeredgewidth(), linestyle="none", zorder=3)
    fig.text(.055, .057,
             "Oracle additions use exact GT crossing times and only bees with no legal match to a native event.",
             color=INK, fontsize=8.5)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="png", dpi=300)


def synthetic_summary():
    values = [(27, 18, .68, .75), (23, 21, .77, .74), (25, 25, .72, .72),
              (21, 19, .83, .85), (18, 12, .81, .80), (22, 15, .79, .84)]
    rows = []
    for idx, (w0, w1, f0, f1) in enumerate(values):
        fold, seed = (33 if idx < 3 else 35), idx % 3
        rows.append(dict(fold=fold, seed=seed, clip=CLIPS[fold], baseline_w=w0,
                         oracle_w=w1, gain_pp=w0-w1, baseline_f1=f0, oracle_f1=f1))
    return {"synthetic": True, "rows": rows}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--report", "--summary", dest="summary", type=Path,
                        help="JSON written by replay_capacity.py replay --report (or a publication summary)")
    inputs.add_argument("--synthetic", action="store_true")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    summary = synthetic_summary() if args.synthetic else json.loads(args.summary.read_text(encoding="utf-8"))
    plot(summary, args.out)
