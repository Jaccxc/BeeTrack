"""Plot the registered Study 126 arms and individual real-detection seeds."""
import argparse
import gzip
import json
from pathlib import Path

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from replay126 import replay

INK, PALETTE = "#25313a", {"G": "#59636e", "E": "#2878b5", "T": "#16834a", "C": "#9467bd", "K": "#d9822b", "U": "#59636e", "M": "#2878b5"}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("evidence", type=Path)
    ap.add_argument("output", type=Path)
    args = ap.parse_args()
    with gzip.open(args.evidence, "rt", encoding="utf-8") as f:
        records = json.load(f)["records"]
    data = {}
    for r in records:
        m = replay(r["graph_evidence"])
        if (m["w30_numerator"] != r["window"]["integer_absolute_error_total"]
                or m["w30_denominator"] != r["window"]["gt_denominator"]):
            raise ValueError("chart input does not replay to the archived integer score")
        data.setdefault(r["case"], {}).setdefault(r["family"], {}).setdefault(r["arm"], []).append(
            (r.get("seed"), m["w30_percent"], m["f1"]["0.5"]["f1"]))
    heads = sorted(data)
    fig = Figure(figsize=(10.2, 6.2), facecolor="white")
    FigureCanvasAgg(fig)
    axes = [fig.add_subplot(2, 2, i + 1) for i in range(4)]
    fig.text(.055, .965, "Study 126 · W30 and event F1@0.5", fontsize=16, fontweight="bold", color=INK)
    fig.text(.055, .93, "Per-head means; open dots show individual real-detection seeds", fontsize=9, color=INK)
    for h, head in enumerate(heads):
        for j, fam in enumerate(("gt", "real")):
            ax = axes[h * 2 + j]
            arms = ["G", "E", "T", "C", "K"] if fam == "gt" else ["U", "M", "E", "T", "K"]
            ax.set_title(f"{head} · {'GT frames' if fam == 'gt' else 'Real detections'}", loc="left", fontsize=10, color=INK)
            ax.set_xticks(range(len(arms)), arms)
            ax.set_ylabel("W30 (%)")
            ax.grid(axis="y", color="#e8ecef", linewidth=.8)
            ax.spines[["top", "right"]].set_visible(False)
            ax.set_xlim(-.55, len(arms) - .45)
            for xi, arm in enumerate(arms):
                vals = sorted(data[head].get(fam, {}).get(arm, []), key=lambda x: -1 if x[0] is None else x[0])
                if not vals:
                    raise ValueError(f"missing {head} {fam}:{arm}")
                ws = [v[1] for v in vals]
                color = PALETTE[arm]
                f1 = sum(v[2] for v in vals) / len(vals)
                if any(w is None for w in ws):
                    if not all(w is None for w in ws):
                        raise ValueError("mixed defined and undefined W30 values within an arm")
                    ax.text(xi, .04, f"W n/a\nF1 {f1:.2f}", transform=ax.get_xaxis_transform(),
                            ha="center", va="bottom", fontsize=7, color=INK)
                    continue
                mean = sum(ws) / len(ws)
                if len(ws) > 1:
                    ax.vlines(xi, min(ws), max(ws), color=color, linewidth=2, alpha=.7, zorder=2)
                    ax.scatter([xi] * len(ws), ws, facecolors="white", edgecolors=color, s=42, zorder=3)
                ax.scatter([xi], [mean], color=color, s=42, marker="D", zorder=4)
                ax.annotate(f"{mean:.1f}%\nF1 {f1:.2f}", (xi, mean), xytext=(0, 7), textcoords="offset points",
                            ha="center", va="bottom", fontsize=7, color=INK)
            ax.margins(y=.25)
    fig.legend([Line2D([], [], marker="D", color="none", markerfacecolor=INK, markeredgecolor=INK, markersize=6),
                Line2D([], [], marker="o", color="none", markerfacecolor="white", markeredgecolor=INK, markersize=6)],
               ["Mean", "Individual seed"], loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(.5, .02))
    fig.subplots_adjust(left=.09, right=.98, top=.88, bottom=.13, hspace=.42, wspace=.25)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=240, facecolor="white")


if __name__ == "__main__":
    main()
