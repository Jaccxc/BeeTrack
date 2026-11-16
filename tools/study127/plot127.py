"""Plot the registered §127 event-origin breakdown from its integer replay bundle."""
from __future__ import annotations

import argparse
import gzip
import json
from collections import Counter
from pathlib import Path

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from replay_counts127 import count_cell

DEFAULT_BUNDLE = Path("docs/study127_attribution.json.gz")
DEFAULT_OUTPUT = Path("runs/study127_origins.png")
INK = "#24313a"
COLORS = {
    "transient": "#16834a",
    "different": "#2878b5",
    "competition": "#d9822b",
    "no_owner": "#687782",
    "other": "#9467bd",
}
CLASSES = [
    ("no_owner", "No assigned owner at ≥1 endpoint"),
    ("competition", "Competing owner"),
    ("different", "Different GT bee segments"),
    ("transient", "Same bee, net 0 (confirmed transit)"),
    ("other", "Other classifications"),
]


def cell_index(cells: list[dict]) -> dict[tuple, dict]:
    index = {}
    for cell in cells:
        key = (cell["arm"], cell["sequence"], cell.get("seed"), round(float(cell["threshold"]), 1))
        if key in index:
            raise ValueError(f"duplicate cell {key}")
        # The public rows are per-event evidence. Rebuild declared aggregate counters first.
        count_cell(cell)
        index[key] = cell
    expected = set()
    for seq in ("BEE24-33-head", "BEE24-35-head"):
        expected.update((arm, seq, None, t) for arm in ("C", "G") for t in (.5, .3))
        expected.update(("U", seq, seed, t) for seed in range(3) for t in (.5, .3))
    if set(index) != expected:
        raise ValueError(f"expected complete 20-cell §127 bundle, found {len(index)} cells")
    return index


def classes_for(events: list[dict], *, transient_in_c_only: bool) -> dict[str, int]:
    counts = Counter()
    for event in events:
        cat = event["category"]
        if cat == "01_no_owner":
            label = "no_owner"
        elif cat == "02_ambiguous_owner":
            label = "competition"
        elif cat == "03_wrong_side_geometry":
            label = "other"
        elif cat == "04_different_bees":
            label = "different"
        elif cat == "05_same_bee_genuine_transient_net0":
            label = "transient"
        elif cat == "05b_same_bee_net0_outside_C":
            label = "other" if transient_in_c_only else "transient"
        elif cat in ("06_same_bee_official_direction", "07_same_bee_opposite_net"):
            label = "other"
        else:
            raise ValueError(f"unknown §127 category {cat}")
        counts[label] += 1
    return {key: counts[key] for key, _ in CLASSES}


def label_c33(index: dict[tuple, dict], threshold: float) -> dict[str, int]:
    cell = index[("C", "BEE24-33-head", None, threshold)]
    counts = classes_for(cell["events"], transient_in_c_only=True)
    if cell["predicted_events"] != 67 or sum(counts.values()) != 67:
        raise ValueError("C/33 reference must have the fixed 67-event denominator")
    return counts


def no_edge_u(index: dict[tuple, dict], sequence: str, seed: int) -> tuple[dict[str, int], int]:
    cell = index[("U", sequence, seed, .5)]
    no_edge = [e for e in cell["events"] if e["graph_status"] == "no_legal_edge"]
    counts = classes_for(no_edge, transient_in_c_only=False)
    if sum(counts.values()) != len(no_edge):
        raise AssertionError("no-edge categories must partition their fixed denominator")
    return counts, len(no_edge)


def plot(index: dict[tuple, dict], output: Path) -> None:
    fig = Figure(figsize=(11.6, 8.8), facecolor="white")
    FigureCanvasAgg(fig)
    ax_top = fig.add_axes([.16, .70, .78, .13])
    ax_bot = fig.add_axes([.28, .25, .66, .36])
    fig.text(.055, .96, "Origins of P0 events with no official GT edge", color=INK,
             fontsize=16, fontweight="bold", va="top")
    fig.text(.055, .915, "§127 endpoint attribution · IoU 0.5 primary; C/33 also shows IoU 0.3 sensitivity",
             color=INK, fontsize=9, va="top")

    # The registered example is all 67 C/33 predictions, shown at both event IoUs.
    c_rows = [("C / 33 · IoU 0.5", label_c33(index, .5)),
              ("C / 33 · IoU 0.3", label_c33(index, .3))]
    visible_c = ["no_owner", "competition", "different", "transient", "other"]
    for yi, (name, counts) in enumerate(c_rows):
        total = sum(counts.values())
        shown = {key: counts[key] for key in visible_c if key != "other"}
        shown["other"] = total - sum(shown.values())
        if shown["other"] < 0:
            raise ValueError("selected C/33 classes exceed its fixed event count")
        left = 0.0
        for cls in visible_c:
            value = shown[cls]
            width = value / total if total else 0
            ax_top.barh(yi, width, left=left, height=.54, color=COLORS[cls], edgecolor="white", linewidth=.6)
            if value:
                ax_top.text(left + width / 2, yi, str(value), ha="center", va="center",
                            color="white" if width > .09 else INK, fontsize=9, fontweight="bold")
            left += width
        ax_top.text(1.012, yi, f"n={total}", va="center", fontsize=9, color=INK)
    ax_top.set_yticks(range(2), [row[0] for row in c_rows], fontsize=9)
    ax_top.set_xlim(0, 1.05)
    ax_top.set_xticks([0, .25, .5, .75, 1], ["0", "25", "50", "75", "100%"])
    ax_top.set_title("C/33: all 67 P0 predictions", loc="left", fontsize=11, color=INK, pad=7)
    ax_top.grid(axis="x", color="#e7ecef", linewidth=.7, zorder=0)
    ax_top.set_axisbelow(True)
    ax_top.spines[["top", "right", "left"]].set_visible(False)
    ax_top.tick_params(axis="y", length=0, pad=8)

    u_rows = []
    for hive, seq in ((33, "BEE24-33-head"), (35, "BEE24-35-head")):
        for seed in range(3):
            counts, n = no_edge_u(index, seq, seed)
            u_rows.append((f"{hive} · s{seed} · n={n}", counts, n))
    display_order = ["no_owner", "competition", "different", "transient", "other"]
    y_positions = list(range(len(u_rows)))
    for yi, (name, counts, n) in zip(y_positions, u_rows):
        left = 0.0
        for cls in display_order:
            value = counts[cls]
            width = value / n if n else 0
            ax_bot.barh(yi, width, left=left, height=.62, color=COLORS[cls], edgecolor="white", linewidth=.5)
            if value and width >= .055:
                ax_bot.text(left + width / 2, yi, str(value), ha="center", va="center",
                            fontsize=8, color="white" if width > .11 else INK, fontweight="bold")
            left += width
        ax_bot.text(1.025, yi, " / ".join(str(counts[k]) for k in display_order),
                    ha="left", va="center", fontsize=7.5, color=INK)
    ax_bot.set_yticks(y_positions, [r[0] for r in u_rows], fontsize=8.5)
    ax_bot.invert_yaxis()
    ax_bot.set_xlim(0, 1.53)
    ax_bot.set_xticks([0, .25, .5, .75, 1], ["0", "25", "50", "75", "100%"])
    ax_bot.set_title("U: composition of official no-edge events at IoU 0.5", loc="left", fontsize=11, color=INK, pad=8)
    ax_bot.grid(axis="x", color="#e7ecef", linewidth=.7, zorder=0)
    ax_bot.set_axisbelow(True)
    ax_bot.spines[["top", "right", "left"]].set_visible(False)
    ax_bot.tick_params(axis="y", length=0, pad=8)
    ax_bot.text(1.025, -.25, "Counts: no-owner / competing / different / net-0 / other",
                ha="left", va="bottom", fontsize=7, color=INK)

    legend = [Line2D([], [], color=COLORS[k], marker="s", linestyle="none", markersize=8, label=label)
              for k, label in CLASSES]
    fig.legend(handles=legend, loc="lower center", bbox_to_anchor=(.53, .025), ncol=4,
               frameon=False, fontsize=8, columnspacing=1.2, handletextpad=.45)
    fig.text(.28, .19,
             "Bars show shares of each no-edge denominator; segment labels are counts. No owner = ≥1 endpoint lacks an assigned IoU owner.",
             fontsize=7.5, color=INK, va="top")
    fig.text(.28, .162,
             "Net 0 combines same-bee transits inside/outside C for U; C/33 counts only confirmed C members.",
             fontsize=7.5, color=INK, va="top")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160, facecolor="white")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("bundle", nargs="?", type=Path, default=DEFAULT_BUNDLE)
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = ap.parse_args()
    with gzip.open(args.bundle, "rt", encoding="utf-8") as f:
        bundle = json.load(f)
    if bundle.get("format") != "study127-event-attribution-v1":
        raise ValueError("unsupported public integer event package")
    plot(cell_index(bundle["cells"]), args.output)
    print(args.output)


if __name__ == "__main__":
    main()
