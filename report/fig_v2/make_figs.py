"""Charts for report_v2_en.html. Every number is typed in by hand from report/RESULTS_v2.md (table named in each
block); nothing is read from runs/. Run from anywhere:  .venv/Scripts/python report/fig_v2/make_figs.py
The asserts check that each typed-in mean and range matches its three member values, so a typo fails loudly."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).parent
OURS, BASE, INK, MUTE, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e4e4e0"
plt.rcParams.update({"font.size": 8, "axes.edgecolor": MUTE, "axes.labelcolor": INK, "xtick.color": MUTE,
                     "ytick.color": INK, "axes.spines.top": False, "axes.spines.right": False})


def check(rows):
    """rows: (label, three member values, mean, range, is_baseline); mean and range as printed in RESULTS_v2."""
    for name, v, m, r, _ in rows:   # 0.011: the log rounds members and range separately (35-head E3: 11.81 vs 11.82)
        if not v:                   # E3 on val + dev is one number, no members
            continue
        assert abs(sum(v) / 3 - m) < 0.011, (name, sum(v) / 3, m)
        if r is not None:
            assert abs(max(v) - min(v) - r) < 0.011, (name, max(v) - min(v), r)
            if abs(max(v) - min(v) - r) > 0.005:
                print("rounding:", name, round(max(v) - min(v), 3), "printed", r)


def panel(ax, rows, title, ref, xmax):
    """Bars = mean, dots = the three members (detectors, linkers or seed groups), dashed line = P0 mean."""
    y = range(len(rows))[::-1]
    for yi, (name, v, m, _, base) in zip(y, rows):
        ax.barh(yi, m, height=0.62, color=BASE if base else OURS, alpha=0.85, zorder=2)
        ax.plot(v, [yi] * len(v), "o", ms=3.2, mfc="white", mec=INK, mew=0.8, zorder=3)
        ax.text(max((*v, m)) + xmax * 0.02, yi, f"{m:.2f}", va="center", fontsize=7.5, color=INK, zorder=4,
                bbox=dict(boxstyle="square,pad=0.1", fc="white", ec="none"))
    ax.axvline(ref, color=INK, lw=0.9, ls="--", zorder=1)
    ax.set_yticks(list(y), [r[0] for r in rows])
    ax.set_xlim(0, xmax * 1.13)
    ax.grid(axis="x", color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.set_title(title, fontsize=8.5, loc="left", color=INK)
    ax.set_xlabel("30 s window error (%), lower is better")


def legend(fig, members):
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    h = [Patch(color=OURS, alpha=0.85, label="P0 and its side arms"),
         Patch(color=BASE, alpha=0.85, label="baselines"),
         Line2D([], [], ls="", marker="o", ms=3.2, mfc="white", mec=INK, label=members),
         Line2D([], [], color=INK, lw=0.9, ls="--", label="P0 mean")]
    fig.legend(handles=h, loc="lower center", ncol=4, frameon=False, fontsize=7.5, bbox_to_anchor=(0.5, 0.0))


# ---- Figure 2: RESULTS_v2 Table A (val + dev, 486 GT events). Members: 3 detectors (single arms) or 3 linkers (E3 arms).
A = [("P0", (18.89, 17.34, 19.15), 18.46, 1.81, False),
     ("P1", (18.82, 16.24, 18.90), 17.99, 2.66, False),
     ("F8", (15.81, 15.51, 16.83), 16.05, 1.32, False),
     ("E3", (), 17.07, None, False),
     ("E3x F8", (15.48, 15.40, 15.72), 15.53, None, False),
     ("E3x P1", (16.26, 16.32, 16.95), 16.51, None, False),
     ("E3x P1x F8", (15.67, 15.22, 15.64), 15.51, None, False),
     ("E3x P1↓x F8", (15.77, 15.51, 15.94), 15.74, None, False),
     ("Paper's tracker A", (36.63, 34.02, 35.10), 35.25, 2.61, True),
     ("Calibrated ByteTrack", (33.78, 29.09, 41.60), 34.82, 12.51, True),
     ("Default ByteTrack", (41.37, 38.13, 41.04), 40.18, 3.24, True)]
check(A)
A = [(n.replace("x ", "×"), *r) for n, *r in A]
fig, ax = plt.subplots(figsize=(7.0, 3.0), dpi=220)
panel(ax, A, "Seen hives (val + dev, 486 GT events)", 18.46, max(max((*r[1], r[2])) for r in A))
legend(fig, "3 detectors (E3 rows: 3 linkers)")
fig.tight_layout(rect=(0, 0.07, 1, 1))
fig.savefig(OUT / "fig_seen.png")
plt.close(fig)

# ---- Figure 3: RESULTS_v2 Table B (heads) and Table C (tails). Members: seed groups 0-2 / 3-5 / 6-8.
HEADS = {"Leave out 35: 35-head (129 GT events)": [
    ("P0 single", (17.76, 19.93, 28.85), 22.18, 11.09, False),
    ("E3", (15.87, 17.81, 27.68), 20.45, 11.82, False),
    ("E3x F8", (15.07, 15.29, 20.80), 17.05, 5.73, False),
    ("E3x P1", (8.65, 11.94, 12.63), 11.07, 3.98, False),
    ("E3x P1x F8", (9.26, 11.76, 12.80), 11.28, 3.54, False),
    ("E3x P1↓x F8 (post hoc)", (9.26, 10.54, 10.98), 10.26, 1.72, False),
    ("Calibrated ByteTrack E3", (42.84, 43.09, 52.09), 46.01, 9.25, True),
    ("Default ByteTrack single", (27.27, 25.58, 27.84), 26.90, 2.26, True)],
    "Leave out 33: 33-head (184 GT events)": [
    ("P0 single", (23.27, 26.50, 21.94), 23.90, 4.55, False),
    ("E3", (16.80, 23.95, 17.13), 19.29, 7.15, False),
    ("E3x F8", (17.42, 17.62, 18.35), 17.80, 0.92, False),
    ("E3x P1", (20.57, 22.87, 22.40), 21.95, 2.31, False),
    ("E3x P1x F8", (16.05, 22.07, 17.80), 18.64, 6.02, False),
    ("E3x P1↓x F8 (post hoc)", (17.42, 17.62, 18.35), 17.80, 0.92, False),
    ("Calibrated ByteTrack E3", (41.32, 47.48, 35.51), 41.44, 11.97, True),
    ("Default ByteTrack single", (35.92, 27.46, 36.01), 33.13, 8.55, True)]}
TAILS = {"Leave out 35: 35-tail (37 GT events)": [
    ("P0 single", (25.59, 20.24, 30.21), 25.35, 9.97, False),
    ("E3", (21.26, 17.84, 27.27), 22.12, 9.43, False),
    ("E3x F8", (13.03, 12.27, 19.41), 14.91, 7.14, False),
    ("E3x P1", (16.28, 9.49, 14.41), 13.39, 6.79, False),
    ("E3x P1↓x F8", (6.90, 8.92, 4.83), 6.88, 4.09, False),
    ("Calibrated ByteTrack E3", (53.28, 47.23, 56.26), 52.26, 9.03, True)],
    "Leave out 33: 33-tail (50 GT events)": [
    ("P0 single", (26.09, 29.07, 30.89), 28.68, 4.80, False),
    ("E3", (13.24, 21.56, 24.18), 19.66, 10.93, False),
    ("E3x F8", (17.57, 18.71, 21.12), 19.13, 3.55, False),
    ("E3x P1", (10.44, 20.84, 17.02), 16.10, 10.40, False),
    ("E3x P1↓x F8", (17.57, 18.71, 21.12), 19.13, 3.55, False),
    ("Calibrated ByteTrack E3", (14.58, 24.37, 26.80), 21.92, 12.22, True)]}

# criterion 1 on the tails as printed in RESULTS_v2 section 4: d = arm - P0 single per group
for fold, (mean_d, thr) in zip(TAILS, ((-18.46, 9.97), (-9.55, 4.80))):
    p0, arm = TAILS[fold][0], TAILS[fold][4]
    d = [a - b for a, b in zip(arm[1], p0[1])]
    assert abs(sum(d) / 3 - mean_d) < 0.011 and abs(max(arm[3], p0[3]) - thr) < 1e-9, (fold, d)

for name, tab in (("fig_heads.png", HEADS), ("fig_tails.png", TAILS)):
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.9 if tab is HEADS else 2.4), dpi=220, sharey=True)
    xmax = max(max(r[1]) for rows in tab.values() for r in rows)   # same x scale in both folds
    for ax, (title, rows) in zip(axs, tab.items()):
        check(rows)
        panel(ax, [(n.replace("x ", "×"), *r) for n, *r in rows], title, rows[0][2], xmax)
    axs[1].tick_params(labelleft=False)
    legend(fig, "seed groups 0-2 / 3-5 / 6-8")
    fig.tight_layout(rect=(0, 0.09 if tab is HEADS else 0.11, 1, 1))
    fig.savefig(OUT / name)
    plt.close(fig)
print("wrote", *sorted(p.name for p in OUT.glob("*.png")))
