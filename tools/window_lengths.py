"""Log §154 (report only, issue #4): the counting arms scored with windows of 10, 30, 60 and 120 s and with the whole
video as one window, from the cached detections. L s windows are scored at L start offsets one second apart
(beetrack.p0.window_err); window error % = 100 x summed errors / all GT events. The main metric stays 30 s.
  default  val + dev, the four Table A arms of tools/repro_table_a.py on its three detectors (P0 = mean over the three
           cross-fitted link models), 486 GT events; range = max - min over the detectors.
  --heads  leave-one-hive-out heads (§95): loho_no<h>_s0..8 on BEE24-<h>-head (dev), one seed at a time: P0 with
           m_dev_strict.json, default ByteTrack, calibrated ByteTrack fit B (k = true / predicted totals of default
           ByteTrack on the other long video's head and tail and the four short val videos); range = max - min over seeds.
Asserts: the 30 s column reproduces Table A / §95; no error grows with the window length (the window grids nest, so
this follows from the triangle inequality); the whole-video error equals the one from per-video in / out totals. CPU only.
  PYTHONPATH=. uv run python tools/window_lengths.py [--heads] [--out DIR]
Writes window_lengths[_heads].txt (the printed tables) and .json (every number) to DIR, nothing else."""
import argparse
import json
import tempfile
import time
from pathlib import Path

import numpy as np

from beetrack.common import write_mot
from beetrack.evaluate import load_tracks
from beetrack.p0 import bytetrack, calibrated, dets, factor, link_model, nframes, p0, pooled, score, stage1, totals, window_err
from repro_table_a import DETS, EXP, NEV, SEEDS, VALDEV

WINS = (10, 30, 60, 120, None)
NAMES = ("10 s", "30 s", "60 s", "120 s", "whole")
OTHER = {"35": "33", "33": "35"}       # held-out hive -> the hive of the other long video (seen by the detector)
SHORT = ("BEE24-10", "BEE24-15", "BEE24-24", "BEE24-28")
GROUPS = ((0, 1, 2), (3, 4, 5), (6, 7, 8))
# §95 (score95_all9.txt), 30 s: per seed 0..8, 9-seed mean and range, group means (P0 and BT groups = Table B)
EXP_HEADS = {
    "35": {"P0": dict(seed=(25.43, 12.71, 15.14, 21.29, 15.25, 23.26, 46.51, 12.82, 27.24), mean=22.18, range=33.80,
                      groups=(17.76, 19.93, 28.85), bias=(32, 12, 12, 22, 15, 27, 60, 11, 34)),
           "BT": dict(seed=(23.72, 30.54, 27.55, 25.89, 24.39, 26.46, 32.71, 21.81, 28.99), mean=26.90, range=10.90,
                      groups=(27.27, 25.58, 27.84)),
           "cal BT": dict(seed=(51.78, 39.07, 41.85, 52.04, 47.53, 33.05, 78.64, 36.48, 46.00), mean=47.38, range=45.60,
                          groups=(44.23, 44.20, 53.71))},
    "33": {"P0": dict(seed=(22.17, 24.49, 23.15, 30.36, 20.65, 28.48, 17.10, 28.33, 20.40), mean=23.90, range=13.26,
                      groups=(23.27, 26.50, 21.94), bias=(-7, -35, 16, 39, -1, 31, -12, 24, -23)),
           "BT": dict(seed=(38.37, 37.64, 31.74, 28.12, 26.78, 27.50, 41.96, 33.73, 32.36), mean=33.13, range=15.18,
                      groups=(35.92, 27.46, 36.01)),
           "cal BT": dict(seed=(50.92, 33.46, 44.94, 61.92, 42.16, 47.67, 33.43, 33.08, 53.52), mean=44.57, range=28.84,
                          groups=(43.11, 50.58, 40.01))}}


def scored(rows, sp, ev, tmp):
    """p0.score of the rows as read back from a MOT file (boxes rounded to 2 decimals), as the logged numbers were."""
    f = Path(tmp) / "rows.txt"
    write_mot(f, rows)
    return score(load_tracks(f), sp, ev)


def pct(ss, k=(1, 1), n=NEV):
    """Window error % (100 x summed errors / n) of the scored videos ss, counts x k, at every length of WINS. Asserts that
    it never grows with the window length and that the whole-video error is the one from the per-video totals."""
    e = 100 * np.array([sum(window_err(s["P"], s["G"], s["nf"], k, L) for s in ss) for L in WINS]) / n
    assert all(a >= b - 1e-9 for a, b in zip(e, e[1:])), e
    assert abs(e[-1] - 100 * sum(np.abs(totals(s["P"]) * k - totals(s["G"])).sum() for s in ss) / n) < 1e-9, e
    return e


def table(say, R, B, cols):
    """Per arm: one line per window length (values per column, mean [range]), then the bias. Returns the JSON part."""
    out = {}
    for arm, r in R.items():
        r = np.array(r)                   # (columns, lengths)
        assert (np.diff(r, axis=1) <= 1e-9).all(), (arm, r)
        for j, name in enumerate(NAMES):
            say(f"  {arm:9s} {name:5s}  {' / '.join(f'{x:5.2f}' for x in r[:, j])}  mean {r[:, j].mean():5.2f} [{np.ptp(r[:, j]):5.2f}]")
        say(f"  {arm:9s} bias   {' / '.join(f'{x:+.1f}' for x in B[arm])}  mean {np.mean(B[arm]):+.1f}")
        out[arm] = {cols: r.tolist(), "mean": r.mean(0).tolist(), "range": np.ptp(r, 0).tolist(), "bias": list(map(float, B[arm]))}
    return out


def valdev(tmp, say):
    links = {(sp, k): link_model(f"m_v0_{sp}{s}") for sp in ("val", "dev") for k, s in enumerate(SEEDS)}
    strict = link_model("m_dev_strict")
    R, B, per_link, t0 = {arm: [] for arm in EXP}, {arm: [] for arm in EXP}, [], time.time()
    for d in DETS:
        tl, st, bt = {k: [] for k in range(3)}, [], {"val": [], "dev": []}
        for sp, ev in VALDEV:
            D, nf = dets(d, sp, ev), nframes(sp, ev)
            S1 = stage1(D, nf)
            for k in range(3):
                tl[k].append(scored(p0(S1, links[sp, k]), sp, ev, tmp))
            st.append(scored(p0(S1, strict), sp, ev, tmp))
            bt[sp].append(scored(bytetrack(D, nf), sp, ev, tmp))
        assert sum(s["gt"] for s in st) == NEV
        per = [pct(tl[k]) for k in range(3)]
        per_link.append(per)
        R["P0"].append(np.mean(per, 0))
        R["P0 strict"].append(pct(st))
        R["BT"].append(pct(bt["val"] + bt["dev"]))
        R["cal BT"].append(np.array([calibrated(bt, L)[0]["win"] for L in WINS]))
        # calibrated() at every length = val counts x k(dev) + dev counts x k(val), each part checked by pct
        assert np.allclose(R["cal BT"][-1], pct(bt["val"], factor(bt["dev"])) + pct(bt["dev"], factor(bt["val"])), rtol=0, atol=1e-9)
        B["P0"].append(np.mean([pooled(tl[k])["bias"] for k in range(3)]))
        B["P0 strict"].append(pooled(st)["bias"])
        B["BT"].append(pooled(bt["val"] + bt["dev"])["bias"])
        B["cal BT"].append(calibrated(bt)[0]["bias"])
        print(f"{d}: done, {(time.time() - t0) / 60:.1f} min", flush=True)

    for arm, e in EXP.items():          # the 30 s column is Table A
        w = [float(x[1]) for x in R[arm]]
        assert [round(x, 2) for x in w] == list(e["win"]), (arm, w)
        assert (round(float(np.mean(w)), 2), round(float(np.ptp(w)), 2)) == (e["mean"], e["range"]), (arm, w)
        assert [round(float(x), 1) for x in B[arm]] == list(e["bias"]), (arm, B[arm])
    assert [tuple(round(float(p[1]), 2) for p in q) for q in per_link] == list(EXP["P0"]["links"])

    say(f"val + dev ({NEV} GT events): window error % per detector ({' / '.join(DETS)}), mean [range]; bias = predicted - GT "
        f"events (calibrated: k x predicted)")
    out = table(say, R, B, "per_det")
    out["P0"]["per_link"] = np.array(per_link).tolist()      # (detector, link model, length)
    say("self-checks passed: 30 s = Table A (every detector, mean, range, bias, P0 per link model); no error grows with the "
        "window length (every arm, detector and link model); whole video = per-video totals")
    return dict(windows=NAMES, dets=DETS, gt_events=NEV, arms=out)


def heads(tmp, say):
    strict = link_model("m_dev_strict")
    res = {}
    for h, o in OTHER.items():
        tgt, t0 = f"BEE24-{h}-head", time.time()
        fit = [("dev", f"BEE24-{o}-head"), ("val", f"BEE24-{o}-tail")] + [("val", ev) for ev in SHORT]
        R, B, K = {arm: [] for arm in EXP_HEADS[h]}, {arm: [] for arm in EXP_HEADS[h]}, []
        for s in range(9):
            tag = f"loho_no{h}_s{s}"
            D, nf = dets(tag, "dev", tgt), nframes("dev", tgt)
            T, Bt = scored(p0(stage1(D, nf), strict), "dev", tgt, tmp), scored(bytetrack(D, nf), "dev", tgt, tmp)
            k = factor([scored(bytetrack(dets(tag, sp, ev), nframes(sp, ev)), sp, ev, tmp) for sp, ev in fit])
            n = T["gt"]
            R["P0"].append(pct([T], n=n))
            R["BT"].append(pct([Bt], n=n))
            R["cal BT"].append(pct([Bt], k, n))
            B["P0"].append(T["pred"] - n)
            B["BT"].append(Bt["pred"] - n)
            B["cal BT"].append(float((k * totals(Bt["P"])).sum()) - n)
            K.append(k.tolist())
            print(f"fold {h} seed {s}: done, {(time.time() - t0) / 60:.1f} min", flush=True)

        for arm, e in EXP_HEADS[h].items():      # the 30 s column is §95
            w = [float(x[1]) for x in R[arm]]
            assert [round(x, 2) for x in w] == list(e["seed"]), (h, arm, w)
            assert (round(np.mean(w), 2), round(float(np.ptp(w)), 2)) == (e["mean"], e["range"]), (h, arm, w)
            assert [round(float(np.mean([w[i] for i in g])), 2) for g in GROUPS] == list(e["groups"]), (h, arm, w)
            assert "bias" not in e or B[arm] == list(e["bias"]), (h, arm, B[arm])

        say(f"fold {h}: {tgt} (dev, {n} GT events), detectors loho_no{h}_s0..8; calibrated ByteTrack fit on "
            f"{', '.join(ev for _, ev in fit)}")
        say("  window error % per seed (0..8), 9-seed mean [range]; bias = predicted - GT events (calibrated: k x predicted)")
        res[h] = dict(target=tgt, gt_events=n, fit=fit, k=K, arms=table(say, R, B, "per_seed"))
    say("self-checks passed: 30 s = §95 (every seed, 9-seed mean and range, group means, P0 bias); no error grows with the "
        "window length (every arm and seed); whole video = per-video totals")
    return dict(windows=NAMES, folds=res)


def main():
    ap = argparse.ArgumentParser(description="Window error at 10, 30, 60, 120 s and the whole video (log §154).")
    ap.add_argument("--heads", action="store_true", help="the leave-one-hive-out heads instead of val + dev")
    ap.add_argument("--out", type=Path, default=Path("D:/beetrack_runs/w154"), help="folder for the .txt and .json")
    a = ap.parse_args()
    t0, lines = time.time(), []

    def say(s):
        print(s, flush=True)
        lines.append(s)

    with tempfile.TemporaryDirectory() as tmp:
        r = (heads if a.heads else valdev)(tmp, say)
    a.out.mkdir(parents=True, exist_ok=True)
    name = "window_lengths_heads" if a.heads else "window_lengths"
    (a.out / f"{name}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (a.out / f"{name}.json").write_text(json.dumps(r, indent=1), encoding="utf-8")
    print(f"wrote {a.out / name}.txt and .json; runtime {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
