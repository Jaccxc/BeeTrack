"""Log §155 (report only, issue #3): field calibration. On a new hive someone counts the first 60 s by hand, and the in
and out counts after that are multiplied by k60 = true / predicted totals of those 60 s (per direction, p0.factor's form).
Cut at frame 1500: calibration part = events at frame <= 1500; evaluation part = the later events, re-based to frame 1,
scored as a video of its own (2450 frames) with the 30 s window error (p0.window_err), % of its GT events. Events are
dated as in p0.events (last boundary crossing) on the whole video's tracks.
  heads: the leave-one-hive-out heads as in tools/window_lengths.py --heads (loho_no<h>_s0..8 on BEE24-<h>-head, one seed
    at a time). Arms, evaluation part only: 甲 P0 (m_dev_strict.json); 乙 P0 x k60; 丙 default ByteTrack x its own k60;
    丁 calibrated ByteTrack fit B (k from the seen val + dev videos, no field data); 戊 rate reference, no tracker: per
    window, the calibration part's GT in (out) per frame x the video frames in that window.
    Criterion (§95 criterion 1, single-seed version): d = 乙 - other per seed, PASS if |mean d| > max(9-seed range of both).
  tracking side: both dev videos with the same cut, the seen detectors of Table A; 甲 and 乙 = mean over the three m_v0_dev*
    link models (乙: each link model x its own k60), 丙, 戊; pooled over the two videos (211 GT events); threshold 2 points.
Asserts: 乙 with k = (1, 1) is 甲; the uncut heads reproduce §95 per seed (P0, ByteTrack, calibrated ByteTrack); the GT
counts of both parts; the rate reference on a hand case, its window counts summing to rate x frames. CPU only.
  PYTHONPATH=. uv run python tools/field_calibration.py [--out DIR]
Writes field_calibration.txt (the printed tables) and .json to DIR (default D:/beetrack_runs/w155). Tracker outputs are
kept as MOT files in DIR/cache/<name>/<split>/<video>.txt and always scored as read back from there."""
import argparse
import json
import sys
import time
from functools import cache
from pathlib import Path

import numpy as np

from beetrack.common import write_mot
from beetrack.evaluate import load_tracks
from beetrack.p0 import FPS, WIN, bytetrack, dets, factor, link_model, nframes, p0, per_window, score, stage1, totals, window_err
from repro_table_a import DETS, SEEDS, VALDEV
from window_lengths import EXP_HEADS, OTHER, SHORT

CUT, NF = 1500, 3950
NE = NF - CUT
GT_CUT = {"BEE24-35-head": ((23, 21), (42, 43)), "BEE24-33-head": ((35, 23), (60, 66))}   # §155 GT in / out: calibration, evaluation
ARMS = {"甲": "P0", "乙": "P0 x k60", "丙": "BT x k60", "丁": "cal BT B", "戊": "rate ref"}


def rate_err(rate, G, nf, win=WIN):
    """window_err of the tracker-free prediction rate (in, out per frame) x the video frames inside each window, with
    window_err's windows and offsets (at offset o the first window is o seconds shorter)."""
    w, e = win * FPS, []
    for o in range(win):
        s, n = o * FPS, nf + o * FPS
        b = np.arange(int(np.ceil(n / w)))
        f = np.minimum((b + 1) * w, n) - np.maximum(b * w, s)        # frames of the video in each window
        assert f.min() > 0 and f.sum() == nf, f                       # so the window counts sum to rate x nf
        e.append(np.abs(f[:, None] * np.asarray(rate) - per_window(G + [s, 0], n, w)).sum())
    return float(np.mean(e))


# 50-frame windows on 100 frames, one 'in' at frame 10: offset 0 windows of 50 / 50 frames, error 0 + 1; offset 1
# (25 frames later) 25 / 50 / 25 frames, error 0.5 + 1 + 0.5; mean 1.5
assert abs(rate_err((0.02, 0), np.array([[10, 1]]), 100, 2) - 1.5) < 1e-12


def split(E):
    """(calibration part, evaluation part re-based to frame 1) of the events E."""
    return E[E[:, 0] <= CUT], E[E[:, 0] > CUT] - [CUT, 0]


def tracked(cdir, name, sp, ev, run):
    """p0.score of a tracker output as read back from its MOT file cdir/name/sp/ev.txt, written once from run()."""
    f = cdir / name / sp / f"{ev}.txt"
    if not f.exists():
        t = f.with_suffix(".tmp")
        write_mot(t, run())
        t.replace(f)
    return score(load_tracks(f), sp, ev)


def field(T, B, ev, kB=None):
    """One video, evaluation part: window errors (not normalised) and biases (k x predicted - GT events) of the arms, and
    k60 of P0 and ByteTrack. T, B = p0.score of P0 and default ByteTrack on the whole video; kB = calibrated ByteTrack's k."""
    (Pc, Pe), (Bc, Be), (Gc, Ge) = split(T["P"]), split(B["P"]), split(T["G"])
    assert T["nf"] == NF and (B["G"] == T["G"]).all() and (tuple(totals(Gc)), tuple(totals(Ge))) == GT_CUT[ev], ev
    kp, kb = factor([dict(P=Pc, G=Gc)]), factor([dict(P=Bc, G=Gc)])
    pred = {"甲": (Pe, np.ones(2)), "乙": (Pe, kp), "丙": (Be, kb)} | ({} if kB is None else {"丁": (Be, kB)})
    e = {a: window_err(P, Ge, NE, k) for a, (P, k) in pred.items()}
    b = {a: float((k * totals(P)).sum()) - len(Ge) for a, (P, k) in pred.items()}
    assert e["甲"] == window_err(Pe, Ge, NE)          # 乙's path with k = (1, 1) is plain P0
    rate = totals(Gc) / CUT
    e["戊"], b["戊"] = rate_err(rate, Ge, NE), float(rate.sum() * NE) - len(Ge)
    return e, b, kp, kb


def show(say, V, Bs):
    """Per arm: values per column, mean [range]; then the biases."""
    for a, v in V.items():
        say(f"  {a} {ARMS[a]:9s} {' / '.join(f'{x:6.2f}' for x in v)}  mean {np.mean(v):6.2f} [{np.ptp(v):6.2f}]")
    say("  bias (k x predicted - GT events, evaluation part)")
    for a, v in Bs.items():
        say(f"  {a} {ARMS[a]:9s} {' / '.join(f'{x:+6.1f}' for x in v)}  mean {np.mean(v):+6.1f}")
    return {a: dict(values=list(map(float, V[a])), mean=float(np.mean(V[a])), range=float(np.ptp(V[a])), bias=list(map(float, Bs[a])),
                    bias_mean=float(np.mean(Bs[a]))) for a in V}


def kfmt(ks):
    return ", ".join(f"{k[0]:.3f}/{k[1]:.3f}" for k in ks)


def heads(cdir, say):
    strict, res = link_model("m_dev_strict"), {}
    for h, o in OTHER.items():
        tgt, t0 = f"BEE24-{h}-head", time.time()
        fit = [("dev", f"BEE24-{o}-head"), ("val", f"BEE24-{o}-tail")] + [("val", ev) for ev in SHORT]
        n = sum(GT_CUT[tgt][1])
        V, Bs, K = {a: [] for a in ARMS}, {a: [] for a in ARMS}, {"P0": [], "BT": [], "fit B": []}
        for s in range(9):
            tag = f"loho_no{h}_s{s}"
            D, nf = dets(tag, "dev", tgt), nframes("dev", tgt)
            T = tracked(cdir, f"p0s_{tag}", "dev", tgt, lambda: p0(stage1(D, nf), strict))
            B = tracked(cdir, f"bt_{tag}", "dev", tgt, lambda: bytetrack(D, nf))
            kB = factor([tracked(cdir, f"bt_{tag}", sp, ev, lambda: bytetrack(dets(tag, sp, ev), nframes(sp, ev))) for sp, ev in fit])
            for arm, x in (("P0", T["err"]), ("BT", B["err"]), ("cal BT", window_err(B["P"], B["G"], nf, kB))):   # uncut = §95
                assert round(100 * x / T["gt"], 2) == EXP_HEADS[h][arm]["seed"][s], (h, s, arm, x)
            e, b, kp, kb = field(T, B, tgt, kB)
            for a in ARMS:
                V[a].append(100 * e[a] / n)
                Bs[a].append(b[a])
            for key, k in (("P0", kp), ("BT", kb), ("fit B", kB)):
                K[key].append(k.tolist())
            print(f"fold {h} seed {s}: done, {(time.time() - t0) / 60:.1f} min", flush=True)

        (ci, co), (ei, eo) = GT_CUT[tgt]
        say(f"fold {h}: {tgt} (dev), loho_no{h}_s0..8; calibration part frames 1-{CUT} GT in {ci} out {co}; evaluation part "
            f"frames {CUT + 1}-{NF} ({NE} frames) GT in {ei} out {eo} ({n})")
        say("  evaluation part, 30 s window error % per seed (0..8), 9-seed mean [range]")
        r = dict(target=tgt, fit=fit, gt_cal=(ci, co), gt_eval=(ei, eo), arms=show(say, V, Bs), k60_P0=K["P0"], k60_BT=K["BT"], k_fitB=K["fit B"])
        for key in K:
            say(f"  k {key:5s} (in/out) per seed: {kfmt(K[key])}")
        r["criterion"] = {}
        for x in ("甲", "丙", "丁", "戊"):
            d, lim = np.subtract(V["乙"], V[x]), max(np.ptp(V["乙"]), np.ptp(V[x]))
            ok = bool(abs(d.mean()) > lim)
            say(f"  乙 vs {x}: mean d (乙 - {x}) {d.mean():+.2f}; |mean d| {abs(d.mean()):.2f} > max(range 乙 {np.ptp(V['乙']):.2f}, "
                f"range {x} {np.ptp(V[x]):.2f}) = {lim:.2f}: {'PASS' if ok else 'FAIL'}; mean d negative: "
                f"{'yes' if d.mean() < 0 else 'no'}; 乙 < {x} in {(d < 0).sum()} / 9 seeds")
            r["criterion"][x] = dict(d=d.tolist(), mean_d=float(d.mean()), limit=float(lim), passed=ok, better=int((d < 0).sum()))
        res[h] = r
    return res


def tracking(cdir, say):
    links = [link_model(f"m_v0_dev{s}") for s in SEEDS]
    devs = [ev for sp, ev in VALDEV if sp == "dev"]
    n = sum(sum(GT_CUT[ev][1]) for ev in devs)
    V, Bs, K = {a: [] for a in ARMS if a != "丁"}, {a: [] for a in ARMS if a != "丁"}, []
    t0 = time.time()
    for d in DETS:
        R = {}
        for ev in devs:
            D, nf = dets(d, "dev", ev), nframes("dev", ev)
            S1 = cache(lambda: stage1(D, nf))
            B = tracked(cdir, f"bt_{d}", "dev", ev, lambda: bytetrack(D, nf))
            R[ev] = [field(tracked(cdir, f"p0_{d}_l{j}", "dev", ev, lambda: p0(S1(), links[j])), B, ev) for j in range(3)]
        for a in V:          # 甲, 乙: mean over the link models of the pooled values; 丙, 戊 do not use them
            js = range(3) if a in ("甲", "乙") else [0]
            V[a].append(100 * np.mean([sum(R[ev][j][0][a] for ev in devs) for j in js]) / n)
            Bs[a].append(float(np.mean([sum(R[ev][j][1][a] for ev in devs) for j in js])))
        K.append({ev: {"P0": [R[ev][j][2].tolist() for j in range(3)], "BT": R[ev][0][3].tolist()} for ev in devs})
        print(f"{d}: done, {(time.time() - t0) / 60:.1f} min", flush=True)

    say(f"tracking side: dev {' + '.join(devs)}, same cut; seen detectors; evaluation parts pooled ({n} GT events)")
    say(f"  evaluation part, 30 s window error % per detector ({' / '.join(DETS)}), mean [range]")
    r = dict(videos=devs, gt_eval=n, dets=DETS, arms=show(say, V, Bs), k60=K)
    for d, k in zip(DETS, K):
        say(f"  k60 {d}: " + "; ".join(f"{ev} P0 links {kfmt(k[ev]['P0'])}, BT {kfmt([k[ev]['BT']])}" for ev in devs))
    dm = float(np.mean(V["乙"]) - np.mean(V["甲"]))
    say(f"  乙 - 甲 on the detector mean: {dm:+.2f}; |d| > 2 points: {'yes' if abs(dm) > 2 else 'no'}")
    r["d"] = dm
    return r


def main():
    ap = argparse.ArgumentParser(description="Field calibration with the first 60 s (log §155).")
    ap.add_argument("--out", type=Path, default=Path("D:/beetrack_runs/w155"), help="folder for the .txt, .json and cache/")
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    t0, lines = time.time(), []

    def say(s):
        print(s, flush=True)
        lines.append(s)

    r = dict(heads=heads(a.out / "cache", say), tracking=tracking(a.out / "cache", say))
    say("self-checks passed: uncut heads = §95 per seed (P0, BT, calibrated BT); GT counts of both parts (§155); 乙 with "
        "k = (1, 1) = 甲 (every video); rate reference hand case and window counts summing to rate x frames (every offset)")
    (a.out / "field_calibration.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (a.out / "field_calibration.json").write_text(json.dumps(r, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {a.out / 'field_calibration'}.txt and .json; runtime {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
