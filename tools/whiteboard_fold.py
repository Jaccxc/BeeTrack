"""Log §157 (measurement only, issue #4, #3): the third leave-one-scene-out fold, the whiteboard. Detectors loho_noW_s0..8
(never saw a whiteboard clip), imgsz 640, on the 13 whiteboard clips: val BEE24-10, 15, train BEE24-01..09, 11, 14 (25 fps,
1600 frames, 137 GT events: in 77, out 60). The train clips have no polygon in zones.json (left unchanged); POLY below
derives them from the BEE24-10 / 15 polygons by the phase-correlation shifts of the registration.
  甲 P0: stage 1 -> TrackletLink 0.15 with beetrack/models/m_noW.json (§64, fitted without any whiteboard sequence) ->
     drop tracks under 3 boxes -> events (beetrack/p0.py), per seed.
  乙 E3 x F8: per group of 3 seeds x the 8 frame symmetries (24 members), window counts averaged (§147); groups 0-2, 3-5, 6-8.
  丙 default ByteTrack, per seed.   丁 calibrated ByteTrack fit B (§95): k = true / predicted totals of the same detector's
     default ByteTrack on the seen BEE24-24, 28, 33-head, 33-tail, 35-head, 35-tail; criterion 2 uses its E3.
  report only: E3 (3 seeds' P0 counts averaged) and F8 (one seed, 8 symmetries averaged).
Score: 30 s window error (p0.window_err; averaged arms: the members' mean window counts) summed over the 13 clips, % of 137.
Also bias (k x predicted - GT events) and event F1@0.5 (averaged arms: member mean). Criteria as §151 / §152: d = arm - base,
PASS if mean d < 0 and |mean d| > max(range of both); P0 vs 丁 (and 丙, report) over the 9 seeds, criterion 1 = 乙 vs 甲
and criterion 2 = 乙 vs 丁 E3 over the 3 groups.
Asserts: GT 137 events (77 / 60) and GT-as-prediction scores 0 with F1 1 on every clip; m_noW.json's SHA-256; on BEE24-35-head
with loho_no35_s0..8 and m_dev_strict.json (§95): per seed P0, default and calibrated ByteTrack (2 decimals), P0 bias, fit B
k (3 decimals), 9-seed mean and range, group means, E3 and calibrated ByteTrack E3; k = (1, 1) / one-member identities;
this tool's scorer = p0.score where zones.json has the polygon; run: seed 0's BEE24-10 rerun byte-identical to its cache.
  PYTHONPATH=. uv run python tools/whiteboard_fold.py selfcheck|run [--out DIR]          (CPU only)
selfcheck runs on the files that exist now; run needs every detection file of t157/job157.sh and stops, before reading
any of them, if one is missing. Writes whiteboard_fold[_selfcheck].txt and .json to DIR (default D:/beetrack_runs/w157);
tracker outputs are kept as MOT files in DIR/cache/<name>/<split>/<video>.txt and always scored as read back from there.
The fold is a parameter F (default WB, the whiteboard, output unchanged); tools/gray_fold.py (§159) passes the gray board."""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from beetrack.common import RUNS, write_mot
from beetrack.evaluate import load_tracks, match_events
from beetrack.link import model
from beetrack.p0 import (FPS, MODELS, WIN, ZONE, bytetrack, dets, events, factor, gt, link_model, nframes, p0, per_window,
                         pooled, score, stage1, totals, window_err)
from window_lengths import EXP_HEADS, GROUPS, SHORT

SEEDS = range(9)
VAR8 = ("id", "lr", "ud", "rot", "tr", "atr", "cw", "ccw")               # score144.VAR8
WB_VAL = ("BEE24-10", "BEE24-15")
WB_TRAIN = tuple(f"BEE24-{i:02d}" for i in (*range(1, 10), 11, 14))
CLIPS = [("train", ev) for ev in WB_TRAIN] + [("val", ev) for ev in WB_VAL]
NF_W = dict.fromkeys(WB_TRAIN, 75) | {"BEE24-14": 360, "BEE24-10": 250, "BEE24-15": 240}
GT_W = (77, 60)
P10, P15 = np.array(ZONE["BEE24-10"]), np.array(ZONE["BEE24-15"])
POLY = {ev: P10 for ev in ("BEE24-01", "BEE24-02", "BEE24-03", "BEE24-04", "BEE24-05", "BEE24-06", "BEE24-08", "BEE24-11")}
POLY = {k: v.tolist() for k, v in (POLY | {"BEE24-07": P15 + [-1, 79], "BEE24-09": P15 + [-1, 79], "BEE24-14": P15,
                                           "BEE24-10": P10, "BEE24-15": P15}).items()}
FIT_W = [("val", "BEE24-24"), ("val", "BEE24-28"), ("dev", "BEE24-33-head"), ("val", "BEE24-33-tail"), ("dev", "BEE24-35-head"),
         ("val", "BEE24-35-tail")]
M_NOW_SHA = "af0bfea1e3b2a8c90e3e0b1022c4e36161a88e20be3b122fb3a4e6323bb8a0a1"   # scratch loho/m_noW.json; refit identical
RERUN, CACHED = (RUNS / "dets" / t / "val" / "BEE24-10.npy" for t in ("loho_noW_s0_re157", "loho_noW_s0"))
# §95 (t95/score95_all9.txt), fold 35: fit B k (in, out) per seed; E3 and calibrated ByteTrack E3 per group
K95 = ((1.988, 1.396), (1.964, 1.422), (2.062, 1.632), (1.941, 1.535), (1.988, 1.582), (1.875, 1.384), (2.037, 1.535),
       (1.833, 1.325), (1.964, 1.490))
E3_95, BT3_95 = (15.87, 17.81, 27.68), (42.84, 43.09, 52.09)
assert sorted(POLY) == sorted(WB_TRAIN + WB_VAL) and not any(ev in ZONE for ev in WB_TRAIN)
assert not any(ev in POLY or ev.startswith(("BEE24-12", "BEE24-13", "BEE24-16", "BEE24-34", "BEE24-36")) for _, ev in FIT_W)
# the fold: its clips, frames, GT totals, polygons, fit B videos, detector tag, link model, rerun check, names
WB = SimpleNamespace(name="whiteboard", tag="loho_noW", clips=CLIPS, nf=NF_W, gt=GT_W, frames=1600, poly=POLY, fit=FIT_W,
                     linker=MODELS / "m_noW.json", sha=M_NOW_SHA, rerun=RERUN, cached=CACHED, p0c="p0w_",
                     job="t157/job157.sh", file="whiteboard_fold", out=Path("D:/beetrack_runs/w157"), log="§157")


def avg_err(Ps, G, nf, ks=None, win=WIN):
    """window_err of the members' mean window counts (§95 E3, §147 F8): member i's in / out counts x ks[i], averaged per
    window (fractions allowed), |mean - true| summed over the windows, mean over the win start offsets."""
    ks = np.ones((len(Ps), 2)) if ks is None else np.asarray(ks, float)
    w = win * FPS
    return float(np.mean([np.abs(np.mean([per_window(P + [o * FPS, 0], nf + o * FPS, w) * k for P, k in zip(Ps, ks)], 0)
                                 - per_window(G + [o * FPS, 0], nf + o * FPS, w)).sum() for o in range(win)]))


# score95's check: one member over-counts an 'in', one misses it, their mean is exact
_G, _a, _b = np.array([[10, 1], [20, -1]]), np.array([[10, 1], [12, 1], [20, -1]]), np.array([[20, -1]])
assert [avg_err([_a], _G, 100), avg_err([_b], _G, 100), avg_err([_a, _b], _G, 100)] == [1.0, 1.0, 0.0]
assert avg_err([_a, _a], _G, 100, [(2, 1), (2, 1)]) == window_err(_a, _G, 100, (2, 1))


def mot(cdir, name, sp, ev, run):
    """Tracker rows as read back from the MOT file cdir/name/sp/ev.txt, written once from run() (field_calibration.tracked)."""
    f = cdir / name / sp / f"{ev}.txt"
    if not f.exists():
        t = f.with_suffix(".tmp")
        write_mot(t, run())
        t.replace(f)
    return load_tracks(f)


def wscore(pr, sp, ev, F=WB):
    """p0.score with the polygon F.poly[ev] (the fold's train clips have none in zones.json)."""
    nf, poly, g = nframes(sp, ev), F.poly[ev], gt(sp, ev)
    P, G = events(pr, poly), events(g, poly)
    m = match_events(g, pr, poly)
    assert (m["pred"], m["gt"]) == (len(P), len(G)), (sp, ev)
    return dict(err=window_err(P, G, nf), gt=len(G), pred=len(P), tp=m["tp"], P=P, G=G, nf=nf)


def same(a, b):
    return all(np.array_equal(a[x], b[x]) for x in ("err", "gt", "pred", "tp", "P", "G", "nf"))


def dtag(s, sp, v="id", F=WB):
    """Detection tag of <F.tag>_s<s> at 640 on a clip of the fold of split sp, symmetry v (job157.sh's names)."""
    t = f"{F.tag}_s{s}" + ("_tr_i640" if sp == "train" else "")
    return t if v == "id" else f"{t}_f{v}"


def needed(F=WB):
    f = [RUNS / "dets" / dtag(s, sp, v, F) / sp / f"{ev}.npy" for s in SEEDS for sp, ev in F.clips for v in VAR8]
    return f + [RUNS / "dets" / f"{F.tag}_s{s}" / sp / f"{ev}.npy" for s in SEEDS for sp, ev in F.fit] + [F.rerun]


def rng(x):
    return float(np.ptp(x))


def line(vals, fmt="6.2f"):
    return " / ".join(format(x, fmt) for x in vals)


def gt_check(say, F=WB):
    """GT events per clip with F.poly; GT scored as the prediction must be exact."""
    tot, n = np.zeros(2, int), 0
    for sp, ev in F.clips:
        assert nframes(sp, ev) == F.nf[ev], (ev, nframes(sp, ev))
        s = wscore(gt(sp, ev), sp, ev, F)
        assert s["err"] == 0 and s["pred"] == s["tp"] == s["gt"], (ev, s["err"], s["tp"])
        tot += totals(s["G"])
        n += s["nf"]
    assert tuple(tot) == F.gt and n == F.frames, (tot, n)
    say(f"{F.name}: {len(F.clips)} clips, {n} frames, GT in {tot[0]} out {tot[1]} ({tot.sum()} events); GT as prediction: "
        f"error 0, F1 1 on every clip")
    return int(tot.sum())


def l35(cdir, say, F=WB):
    """§95 fold 35 on BEE24-35-head through this tool's scoring path (P0 with m_dev_strict.json there, as §95)."""
    strict, tgt, E = link_model("m_dev_strict"), "BEE24-35-head", EXP_HEADS["35"]
    fit = [("dev", "BEE24-33-head"), ("val", "BEE24-33-tail")] + [("val", ev) for ev in SHORT]
    nf, V, Bs, K, T, B, t0 = nframes("dev", tgt), {a: [] for a in E}, [], [], [], [], time.time()
    for s in SEEDS:
        tag = f"loho_no35_s{s}"
        T.append(score(mot(cdir, f"p0s_{tag}", "dev", tgt, lambda: p0(stage1(dets(tag, "dev", tgt), nf), strict)), "dev", tgt))
        B.append(score(mot(cdir, f"bt_{tag}", "dev", tgt, lambda: bytetrack(dets(tag, "dev", tgt), nf)), "dev", tgt))
        R = {(sp, ev): mot(cdir, f"bt_{tag}", sp, ev, lambda: bytetrack(dets(tag, sp, ev), nframes(sp, ev))) for sp, ev in fit}
        Fs = [score(r, sp, ev) for (sp, ev), r in R.items()]
        assert all(same(wscore(r, sp, ev, F), score(r, sp, ev)) for (sp, ev), r in R.items() if ev in F.poly)  # 10, 15
        k = factor(Fs)
        t, b, n = T[-1], B[-1], T[-1]["gt"]
        assert avg_err([t["P"]], t["G"], nf) == t["err"] and window_err(b["P"], b["G"], nf, (1, 1)) == b["err"], s
        cal = window_err(b["P"], b["G"], nf, k)
        assert abs(avg_err([b["P"]], b["G"], nf, [k]) - cal) < 1e-9, s
        for arm, x in (("P0", t["err"]), ("BT", b["err"]), ("cal BT", cal)):
            V[arm].append(100 * x / n)
            assert round(V[arm][-1], 2) == E[arm]["seed"][s], (s, arm, V[arm][-1])
        assert t["pred"] - n == E["P0"]["bias"][s] and [f"{x:.3f}" for x in k] == [f"{x:.3f}" for x in K95[s]], (s, t["pred"] - n, k)
        Bs.append(t["pred"] - n)
        K.append(k)
        print(f"self-check fold 35 seed {s}: done, {(time.time() - t0) / 60:.1f} min", flush=True)
    for arm, e in E.items():
        assert (round(float(np.mean(V[arm])), 2), round(rng(V[arm]), 2)) == (e["mean"], e["range"]), arm
        assert [round(float(np.mean([V[arm][i] for i in g])), 2) for g in GROUPS] == list(e["groups"]), arm
    e3 = [100 * avg_err([T[i]["P"] for i in g], T[0]["G"], nf) / n for g in GROUPS]
    bt3 = [100 * avg_err([B[i]["P"] for i in g], T[0]["G"], nf, [K[i] for i in g]) / n for g in GROUPS]
    assert [round(x, 2) for x in e3] == list(E3_95) and [round(x, 2) for x in bt3] == list(BT3_95), (e3, bt3)
    say(f"self-check fold 35: {tgt} ({n} GT events), loho_no35_s0..8, P0 with m_dev_strict.json, fit B on "
        f"{', '.join(ev for _, ev in fit)}; window error % per seed (0..8), mean [range]")
    for arm in E:
        say(f"  {arm:7s} {line(V[arm])}  mean {np.mean(V[arm]):6.2f} [{rng(V[arm]):6.2f}]")
    say(f"  P0 bias {line(Bs, '+d')}; fit B k (in/out) {', '.join(f'{a:.3f}/{b:.3f}' for a, b in K)}")
    say(f"  groups: E3 {line(e3)}, calibrated BT E3 {line(bt3)}")
    v1, v2 = [ev for _, ev in fit if ev in F.poly]               # the fold's two val clips: BEE24-10, 15 (whiteboard)
    say("  = §95 (per seed P0 / BT / calibrated BT at 2 decimals, P0 bias, k at 3 decimals, 9-seed mean and range, group "
        f"means, E3, calibrated BT E3); one-member and k = (1, 1) identities hold; this scorer = p0.score on {v1} / "
        f"{v2.removeprefix('BEE24-')}")
    return dict(target=tgt, fit=fit, arms={a: list(map(float, v)) for a, v in V.items()}, bias=Bs, k=[k.tolist() for k in K],
                e3=e3, cal_bt_e3=bt3)


def seed_w(cdir, s, prob, F=WB):
    """<F.tag>_s<s> on the fold's clips: wscore of P0 for the 8 symmetries and of default ByteTrack per clip; fit B k."""
    P, B = {v: [] for v in VAR8}, []
    for sp, ev in F.clips:
        nf = nframes(sp, ev)
        for v in VAR8:
            t = dtag(s, sp, v, F)
            P[v].append(wscore(mot(cdir, f"{F.p0c}{t}", sp, ev, lambda: p0(stage1(dets(t, sp, ev), nf), prob)), sp, ev, F))
        t = dtag(s, sp, F=F)
        B.append(wscore(mot(cdir, f"bt_{t}", sp, ev, lambda: bytetrack(dets(t, sp, ev), nf)), sp, ev, F))
    tag = f"{F.tag}_s{s}"
    k = factor([score(mot(cdir, f"bt_{tag}", sp, ev, lambda: bytetrack(dets(tag, sp, ev), nframes(sp, ev))), sp, ev)
                for sp, ev in F.fit])
    return P, B, k


def whiteboard(cdir, say, n, prob, F=WB):
    """The fold's arms and criteria (named for §157; F picks the fold)."""
    CLIPS, FIT_W, GT_W = F.clips, F.fit, F.gt
    pct = lambda e: 100 * e / n
    S, t0 = [], time.time()
    for s in SEEDS:
        P, B, k = seed_w(cdir, s, prob, F)
        for c, (b, p) in enumerate(zip(B, P["id"])):          # one-member, k = (1, 1) identities, every clip
            assert avg_err([p["P"]], p["G"], p["nf"]) == p["err"] and window_err(b["P"], b["G"], b["nf"], (1, 1)) == b["err"], c
            assert abs(avg_err([b["P"]], b["G"], b["nf"], [k]) - window_err(b["P"], b["G"], b["nf"], k)) < 1e-9, c
        f8 = pct(sum(avg_err([P[v][c]["P"] for v in VAR8], P["id"][c]["G"], P["id"][c]["nf"]) for c in range(len(CLIPS))))
        po, bo = pooled(P["id"]), pooled(B)
        S.append(dict(P=P, B=B, k=k, p0=po["win"], p0_bias=po["bias"], p0_f1=po["f1"], p0_tot=sum(totals(p["P"]) for p in P["id"]),
                      p0_tp=sum(p["tp"] for p in P["id"]), f8=f8, f8_bias=float(np.mean([pooled(P[v])["bias"] for v in VAR8])),
                      bt=bo["win"], bt_bias=bo["bias"], bt_f1=bo["f1"],
                      cal=pct(sum(window_err(b["P"], b["G"], b["nf"], k) for b in B)),
                      cal_bias=sum(float((k * totals(b["P"])).sum()) for b in B) - n))
        print(f"{F.name} seed {s}: done, {(time.time() - t0) / 60:.1f} min", flush=True)

    say(f"{F.name} fold: {F.tag}_s0..8 at 640, link model {F.linker.name}; calibrated ByteTrack fit B on "
        f"{', '.join(ev for _, ev in FIT_W)}")
    say(f"  per seed (0..8): 30 s window error % ({len(CLIPS)} clips summed / {n}), 9-seed mean [range]")
    A = {"甲 P0": "p0", "丙 BT": "bt", "丁 cal BT B": "cal", "F8 (report)": "f8"}
    for name, key in A.items():
        v = [x[key] for x in S]
        say(f"  {name:12s} {line(v)}  mean {np.mean(v):6.2f} [{rng(v):6.2f}]")
    say("  bias (k x predicted - GT events; F8: mean over the 8 symmetries)")
    for name, key in A.items():
        v = [x[f'{key}_bias'] for x in S]
        say(f"  {name:12s} {line(v, '+6.1f')}  mean {np.mean(v):+6.1f}")
    say(f"  甲 P0 predicted in / out: {', '.join(f'{a}/{b}' for a, b in (x['p0_tot'] for x in S))} (GT {GT_W[0]}/{GT_W[1]})")
    say(f"  event F1@0.5: 甲 {line([x['p0_f1'] for x in S], '.3f')}  mean {np.mean([x['p0_f1'] for x in S]):.3f}; 丙 (= 丁's events) "
        f"{line([x['bt_f1'] for x in S], '.3f')}  mean {np.mean([x['bt_f1'] for x in S]):.3f}")
    say(f"  fit B k (in/out) per seed: {', '.join(f'{a:.3f}/{b:.3f}' for a, b in (x['k'] for x in S))}")

    G = {a: [] for a in ("甲 P0 single", "E3 (report)", "乙 E3 x F8", "F8 single (report)", "丙 BT single", "丁 cal BT single",
                         "丁 cal BT E3")}
    Gb, Gf = {a: [] for a in ("E3 (report)", "乙 E3 x F8", "丁 cal BT E3")}, {a: [] for a in ("E3 (report)", "乙 E3 x F8")}
    for M in GROUPS:
        for a, key in (("甲 P0 single", "p0"), ("F8 single (report)", "f8"), ("丙 BT single", "bt"), ("丁 cal BT single", "cal")):
            G[a].append(float(np.mean([S[s][key] for s in M])))
        G["E3 (report)"].append(pct(sum(avg_err([S[s]["P"]["id"][c]["P"] for s in M], g["G"], g["nf"])
                                       for c, g in enumerate(S[0]["P"]["id"]))))
        G["乙 E3 x F8"].append(pct(sum(avg_err([S[s]["P"][v][c]["P"] for s in M for v in VAR8], g["G"], g["nf"])
                                     for c, g in enumerate(S[0]["P"]["id"]))))
        G["丁 cal BT E3"].append(pct(sum(avg_err([S[s]["B"][c]["P"] for s in M], g["G"], g["nf"], [S[s]["k"] for s in M])
                                       for c, g in enumerate(S[0]["P"]["id"]))))
        Gb["E3 (report)"].append(float(np.mean([S[s]["p0_bias"] for s in M])))
        Gb["乙 E3 x F8"].append(float(np.mean([pooled(S[s]["P"][v])["bias"] for s in M for v in VAR8])))
        Gb["丁 cal BT E3"].append(float(np.mean([S[s]["cal_bias"] for s in M])))
        Gf["E3 (report)"].append(float(np.mean([S[s]["p0_f1"] for s in M])))
        Gf["乙 E3 x F8"].append(float(np.mean([pooled(S[s]["P"][v])["f1"] for s in M for v in VAR8])))
    say("  per group (seeds 0-2 / 3-5 / 6-8): window error %, mean, range")
    for a, v in G.items():
        say(f"  {a:18s} {line(v)}  mean {np.mean(v):6.2f}  range {rng(v):6.2f}")
    say("  group bias (mean member counts x k - GT events): " + "; ".join(f"{a} {line(v, '+.1f')}" for a, v in Gb.items()))
    say("  group event F1@0.5 (member mean): " + "; ".join(f"{a} {line(v, '.3f')}" for a, v in Gf.items()))

    def crit(name, A_, B_, la, lb):
        d, bar = np.subtract(A_, B_), max(rng(A_), rng(B_))
        ok = bool(d.mean() < 0 and -d.mean() > bar)
        say(f"  {name}: d = {la} - {lb} [{line(d, '+.2f')}] mean {d.mean():+.2f}; bar max(range {la} {rng(A_):.2f}, range {lb} "
            f"{rng(B_):.2f}) = {bar:.2f}: {'PASS' if ok else 'no'} (margin {-d.mean() - bar:+.3f}); {la} lower in "
            f"{(d < 0).sum()} / {len(d)}")
        return dict(d=d.tolist(), mean_d=float(d.mean()), bar=bar, passed=ok, lower=int((d < 0).sum()))

    say("  criteria (§151 / §152 arithmetic)")
    one = lambda key: [x[key] for x in S]
    C = {"P0 vs 丁 (9 seeds)": crit("P0 vs 丁 (9 seeds)", one("p0"), one("cal"), "甲", "丁"),
         "P0 vs 丙 (9 seeds, report)": crit("P0 vs 丙 (9 seeds, report)", one("p0"), one("bt"), "甲", "丙"),
         "criterion 1": crit("criterion 1", G["乙 E3 x F8"], G["甲 P0 single"], "乙", "甲"),
         "criterion 2": crit("criterion 2", G["乙 E3 x F8"], G["丁 cal BT E3"], "乙", "丁 E3")}
    keep = ("p0", "p0_bias", "p0_f1", "p0_tp", "f8", "f8_bias", "bt", "bt_bias", "bt_f1", "cal", "cal_bias")
    seeds = [{x: float(r[x]) for x in keep} | dict(k=r["k"].tolist(), p0_in_out=[int(a) for a in r["p0_tot"]],
             per_clip={ev: dict(p0=r["P"]["id"][c]["err"], p0_pred=r["P"]["id"][c]["pred"], gt=r["P"]["id"][c]["gt"],
                                bt=r["B"][c]["err"], bt_pred=r["B"][c]["pred"]) for c, (_, ev) in enumerate(CLIPS)}) for r in S]
    return dict(clips=CLIPS, gt_events=n, fit=FIT_W, seeds=seeds, groups=G, group_bias=Gb, group_f1=Gf, criteria=C)


def main(F=WB):
    ap = argparse.ArgumentParser(description=f"Leave-{F.name} fold (log {F.log}).")
    ap.add_argument("mode", choices=["selfcheck", "run"])
    ap.add_argument("--out", type=Path, default=F.out, help="folder for the .txt, .json and cache/")
    ap.add_argument("--linker", type=Path, default=F.linker, help=f"the link model file (its SHA-256 must be {F.sha[:16]}...)")
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    miss = [p for p in needed(F) if not p.exists()]
    if a.mode == "run" and miss:
        print(f"run: {len(miss)} of {len(needed(F))} detection files missing ({F.job} makes them); nothing read. First:")
        print("\n".join(f"  {p.relative_to(RUNS.parent).as_posix()}" for p in miss[:10]))
        raise SystemExit(1)
    t0, lines = time.time(), []

    def say(s):
        print(s, flush=True)
        lines.append(s)

    raw = a.linker.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == F.sha, f"{a.linker}: SHA-256 is not {F.linker.name}'s"
    r = {f"{F.linker.stem}_sha256": F.sha}
    if a.mode == "run":
        assert F.rerun.read_bytes() == F.cached.read_bytes(), f"seed 0's {F.rerun.stem} rerun differs from the cached detections"
        say(f"self-check: {F.rerun.relative_to(RUNS.parent).as_posix()} byte-identical to {F.cached.relative_to(RUNS.parent).as_posix()}")
    n = gt_check(say, F)
    r["l35"] = l35(a.out / "cache", say, F)
    if a.mode == "selfcheck":
        say(f"run inputs: {len(needed(F)) - len(miss)} of {len(needed(F))} detection files exist, {len(miss)} missing")
        r["missing"] = [p.relative_to(RUNS.parent).as_posix() for p in miss]
    else:
        r[F.name] = whiteboard(a.out / "cache", say, n, model(json.loads(raw)), F)
    say(f"self-checks passed ({a.mode}); {F.linker.name} SHA-256 {F.sha[:16]}...")
    name = F.file + ("_selfcheck" if a.mode == "selfcheck" else "")
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / f"{name}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (a.out / f"{name}.json").write_text(json.dumps(r, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {a.out / name}.txt and .json; runtime {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
