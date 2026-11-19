"""§158: in / out counts after each P0 stage on the 13 whiteboard clips, GT boxes (score 1) and loho_noW_s0..s2 (640);
GT 'in' events attributed to the first stage that loses them (GT-box input). CPU only, no test.
  PYTHONPATH=. CUDA_VISIBLE_DEVICES=-1 uv run python <this> <out.json>        (repo root)"""
import json
import sys

import numpy as np
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, "tools")
import whiteboard_fold as wf  # noqa: E402
from beetrack.evaluate import by_track, frame_match, zone_tracks  # noqa: E402
from beetrack.link import link  # noqa: E402
from beetrack.p0 import MIN_LEN, THR  # noqa: E402
from beetrack.track import drop_short  # noqa: E402

PROB = wf.link_model("m_noW")
STAGES = ("1 GT tracks", "2 GT tracks >= 3 boxes", "3 stage-1 tracklets", "4 + TrackletLink", "5 P0")


def matched(gt, pr, poly, tol=5):
    """GT track ids whose event is matched (match_events' rule and assignment), and the GT event direction per id."""
    gt, pr = by_track(np.asarray(gt, float)), by_track(np.asarray(pr, float))
    G, P, under = zone_tracks(gt, poly), zone_tracks(pr, poly), frame_match(gt, pr)
    gev, pev = [g for g, t in G.items() if t[0]], [p for p, t in P.items() if t[0]]
    col = {g: k for k, g in enumerate(gev)}
    C = np.zeros((len(pev), len(gev)), int)
    for r, p in enumerate(pev):
        d, s, e, cross = P[p]
        for j, dj in cross:
            for g in {under[j - 1], under[j]} - {-1} if dj == d else ():
                if G[g][0] == d and any(dg == d and abs(gt[i, 0] - pr[j, 0]) <= tol for i, dg in G[g][3]):
                    C[r, col[g]] = 1
    ri, ci = linear_sum_assignment(C, maximize=True)
    return {gev[c] for r, c in zip(ri, ci) if C[r, c]}, {g: G[g][0] for g in gev}


def counts(rows, poly):
    ev = [t[0] for t in zone_tracks(by_track(np.asarray(rows, float)), poly).values()] if len(rows) else []
    return [ev.count("in"), ev.count("out")]


def chain(D, nf):
    s1 = wf.stage1(D, nf)
    s4 = np.array(sorted(link(s1.tolist(), THR, PROB).tolist()), float)
    s5 = np.array(drop_short(s4.tolist(), MIN_LEN, 0.0), float)
    return s1, s4, s5


def main():
    out, tot = {}, {}
    lost = {"in": {k: 0 for k in STAGES[1:]}, "out": {k: 0 for k in STAGES[1:]}}
    detail15 = []
    for sp, ev in wf.CLIPS:
        poly, nf, g = wf.POLY[ev], wf.nframes(sp, ev), wf.gt(sp, ev)
        g2 = np.array(drop_short(np.c_[g[:, :6], np.ones(len(g))].tolist(), MIN_LEN, 0.0), float)
        od = np.c_[g[:, 0], g[:, 2], g[:, 3], g[:, 2] + g[:, 4], g[:, 3] + g[:, 5], np.ones(len(g))]
        rec = {"gt_boxes": {}, "real": {}}
        S = chain(od, nf)
        for name, rows in zip(STAGES, (g, g2, *S)):
            rec["gt_boxes"][name] = counts(rows, poly)
        for s in range(3):
            R = chain(wf.dets(wf.dtag(s, sp), sp, ev), nf)
            rec["real"][f"s{s}"] = {name: counts(rows, poly) for name, rows in zip(STAGES[2:], R)}
        # attribution, GT-box input: first stage (2..5) whose output no longer matches the GT event
        _, dirs = matched(g, g, poly)
        alive = set(dirs)
        for name, rows in zip(STAGES[1:], (g2, *S)):
            m, _ = matched(g, rows, poly)
            for gid in alive - m:
                lost[dirs[gid]][name] += 1
                if ev == "BEE24-15" and name == STAGES[2] and dirs[gid] == "in":
                    detail15.append(describe(g, S[0], gid, poly))
            alive &= m
        out[ev] = rec
        for name, v in rec["gt_boxes"].items():
            tot.setdefault(("gt_boxes", name), np.zeros(2, int))
            tot[("gt_boxes", name)] += v
        for s, d in rec["real"].items():
            for name, v in d.items():
                tot.setdefault((f"real {s}", name), np.zeros(2, int))
                tot[(f"real {s}", name)] += v
        print(ev, json.dumps(rec, ensure_ascii=False), flush=True)
    print("\ntotals over the 13 clips (in, out):")
    for (inp, name), v in tot.items():
        print(f"  {inp:9s} {name:24s} {v.tolist()}")
    print("GT-box input, GT events lost first at each stage:", json.dumps(lost))
    print("BEE24-15 'in' events lost at stage 1 (GT boxes):")
    for d in detail15:
        print("  ", d)
    json.dump(dict(clips=out, totals={f"{a} | {b}": v.tolist() for (a, b), v in tot.items()}, lost=lost, detail15=detail15),
              open(sys.argv[1], "w", encoding="utf-8"), indent=1, ensure_ascii=False)


def describe(g, s1, gid, poly):
    """The GT bee's frames, inside flags (I / o), the stage-1 tracklet id under each box (-1 none) and box areas."""
    import cv2
    gb, sb = by_track(np.asarray(g, float)), by_track(np.asarray(s1, float))
    tl = frame_match(sb, gb)                       # stage-1 tracklet id under every GT row
    k = gb[:, 1] == gid
    rows = gb[k]
    ins = "".join("I" if cv2.pointPolygonTest(np.asarray(poly, np.float32), (float(x), float(y)), False) >= 0 else "o"
                  for x, y in rows[:, 2:4] + rows[:, 4:6] / 2)
    return dict(gt_id=int(gid), frames=[int(rows[0, 0]), int(rows[-1, 0])], inside=ins, tracklet=[int(x) for x in tl[k]],
                area=[int(a) for a in rows[:, 4] * rows[:, 5]])


if __name__ == "__main__":
    main()
