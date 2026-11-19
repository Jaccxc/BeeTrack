"""§158 description: GT box-centre speed (body lengths per frame = displacement / sqrt(area)) at each GT event's last
crossing, 'in' vs 'out', and whether real P0 (loho_noW_s0..s2) matches the event. Whiteboard 13 clips; CPU.
Argument 'gray' (§159): the gray-board fold instead, loho_noG_s0..s8 from the gray_fold.py run cache."""
import sys
import numpy as np
sys.path.insert(0, "tools")
import whiteboard_fold as wf
import gray_fold as gf
from beetrack.evaluate import by_track, zone_tracks
from beetrack.p0 import load_tracks
from whiteboard_stages import matched
F, NS = (gf.GRAY, 9) if sys.argv[1:] == ["gray"] else (wf.WB, 3)
sp_ = {"in": [], "out": []}; hit = {"in": [], "out": []}
for sp, ev in F.clips:
    poly, g = F.poly[ev], by_track(wf.gt(sp, ev).astype(float))
    G = zone_tracks(g, poly)
    M = [matched(g, load_tracks(F.out / "cache" / f"{F.p0c}{wf.dtag(s, sp, F=F)}" / sp / f"{ev}.txt"), poly)[0] for s in range(NS)]
    for gid, (d, s, e, cross) in G.items():
        if not d: continue
        j = [k for k, dk in cross if dk == d][-1]
        c = g[:, 2:4] + g[:, 4:6] / 2
        v = np.linalg.norm(c[j] - c[j - 1]) / np.sqrt(g[j, 4] * g[j, 5]) if j > s else np.nan
        sp_[d].append(v); hit[d].append(np.mean([gid in m for m in M]))
for d in ("in", "out"):
    v, h = np.array(sp_[d]), np.array(hit[d])
    fast = v > 0.5
    print(f"{d:3s} n={len(v)} speed median {np.nanmedian(v):.2f} body/frame, share > 0.5: {np.mean(fast):.2f}; "
          f"found by real P0 (s0-{NS - 1} mean): all {h.mean():.2f}, fast {h[fast].mean() if fast.any() else float('nan'):.2f}, slow {h[~fast].mean():.2f}")
