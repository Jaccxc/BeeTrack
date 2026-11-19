"""Log §159 (measurement only, issue #4, #3): the fourth leave-one-scene-out fold, the gray board. Detectors loho_noG_s0..8
(trained without BEE24-17..32), imgsz 640, on the 16 gray-board clips: train BEE24-17..23, 25..27, 29..32, val BEE24-24, 28
(25 fps, 75 frames each: 1200 frames, 152 GT events: in 61, out 91). Scored exactly as §157 by tools/whiteboard_fold.py
with the fold GRAY:
  甲 P0 with m_noG.json (python -m beetrack.link --dets y11s_640 --fit train val --skip BEE24-17 ... BEE24-32, the 16 gray
     sequences; CPU; a refit is byte-identical), per seed.   乙 E3 x F8 per group of 3 seeds (0-2, 3-5, 6-8).
  丙 default ByteTrack.   丁 calibrated ByteTrack fit B: k from the same detector's default ByteTrack on the seen val
     BEE24-10, 15, dev 33-head, val 33-tail, dev 35-head, val 35-tail (the whiteboard fold used 24, 28: held out here).
Polygons: BEE24-24, 28 from zones.json (unchanged); the 14 train clips get BEE24-28's polygon moved by the rounded phase
correlation offset of their first frame (t159/gray_poly.py: cv2.phaseCorrelate against 24 and 28, 28 responds higher on
all 14, response 0.47-0.60; BEE24-19, 25 by (-1, +1), the others by (0, 0)).
Asserts as whiteboard_fold.py: GT totals and GT-as-prediction exact on every clip, m_noG.json's SHA-256, the §95 fold-35
reproduction on BEE24-35-head; run: seed 0's BEE24-24 rerun byte-identical to its cached detections.
  PYTHONPATH=. CUDA_VISIBLE_DEVICES=-1 uv run python tools/gray_fold.py selfcheck|run [--out DIR] [--linker FILE]
Defaults: --out D:/beetrack_runs/g159, --linker beetrack/models/m_noG.json.
run needs every detection file of t159/job159.sh and stops, before reading any of them, if one is missing."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from beetrack.common import RUNS
from beetrack.p0 import MODELS, ZONE
from whiteboard_fold import FIT_W, main

G_VAL = ("BEE24-24", "BEE24-28")
G_TRAIN = tuple(f"BEE24-{i}" for i in (*range(17, 24), 25, 26, 27, *range(29, 33)))
CLIPS_G = [("train", ev) for ev in G_TRAIN] + [("val", ev) for ev in G_VAL]
GT_G = (61, 91)
SHIFT = dict.fromkeys(G_TRAIN, (0, 0)) | {"BEE24-19": (-1, 1), "BEE24-25": (-1, 1)}    # t159/gray_poly.py, from BEE24-28
POLY_G = {ev: (np.array(ZONE["BEE24-28"]) + d).tolist() for ev, d in SHIFT.items()} | {ev: ZONE[ev] for ev in G_VAL}
FIT_G = [("val", "BEE24-10"), ("val", "BEE24-15")] + FIT_W[2:]
M_NOG_SHA = "58d0a446fb578e88b8372176d7e81966a4d27838eafce333cb0496a14099a889"   # t159/m_noG.json; refit identical
RERUN, CACHED = (RUNS / "dets" / t / "val" / "BEE24-24.npy" for t in ("loho_noG_s0_re159", "loho_noG_s0"))
GRAY = SimpleNamespace(name="gray", tag="loho_noG", clips=CLIPS_G, nf=dict.fromkeys(POLY_G, 75), gt=GT_G, frames=1200,
                       poly=POLY_G, fit=FIT_G, linker=MODELS / "m_noG.json", sha=M_NOG_SHA, rerun=RERUN, cached=CACHED,
                       p0c="p0g_", job="t159/job159.sh", file="gray_fold", out=Path("D:/beetrack_runs/g159"), log="§159")
assert len(CLIPS_G) == 16 and sorted(POLY_G) == sorted(G_TRAIN + G_VAL) and not any(ev in ZONE for ev in G_TRAIN)
assert [ev for _, ev in FIT_G] == ["BEE24-10", "BEE24-15", "BEE24-33-head", "BEE24-33-tail", "BEE24-35-head", "BEE24-35-tail"]
assert not any(ev in POLY_G or ev.startswith(("BEE24-12", "BEE24-13", "BEE24-16", "BEE24-34", "BEE24-36")) for _, ev in FIT_G)

if __name__ == "__main__":
    main(GRAY)
