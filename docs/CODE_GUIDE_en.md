# BeeTrack code guide

Any team member can be asked to explain any part of this code. This guide covers what each file does, why it was designed that way, and how to run it, and ends with likely instructor questions and short answers. Read it once alongside the source; each module is under 160 lines.

## 1. The pipeline in one picture

```
data/BEE24 (MOT gt.txt + jpg frames)
   │ prep.py ──► runs/data/{train,val,test}.txt, bee24.yaml, split.json      (detector lists)
   │         └─► runs/gt/{val,test,test_step2,3,5}/<seq>/gt/gt.txt + seqinfo.ini  (TrackEval GT)
   ▼
train.py ──► runs/detect/y11s_<imgsz>/weights/best.pt
   ▼
detect.py ──► runs/dets/<tag>/<split>/<seq>.npy   (frame, x1, y1, x2, y2, score) + <seq>.json timing
   ▼
track.py ──► runs/trackers/<split>/<name>/data/<seq>.txt (MOT format) + timing.json
   ▼
evaluate.py ──► HOTA / IDF1 / MOTA / IDSW / count error, runs/eval/<split>/<name>.json
sweep.py    ──► many tracker settings at once on val, runs/eval/val/sweep_<grid>_<dets>.json
analytics.py ─► runs/figs/<seq>_<tracker>_{trajectories,heatmap,counts,lengths}.png
demo.py     ──► runs/demo/<name>_<tracker>.mp4 + .csv   (online: detect + track per frame)
```

Everything is run from the repository root as `uv run python -m beetrack.<module> ...`. Set up the environment once with `uv sync` (it reads `uv.lock`), then fetch the data with `uv run python tools/fetch_bee24.py`.

## 2. Module by module

### `common.py`: paths, split, MOT I/O
- `ROOT`, `DATA`, `RUNS`: all paths derive from the package location, so the code runs from any clone.
- `split_specs(split)` is the **single definition of the data split**. It returns `(eval_name, source_sequence, first_frame, last_frame)` tuples:
  - `test`: the 5 official test videos, whole.
  - `val`: `VAL_WHOLE` = BEE24-10/15/24/28, plus for every long training video (≥ `LONG` = 1000 frames, i.e. BEE24-33 and -35) its last `VAL_TAIL` = 1000 frames, named `<seq>-tail`.
  - `train`: everything else; long videos stop `GAP` = 50 frames before their validation tail.
- `gt_slice` cuts ground truth to a frame range and renumbers it from 1 (the val tails start at frame 1 in TrackEval).
- `write_mot` writes MOTChallenge rows `frame,id,x,y,w,h,score,-1,-1,-1`.

**Why this split**: test must never be touched (fairness rule 7B). The proposal planned whole sequences for val, but the short training videos are sparse while test 34/36 are 5,000-frame dense videos; tuning on sparse data would pick the wrong settings. The tails add long dense segments; the 50-frame gap avoids near-duplicate neighbouring frames leaking across the boundary.

### `prep.py`: labels, image lists, evaluation GT
- `yolo_labels`: one `.txt` next to each frame (`0 cx cy w h`, normalised), boxes clipped to the image, < 1 px boxes dropped, empty file for bee-free frames (valid negatives).
- Image lists: short videos use every frame; long videos every 5th frame for train (`STRIDE_LONG`) and every 10th for val/test (`STRIDE_VAL_LONG`), because 25 fps neighbours are near-duplicates. Result: 3,740 / 840 / 1,762 images.
- `write_eval_gt`: TrackEval folders. **Duplicate-ID repair**: BEE24-14/15/33/35 give one ID to two bees in the same frame (993 rows); TrackEval refuses that, so the extra occurrence keeps its box and gets a fresh ID. Test sequences have none. With `step=k` it also writes the frame-drop GT (`test_step2/3/5`), keeping frames 1, 1+k, ... and renumbering them, the same rule `track.run_sequence` uses.
- The final loop **asserts that no frame belongs to two splits**.
- Run: `uv run python -m beetrack.prep`

### `train.py`: detector fine-tuning
- YOLO11-s COCO weights → one class "bee". Fixed budget 50 epochs, patience 15, seed 0, `deterministic=True`; batch 16 at 640, 4 at 1280 (8 GB GPU); mosaic with `close_mosaic=10`, horizontal flip only (bees at an entrance have a meaningful up/down direction).
- The **same budget for both input sizes** makes Experiment 2 fair.
- Run: `uv run python -m beetrack.train --imgsz 640` (and `--imgsz 1280`).
- Result so far: 640 px early-stopped after 22 epochs (0.32 h), val mAP50 0.946, mAP50-95 0.513.

### `detect.py`: detection cache, corruptions, oracle
- Runs the detector **once** per frame of val or test and saves `frame, x1, y1, x2, y2, score` to `.npy`. Every tracker then reads the same file, so Experiment 1 changes only the tracker (no FP16 or run-to-run differences in detection).
- `MIN_SCORE = 0.05`: low-score boxes are kept because ByteTrack's second stage uses boxes between 0.1 and 0.25.
- `CORRUPT`: robustness conditions applied before detection: `dark` (×0.4), `bright` (×1.6 + 20), `blur` (9-px horizontal kernel), `noise` (Gaussian σ = 15, seed 0). One at a time, test only, no retraining.
- `--weights gt`: **oracle** detector, GT boxes with score 1. It isolates association errors: default ByteTrack on val oracle gets only HOTA 71.3 / IDF1 76.0 / IDSW 431.
- Timing: `torch.cuda.synchronize()` around each call; first 10 frames are dropped as warm-up; the time includes image decode.
- Run: `uv run python -m beetrack.detect --weights runs/detect/y11s_640/weights/best.pt --imgsz 640 --split test [--corrupt dark]`, oracle: `--weights gt --split val`.

### `track.py`: baseline and our tracker
- `BYTETRACK`: the Ultralytics default `bytetrack.yaml` values (baseline).
- `OURS`: the same plus `track_buffer=60, buffer=0.3, grow=0.1, speed=0.5, damp=0.9, q_vel=1/80` (defaults; final values come from the val sweep).
- `Dets`: a tiny adapter so a NumPy array looks like the Ultralytics `Results.boxes` object `BYTETracker.update` expects.
- `BeeTracker(BYTETracker)` overrides three methods:
  - `get_dists` → **buffered IoU**. Track buffer `r = buffer + grow·min(lost, 10) + speed·min(|v|/h, 1)`; detection buffer `buffer`. Both boxes are enlarged by r × their width/height on every side, then IoU, then fused with the detection score (`fuse_score`) as ByteTrack does. Used in the first (high-score) stage and for unconfirmed tracks; the low-score second stage keeps plain IoU.
  - `multi_predict` → **velocity damping**: lost tracks have velocity × `damp` before prediction; also sets the shared Kalman filter's velocity noise.
  - `get_kalmanfilter` → **process noise** `q_vel` (ByteTrack 1/160).
- `run_sequence(dets, n, tracker, step)`: feeds frames 1, 1+step, ... to the tracker, renumbers output frames, and times only the tracker.
- Run: `uv run python -m beetrack.track --dets y11s_640 --split test --tracker ours` (or `bytetrack`; `--set buffer=0.2 track_buffer=90 --name x` to override; `--step 3` for frame dropping, output goes to `test_step3`).

**Why buffered IoU (the reasoning to explain)**: a bee box is ~46 px. If a bee moves more than its own width between frames, the predicted box and the detection do not overlap; IoU = 0 and no threshold can match them, so the track is lost and restarted with a new ID. Enlarging boxes restores overlap. Making the enlargement **adaptive** keeps it small for slow bees at the crowded entrance (avoids merging neighbours) and large only for fast or lost tracks, where position uncertainty really is larger.

### `evaluate.py`: metrics
- `trackeval_metrics`: TrackEval `MotChallenge2DBox`, metrics HOTA, CLEAR, Identity. Settings that matter: `DO_PREPROC=False` (MOT17 preprocessing removes distractor classes; our GT has only bees, all marked valid), `SKIP_SPLIT_FOL=True`, our own seqmap. The class name "pedestrian" is just TrackEval's label for class 1. The small block at the top restores `np.float` etc., which TrackEval uses but NumPy ≥ 1.24 removed.
- `crossings(rows, line_y)`: **counting protocol**. Per track sorted by frame: sign of (centre y − line). Zeros are dropped; each sign change = one crossing; upward (sign goes + → −, i.e. y decreases) = in, downward = out. The same function is used for GT and predictions.
- `count_metrics`: line at mid-height (`imHeight/2` from `seqinfo.ini`); per-sequence error `|in_pred − in_gt| + |out_pred − out_gt|`.
- Run: `uv run python -m beetrack.evaluate --split test --trackers bytetrack ours` (also `--split val`, `--split test_step2`).

### `sweep.py`: tuning and ablation (val only)
- `GRIDS`: `ablation` (each row turns one component of ours back to ByteTrack's value), `buffer` (24), `motion` (27), `thresh` (36).
- Runs each setting on cached val detections, evaluates, prints a table sorted by HOTA, saves JSON, and deletes the temporary tracker outputs.
- Run: `uv run python -m beetrack.sweep --dets y11s_640 --grid ablation` (`--base buffer=0.2` applies an override to every "ours" row).

### `analytics.py`: figures
- For one sequence: trajectories of the first 60 s, occupancy heatmap, in/out per 20-s bin (GT vs prediction, using the same crossing rule), track-length histogram (fragmentation = many short tracks), median speed.
- Run: `uv run python -m beetrack.analytics --split test --tracker ours --seq BEE24-36`

### `demo.py`: the working application
- Online path: reads a video or frame folder, detects (`conf=0.05`, FP16) and tracks frame by frame, draws boxes, IDs, 1-s trails, the counting line and a HUD with IN/OUT, number of bees and FPS (rolling 30 frames over detect + track), and writes `.mp4` + `.csv`.
- The live counter uses the same sign-change rule as `evaluate.crossings`, kept incrementally per ID.
- Run: `uv run python -m beetrack.demo --source data/BEE24/test/BEE24-36/img1 --tracker ours [--line 0.5 --max-frames 500]`

## 3. Full reproduction order
```
uv sync
uv run python tools/fetch_bee24.py
uv run python -m beetrack.prep
uv run python -m beetrack.train --imgsz 640
uv run python -m beetrack.detect --weights runs/detect/y11s_640/weights/best.pt --imgsz 640 --split val
uv run python -m beetrack.sweep --dets y11s_640 --grid buffer      # and motion, thresh, ablation
uv run python -m beetrack.detect --weights runs/detect/y11s_640/weights/best.pt --imgsz 640 --split test
uv run python -m beetrack.track --dets y11s_640 --split test --tracker bytetrack
uv run python -m beetrack.track --dets y11s_640 --split test --tracker ours
uv run python -m beetrack.evaluate --split test --trackers bytetrack ours
uv run python -m beetrack.analytics --tracker ours --seq BEE24-36
uv run python -m beetrack.demo --source data/BEE24/test/BEE24-36/img1
```
Robustness: `detect.py --corrupt <c>` then `track.py --dets y11s_640_<c>`; frame dropping: `track.py --step k` then `evaluate.py --split test_step<k>`.
Report PDFs: `powershell -ExecutionPolicy Bypass -File report\build.ps1`.

## 4. Likely instructor questions

**Q: What is your improvement, in one sentence?**
Motion-adaptive association: ByteTrack's IoU is computed on enlarged boxes whose enlargement grows with a track's speed and frames lost, plus velocity damping for lost tracks, larger Kalman velocity noise and a longer lost-track buffer.

**Q: How do you know the tracker, not the detector, is the problem?**
Oracle experiment: with ground-truth boxes as detections, default ByteTrack still gets only HOTA 71.3, IDF1 76.0 and 431 ID switches on val. Those errors are all association.

**Q: Is the comparison fair?**
Yes. Both trackers read the same cached detections (`runs/dets/...npy`), same thresholds, same test set and evaluation code. All tuning used val only; the test set is evaluated once per final configuration.

**Q: Did you tune on the test set?**
No. `sweep.py` defaults to `--split val`, and parameters are frozen before running test.

**Q: Why not use appearance features (DeepSORT, BoT-SORT ReID)?**
All bees look nearly identical and are ~46 px, so a ReID embedding has little to distinguish; the failure we observed is geometric (no overlap), which a motion fix addresses directly and stays real-time.

**Q: Why not just set a large fixed buffer?**
At the crowded entrance a large buffer makes neighbouring slow bees overlap and swaps their IDs. The adaptive buffer is large only where uncertainty is large (fast or lost tracks). The ablation row "− adaptive" tests exactly this.

**Q: Why does the second (low-score) stage keep plain IoU?**
Low-score boxes are more often false positives; accepting them only on genuine overlap avoids attaching tracks to noise.

**Q: What do HOTA, IDF1, MOTA measure?**
MOTA = 1 − (FN + FP + IDSW)/GT, dominated by detection. IDF1 = F1 of identity-matched detections, measures how long IDs stay correct. HOTA = √(DetA·AssA) averaged over IoU thresholds 0.05–0.95, balancing detection and association.

**Q: Why `DO_PREPROC=False` in TrackEval?**
MOT17 preprocessing removes boxes matched to distractor classes (e.g. static people). Our GT has a single class with every box valid, so there is nothing to remove.

**Q: How is counting defined without count labels?**
Line at mid-height; a track's box centre changing side is one crossing (up = in, down = out). Exactly the same rule on GT and predictions; error = |Δin| + |Δout| per sequence. Test GT has 398 in / 355 out = 753 crossings.

**Q: Why is the validation set not only whole sequences, as in the proposal?**
Short training videos are sparse; test 34/36 are long and dense. We added the last 1000 frames of the two long training videos, separated by a 50-frame gap, so tuning sees dense long segments without leakage.

**Q: What was the duplicate-ID problem?**
993 rows in BEE24-14/15/33/35 reuse one ID for two bees in one frame; TrackEval rejects it. In evaluation GT the extra occurrence gets a fresh ID. Test has none, so test results are unaffected.

**Q: Why fine-tune instead of using COCO YOLO directly?**
COCO has no bee class.

**Q: Why 640 vs 1280 for Experiment 2, and is batch size a confound?**
Bees are small; higher resolution may help detection at 4× compute. Batch must drop to 4 at 1280 to fit 8 GB; that is an unavoidable consequence of the factor studied and is reported.

**Q: How is FPS measured?**
CUDA-synchronised wall clock; detector time per frame (incl. image decode, excluding 10 warm-up frames) plus tracker time per frame; demo shows a rolling 30-frame FPS of detect + track.

**Q: What did AI do?**
Claude (Anthropic) assisted with code, experiments and writing; it is disclosed in the report. Every member reviewed the code and can explain it; all numbers come from our own runs.
