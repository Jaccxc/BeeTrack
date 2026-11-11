# BeeTrack final numbers (source of truth for the report)

All tracking numbers: TrackEval, combined over sequences. Parameters were tuned on **val only**; test was run once per setting.
Detections cached once per detector and shared by all trackers (conf 0.05, NMS IoU 0.7, FP16).

## Counting protocols
- **Entrance zone (main):** one polygon per camera (`beetrack/zones.json`), drawn from the first frame only, without labels or predictions.
  A track whose first box centre is outside the zone and last centre inside = 1 *in*; inside → outside = 1 *out*; same rule for GT and predictions.
  Error = |pred_in − gt_in| + |pred_out − gt_out|, summed over sequences. GT crossings: val 173 (90 in / 83 out), test 341.
- **Mid-height line, banded:** hysteresis band ±0.05·H (fixed a priori). GT crossings: val 95, test 316.
- **Plain line (band 0):** counts every wiggle across the line; GT crossings val 298, test 753. Reported only to show why the band/zone are needed.
- Zone counting cut the relative counting error of *ours* on test from 83% (banded line) to 35%.

## Detector (YOLO11-s, Ultralytics 8.4.208, single RTX 4060 Laptop 8 GB)
| model | train | best epoch / stop | val mAP50 | val mAP50-95 |
|---|---|---|---|---|
| 640, batch 16 | 0.32 h | best ep. 7, early stop at ep. 22 (patience 15) | 0.946 | 0.513 |
| 1280, batch 2 (nbs 64 → same effective batch) | 1.97 h (~3.5 min/epoch) | best ep. 19, early stop at ep. 34 | 0.955 | 0.536 |

Detection P / R / mAP on val and test (`runs/eval/detection.json`, Ultralytics val, conf 0.001 default):
| model | split | P | R | mAP50 | mAP50-95 |
|---|---|---|---|---|---|
| 640 | val | 0.935 | 0.897 | 0.946 | 0.513 |
| 640 | test | 0.886 | 0.838 | 0.905 | 0.472 |
| 1280 | val | 0.939 | 0.910 | 0.955 | 0.536 |
| 1280 | test | 0.901 | 0.878 | 0.927 | 0.502 |

## Tracker parameters
- ByteTrack (Ultralytics bytetrack.yaml): track_high 0.25, track_low 0.1, new_track 0.25, buffer 30, match 0.8, fuse_score.
- ByteTrack-tuned: same, new_track_thresh 0.5 (the one threshold change ours also uses, for a fair comparison).
- **Ours (BeeTracker):** new_track 0.5, track_buffer 60, buffered IoU with adaptive margin r = 0.1 + 0.1·frames_lost + 0.5·min(speed/h, 1), velocity damping 0.8 for lost tracks, Kalman velocity noise 1/80 (default 1/160).
- ours_count: ours + causal tracklet stitching (gap ≤ 10 frames, ≤ 4 box heights). Helps line counting, hurts zone counting and HOTA → not used for the main zone result.
- Tried and rejected on val: BIoU in the second (low-score) stage, gap interpolation, short-track filter (16 settings of min length 5–50 × mean score 0–0.7; HOTA never above 55.7, MOTA and zone count worse with stronger filtering).

## Main test results (640 detector)
| tracker | HOTA | IDF1 | MOTA | IDSW | zone err /341 | banded line /316 | plain line /753 |
|---|---|---|---|---|---|---|---|
| ByteTrack | 40.2 | 48.1 | 15.0 | 2186 | 122 | 263 | 479 |
| ByteTrack-tuned | 45.2 | 56.4 | 43.2 | 1382 | 139 | 264 | 509 |
| **Ours** | **45.2** | 55.7 | **46.7** | 1314 | **119** | 262 | 492 |
| Ours + stitching | 42.2 | 52.0 | 46.9 | **952** | 164 | **192** | **359** |
| Oracle dets + ours | 79.2 | 83.9 | 96.2 | 904 | 100 | 186 | 312 |
| Oracle dets + ours + stitching | 74.8 | 76.5 | 96.4 | 715 | 163 | 110 | 212 |

Oracle on val: ByteTrack 71.3 HOTA, ours 71.0 (association alone is not where ours gains; gains come with noisy detections).
Even with GT boxes the zone error is 29% → remaining counting error is mainly track fragmentation in the crowded entrance, not detection.

## Val ablation (640 dets)
| setting | HOTA | AssA | IDF1 | MOTA | IDSW | banded /95 | zone /173 |
|---|---|---|---|---|---|---|---|
| ours | 55.7 | 51.6 | 72.5 | 78.3 | 407 | 78 | 83 |
| − buffered IoU (plain IoU) | 53.5 | 47.8 | 68.3 | 77.8 | 518 | 77 | 89 |
| − adaptive margin (fixed 0.1) | 53.9 | 48.2 | 69.5 | 78.2 | 489 | 74 | 87 |
| − velocity damping | 55.5 | 51.2 | 71.9 | 78.2 | 413 | 80 | 86 |
| default Kalman Q | 55.5 | 51.1 | 72.0 | 78.6 | 392 | 78 | 88 |
| default new-track thresh (0.25) | 55.4 | 51.2 | 71.1 | 77.1 | 615 | 78 | 74 |
| track_buffer 30 | 55.7 | 51.6 | 72.5 | 78.3 | 401 | 79 | 81 |
| ByteTrack-tuned | 54.8 | 50.0 | 71.0 | 78.5 | 427 | 72 | 82 |
| ByteTrack | 53.7 | 48.5 | 68.6 | 76.6 | 632 | 76 | 76 |

Per-sequence zone errors, ours on val (GT in/out → pred in/out, err): BEE24-10 6/7→2/6 (5); 15 30/26→2/19 (35); 24 1/4→3/2 (4); 28 4/8→5/6 (3); 33-tail 29/21→13/16 (21); 35-tail 20/17→10/12 (15).

## Experiment 2: input resolution (test)
| | HOTA | IDF1 | MOTA | IDSW | zone /341 | banded /316 | det ms/frame* |
|---|---|---|---|---|---|---|---|
| ours 640 | 45.2 | 55.7 | 46.7 | 1314 | 119 | 262 | 17.0 |
| ours 1280 | 48.0 | 60.0 | 60.3 | 1227 | 120 | 252 | 18.5 |
| ByteTrack-tuned 1280 | 47.8 | 60.8 | 59.2 | 1322 | 135 | 261 | 18.5 |
| ours + stitching 1280 | 43.9 | 53.8 | 60.5 | 928 | 157 | 180 | 18.5 |
Val at 1280: ours 56.9 HOTA / zone 75; ByteTrack-tuned 57.1 / 87; ours+stitch 54.5 / 98.
*idle machine, frame-weighted mean over test sequences, includes JPEG decode (decode-bound: 1280 only ~9% slower).

## Speed
- Detector (640): ~17 ms/frame incl. decode; tracker (ours): 1.63 ms/frame (ByteTrack 1.69 ms at 1280 dets, ours 1.86).
- Live demo on BEE24-36 (detector + tracker, excluding decode, 640): mean 66 FPS; real-time at 25 fps with large margin.

## Robustness 1: frame drop (test, every k-th frame, GT subsampled the same way)
| k | tracker | HOTA | IDF1 | MOTA | IDSW | zone err/GT | banded/GT |
|---|---|---|---|---|---|---|---|
| 2 | ByteTrack | 39.4 | 47.8 | 13.6 | 1406 | 179/325 | 249/287 |
| 2 | ByteTrack-tuned | 43.9 | 54.8 | 40.8 | 919 | 191/325 | 249/287 |
| 2 | Ours | 43.5 | 53.5 | 43.0 | 936 | 184/325 | 250/287 |
| 2 | Ours+stitch | 39.4 | 47.0 | 43.3 | 679 | 210/325 | 212/287 |
| 3 | ByteTrack | 38.0 | 45.7 | 12.8 | 1273 | 182/309 | 224/256 |
| 3 | ByteTrack-tuned | 42.1 | 52.3 | 38.6 | 931 | 192/309 | 222/256 |
| 3 | Ours | 44.0 | 55.1 | 40.3 | 750 | 184/309 | 223/256 |
| 3 | Ours+stitch | 39.4 | 46.7 | 40.7 | 524 | 222/309 | 190/256 |
| 5 | ByteTrack | 35.1 | 40.8 | 10.6 | 1418 | 163/268 | 190/207 |
| 5 | ByteTrack-tuned | 39.9 | 48.3 | 33.7 | 1160 | 165/268 | 187/207 |
| 5 | Ours | 42.0 | 51.6 | 37.8 | 734 | 173/268 | 197/207 |
| 5 | Ours+stitch | 38.4 | 45.2 | 38.5 | 503 | 200/268 | 169/207 |
Ours degrades most gracefully in HOTA/IDSW (adaptive margin grows with speed and frames lost): at k=5, +2.1 HOTA and −37% IDSW vs ByteTrack-tuned.

## Robustness 2: synthetic image corruptions (test, 640 detector, applied before detection)
dark = ×0.4; bright = ×1.6+20; blur = 1×9 horizontal motion blur; noise = Gaussian σ15.
| corruption | tracker | HOTA | IDF1 | MOTA | IDSW | zone /341 | banded /316 |
|---|---|---|---|---|---|---|---|
| none | ByteTrack-tuned | 45.2 | 56.4 | 43.2 | 1382 | 139 | 264 |
| none | Ours | 45.2 | 55.7 | 46.7 | 1314 | 119 | 262 |
| dark | ByteTrack-tuned | 47.3 | 59.6 | 51.5 | 1314 | 145 | 270 |
| dark | Ours | 46.4 | 57.5 | 54.4 | 1268 | 118 | 263 |
| bright | ByteTrack-tuned | 41.9 | 51.1 | 35.1 | 1426 | 174 | 265 |
| bright | Ours | 41.2 | 49.7 | 38.2 | 1352 | 141 | 255 |
| blur | ByteTrack-tuned | 43.2 | 53.5 | 45.8 | 1317 | 156 | 271 |
| blur | Ours | 41.6 | 50.4 | 48.6 | 1423 | 119 | 268 |
| noise | ByteTrack-tuned | 37.4 | 45.2 | 8.1 | 1463 | 166 | 270 |
| noise | Ours | 38.1 | 46.0 | 20.9 | 1533 | 127 | 263 |
Darkening *raises* HOTA/MOTA: fewer detections of unlabeled bees in the dark entrance of BEE24-34 (fewer FP). Noise hurts most. Ours keeps the best zone count under every corruption.

## Robustness 3: density (test, per sequence; `runs/eval/test/density.json`)
| seq | GT bees/frame | GT tracks | ByteTrack HOTA | ByteTrack-tuned HOTA | Ours HOTA (IDSW, zone err/GT) | Ours 1280 HOTA |
|---|---|---|---|---|---|---|
| BEE24-12 | 7.9 | 19 | 71.0 | 71.1 | 71.8 (4, 2/4) | 69.9 |
| BEE24-36 | 10.5 | 163 | 42.5 | 49.1 | 46.4 (265, 19/81) | 44.2 |
| BEE24-16 | 14.8 | 80 | 53.9 | 57.8 | 57.1 (65, 37/63) | 62.2 |
| BEE24-34 | 20.7 | 1143 | 36.9 | 41.2 | 42.5 (947, 52/172) | 47.6 |
| BEE24-13 | 25.0 | 54 | 67.8 | 67.2 | 66.3 (33, 9/21) | 66.6 |
Difficulty follows track turnover (bees entering/leaving, occlusion at the entrance) more than raw density: BEE24-13 is the densest but bees mostly stay, so HOTA is high.

## Failure analysis (figures in runs/figs/)
- `failure_fp_BEE24-34.png` (frame 391: 21 unmatched predictions vs 23 GT): many "false positives" are real bees whose GT boxes are much looser than our tight boxes (IoU < 0.5), plus unlabeled bees on flowers/ground at the frame border. BEE24-34 alone has ~51k of ours' FP.
- `failure_idsw_BEE24-36.png` (frames 6 → 13): ID switch inside the entrance cluster; touching bees occlude each other and the track restarts with a new ID → the zone rule sees two half-tracks and misses the crossing.
- Counting errors are almost all undercounts: bees vanish into the dark entrance (GT follows them in), fast bees break tracks.

## Analytics / demo (BEE24-36, ours)
- Zone counts: GT 49 in / 32 out; ours 40 in / 22 out (offline analytics == live demo count).
- GT 163 tracks, median 2.24 s; ours 390 tracks, median 0.64 s (fragmentation); median speed 73 px/s.
- Figures: runs/figs/BEE24-36_ours_{trajectories,heatmap,counts,lengths}.png; demo video runs/demo/BEE24-36_ours.mp4 (H.264), still runs/demo/BEE24-36_ours.jpg.

## ReID probe (val, GT boxes; `beetrack/reid_probe.py`, `runs/eval/reid_probe.json`)
Task: given a bee's crop at frame t, pick the same bee among the GT bees at frame t+gap (300 frame pairs per gap).
"Near" = only queries with another bee within 4 box heights; rank-1 accuracy in %.
| gap (frames) | appearance, ImageNet R18 | appearance, fine-tuned R18 | position only | appearance + position | chance | queries |
|---|---|---|---|---|---|---|
| 5 | 66.5 | 73.9 | 93.3 | 92.8 | 11.8 | 4490 |
| 10 | 56.2 | 64.6 | 89.5 | 89.0 | 12.1 | 4356 |
| 25 | 47.9 | 53.9 | 80.6 | 79.4 | 13.6 | 3850 |
| 50 | 43.9 | 49.0 | 69.7 | 69.5 | 15.7 | 3026 |
Fine-tuning: ResNet-18, 64 px crops, 3000 steps of contrastive loss on train-split frame pairs (negatives = other bees in the same frame).
Appearance is far above chance but below plain position at every gap, and adding it to position (equal weight, not tuned) gives no gain.
Crops include background, so part of the appearance score is location leaking in; true identity cues are weaker still.
Conclusion: a ReID branch was not integrated into the tracker.
