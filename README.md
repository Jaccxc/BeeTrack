# BeeTrack

Detecting, tracking and counting honeybees at the hive entrance — DLIA (M5261) final project, Project 1: Real-Time Multi-Object Detection and Tracking.

- **Detector**: YOLO11-s (COCO-pretrained, fine-tuned on BEE24)
- **Baseline tracker**: Ultralytics ByteTrack, default settings
- **Ours**: ByteTrack with motion-adaptive association — buffered IoU whose search area grows with a track's speed and with the frames it has been lost, a bee-tuned Kalman process noise and a longer lost-track buffer
- **Analytics**: trajectories, occupancy heatmap, in/out counting at a virtual line
- **Data**: [BEE24](https://holmescao.github.io/datasets/BEE24) (36 videos, MOT-format boxes with track IDs)

## Setup ([uv](https://docs.astral.sh/uv/), Python 3.11)

```bash
uv sync                               # creates .venv from uv.lock; CUDA 12.8 PyTorch on Windows/Linux, PyPI build on macOS
uv run python tools/fetch_bee24.py    # downloads BEE24 to data/BEE24 (2.3 GB)
```

Run every command below as `uv run python -m ...` from the repository root.

## Pipeline

| Step | Command | Output |
|---|---|---|
| 1. Split + labels | `python -m beetrack.prep` | `runs/data/` (YOLO lists, `bee24.yaml`, `split.json`), `runs/gt/` (TrackEval GT) |
| 2. Train detector | `python -m beetrack.train --imgsz 640` | `runs/detect/y11s_640/weights/best.pt` |
| 3. Cache detections | `python -m beetrack.detect --weights runs/detect/y11s_640/weights/best.pt --split test` | `runs/dets/y11s_640/test/*.npy` |
| 4. Track | `python -m beetrack.track --dets y11s_640 --split test --tracker ours` | `runs/trackers/test/ours/data/*.txt` |
| 5. Evaluate | `python -m beetrack.evaluate --split test --trackers bytetrack ours` | HOTA / IDF1 / MOTA / IDSW / count error, `runs/eval/` |
| 6. Tune / ablate (val only) | `python -m beetrack.sweep --dets y11s_640 --grid ablation` | `runs/eval/val/sweep_*.json` |
| 7. Analytics | `python -m beetrack.analytics --tracker ours --seq BEE24-36` | `runs/figs/` |
| 8. Demo | `python -m beetrack.demo --source data/BEE24/test/BEE24-36/img1` | annotated video + tracks CSV in `runs/demo/` |

Detections are computed once per detector and reused by every tracker setting, so tracker comparisons change only the tracker.

**Split**: test = the 5 official BEE24 test videos (never used for training or tuning). Validation = BEE24-10/15/24/28 plus the last 1000 frames of the two long training videos (a 50-frame gap separates them from training frames).

**Robustness**: `detect.py --corrupt dark|bright|blur|noise` (synthetic image changes on test frames) and `track.py --step k` with `evaluate.py --split test_step<k>` (frame dropping).

## Layout

```
beetrack/   common (paths, split, MOT I/O) · prep · train · detect · track · evaluate · sweep · analytics · demo
tools/      fetch_bee24.py
proposal/   proposal deck generator (make_figs.py, build_deck.js, build.ps1)
```

## Disclosures

External model: Ultralytics YOLO11-s COCO weights (AGPL-3.0). Dataset: BEE24 (Cao et al.), cite the dataset paper. Evaluation: TrackEval. AI assistance (Claude) was used for coding and writing and is disclosed in the report.
