"""Fine-tune a COCO-pretrained YOLO11-s on BEE24 (one class: bee).
run: uv run python -m beetrack.train --imgsz 640     (and --imgsz 1280 for Experiment 2)
weights end up in runs/detect/y11s_<imgsz>/weights/best.pt
"""
import argparse

from ultralytics import YOLO

from .common import RUNS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--epochs", type=int, default=50)   # same budget for both input sizes (Experiment 2 fairness)
    ap.add_argument("--batch", type=int, default=0, help="0 = 16 at 640, 4 at 1280 (8 GB GPU)")
    ap.add_argument("--model", default="yolo11s.pt")
    a = ap.parse_args()
    batch = a.batch or (16 if a.imgsz <= 640 else 4)
    YOLO(a.model).train(
        data=str(RUNS / "data" / "bee24.yaml"), imgsz=a.imgsz, epochs=a.epochs, batch=batch,
        project=str(RUNS / "detect"), name=f"y11s_{a.imgsz}", exist_ok=True,
        seed=0, deterministic=True, workers=4, patience=15, plots=True,
        fliplr=0.5, flipud=0.0, mosaic=1.0, close_mosaic=10,
    )


if __name__ == "__main__":
    main()
