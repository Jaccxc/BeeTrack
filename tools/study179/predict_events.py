"""Export safe numeric bundles and predict boundary events without GT or training."""
import argparse
import json
from pathlib import Path
import re
import tempfile

import cv2
import numpy as np
import torch

import strip_data
import strip_model
import strip_score


def _metadata(arm, threshold):
    if arm not in ("T", "S") or not np.isfinite(threshold):
        raise ValueError("Invalid arm or threshold")
    percent = round(float(threshold) * 100)
    if not 0 <= percent <= 101 or abs(percent / 100 - threshold) > 1e-8:
        raise ValueError("Threshold must be an integer percent from 0 to 101")
    return np.array(0 if arm == "T" else 1, np.int64), np.array(percent, np.int64)


def _state(arrays):
    expected = strip_model._network().state_dict()
    if set(arrays) != set(expected):
        raise ValueError("Unexpected tensor keys")
    result = {}
    for key, reference in expected.items():
        value = np.asarray(arrays[key])
        if value.dtype != np.float32 or value.shape != tuple(reference.shape) or not np.isfinite(value).all():
            raise ValueError(f"Invalid tensor {key}")
        result[key] = torch.from_numpy(value.copy())
    return result


def export_bundle(checkpoint, arm, threshold, dest):
    """Export a local weights-only PyTorch checkpoint to a safe numeric NPZ."""
    dest = Path(dest)
    if dest.exists():
        raise FileExistsError(dest)
    arm_code, threshold_percent = _metadata(arm, threshold)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    arrays = {key: tensor.detach().cpu().numpy() for key, tensor in state.items()}
    _state(arrays)
    with dest.open("xb") as stream:
        np.savez_compressed(stream, **arrays, arm_code=arm_code, threshold_percent=threshold_percent)


def load_bundle(path):
    with np.load(path, allow_pickle=False) as bundle:
        expected = set(strip_model._network().state_dict()) | {"arm_code", "threshold_percent"}
        if len(bundle.files) != len(expected) or set(bundle.files) != expected:
            raise ValueError("Unexpected bundle keys")
        code, percent = bundle["arm_code"], bundle["threshold_percent"]
        if (code.shape != () or percent.shape != () or code.dtype.kind not in "iu"
                or percent.dtype.kind not in "iu" or int(code) not in (0, 1)
                or not 0 <= int(percent) <= 101):
            raise ValueError("Invalid integer metadata")
        arm = "T" if int(code) == 0 else "S"
        model = strip_model._network().cpu()
        model.load_state_dict(_state({k: bundle[k] for k in expected - {"arm_code", "threshold_percent"}}))
    return model, arm, int(percent) / 100


def frame_files(directory):
    files = list(Path(directory).iterdir())
    images = [p for p in files if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg")]
    if not images or any(re.fullmatch(r"[0-9]+", p.stem) is None for p in images):
        raise ValueError("Expected numeric JPEG filenames")
    ordered = sorted((int(p.stem), p) for p in images)
    indices = [index for index, _ in ordered]
    if indices[0] < 1 or indices != list(range(indices[0], indices[0] + len(indices))):
        raise ValueError("JPEG indices must be positive, unique and contiguous")
    return ordered


def predict(weights, frames, polygon, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    torch.set_num_threads(2)
    cv2.setNumThreads(2)
    model, arm, threshold = load_bundle(weights)
    poly = json.loads(Path(polygon).read_text())
    if not isinstance(poly, list) or any(not isinstance(p, list) or len(p) != 2 for p in poly):
        raise ValueError("Polygon JSON must be an explicit list of vertex pairs")
    maps = strip_data.strip_maps(poly)
    ordered = frame_files(frames)
    with tempfile.TemporaryDirectory(prefix="boundary-predict-", dir=output.parent) as temporary:
        bank = np.lib.format.open_memmap(Path(temporary) / "strips.npy", mode="w+", dtype=np.uint8,
                                        shape=(len(ordered), 32, 21, 21))
        try:
            for index, (_, path) in enumerate(ordered):
                gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
                if gray is None:
                    raise OSError(f"Cannot decode {path}")
                bank[index] = strip_data.sample_strips(gray, maps)
            bank.flush()
            scores = strip_model.predict_model(model, bank, 1, len(bank), arm)
            events = strip_score.decode(scores, threshold, start=ordered[0][0])
        finally:
            bank._mmap.close()
            del bank
    result = dict(arm=arm, threshold=threshold, first_frame=ordered[0][0], last_frame=ordered[-1][0],
                  event_format=["source_frame", "boundary_cell_0_to_31", "direction_0_in_1_out"],
                  events=events)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("weights", "frames", "polygon", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    predict(args.weights, args.frames, args.polygon, args.output)
