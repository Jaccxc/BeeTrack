"""CPU-only directional optical flux at the registered entrance polygon.

From the project root with PYTHONPATH set to it, pass an allowed clip name
and an output directory. The .npz stores two flux columns (in, out) for every
local frame; row 0 is zero because no preceding frame exists within the clip.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import time

import cv2
import numpy as np


CLIPS = [f"BEE24-{hive}-{part}" for hive in (33, 35) for part in ("head", "tail")]
FLOW_ARGS = dict(pyr_scale=.5, levels=3, winsize=15, iterations=3,
                 poly_n=5, poly_sigma=1.2, flags=0)
LONG_EDGE = 480


class Geometry:
    """Precompute positions, arc weights, interpolation and camera grid once."""

    def __init__(self, shape, poly):
        height, width = shape
        vertices = np.asarray(poly, dtype=np.float64)
        if vertices.ndim != 2 or vertices.shape[1] != 2 or len(vertices) < 3 or not np.isfinite(vertices).all():
            raise ValueError("Invalid entrance polygon")
        edges = np.roll(vertices, -1, axis=0) - vertices
        area2 = np.sum(vertices[:, 0] * edges[:, 1] - vertices[:, 1] * edges[:, 0])
        if not area2:
            raise ValueError("Entrance polygon has zero signed area")
        points, normals, arc = [], [], []
        for start, delta in zip(vertices, edges):
            length = np.hypot(*delta)
            if length == 0:
                continue
            pieces = math.ceil(length)
            mid = start + (np.arange(pieces)[:, None] + .5) / pieces * delta
            inside = ((mid[:, 0] >= 2) & (mid[:, 0] <= width - 3)
                      & (mid[:, 1] >= 2) & (mid[:, 1] <= height - 3))
            points.extend(mid[inside])
            normal = np.sign(area2) * np.array([-delta[1], delta[0]]) / length
            normals.extend([normal] * int(inside.sum()))
            arc.extend([length / pieces] * int(inside.sum()))
        points = np.asarray(points).reshape(-1, 2)
        self.normals = np.asarray(normals).reshape(-1, 2)
        self.arc = np.asarray(arc)
        self.x, self.y = np.floor(points).astype(np.intp).T
        self.fx, self.fy = (points - np.floor(points)).T
        polygon32 = vertices.astype(np.float32)
        grid = [(y, x) for y in range(0, height, 4) for x in range(0, width, 4)
                if cv2.pointPolygonTest(polygon32, (x, y), True) <= -8]
        self.gy, self.gx = np.asarray(grid, dtype=np.intp).reshape(-1, 2).T

    def integrate(self, flow, camera):
        x, y = self.x, self.y
        fx, fy = self.fx[:, None], self.fy[:, None]
        sampled = ((1-fy) * ((1-fx) * flow[y, x] + fx * flow[y, x+1])
                   + fy * ((1-fx) * flow[y+1, x] + fx * flow[y+1, x+1]))
        normal_flow = ((sampled - camera) * self.normals).sum(axis=1)
        return np.array([np.maximum(normal_flow, 0) @ self.arc,
                         np.maximum(-normal_flow, 0) @ self.arc])


def pair_flux(before, after, geom):
    """Return [in, out] pixel-area flux, estimated camera [dx, dy], and fallback."""
    flow = cv2.calcOpticalFlowFarneback(before, after, None, **FLOW_ARGS)
    gx = cv2.Sobel(before, cv2.CV_32F, 1, 0, ksize=3)[geom.gy, geom.gx]
    gy = cv2.Sobel(before, cv2.CV_32F, 0, 1, ksize=3)[geom.gy, geom.gx]
    power = gx * gx + gy * gy
    # Row-major grid order breaks ties in squared gradient magnitude.
    selected = np.argsort(-power, kind="stable")[:math.ceil(len(power) / 4)]
    selected = selected[power[selected] > 0]
    fallback = len(selected) < 32
    camera = np.zeros(2, dtype=np.float32) if fallback else np.median(
        flow[geom.gy[selected], geom.gx[selected]], axis=0)
    return geom.integrate(flow, camera), camera, fallback


def work_size(shape):
    h, w = shape
    scale = LONG_EDGE / max(shape)
    return max(1, round(w * scale)), max(1, round(h * scale))


def extract(clip, outdir):
    from beetrack.common import DATA, split_specs
    from beetrack.evaluate import ZONES

    if clip not in CLIPS:
        raise ValueError("Expected one of the four registered clips")
    split = "dev" if clip.endswith("-head") else "val"
    [(name, source, first, last)] = [r for r in split_specs(split) if r[0] == clip]
    polygon = np.asarray(json.loads(ZONES.read_text(encoding="utf-8"))[name], dtype=np.float64)
    folder = DATA / "train" / source / "img1"
    flux = np.zeros((last - first + 1, 2), dtype=np.float64)
    camera = np.zeros_like(flux, dtype=np.float32)
    fallback = np.zeros(len(flux), dtype=bool)
    frames_digest = hashlib.sha256()
    start = time.perf_counter()
    for i, number in enumerate(range(first, last + 1)):
        filename = f"{number:06d}.jpg"
        raw = (folder / filename).read_bytes()
        frames_digest.update(f"{hashlib.sha256(raw).hexdigest()}  {filename}\n".encode("ascii"))
        img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_GRAYSCALE | cv2.IMREAD_IGNORE_ORIENTATION)
        if img is None:
            raise ValueError(f"Cannot decode {source}/{filename}")
        if i == 0:
            source_shape = img.shape
            w, h = work_size(source_shape)
            scaled_polygon = polygon * [w / source_shape[1], h / source_shape[0]]
            geom = Geometry((h, w), scaled_polygon)
        if img.shape != source_shape:
            raise ValueError(f"Frame dimensions changed at {source}/{filename}")
        current = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
        if i:
            flux[i], camera[i], fallback[i] = pair_flux(previous, current, geom)
        previous = current
        if (i+1) % 250 == 0:
            print(f"{clip} {i+1}/{len(flux)} frames, {time.perf_counter()-start:.1f}s", flush=True)
    if not (np.isfinite(flux).all() and (flux >= 0).all() and np.isfinite(camera).all()):
        raise ValueError("Non-finite flux or camera motion")
    meta = dict(clip=clip, split=split, source=source, source_first=first, source_last=last,
                nframes=len(flux), fps=25, source_wh=list(source_shape[::-1]), work_wh=[w, h],
                polygon_xy=polygon.tolist(), boundary_samples=len(geom.arc), arc_length=float(sum(geom.arc)),
                background_grid_size=len(geom.gx), fallback_pairs=int(fallback.sum()),
                frame_pairs=len(flux)-1, columns=["in", "out"], frame_index="source_first + row",
                flow_parameters=FLOW_ARGS, opencv=cv2.__version__, long_edge=LONG_EDGE,
                frames_manifest_sha256=frames_digest.hexdigest(),
                frame_manifest_format="SHA-256: lower-case hex digest of each JPG, two spaces, six-digit JPG filename, LF, ordered by frame",
                flux_f64le_sha256=hashlib.sha256(flux.astype("<f8", copy=False).tobytes()).hexdigest())
    outdir.mkdir(parents=True, exist_ok=True)
    dest = outdir / f"{clip}.npz"
    with tempfile.NamedTemporaryFile(dir=outdir, delete=False, suffix=".tmp") as stream:
        tmp = Path(stream.name)
        try:
            np.savez_compressed(stream, flux=flux, camera_xy=camera, camera_fallback=fallback,
                                metadata=np.asarray(json.dumps(meta, sort_keys=True, allow_nan=False)))
        except BaseException:
            stream.close()
            tmp.unlink(missing_ok=True)
            raise
    try:
        os.replace(tmp, dest)
    finally:
        tmp.unlink(missing_ok=True)
    print(f"{clip} saved {len(flux)} frames, {len(geom.arc)} boundary samples, "
          f"{meta['fallback_pairs']}/{meta['frame_pairs']} camera fallbacks; {dest}", flush=True)


def self_check():
    poly = np.array([[15.2, 13.4], [66.8, 16.5], [69., 53.], [20.4, 60.]])
    geom = Geometry((80, 80), poly)
    reverse = Geometry((80, 80), poly[::-1])
    y, x = np.mgrid[:80, :80]
    flow = np.dstack((.2*x + .3*y, .5*x - .1*y)).astype(np.float32)
    inward = geom.integrate(flow, np.zeros(2))
    assert np.allclose(inward, reverse.integrate(flow, np.zeros(2)), rtol=1e-12)
    points = np.stack((geom.x+geom.fx, geom.y+geom.fy), axis=1)
    known = np.stack((.2*points[:, 0] + .3*points[:, 1], .5*points[:, 0] - .1*points[:, 1]), axis=1)
    normal = np.sum(known * geom.normals, axis=1)
    assert np.allclose(inward, [np.maximum(normal, 0) @ geom.arc, np.maximum(-normal, 0) @ geom.arc])
    border = Geometry((80, 80), [[0, 0], [79, 0], [79, 79], [0, 79]])
    assert len(border.arc) == 0
    assert np.array_equal(border.integrate(flow, [1, 1]), [0, 0])
    blank = np.zeros((80, 80), dtype=np.uint8)
    masses, camera, fallback = pair_flux(blank, blank, geom)
    assert fallback and np.array_equal(camera, [0, 0]) and np.array_equal(masses, [0, 0])
    assert work_size((1080, 1920)) == (480, 270)
    print("Synthetic geometry, winding, interpolation, border and camera-fallback checks passed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("clip", choices=[*CLIPS, "selfcheck"])
    parser.add_argument("outdir", type=Path, nargs="?")
    args = parser.parse_args()
    cv2.setNumThreads(1)
    cv2.ocl.setUseOpenCL(False)
    if args.clip == "selfcheck":
        self_check()
    else:
        if args.outdir is None:
            parser.error("provide an output directory for a study clip")
        extract(args.clip, args.outdir)
