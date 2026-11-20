"""CPU boundary strips and deployed-protocol event labels. Synthetic check: --self-check."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def _polygon(poly):
    p = np.asarray(poly, dtype=float)
    if p.ndim != 2 or p.shape[1] != 2 or not np.isfinite(p).all():
        raise ValueError("Expected finite polygon coordinates")
    if len(p) > 1 and np.array_equal(p[0], p[-1]):
        p = p[:-1]
    if len(p) < 3 or np.any(np.linalg.norm(np.roll(p, -1, axis=0) - p, axis=1) == 0):
        raise ValueError("Polygon requires three vertices and nonzero edges")
    area = np.sum(p[:, 0] * np.roll(p[:, 1], -1) - p[:, 1] * np.roll(p[:, 0], -1))
    if abs(area) < 1e-10:
        raise ValueError("Polygon has zero area")
    if area < 0:
        p = p[::-1]
    return np.roll(p, -np.lexsort((p[:, 1], p[:, 0]))[0], axis=0)


def ring_geometry(poly, n=32):
    """Uniform perimeter mid-bin centers and inward normals, invariant to vertex order reversal."""
    if not isinstance(n, int) or n < 1:
        raise ValueError("n must be a positive integer")
    p = _polygon(poly)
    edges = np.roll(p, -1, axis=0) - p
    lengths = np.linalg.norm(edges, axis=1)
    ends = np.cumsum(lengths)
    distance = (np.arange(n) + .5) * ends[-1] / n
    idx = np.searchsorted(ends, distance, side="right")
    tangent = edges[idx] / lengths[idx, None]
    centers = p[idx] + tangent * (distance - np.r_[0., ends[:-1]][idx])[:, None]
    return centers, np.c_[-tangent[:, 1], tangent[:, 0]]


def strip_maps(poly):
    centers, normals = ring_geometry(poly)
    tangents = np.c_[normals[:, 1], -normals[:, 0]]
    row, col = np.meshgrid(np.arange(-10, 11) * 4., np.arange(-10, 11) * 4., indexing="ij")
    xy = centers[:, None, None, :] + row[None, :, :, None] * normals[:, None, None, :] + col[None, :, :, None] * tangents[:, None, None, :]
    return xy[..., 0].astype(np.float32), xy[..., 1].astype(np.float32)


def sample_strips(gray, maps):
    """(32,21,21) uint8, columns tangent and rows inward at four-pixel pitch."""
    if gray.ndim != 2 or gray.dtype != np.uint8:
        raise ValueError("Expected uint8 grayscale image")
    mx, my = maps
    return cv2.remap(gray, mx.reshape(-1, 21), my.reshape(-1, 21), cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REFLECT_101).reshape(32, 21, 21)


def _context(repo, seq, part):
    if seq not in ("BEE24-33", "BEE24-35") or part not in ("head", "tail"):
        raise ValueError("Only explicitly allowed 33/35 head/tail clips")
    from beetrack.common import ROOT, split_specs
    repo = Path(repo).resolve()
    if repo != ROOT.resolve():
        raise ValueError("repo must match the imported beetrack checkout")
    if part == "head":
        first, last, split = 1, 3950, "dev"
    else:
        _, _, first, last = next(s for s in split_specs("val") if s[0] == f"{seq}-tail")
        split = "val"
    poly = json.loads((repo / "beetrack/zones.json").read_text())[f"{seq}-{part}"]
    return repo, first, last, split, poly


def cache_images(repo, seq, part, out_npypath):
    """Stream allowed source images into a uint8 .npy memmap; refuse overwriting."""
    repo, first, last, _, poly = _context(repo, seq, part)
    out = Path(out_npypath)
    if out.exists():
        raise FileExistsError(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    result = np.lib.format.open_memmap(out, mode="w+", dtype=np.uint8, shape=(last-first+1, 32, 21, 21))
    maps = strip_maps(poly)
    for i, frame in enumerate(range(first, last+1)):
        path = repo / "data/BEE24/train" / seq / "img1" / f"{frame:06d}.jpg"
        gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise OSError(f"Cannot read image {path}")
        result[i] = sample_strips(gray, maps)
    result.flush()
    return result


def crossing_cell(start, arrival, poly, centers=None):
    """Return (nearest ring cell, fallback flag); last intersection along motion wins."""
    p = _polygon(poly)
    a, b = np.asarray(start, float), np.asarray(arrival, float)
    motion = b-a
    cross = lambda x, y: x[0]*y[1] - x[1]*y[0]
    hits = []
    for j, (u, v) in enumerate(zip(p, np.roll(p, -1, axis=0))):
        edge = v-u
        denominator = cross(motion, edge)
        if abs(denominator) < 1e-10:
            continue
        t, s = cross(u-a, edge)/denominator, cross(u-a, motion)/denominator
        if -1e-9 <= t <= 1+1e-9 and -1e-9 <= s <= 1+1e-9:
            hits.append((float(np.clip(t, 0, 1)), -j))
    point = a + max(hits)[0]*motion if hits else (a+b)/2
    centers = ring_geometry(p)[0] if centers is None else centers
    return int(np.argmin(np.sum((centers-point)**2, axis=1))), not bool(hits)


def labels_from_rows(rows, poly):
    """Return (event tuples, fallback count), using the public deployed GT protocol."""
    from beetrack.evaluate import by_track, zone_tracks
    rows = np.asarray(rows, float).reshape(-1, 6)
    if not len(rows):
        return [], 0
    rows = by_track(rows)
    centers = ring_geometry(poly)[0]
    events, fallbacks = [], 0
    for direction, _, _, crossings in zone_tracks(rows, poly).values():
        if direction:
            j = [j for j, d in crossings if d == direction][-1]
            xy = rows[j-1:j+1, 2:4] + rows[j-1:j+1, 4:6]/2
            cell, fallback = crossing_cell(*xy, poly, centers)
            events.append((int(rows[j, 0]), cell, 0 if direction == "in" else 1))
            fallbacks += int(fallback)
    return sorted(events), fallbacks


def event_cells(repo, seq, part):
    """Integer (local one-based frame, cell, in=0/out=1) labels from prepared GT."""
    repo, _, _, split, poly = _context(repo, seq, part)
    from beetrack.evaluate import load_tracks
    rows = load_tracks(repo / "runs/gt" / split / f"{seq}-{part}" / "gt/gt.txt")
    return labels_from_rows(rows[:, :6], poly)[0]


def self_check():
    square = [[50, 50], [150, 50], [150, 150], [50, 150]]
    centers, normals = ring_geometry(square)
    rev = ring_geometry(square[::-1])
    assert np.allclose(centers, rev[0]) and np.allclose(normals, rev[1])
    assert np.allclose(np.linalg.norm(normals, axis=1), 1)
    assert all(cv2.pointPolygonTest(np.array(square, np.float32), tuple(q), False) > 0 for q in centers+normals)
    maps = strip_maps(square)
    gray = np.indices((220, 220))[0].astype(np.uint8)
    strips = sample_strips(gray, maps)
    assert strips.shape == (32, 21, 21) and strips.dtype == np.uint8
    assert strips[0, 11, 10] - strips[0, 10, 10] == 4
    assert strips[0, 10, 11] == strips[0, 10, 10]
    assert np.array_equal(sample_strips(gray, strip_maps(square[::-1])), strips)
    assert not crossing_cell([0, 100], [100, 100], square)[1]
    assert not crossing_cell([0, 0], [100, 100], square)[1]
    assert crossing_cell([0, 0], [10, 10], square)[1]
    concave = [[0,0], [100,0], [100,100], [70,100], [70,30], [30,30], [30,100], [0,100]]
    c, flag = crossing_cell([-10,50], [110,50], concave)
    assert not flag and np.linalg.norm(ring_geometry(concave)[0][c]-[100,50]) < 10
    row = lambda f, i, x: [f,i,x-1,99,2,2]
    rows = [row(1,1,0), row(2,1,100), row(3,1,0), row(4,1,100), row(1,2,100), row(2,2,200)]
    events, fallback = labels_from_rows(rows, square)
    assert [(f,d) for f,_,d in events] == [(2,1),(4,0)] and fallback == 0
    assert labels_from_rows([], square) == ([],0)
    print("Synthetic geometry, strips, intersections and deployed event checks passed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-check", action="store_true", required=True)
    parser.parse_args()
    self_check()
