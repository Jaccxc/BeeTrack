"""Verify a §114 candidate manifest from its native detector cache, P0 MOT and entrance polygon."""

import argparse
from collections import defaultdict
import gzip
import hashlib
import json
import math
from pathlib import Path

import cv2
import numpy as np


def hundredths(value):
    return int(f"{float(value):.2f}".replace(".", ""))


def nms_positions(boxes):
    """Original per-frame input positions, with stable FP32 score ties."""
    xy = boxes[:, :4]
    w = np.maximum(0, np.minimum(xy[:, None, 2], xy[:, 2]) - np.maximum(xy[:, None, 0], xy[:, 0]))
    h = np.maximum(0, np.minimum(xy[:, None, 3], xy[:, 3]) - np.maximum(xy[:, None, 1], xy[:, 1]))
    inter = w * h
    areas = (xy[:, 2] - xy[:, 0]) * (xy[:, 3] - xy[:, 1])
    blocked_by = inter / (areas[:, None] + areas - inter + np.float32(1e-9)) >= np.float32(.5)
    hidden = np.zeros(len(boxes), bool)
    kept = []
    for i in np.argsort(-boxes[:, 4], kind="stable"):
        if not hidden[i]:
            kept.append(i)
            hidden |= blocked_by[i]
    return np.array(sorted(kept), dtype=np.int64)


def group_frames(raw, polygon):
    """Index post-NMS observations; preserve within-frame positions before suppression."""
    order = np.argsort(raw[:, 0], kind="stable")
    times, firsts = np.unique(raw[order, 0], return_index=True)
    blocks = {}
    for i, t in enumerate(times):
        orig = order[firsts[i]:firsts[i + 1] if i + 1 < len(times) else len(raw)]
        data = raw[orig, 1:6].astype(np.float32)
        if np.any(data[:, 2:4] < data[:, :2]) or not np.isfinite(data).all():
            raise ValueError("Invalid FP32 detector box.")
        kept = nms_positions(data)
        obs = data[kept].copy()
        obs[:, 2:4] -= obs[:, :2]
        boxes = np.array([[hundredths(v) for v in row[:4]] for row in obs], dtype=np.int64)
        if np.any((boxes[:, 2:] <= 0).any(axis=1) & (obs[:, 4]>=.25)):
            raise ValueError("An eligible post-NMS detector box has zero width or height at MOT precision.")
        centers2 = 2 * boxes[:, :2] + boxes[:, 2:4]
        # The official side function tests a float32 polygon and a float center.
        centers = boxes[:, :2] / 100 + boxes[:, 2:4] / 200
        inside = np.fromiter((cv2.pointPolygonTest(polygon, (x,y), False) >= 0 for x,y in centers),
                             bool, count=len(obs))
        side = [np.flatnonzero((inside == s) & (obs[:, 4] >= .25)) for s in (False, True)]
        scales = np.sqrt(boxes[:, 2] * boxes[:, 3])
        blocks[int(t)] = (orig[kept], kept, boxes, centers2, scales, side)
    return blocks


def native_swaps(mot, polygon, pairs):
    from beetrack.evaluate import by_track, zone_tracks

    trajectory = by_track(mot[:, :6])
    # A MOT is stored at 0.01 pixel; apply the same convention to callers
    # providing an in-memory array with additional precision.
    if trajectory.size:
        rounded=np.array([[hundredths(v) for v in row] for row in trajectory[:,2:6]])
        if np.any(rounded[:,2:4]<0):
            raise ValueError("Invalid serialized native MOT box.")
        trajectory[:,2:6]=rounded/100
    native = []
    def key(i):
        row = trajectory[i]
        return (int(row[0]), *(hundredths(v) for v in row[2:6]))
    for tid, (event, _, _, flips) in zone_tracks(trajectory, polygon).items():
        last = None
        for i, d in flips:
            k = (key(i - 1), key(i), 1 if d == "in" else -1)
            if min(*k[0][3:5], *k[1][3:5]) <= 0:
                raise ValueError("A native crossing endpoint has zero area at MOT precision.")
            pairs[k][0].add(tid)
            if d == event:
                last = k
        if event:
            if last is None:
                raise ValueError("A native event is missing its official swap.")
            native.append((tid, last[2] * last[1][0], last))
    return native


def raw_swaps(blocks, pairs):
    for t, (ids0, _, boxes0, centers0, radii0, side0) in blocks.items():
        for dt in (1,2,3):
            later = blocks.get(t + dt)
            if later is None:
                continue
            ids1, _, boxes1, centers1, radii1, side1 = later
            for s in (0, 1):
                src, dst = side0[s], side1[1 - s]
                if not len(src) or not len(dst):
                    continue
                delta = centers0[src, None, :] - centers1[dst]
                # Cheap necessary test; the cut and the ranking below use the
                # actual Euclidean distance between exact centipixel centers.
                radius = (radii0[src, None] + radii1[dst]) * (2 * dt)
                ii, jj = np.nonzero(np.max(np.abs(delta), axis=2) <= np.nextafter(radius, np.inf))
                choices = defaultdict(list)
                for a, b in zip(ii, jj):
                    i, j = src[a], dst[b]
                    dx, dy = delta[a,b]
                    dist = math.hypot(int(dx), int(dy)) / (
                        math.sqrt(int(boxes0[i,2] * boxes0[i,3])) +
                        math.sqrt(int(boxes1[j,2] * boxes1[j,3])))
                    if dist <= 2 * dt:
                        choices[i].append((dist, int(ids1[j]), j))
                for i, options in choices.items():
                    u = (t, *map(int,boxes0[i]))
                    for _, raw_id1, j in sorted(options)[:2]:
                        v = (t + dt, *map(int,boxes1[j]))
                        pairs[u, v, 1 if s == 0 else -1][1].add((int(ids0[i]), raw_id1))


def reconstruct(dets, mot, polygon, nframes):
    raw, mot = np.asarray(dets), np.asarray(mot)
    polygon = np.asarray(polygon, np.float32)
    if raw.ndim!=2 or raw.shape[1]<6 or mot.ndim!=2 or mot.shape[1]<6:
        raise ValueError("Expected an N by 6 detector cache and MOT frame,ID,xywh rows.")
    if nframes < 1 or polygon.ndim!=2 or polygon.shape[1]!=2 or len(polygon)<3 or not np.isfinite(polygon).all():
        raise ValueError("Invalid clip length or polygon.")
    for v in (raw, mot[:, :6]):
        if not np.isfinite(v).all() or np.any(v[:,0]!=np.rint(v[:,0])) or np.any((v[:,0]<1)|(v[:,0]>nframes)):
            raise ValueError("Invalid values or frame numbers.")
    if len(np.unique(mot[:,:2], axis=0)) != len(mot) or np.any(mot[:,1]!=np.rint(mot[:,1])):
        raise ValueError("Duplicated or noninteger MOT IDs.")

    pairs = defaultdict(lambda: (set(), set()))
    native = native_swaps(mot, polygon, pairs)
    blocks = group_frames(raw, polygon)
    raw_swaps(blocks, pairs)
    obs = sorted({k for u,v,_ in pairs for k in (u,v)})
    ranks = {key:i for i,key in enumerate(obs)}
    refs = [[] for _ in obs]
    obs_in_frame = defaultdict(set)
    for t,x,y,w,h in obs:
        obs_in_frame[t].add((x,y,w,h))
    for t, keys in obs_in_frame.items():
        if t not in blocks:
            continue
        ids, positions, boxes, _, _, _ = blocks[t]
        for gid, pos, box in zip(ids,positions,boxes):
            b = tuple(box)
            if b in keys:
                refs[ranks[(t,*b)]].append([int(gid),int(pos)])
    for key, links in zip(obs, refs):
        if not links:
            raise ValueError(f"Native swap endpoint {key} has no post-NMS raw provenance.")
    ordered = sorted(pairs, key=lambda k: (ranks[k[0]], ranks[k[1]]))
    arr = []
    for u,v,d in ordered:
        tids, raw_pairs = pairs[u,v,d]
        arr.append([ranks[u],ranks[v],d,sorted(tids),[list(p) for p in sorted(raw_pairs)]])
    indices = {k:i for i,k in enumerate(ordered)}
    return dict(observations=[[*key,r] for key,r in zip(obs,refs)],
                candidates=arr, native_events=sorted([tid,tf,indices[k]] for tid,tf,k in native))


def verify(dets, mot, polygon, nframes, expected_core):
    actual = reconstruct(dets, mot, polygon, nframes)
    for field, rows in actual.items():
        expected = expected_core[field]
        if len(rows) != len(expected):
            raise ValueError(f"{field}: expected {len(expected)} rows; independently reconstructed {len(rows)}")
        for i,(row,exp) in enumerate(zip(rows,expected)):
            if row!=exp:
                raise ValueError(f"{field}[{i}]: expected {exp!r}, reconstructed {row!r}")
    return {k:len(v) for k,v in actual.items()}


def verify_case(case_path, cache_path, mot_path, zones_path, clip=None, nframes=None):
    from beetrack.evaluate import load_tracks
    blob=Path(case_path).read_bytes()
    record=json.loads(gzip.decompress(blob) if blob.startswith(b"\x1f\x8b") else blob)
    source=record["source"]
    if (clip is not None and clip!=source["clip"]) or (nframes is not None and nframes!=source["nframes"]):
        raise ValueError("Supplied clip identity or duration differs from the original case.")
    for k,path in (("det",cache_path),("mot",mot_path),("zones",zones_path)):
        with Path(path).open("rb") as f:
            if hashlib.file_digest(f,"sha256").hexdigest()!=source["hashes"][k]:
                raise ValueError(f"Original {k} input fingerprint differs.")
    polygon=json.loads(Path(zones_path).read_bytes())[source["clip"]]
    return verify(np.load(cache_path,allow_pickle=False),load_tracks(mot_path),polygon,source["nframes"],
                  record["geometry"])


def selfcheck():
    polygon = [[0,0],[10,0],[10,100],[0,100]]
    raw = np.array([[4,-2,5,2,9,.18], [1,-5,5,-1,9,.8], [2,-1,5,3,9,.8],
                    [2,-1,5,3,9,.8], [2,-9,5,-5,9,.2], [3,-5,5,-1,9,.7]], np.float32)
    mot = np.array([[1,11,-5,5,4,4],[2,11,-1,5,4,4],[3,11,-5,5,4,4],
                    [3,12,-5,5,4,4],[4,12,-2,5,4,4]],float)
    geom = reconstruct(raw,mot,polygon,60)
    assert geom["native_events"]==[[12,4,2]]
    assert geom["observations"]==[[1,-500,500,400,400,[[1,0]]],[2,-100,500,400,400,[[2,0]]],
                                   [3,-500,500,400,400,[[5,0]]],[4,-200,500,400,400,[[0,0]]]]
    assert geom["candidates"]==[[0,1,1,[11],[[1,2]]],
                                [1,2,-1,[11],[[2,5]]],[2,3,1,[12],[]]]
    wide=[[0,0],[50,0],[50,100],[0,100]]
    raw2=np.array([[1,-4,20,0,24,.25],[2,0,23,4,27,.4],[2,0,17,4,21,.4],
                   [2,0,20,4,24,.4],[4,20,20,24,24,.4],[5,0,20,4,24,.4]],np.float32)
    blank=np.empty((0,6))
    edges=reconstruct(raw2,blank,wide,6)["candidates"]
    assert sorted(p for edge in edges for p in edge[4])==[[0,1],[0,3],[0,4]]
    low_clipped=np.vstack((raw2,[2,0,40,4,40,.12]))
    assert reconstruct(low_clipped,blank,wide,6)["candidates"]==edges
    assert reconstruct(blank,blank,wide,6)==dict(observations=[],candidates=[],native_events=[])
    assert hundredths(np.float32(1.125))==112 and hundredths(np.float32(1.375))==138


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    sub=ap.add_subparsers(dest="cmd",required=True)
    sub.add_parser("self-check")
    run=sub.add_parser("verify")
    run.add_argument("case",type=Path)
    run.add_argument("cache",type=Path)
    run.add_argument("mot",type=Path)
    run.add_argument("zones",type=Path)
    run.add_argument("--clip")
    run.add_argument("--nframes",type=int)
    args=ap.parse_args()
    if args.cmd=="self-check":
        selfcheck()
        print("Synthetic raw lineage, stable NMS, boundary and native return checks passed.")
    else:
        print(verify_case(args.case,args.cache,args.mot,args.zones,args.clip,args.nframes))
