"""Derived full-frame matching evidence for the two §116 NMS paths."""
import base64
import hashlib
import json

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment


def require(condition, message):
    if not condition:
        raise ValueError(message)


def wire(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                       allow_nan=False) + "\n").encode("utf-8")


def digest(value):
    return hashlib.sha256(wire(value)).hexdigest()


def encode(values, dtype):
    return base64.b64encode(np.asarray(values, dtype=dtype).tobytes(order="C")).decode("ascii")


def decode(value, dtype):
    return np.frombuffer(base64.b64decode(value, validate=True), dtype=dtype)


def near(box, rect):
    x = (box[0] + box[2]) / 2
    y = (box[1] + box[3]) / 2
    return bool(rect[0] <= x <= rect[2] and rect[1] <= y <= rect[3])


def inside(box, polygon):
    center = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
    return cv2.pointPolygonTest(polygon, center, False) >= 0


def roster_frames(labels, rect):
    frames = {}
    for frame, rows in labels["gt_frames"].items():
        frame_ids = [row["gt_tid"] for row in rows]
        require(frame_ids == sorted(set(frame_ids)), "GT rows must be in official ID order")
        frames[frame] = [[r["gt_tid"], r["inside"], near(r["xyxy"], rect)] for r in rows]
    return frames


def survivors(rows):
    return sorted((i for i, row in enumerate(rows) if row[4] >= .25),
                  key=lambda i: (-rows[i][4], *rows[i][:4], i))


def pack_frame(truth, rows):
    # Keep exact post-NMS source rows including sub-.25 scores. All detector outputs are FP32.
    raw = np.asarray(rows, dtype="<f4").reshape(-1, 6)
    require(raw.tolist() == rows and np.isfinite(raw).all(), "Detector rows lost precision")
    chosen = survivors(rows)
    p = np.asarray([rows[i][:4] for i in chosen], dtype=float).reshape(-1, 4)
    # Official frame_match reconstructs xyxy from input MOT xywh. Preserve its rounding.
    p[:, 2:] = p[:, :2] + (p[:, 2:] - p[:, :2])
    g = np.asarray([row["xywh"] for row in truth], dtype=float).reshape(-1, 4)
    g[:, 2:] = g[:, :2] + g[:, 2:]
    inter = np.maximum(np.minimum(g[:, None, 2:], p[None, :, 2:]) -
                       np.maximum(g[:, None, :2], p[None, :, :2]), 0).prod(axis=2)
    ga = (g[:, 2:] - g[:, :2]).prod(axis=1)
    pa = (p[:, 2:] - p[:, :2]).prod(axis=1)
    iou = inter / (ga[:, None] + pa[None, :] - inter + 1e-9)
    ix = np.flatnonzero(iou)
    return {"rows": encode(raw, "<f4"), "positions": encode(ix, "<u4"),
            "ious": encode(iou.ravel()[ix], "<f8")}


def unpack_prediction(evidence):
    return decode(evidence["rows"], "<f4").reshape(-1, 6).tolist()


def solve(ground, evidence, rect, polygon):
    rows = unpack_prediction(evidence)
    chosen = survivors(rows)
    ids = [r[0] for r in ground]
    require(ids == sorted(set(ids)), "Invalid GT ordering")
    require(all(type(flag) is bool for row in ground for flag in row[1:]), "Invalid GT flags")
    index = decode(evidence["positions"], "<u4")
    ious = decode(evidence["ious"], "<f8")
    mat = np.zeros((len(ground), len(chosen)), dtype=float)
    require(len(index) == len(ious) and np.all(index[1:] > index[:-1]) and
            np.all(index < mat.size) and np.all((ious > 0) & (ious <= 1)), "Invalid overlap matrix")
    mat.flat[index] = ious
    left, right = linear_sum_assignment(-mat)
    pred_near = [near(rows[j], rect) for j in chosen]
    pred_side = [inside(rows[j], polygon) for j in chosen]
    result = {}
    for threshold in (.5, .3):
        owners = {}
        cells = {r: dict(gt=0, tp=0, pred=0, matched_pred=0) for r in ("inside", "outside")}
        for _, _, flag in ground:
            cells["inside" if flag else "outside"]["gt"] += 1
        for flag in pred_near:
            cells["inside" if flag else "outside"]["pred"] += 1
        for g, p in zip(left, right):
            if mat[g, p] < threshold:
                continue
            gid, gside, gnear = ground[g]
            owners[gid] = dict(index=chosen[p], iou=float(mat[g, p]),
                               predicted_inside=pred_side[p], correct_side=gside == pred_side[p])
            cells["inside" if gnear else "outside"]["tp"] += 1
            cells["inside" if pred_near[p] else "outside"]["matched_pred"] += 1
        result[str(threshold)] = (owners, cells)
    return rows, result


def aggregate_regions(cells):
    result = {}
    for region in ("inside", "outside"):
        result[region] = {key: sum(frame[region][key] for frame in cells)
                          for key in ("gt", "tp", "pred", "matched_pred")}
    result["all"] = {key: result["inside"][key]+result["outside"][key] for key in result["inside"]}
    for c in result.values():
        c.update(fn=c["gt"]-c["tp"], fp=c["pred"]-c["matched_pred"],
                 recall=c["tp"]/c["gt"] if c["gt"] else None,
                 precision=c["matched_pred"]/c["pred"] if c["pred"] else None)
    return result


def replay_arm(clip, evidence_by_frame, rect, pool, scored=None):
    polygon = np.array(clip["polygon"], dtype=np.float32)
    truths = clip["frames"]
    require(truths.keys() == evidence_by_frame.keys(), "Incomplete requested frame set")
    rows, outputs = {}, {}
    for f, gt in truths.items():
        rows[f], outputs[f] = solve(gt, pool[evidence_by_frame[f]], rect, polygon)
        if scored is not None:
            official = scored["frames"][f]["matches"]
            for tau in ("0.5", "0.3"):
                matches = outputs[f][tau][0]
                reference = {int(k): v for k, v in official.items() if tau == "0.3" or v["owner05"]}
                require(matches.keys() == reference.keys() and
                        all(all(v == reference[k][field] for field, v in m.items())
                            for k, m in matches.items()), "Disagree with official global assignments")
    reports = {}
    for tau, flag in (("0.5", "visible05"), ("0.3", "visible03")):
        ends = {}
        for e in clip["events"]:
            tid = e["gt_tid"]
            outcome = []
            for key in ("before", "arrival"):
                frame = str(e[key])
                g = next(g for g in truths[frame] if g[0] == tid)
                require(g[1] == ((e["dir"] == "in") == (key == "arrival")), "GT direction mismatch")
                owner = outputs[frame][tau][0].get(tid)
                outcome.append(bool(owner and owner["correct_side"]))
            ends[str(tid)] = outcome
        cells = aggregate_regions([fr[tau][1] for fr in outputs.values()])
        if scored is not None:
            require(cells == scored["regions"][tau], "Disagree with official regional costs")
            require(ends.keys() == {str(e["gt_tid"]) for e in scored["events"]} and
                    all(all(ends[str(e["gt_tid"])]) == e[flag] for e in scored["events"]),
                    "Disagree with official paired endpoints")
        reports[tau] = {"ends": ends, "regions": cells}
    return reports, rows
