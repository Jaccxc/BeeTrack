"""Unlabelled local MOT/RGB features; identical extraction for both arms.

MOT is frame,id,x,y,w,h[,ignored columns]. RGB reader(frame) returns native
uint8 RGB or None. No GT, score, clip/hive identity, model, or external asset.
Positive swap_evidence means exchanging the two endpoint identities has the
smaller appearance cost. compare() returns HIGH minus LOW, not a decision.
"""
from collections import OrderedDict
import itertools
import json

import numpy as np


GEOMETRY_NAMES = ["log_track_count", "gap_fraction", "normalized_speed",
                  "normalized_velocity_change", "abs_log_area_rate"]
FEATURE_NAMES = GEOMETRY_NAMES + ["log_sampled_close_pairs", "appearance_valid_fraction",
    "appearance_informative_fraction", "swap_evidence_mean", "swap_evidence_absolute_mean"]
RADIUS = 2.0
HALF_GAP = 4
MAX_PAIRS = 16
MAX_GAP = 25  # Official by_track: split only gaps strictly greater than25.
MIN_CONTRAST = .02
MIN_CHANGE = 1 / 255
MIN_MARGIN = 1 / 255


def _mot(mot):
    a = np.asarray(mot)
    if a.ndim != 2 or a.shape[1] < 6:
        raise ValueError("MOT must be Nx6+: frame,id,x,y,w,h[,ignored]")
    a = np.asarray(a[:, :6], float)
    if (not np.isfinite(a).all() or np.any(a[:, :2] != np.floor(a[:, :2]))
            or np.any(a[:, 0] < 1) or np.any(a[:, 1] < 0) or np.any(a[:, 4:6] <= 0)
            or len(np.unique(a[:, :2], axis=0)) != len(a)):
        raise ValueError("finite positive boxes and unique integral frame/ID required")
    return a


class _Frames:
    """Four cached RGB snapshots; never retain a whole video in memory."""
    def __init__(self, reader):
        self.reader, self.cache, self.reads = reader, OrderedDict(), 0

    def __call__(self, frame):
        frame = int(frame)
        if frame in self.cache:
            self.cache.move_to_end(frame)
            return self.cache[frame]
        if self.reader is None:
            return None
        try:
            value = self.reader(frame)
        except (FileNotFoundError, KeyError):
            value = None
        self.reads += 1
        if value is not None:
            value = np.asarray(value)
            if value.ndim != 3 or value.shape[2] != 3 or value.dtype != np.uint8:
                raise ValueError("reader must return RGB uint8 HxWx3, or None")
            value = value.copy()
        self.cache[frame] = value
        if len(self.cache) > 4:
            self.cache.popitem(last=False)
        return value


def _crop(image, row, inset=.25):
    # Pixel centres inside the fixed middle 50% rectangle, without padding,
    # border clipping, resizing, date search or appearance-based selection.
    x, y, w, h = row[2:6]
    x0, x1 = np.ceil(np.array([x + inset*w, x + (1-inset)*w]) - .5).astype(int)
    y0, y1 = np.ceil(np.array([y + inset*h, y + (1-inset)*h]) - .5).astype(int)
    if x0 < 0 or y0 < 0 or x1 > image.shape[1] or y1 > image.shape[0] or x1 <= x0 or y1 <= y0:
        return None
    return image[y0:y1, x0:x1].astype(float) / 255


def _describe(crop):
    pixels = crop.reshape(-1, 3)
    return np.r_[pixels.mean(axis=0), pixels.std(axis=0)]


def _distance(a, b):
    return float(np.abs(a - b).mean())


def _appearance_pair(rows, images):
    """The unchanged four-location descriptor/gate kernel for both callers."""
    result = dict(state="missing_RGB", valid=False, informative=False, vote=0)
    if any(image is None for image in images):
        return result
    if images[0].shape != images[1].shape:
        result["state"] = "RGB_shape_changed"
        return result
    crops = [_crop(images[side], row) for side in (0, 1) for row in rows[2*side:2*side+2]]
    if any(crop is None for crop in crops):
        result["state"] = "centre_crop_outside_or_empty"
        return result
    p, q, r, s = map(_describe, crops)
    aligned = _distance(p, r) + _distance(q, s)
    swapped = _distance(p, s) + _distance(q, r)
    margin = aligned-swapped
    contrast = [_distance(p, q), _distance(r, s)]
    changes = []
    for row in rows:
        c0, c1 = _crop(images[0], row, 0), _crop(images[1], row, 0)
        if c0 is None or c1 is None:
            result["state"] = "temporal_support_outside_or_empty"
            return result
        changes.append(float(np.abs(c0-c1).mean()))
    change = float(np.mean(changes))
    result.update(valid=True, aligned_cost=aligned, exchanged_cost=swapped, raw_margin=margin,
                  pair_contrast=contrast, temporal_change=change,
                  temporal_change_by_fixed_box=changes)
    if min(contrast) < MIN_CONTRAST:
        result["state"] = "same_appearance_no_information"
    elif change < MIN_CHANGE:
        result["state"] = "static_image_no_information"
    elif abs(margin) <= MIN_MARGIN:
        result["state"] = "equal_assignment_cost_no_information"
    else:
        result.update(state="informative_exchange_preferred" if margin > 0 else "informative_original_preferred",
                      informative=True, vote=-1 if margin > 0 else 1)
    return result


def _fixed_pair(mot, ids, before, after, frames):
    if (isinstance(before, bool) or isinstance(after, bool) or int(before) != before or int(after) != after
            or before < 1 or after-before != 2*HALF_GAP):
        raise ValueError("fixed dates must be positive integers exactly8frames apart")
    if len(ids) != 2 or any(isinstance(v, bool) or int(v) != v for v in ids) or ids[0] == ids[1]:
        raise ValueError("two distinct fixed native IDs required")
    a = _mot(mot)
    result = dict(ids=[int(v) for v in ids], endpoint_frames=[int(before), int(after)],
                  state="missing_endpoint", valid=False, informative=False, vote=0)
    rows = []
    for date in (before, after):
        for identity in ids:
            selected = a[(a[:, 0] == date) & (a[:, 1] == identity)]
            if len(selected) != 1:
                return result
            rows.append(selected[0])
    result["fixed_box_rows"] = [row.tolist() for row in rows]
    result.update(_appearance_pair(rows, [frames(before), frames(after)]))
    return result


def evaluate_fixed_pair(mot, ids, before, after, read_rgb):
    """Evaluate a caller-frozen pair/date, without contact/date selection.

    Same middle50% descriptors and SAME four full-box temporal support regions
    as extract(). Positive raw_margin favours exchange; vote+1 prefers original,
    vote-1 exchange, vote0 is uninformative/invalid. No GT correctness is inferred.
    """
    frames = _Frames(read_rgb)
    result = _fixed_pair(mot, ids, before, after, frames)
    result["RGB_reads"] = frames.reads
    return result


def probe_fixed_pair(mot, ids, before, after, read_rgb, *, frame_shift=None):
    """Main+shifted RGB control with the exact SAME four original box rows.

    Caller fixes frame_shift (e.g.+25 or-25 based only on known clip bounds).
    No lookup at shifted GT/MOT dates, no pair/date reselection, no fallback.
    """
    if frame_shift is not None and (isinstance(frame_shift, bool) or int(frame_shift) != frame_shift):
        raise ValueError("explicit integral RGB frame shift required")
    frames = _Frames(read_rgb)
    main = _fixed_pair(mot, ids, before, after, frames)
    shifted = None
    if frame_shift is not None:
        frame_shift = int(frame_shift)
        shifted = _fixed_pair(mot, ids, before, after, lambda date: frames(date+frame_shift))
        shifted["RGB_endpoint_frames"] = [int(before)+frame_shift, int(after)+frame_shift]
    return dict(main=main, shifted=shifted, frame_shift=frame_shift, RGB_reads=frames.reads,
                control="RGB dates shifted only; same original four boxes and fixed pair/date")


def _extract(mot, frames):
    a = _mot(mot)
    tracks = []
    for identity in np.unique(a[:, 1]):
        rows = a[a[:, 1] == identity]
        rows = rows[np.argsort(rows[:, 0], kind="stable")]
        cuts = np.flatnonzero(np.diff(rows[:, 0]) > MAX_GAP) + 1
        for run, segment in enumerate(np.split(rows, cuts)):
            # Tuple keys cannot alias another native ID's official +1e5 ID.
            # Geometry-only order also makes ID renaming/input order irrelevant.
            signature = tuple(segment[:, [0, 2, 3, 4, 5]].ravel())
            tracks.append((signature, (int(identity), run), segment))
    tracks.sort(key=lambda t: t[0])
    x = np.zeros(len(FEATURE_NAMES))
    valid = np.zeros(len(FEATURE_NAMES), bool)
    x[0], valid[0] = np.log1p(len(tracks)), True
    elapsed, missing, speeds, accelerations, area_rates = [], [], [], [], []
    index = {}
    for i, (_, identity, rows) in enumerate(tracks):
        for row in rows:
            index.setdefault(int(row[0]), []).append((i, row))
        if len(rows) < 2:
            continue
        dt = np.diff(rows[:, 0])
        centre = rows[:, 2:4] + rows[:, 4:6] / 2
        size = np.sqrt(rows[:, 4] * rows[:, 5])
        velocity = np.diff(centre, axis=0) / dt[:, None]
        speeds.extend(np.linalg.norm(velocity, axis=1) / ((size[:-1] + size[1:]) / 2))
        elapsed.extend(dt)
        missing.extend(dt - 1)
        area_rates.extend(np.abs(np.diff(np.log(size**2))) / dt)
        if len(velocity) > 1:
            accelerations.extend(np.linalg.norm(np.diff(velocity, axis=0), axis=1)
                                 / ((dt[:-1] + dt[1:]) / 2) / size[1:-1])
    if elapsed:
        x[1], valid[1] = sum(missing) / sum(elapsed), True
    for slot, values in ((2, speeds), (3, accelerations), (4, area_rates)):
        if values:
            x[slot], valid[slot] = np.mean(values), True
    # Closest co-visible date for each pair; exact distance ties keep the first
    # native date. Choose before checking endpoint images/availability.
    minima = {}
    for date in sorted(index):
        for (i, row_i), (j, row_j) in itertools.combinations(index[date], 2):
            pi, pj = row_i[2:4] + row_i[4:6]/2, row_j[2:4] + row_j[4:6]/2
            scale = (np.sqrt(np.prod(row_i[4:6])) + np.sqrt(np.prod(row_j[4:6]))) / 2
            distance = float(np.linalg.norm(pi-pj) / scale)
            key = (i, j)
            if distance <= RADIUS and (key not in minima or distance < minima[key][0]):
                minima[key] = (distance, date)
    candidates = [(date, distance, i, j) for (i, j), (distance, date) in minima.items()]
    candidates.sort(key=lambda c: (c[0], c[1], tracks[c[2]][0], tracks[c[3]][0]))
    selected = candidates[:MAX_PAIRS]
    x[5], valid[5:8] = np.log1p(len(selected)), True
    evidence, margins, measured = [], [], 0
    for date, distance, i, j in selected:
        before, after = date-HALF_GAP, date+HALF_GAP
        item = dict(ids=[tracks[i][1][0], tracks[j][1][0]],
                    id_runs=[dict(original_id=tracks[k][1][0], run=tracks[k][1][1]) for k in (i, j)], contact_frame=date,
                    normalized_distance=distance, endpoint_frames=[before, after], state="missing_endpoint")
        lookups = [{int(row[0]): row for row in tracks[k][2]} for k in (i, j)]
        if not all(t in lookup for lookup in lookups for t in (before, after)):
            evidence.append(item)
            continue
        rows = [lookups[k][t] for t in (before, after) for k in (0, 1)]
        result = _appearance_pair(rows, [frames(before), frames(after)])
        item.update(result)
        measured += int(result["valid"])
        if result["informative"]:
            margins.append(result["raw_margin"])
        evidence.append(item)
    if selected:
        x[6], x[7] = measured/len(selected), len(margins)/len(selected)
    if margins:
        x[8:10] = [np.mean(margins), np.mean(np.abs(margins))]
        valid[8:10] = True
    return dict(features=x, valid=valid, feature_names=FEATURE_NAMES.copy(),
                geometry_features=x[:5].copy(), geometry_valid=valid[:5].copy(),
                runs=[dict(original_id=identity[0], run=identity[1], first_frame=int(rows[0, 0]),
                           last_frame=int(rows[-1, 0]), observations=len(rows)) for _, identity, rows in tracks],
                evidence=evidence, appearance_has_information=bool(margins),
                pair_count=len(candidates), sampled_pair_count=len(selected),
                selection=dict(radius=RADIUS, endpoint_gap=HALF_GAP, maximum_pairs=MAX_PAIRS, split_gap=MAX_GAP,
                               crop="fixed middle50%; no padding/clipping", descriptor="RGB mean+std",
                               rule="per-pair closest co-visible date, first exact tie; first16 by date/distance/geometry; no endpoint fallback"))


def extract(mot, read_rgb=None):
    """One arm; missing feature entries are zero with valid=False, never NaN."""
    frames = _Frames(read_rgb)
    out = _extract(mot, frames)
    out["RGB_reads"] = frames.reads
    return out


def compare(high_mot, low_mot, read_rgb=None):
    """Same extractor on both arms; only common valid values enter high-low."""
    frames = _Frames(read_rgb)
    high, low = _extract(high_mot, frames), _extract(low_mot, frames)
    valid = high["valid"] & low["valid"]
    delta = np.where(valid, high["features"]-low["features"], 0.)
    return dict(features=delta, valid=valid, feature_names=FEATURE_NAMES.copy(),
                geometry_features=delta[:5], geometry_valid=valid[:5],
                high=high, low=low, RGB_reads=frames.reads,
                convention="high-low; positive within-arm swap margin favours exchanging endpoint identities")


def _fixture(swapped=False, same_color=False, bounce=False, static=False):
    rows = []
    frames = {}
    for date in range(1, 18):
        image = np.full((72, 80, 3), 40, np.uint8)
        xa, xb = 20+2*date, 58-2*date
        if bounce and date > 9:
            xa, xb = xb, xa
        physical = [(10, xa, (220, 30, 30)), (20, xb, (220, 30, 30) if same_color else (30, 30, 220))]
        for identity, x, color in physical:
            output_id = 30-identity if swapped and date > 9 else identity
            rows.append([date, output_id, x-4, 30-4, 8, 8, .1])
            image[26:34, int(x)-4:int(x)+4] = color
        if static:
            image[:] = 40
            image[24:36, 20:38] = (220, 30, 30)
            image[24:36, 40:58] = (30, 30, 220)
        frames[date] = image
    return np.array(rows, float), lambda date: frames.get(date)


def selfcheck():
    import hashlib
    good, passing_rgb = _fixture()
    bad, _ = _fixture(swapped=True)
    true_swap = extract(bad, passing_rgb)
    no_swap = extract(good, passing_rgb)
    bounce, bounce_rgb = _fixture(bounce=True)
    no_swap_bounce = extract(bounce, bounce_rgb)
    np.testing.assert_array_equal(bad[np.lexsort((bad[:, 1], bad[:, 0])), :6],
                                  bounce[np.lexsort((bounce[:, 1], bounce[:, 0])), :6])
    np.testing.assert_array_equal(true_swap["geometry_features"], no_swap_bounce["geometry_features"])
    assert true_swap["features"][8] > 0 and no_swap["features"][8] < 0 and no_swap_bounce["features"][8] < 0
    assert true_swap["evidence"][0]["endpoint_frames"] == [5, 13]
    same, same_rgb = _fixture(swapped=True, same_color=True)
    same_result = extract(same, same_rgb)
    static, static_rgb = _fixture(swapped=True, static=True)
    static_result = extract(static, static_rgb)
    baseline = b"".join(r["features"].tobytes()+r["valid"].tobytes()
                        for r in (no_swap, true_swap, no_swap_bounce, same_result, static_result))
    assert hashlib.sha256(baseline).hexdigest() == "5d11792113f6848fd233c7aefa3df5958ca0eaf26234746f123107c91dcd3bfe"
    assert not same_result["appearance_has_information"] and not static_result["appearance_has_information"]
    single = extract(good[good[:, 1] == 10], passing_rgb)
    assert not single["appearance_has_information"] and single["RGB_reads"] == 0
    truncated = extract(bad[bad[:, 0] >= 6], passing_rgb)
    assert truncated["evidence"][0]["state"] == "missing_endpoint"
    missing_image = extract(bad, lambda date: None)
    assert missing_image["evidence"][0]["state"] == "missing_RGB"
    outside = bad.copy(); outside[:, 2] -= 40
    assert extract(outside, passing_rgb)["evidence"][0]["state"] == "centre_crop_outside_or_empty"
    brightness = lambda date: (passing_rgb(date).astype(float) * .75 + 22).astype(np.uint8)
    assert extract(bad, brightness)["features"][8] > 0 and extract(good, brightness)["features"][8] < 0
    # No score feature: arbitrary extra columns are ignored, even nonfinite.
    changed_score = bad.copy(); changed_score[:, 6] = np.nan
    np.testing.assert_array_equal(extract(changed_score, passing_rgb)["features"], true_swap["features"])
    renamed = bad.copy(); renamed[:, 1] = np.where(renamed[:, 1] == 10, 917, 3)
    renamed = renamed[np.random.default_rng(193).permutation(len(renamed))]
    np.testing.assert_array_equal(extract(renamed, passing_rgb)["features"], true_swap["features"])
    ab, ba = compare(bad, good, passing_rgb), compare(good, bad, passing_rgb)
    np.testing.assert_array_equal(ab["valid"], ba["valid"])
    np.testing.assert_array_equal(ab["features"], -ba["features"])
    shifted = bad.copy(); shifted[:, 0] += 950
    np.testing.assert_array_equal(extract(shifted, lambda date: passing_rgb(date-950))["features"], true_swap["features"])
    np.testing.assert_array_equal(extract(bad, passing_rgb)["features"], true_swap["features"])
    # At most16 fixed contacts => at most32 RGB reads/arm; no image search.
    assert true_swap["RGB_reads"] == 2 and ab["RGB_reads"] == 2
    empty = extract(np.empty((0, 6)), passing_rgb)
    assert not empty["appearance_has_information"]
    many = []
    for j in range(18):
        piece = bad.copy(); piece[:, 0] += 20*j; piece[:, 1] += 100*j
        many.append(piece)
    capped = extract(np.concatenate(many), lambda date: passing_rgb((date-1) % 20 + 1))
    assert capped["pair_count"] == 18 and capped["sampled_pair_count"] == MAX_PAIRS and capped["RGB_reads"] == 32
    later = bad.copy(); later[:, 0] += 100
    reused = np.r_[bad, later]
    reused_rgb = lambda date: passing_rgb(date-100 if date > 100 else date)
    reuse_result = extract(reused, reused_rgb)
    assert len(reuse_result["runs"]) == 4 and reuse_result["sampled_pair_count"] == 2
    assert sorted(e["contact_frame"] for e in reuse_result["evidence"]) == [9, 109]
    np.testing.assert_allclose(reuse_result["features"][1:5], true_swap["features"][1:5], atol=1e-15)
    already_split = reused.copy(); already_split[already_split[:, 0] > 100, 1] += 100000
    np.testing.assert_array_equal(extract(already_split, reused_rgb)["features"], reuse_result["features"])
    late_short = reused[(reused[:, 0] <= 17) | (reused[:, 0] >= 108)]
    reads = []
    short_result = extract(late_short, lambda date: (reads.append(date), reused_rgb(date))[1])
    missing_run = next(e for e in short_result["evidence"] if e["contact_frame"] == 109)
    assert missing_run["state"] == "missing_endpoint" and missing_run["endpoint_frames"] == [105, 113]
    assert all(identity["run"] == 1 for identity in missing_run["id_runs"])
    assert reads == [5, 13]  # No older-run or nearby-date substitute.
    collision_track = bad[bad[:, 1] == 10].copy(); collision_track[:, 1] = 100010; collision_track[:, 3] += 40
    collision_result = extract(np.r_[reused, collision_track], reused_rgb)
    keys = {(r["original_id"], r["run"]) for r in collision_result["runs"]}
    assert len(keys) == 5 and (10, 1) in keys and (100010, 0) in keys
    boundary = extract(np.array([[1, 7, 0, 0, 8, 8], [26, 7, 1, 0, 8, 8], [52, 7, 2, 0, 8, 8]], float))
    assert [r["observations"] for r in boundary["runs"]] == [2, 1]
    fixed_bad = evaluate_fixed_pair(bad, (10, 20), 5, 13, passing_rgb)
    fixed_good = evaluate_fixed_pair(good, (10, 20), 5, 13, passing_rgb)
    assert fixed_bad["vote"] == -1 and fixed_good["vote"] == 1
    for key in ("aligned_cost", "exchanged_cost", "raw_margin", "pair_contrast", "temporal_change"):
        np.testing.assert_array_equal(fixed_bad[key], true_swap["evidence"][0][key])
    assert np.mean(fixed_bad["temporal_change_by_fixed_box"]) == fixed_bad["temporal_change"]
    assert evaluate_fixed_pair(same, (10, 20), 5, 13, same_rgb)["vote"] == 0
    fixed_static = evaluate_fixed_pair(static, (10, 20), 5, 13, static_rgb)
    assert fixed_static["vote"] == 0 and fixed_static["temporal_change"] == 0
    no_endpoint = evaluate_fixed_pair(bad[bad[:, 0] != 13], (10, 20), 5, 13, passing_rgb)
    assert no_endpoint["vote"] == 0 and no_endpoint["RGB_reads"] == 0
    requests = []

    def shifted_reader(date):
        requests.append(date)
        return passing_rgb(date) if date <= 17 else static_rgb(date-25)

    probe = probe_fixed_pair(bad, (10, 20), 5, 13, shifted_reader, frame_shift=25)
    assert probe["main"]["vote"] == -1 and probe["shifted"]["vote"] == 0
    assert probe["shifted"]["temporal_change"] == 0
    assert probe["main"]["fixed_box_rows"] == probe["shifted"]["fixed_box_rows"]
    assert requests == [5, 13, 30, 38] and probe["RGB_reads"] == 4
    return dict(feature_names=FEATURE_NAMES, true_ID_swap_margin=float(true_swap["features"][8]),
                correct_pass_margin=float(no_swap["features"][8]), correct_bounce_margin=float(no_swap_bounce["features"][8]),
                same_geometry_opposite_image_evidence="passed", same_color=same_result["evidence"][0]["state"],
                static_background=static_result["evidence"][0]["state"], single_bee="no pair, no RGB reads",
                missing_endpoints="reported without fallback", ID_row_score_date_invariance="passed",
                arm_exchange="exact antisymmetry", RGB_reads_per_toy_pair=true_swap["RGB_reads"],
                brightness_control="same signs under held-out .75x+22 RGB", out_of_image="missing without padding",
                date_cost_cap=dict(eligible=18, sampled=capped["sampled_pair_count"], RGB_reads=capped["RGB_reads"]),
                native_ID_reuse=dict(gap=100, runs=4, already_split_equivalence="exact", tuple_namespace="collision-free",
                    missing_new_run_endpoint=missing_run["endpoint_frames"], old_run_fallback="not used", gap25_not26="passed"),
                fixed_pair_helper=dict(original_vote=fixed_good["vote"], exchange_vote=fixed_bad["vote"],
                    same_color_vote=0, static_vote=0, shifted_control_vote=probe["shifted"]["vote"],
                    RGB_dates=requests, four_temporal_boxes="same original rows; arithmetic mean unchanged",
                    original_feature_values="bitwise unchanged on five original fixtures"),
                limitation="no information for identical appearances/static frames/insufficient dates; motion, illumination, occlusion or local background changes can confound RGB identity similarity")


if __name__ == "__main__":
    print(json.dumps(selfcheck(), indent=2))
