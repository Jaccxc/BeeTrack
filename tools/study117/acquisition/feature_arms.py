"""Six causal P3 classifier arms, sharing the native reciprocal correspondence search.

Put the registered temporal_features module on sys.path. Returned logit tensors are
read-only views for the caller: identical treatments can share the same tensor.
"""

from time import perf_counter_ns

import torch

from temporal_features import best_pair_indices, descriptors


ARM_NAMES = ("U", "A", "Z", "Q", "Zm", "D")
TIMING_FIELDS = ("shared_pair", "class_A", "class_Z", "class_Zm", "class_D", "logit_Q", "zero_gate", "total")


def compute_arms(current, history, logits, past_logits, cls_head, content, arms=ARM_NAMES):
    """Evaluate the six registered arms, or only A for a single-inference deployment."""
    started = perf_counter_ns()
    arms = tuple(arms)
    if arms not in (ARM_NAMES, ("A",)):
        raise ValueError("Expected all six registered arms or a single aligned arm")
    if current.ndim != 4 or current.shape[0] != 1:
        raise ValueError("Expected native single-frame P3 features")
    h, w = current.shape[-2:]
    if logits.shape != (1, 1, h, w) or content.shape != (h, w) or content.dtype != torch.bool:
        raise ValueError("Expected native one-class logits and a boolean P3 content mask")
    if (history is None) != (past_logits is None):
        raise ValueError("History features and logits must be supplied together")
    outputs = dict.fromkeys(arms, logits)
    empty = torch.empty(0, dtype=torch.long, device=logits.device)
    timing = dict.fromkeys(TIMING_FIELDS, 0.0)
    info = dict(current_ids=empty, donor_ids=empty, z_current_ids=empty, timing_ms=timing)

    def finish():
        timing["total"] = (perf_counter_ns()-started)/1e6
        # The prefix through A includes its gate checks, setup, pairing and tower once.
        info.setdefault("a_path_ms", timing["total"])
        return outputs, info

    if history is None:
        return finish()
    if history.shape != current.shape or past_logits.shape != logits.shape:
        raise ValueError("Native t-2 feature and logit shapes must match frame t")

    now = logits[0, 0].sigmoid()
    prior = past_logits[0, 0].sigmoid()
    weak = content & (now >= .05) & (now < .25)
    strong = content & (prior >= .25)
    if not weak.any() or not strong.any():
        return finish()

    stage = perf_counter_ns()
    valid, py, px = best_pair_indices(current, history, content=content)
    iy, ix = torch.where(valid & weak & strong[py, px])
    jy, jx = py[iy, ix], px[iy, ix]
    current_ids = iy*w + ix
    donor_ids = jy*w + jx
    info.update(current_ids=current_ids, donor_ids=donor_ids)
    timing["shared_pair"] = (perf_counter_ns()-stage)/1e6

    # Two arms have identical outputs when they change exactly the same cells with
    # the same native donor cells. This includes single-donor cyclic permutations.
    calculated = []

    def feature_update(arm, y, x, source_y, source_x):
        if not len(y):
            return logits
        targets, donors = y*w + x, source_y*w + source_x
        for previous_targets, previous_donors, result in calculated:
            if torch.equal(targets, previous_targets) and torch.equal(donors, previous_donors):
                return result
        stage = perf_counter_ns()
        features = current.clone()
        features[0, :, y, x] = torch.lerp(current[0, :, y, x], history[0, :, source_y, source_x], .5)
        proposal = cls_head(features)[0, 0, y, x]
        result = logits.clone()
        result[0, 0, y, x] = torch.maximum(logits[0, 0, y, x], proposal)
        calculated.append((targets, donors, result))
        timing["class_" + arm] = (perf_counter_ns()-stage)/1e6
        return result

    outputs["A"] = feature_update("A", iy, ix, jy, jx)
    if arms == ("A",):
        return finish()
    info["a_path_ms"] = (perf_counter_ns()-started)/1e6
    if len(iy):
        outputs["Zm"] = feature_update("Zm", iy, ix, iy, ix)
        outputs["D"] = feature_update("D", iy, ix, jy.roll(1), jx.roll(1))
        stage = perf_counter_ns()
        mixed = torch.lerp(logits[0, 0, iy, ix], past_logits[0, 0, jy, jx], .5)
        q = logits.clone()
        q[0, 0, iy, ix] = torch.maximum(logits[0, 0, iy, ix], mixed)
        outputs["Q"] = q
        timing["logit_Q"] = (perf_counter_ns()-stage)/1e6

    stage = perf_counter_ns()
    if (weak & strong).any():
        c, cv = descriptors(current)
        p, pv = descriptors(history)
        zy, zx = torch.where(weak & strong & cv[0] & pv[0] & ((c*p)[0].sum(0) >= .8))
        info["z_current_ids"] = zy*w + zx
        timing["zero_gate"] = (perf_counter_ns()-stage)/1e6
        outputs["Z"] = feature_update("Z", zy, zx, zy, zx)
    return finish()


def self_check():
    from temporal_features import classify_step

    torch.manual_seed(117)
    torch.set_num_threads(1)
    head = torch.nn.Sequential(torch.nn.Conv2d(8, 6, 3, padding=1), torch.nn.SiLU(),
                               torch.nn.Conv2d(6, 1, 1)).eval()
    modes = dict(A="aligned", Z="zero_motion", Q="past_logit", Zm="same_cell", D="permute")
    scenarios = []
    # Place independently-identifiable moved and stationary anchors on a constant map.
    zero = torch.zeros(1, 8, 7, 9)
    old = zero.clone()
    new = zero.clone()
    for y, x, shift in ((1, 2, 1), (3, 3, 0), (5, 5, -1)):
        signature = torch.randn(8)
        old[0, :, y, x] = signature
        new[0, :, y, x+shift] = .4*signature
    l = torch.full((1,1,7,9), -2.)
    p = torch.full_like(l, 2.)
    keep = torch.ones((7,9), dtype=torch.bool)
    scenarios.append((new, old, l, p, keep))
    partial = keep.clone()
    partial[1, 2] = False
    scenarios.append((new, old, l, p, partial))
    scenarios.append((new, old, l, p, ~keep))
    scenarios.append((new, old, l, l, keep))
    # A tied donor fails the reciprocal margin while the static control remains eligible.
    tied = new.clone()
    tied[0, :, 3, 4] = tied[0, :, 3, 3]
    scenarios.append((tied, old, l, p, keep))
    for _ in range(8):
        old = torch.randn(1,8,7,9)
        new = old.roll(1, dims=3) + .2*torch.randn_like(old)
        l = torch.randn(1,1,7,9) - 2
        p = torch.randn_like(l) - 1
        scenarios.append((new, old, l, p, torch.rand_like(keep, dtype=torch.float)>.15))
    with torch.inference_mode():
        cold, ids = compute_arms(new, None, l, None, head, keep)
        assert all(torch.equal(v, l) for v in cold.values())
        assert all(len(ids[k]) == 0 for k in ("current_ids", "donor_ids", "z_current_ids"))
        assert all(ids["timing_ms"][k] == 0 for k in TIMING_FIELDS if k != "total")
        for index, (new, old, l, p, keep) in enumerate(scenarios):
            orig, orig_past = new.clone(), old.clone()
            result, info = compute_arms(new, old, l, p, head, keep)
            assert tuple(result) == ARM_NAMES and torch.equal(l, result["U"])
            single, single_info = compute_arms(new, old, l, p, head, keep, arms=("A",))
            assert list(single) == ["A"] and torch.equal(single["A"], result["A"])
            assert all(torch.equal(info[k], single_info[k]) for k in ("current_ids", "donor_ids"))
            assert 0 <= info["a_path_ms"] <= info["timing_ms"]["total"]
            assert single_info["a_path_ms"] == single_info["timing_ms"]["total"]
            assert all(single_info["timing_ms"][k] == 0 for k in ("class_Z","class_Zm","class_D","logit_Q","zero_gate"))
            assert sum(info["timing_ms"][k] for k in TIMING_FIELDS[:-1]) <= info["timing_ms"]["total"]
            for arm, mode in modes.items():
                ref, mask = classify_step(new, old, l, p, head, mode, content=keep)
                assert torch.equal(result[arm], ref), (index, mode)
                expected = torch.where(mask.flatten())[0]
                anchor = info["z_current_ids" if arm == "Z" else "current_ids"]
                assert torch.equal(anchor, expected), (index, mode)
            assert torch.equal(orig, new) and torch.equal(orig_past, old)
            if index == 0:
                assert info["current_ids"].tolist() == [12,30,49]
                assert info["donor_ids"].tolist() == [11,30,50]
                assert info["z_current_ids"].tolist() == [30]
            if index == 4:
                assert 30 not in info["current_ids"] and 30 in info["z_current_ids"]
        x, p, l, prior, _ = scenarios[0]
        static = torch.zeros_like(keep)
        static[3,3] = True
        calls = []
        def counted(features):
            calls.append(1)
            return head(features)
        _, info = compute_arms(x, p, l, prior, counted, static)
        times = info["timing_ms"]
        assert len(calls) == 1 and times["class_A"] > 0 and times["logit_Q"] > 0 and times["zero_gate"] > 0
        assert all(times[k] == 0 for k in ("class_Z", "class_Zm", "class_D"))
    print(f"Six arms bit-identical to prototype in {len(scenarios)} synthetic scenes; cold state and donor IDs verified.")


if __name__ == "__main__":
    self_check()
