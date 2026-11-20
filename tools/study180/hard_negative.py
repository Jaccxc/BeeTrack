"""Synthetic-only prototype: paired source hard-negative versus uniform fine-tuning.

Pass model_module=strip_model, or put the existing strip_model module on PYTHONPATH.
No footage, GT files or checkpoints are opened unless explicitly supplied by a caller.
"""
import importlib

import numpy as np
import torch
from torch import nn


def _pool(rng, negative):
    n = len(negative)
    if n >= 8192:
        return negative[rng.choice(n, 8192, replace=False)]
    # Cover every position before repeating when fewer than 8192 exist.
    return negative[np.concatenate([rng.permutation(n) for _ in range((8192+n-1)//n)])[:8192]]


def train_model(bank, events, start, end, base_checkpoint, mode="hard", seed=180,
                model_module=None, input_fn=None):
    """Return (model, audit). Both modes start from the identical trained base.

    Coordinates/candidate pools/positive draws are seed-identical across modes.
    Both arms select 2048 entries from the same 8192-entry candidate pool:
    hard ranks max directional probability; uniform uses a shuffled quarter.
    True-negative eligibility excludes +/-12 frames and circular distance <=2
    around source positives, keeping valid timing/neighbors out of mining.
    Epoch mining is frozen before any updates in that epoch; only source labels
    within the supplied inclusive interval are used. input_fn can instrument all
    bank reads; its signature is the existing strip_model._inputs signature.
    base_checkpoint may also be
    a state_dict, allowing synthetic checks without disk files.
    """
    module = model_module or importlib.import_module("strip_model")
    inputs = input_fn or module._inputs
    module._validate(bank, start, end, "T")
    if mode not in ("hard", "uniform"):
        raise ValueError("mode must be hard or uniform")
    torch.set_num_threads(2)
    torch.manual_seed(seed)
    labels = np.zeros((end-start+1, 32, 2), np.float32)
    for f, c, d in events:
        if any(int(v) != v for v in (f, c, d)) or f < 1 or not 0 <= c < 32 or d not in (0, 1):
            raise ValueError("Invalid source event")
        if start <= f <= end:
            labels[int(f)-start, int(c), int(d)] = 1
    positive = np.argwhere(labels.any(axis=2))
    forbidden = labels.any(axis=2).copy()
    for f, c in positive:
        forbidden[max(0, f-12):min(len(labels), f+13), [(c+d)%32 for d in range(-2, 3)]] = True
    negative = np.argwhere(~forbidden)
    if not len(positive) or not len(negative):
        raise ValueError("Need positive and true-negative positions")
    state = (base_checkpoint if isinstance(base_checkpoint, dict) else
             torch.load(base_checkpoint, map_location="cpu", weights_only=True))
    model = module._network().cpu()
    model.load_state_dict(state, strict=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=.0003)
    criterion = nn.BCEWithLogitsLoss()
    pool_rng, positive_rng, negative_rng = [np.random.default_rng(s) for s in (seed, seed+1, seed+2)]
    uniform_rng = np.random.default_rng(seed+3)
    audit = []
    for epoch in range(4):
        candidates = _pool(pool_rng, negative)
        hardness = np.empty(len(candidates), np.float32)
        model.eval()
        with torch.inference_mode():
            for first in range(0, len(candidates), 256):
                f, c = candidates[first:first+256].T
                x = inputs(bank, f+start-1, c, start, end, "T")
                hardness[first:first+len(f)] = model(x).sigmoid().amax(dim=1).numpy()
        top = np.argsort(-hardness, kind="stable")[:len(candidates)//4]
        uniform = uniform_rng.permutation(len(candidates))[:len(top)]
        selected = candidates[top if mode == "hard" else uniform]
        model.train()
        losses = []
        for _ in range(16):
            pos = positive[positive_rng.integers(len(positive), size=32)]
            neg = selected[negative_rng.integers(len(selected), size=32)]
            f, c = np.concatenate((pos, neg)).T
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(inputs(bank, f+start-1, c, start, end, "T")),
                             torch.from_numpy(labels[f, c]))
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        audit.append(dict(epoch=epoch+1, candidate_count=len(candidates),
                          unique_candidates=len(np.unique(candidates, axis=0)),
                          eligible_count=len(selected),
                          selected_unique_positions=len(np.unique(selected, axis=0)),
                          mean_loss=float(np.mean(losses))))
    return model, audit


def synthetic_check(model_module):
    """Actually fine-tune both paired arms; assert finite loss and learned gradients."""
    rng = np.random.default_rng(180)
    bank = rng.integers(90, 110, (20, 32, 21, 21), dtype=np.uint8)
    bank[:, 4, 3:10, 3:10] = 240
    events = [(f, 4, 0) for f in range(3, 19)]
    torch.manual_seed(179)
    base = model_module._network().state_dict()
    reports = {}
    reads = {}
    for mode in ("hard", "uniform"):
        reads[mode] = []
        def checked_inputs(bank, frames, cells, start, end, arm):
            assert start == 3 and end == 18 and np.all((frames >= 2) & (frames <= 17))
            reads[mode].append((frames.copy(), cells.copy()))
            value = model_module._inputs(bank, frames, cells, start, end, arm)
            altered = bank.copy()
            altered[:2] = 255
            altered[18:] = 0
            assert torch.equal(value, model_module._inputs(altered, frames, cells, start, end, arm))
            return value
        model, audit = train_model(bank, events, 3, 18, base, mode,
                                   model_module=model_module, input_fn=checked_inputs)
        assert all(np.isfinite(e["mean_loss"]) for e in audit)
        assert any(not torch.equal(base[k], v) for k, v in model.state_dict().items())
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
        assert audit[-1]["mean_loss"] < audit[0]["mean_loss"]
        reports[mode] = audit
    # Each epoch: 32 mining batches of 256, then 16 training batches of 64.
    assert len(reads["hard"]) == len(reads["uniform"]) == 4*48
    for epoch in range(4):
        for batch in range(48):
            hard, control = reads["hard"][epoch*48+batch], reads["uniform"][epoch*48+batch]
            if batch < 32:
                assert all(np.array_equal(a, b) for a, b in zip(hard, control))
                for cells in (hard[1], control[1]):
                    assert np.all(np.minimum(abs(cells-4), 32-abs(cells-4)) > 2)
            else:
                assert all(np.array_equal(a[:32], b[:32]) for a, b in zip(hard, control))
                for cells in (hard[1][32:], control[1][32:]):
                    assert np.all(np.minimum(abs(cells-4), 32-abs(cells-4)) > 2)
    assert all(e["eligible_count"] == 2048 for rows in reports.values() for e in rows)
    return reports
