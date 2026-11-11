"""Does appearance identify a bee? Probe on GT boxes, before touching the tracker.
run: uv run python -m beetrack.reid_probe [--steps 3000]
Task (the one a tracker faces after a gap): given a bee's crop at frame t, pick the same bee among the GT bees
at frame t+gap. Rank-1 accuracy of: appearance (ImageNet ResNet-18, and the same net fine-tuned on the train split
with a contrastive loss on exactly this task), nearest position, and chance. Evaluated on the val split only.
writes runs/eval/reid_probe.json
"""
import argparse
import json

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision

from .common import RUNS, frame_path, load_gt, split_specs

SIZE, GAPS, NEAR = 64, (5, 10, 25, 50), 4.0   # crop side px; frame gaps; "near" candidates = within NEAR box heights
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def boxes(src, f):
    """GT rows of one frame, one row per ID (a few train sequences repeat an ID inside a frame)."""
    g = load_gt(src)
    g = g[g[:, 0] == f]
    return g[np.unique(g[:, 1], return_index=True)[1]]


def crops(src, f, rows):
    im = cv2.imread(str(frame_path(src, int(f))))
    h, w = im.shape[:2]
    out = []
    for _, _, x, y, bw, bh in rows:
        x1, y1 = min(max(0, int(x)), w - 2), min(max(0, int(y)), h - 2)   # boxes partly outside the frame stay non-empty
        x2, y2 = max(x1 + 1, min(w, int(x + bw) + 1)), max(y1 + 1, min(h, int(y + bh) + 1))
        c = cv2.resize(im[y1:y2, x1:x2], (SIZE, SIZE))[:, :, ::-1]
        out.append((c / 255.0 - MEAN) / STD)
    return torch.tensor(np.array(out), dtype=torch.float32).permute(0, 3, 1, 2)


def pair(rng, specs, gap, max_n=48):
    """Crops of the bees present in both frame t and t+gap of a random sequence, plus their boxes at both frames."""
    while True:
        _, src, s0, s1 = specs[rng.integers(len(specs))]
        if s1 - s0 <= gap:
            continue
        t = int(rng.integers(s0, s1 - gap + 1))
        a, b = boxes(src, t), boxes(src, t + gap)
        ids = np.intersect1d(a[:, 1], b[:, 1])
        if len(ids) < 2:
            continue
        ids = rng.permutation(ids)[:max_n]
        a, b_same = a[np.searchsorted(a[:, 1], ids)], b[np.searchsorted(b[:, 1], ids)]
        return crops(src, t, a), crops(src, t + gap, b_same), a, b_same


def embed(net, x, dev):
    return F.normalize(net(x.to(dev)), dim=1)


@torch.no_grad()
def evaluate(net, specs, dev, n_pairs, seed=1):
    net.eval()
    res = {}
    for gap in GAPS:
        rng = np.random.default_rng(seed)   # same frame pairs for every model
        hit = dict(app_all=0, pos_all=0, chance_all=0.0, app_near=0, pos_near=0, both_near=0, chance_near=0.0)
        n_all = n_near = 0
        for _ in range(n_pairs):
            xa, xb, a, b = pair(rng, specs, gap)
            sim = (embed(net, xa, dev) @ embed(net, xb, dev).T).cpu().numpy()
            ca, cb = a[:, 2:4] + a[:, 4:6] / 2, b[:, 2:4] + b[:, 4:6] / 2
            dist = np.linalg.norm(ca[:, None] - cb[None], axis=2)          # query position at t vs candidates at t+gap
            near = dist <= NEAR * a[:, 5:6]
            idx = np.arange(len(a))
            hit["app_all"] += int((sim.argmax(1) == idx).sum())
            hit["pos_all"] += int((dist.argmin(1) == idx).sum())
            hit["chance_all"] += 1.0
            n_all += len(a)
            # near: queries whose true match is still nearby and that have at least one other bee nearby
            ok = near[idx, idx] & (near.sum(1) >= 2)
            hit["app_near"] += int((np.where(near, sim, -9).argmax(1) == idx)[ok].sum())
            hit["pos_near"] += int((dist.argmin(1) == idx)[ok].sum())
            both = sim - dist / (NEAR * a[:, 5:6])                         # appearance and position, equal weight
            hit["both_near"] += int((np.where(near, both, -9).argmax(1) == idx)[ok].sum())
            hit["chance_near"] += float((1.0 / near.sum(1).clip(1))[ok].sum())
            n_near += int(ok.sum())
        res[gap] = dict(n_all=n_all, n_near=n_near,
                        **{k: round(100 * v / (n_pairs if k == "chance_all" else n_all if k.endswith("all") else max(n_near, 1)), 1)
                           for k, v in hit.items()})
        res[gap]["chance_all"] = round(100 * n_pairs / n_all, 1)          # mean of 1/n over queries = pairs / queries
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3000, help="fine-tuning steps (one frame pair each); 0 = pretrained only")
    ap.add_argument("--pairs", type=int, default=300, help="val frame pairs per gap")
    ap.add_argument("--weights", default="", help="evaluate saved fine-tuned weights (runs/reid_r18.pt) instead of training")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    net = torchvision.models.resnet18(weights="IMAGENET1K_V1")
    net.fc = torch.nn.Identity()
    net.to(dev)
    val, train = split_specs("val"), split_specs("train")
    out = {"imagenet": evaluate(net, val, dev, a.pairs)}
    if a.weights:
        net.load_state_dict(torch.load(a.weights))
        out["finetuned"] = evaluate(net, val, dev, a.pairs)
    elif a.steps:
        rng = np.random.default_rng(0)
        opt = torch.optim.AdamW(net.parameters(), 1e-4, weight_decay=1e-4)
        net.train()
        for step in range(1, a.steps + 1):
            xa, xb, _, _ = pair(rng, train, int(rng.integers(3, 61)))
            if rng.random() < 0.5:
                xa, xb = xa.flip(3), xb.flip(3)
            logits = embed(net, xa, dev) @ embed(net, xb, dev).T / 0.1   # negatives = the other bees of the same frame
            y = torch.arange(len(xa), device=dev)
            loss = (F.cross_entropy(logits, y) + F.cross_entropy(logits.T, y)) / 2
            opt.zero_grad()
            loss.backward()
            opt.step()
            if step % 500 == 0:
                print(f"step {step} loss {loss.item():.3f}", flush=True)
        out["finetuned"] = evaluate(net, val, dev, a.pairs)
        torch.save(net.state_dict(), RUNS / "reid_r18.pt")
    (RUNS / "eval").mkdir(parents=True, exist_ok=True)
    (RUNS / "eval" / "reid_probe.json").write_text(json.dumps(out, indent=1))
    print("rank-1 % on val   (all = every bee in the frame is a candidate; near = only bees within 4 box heights)")
    print(f"{'model':10s} {'gap':>4s} | {'app_all':>7s} {'pos_all':>7s} {'chance':>6s} | {'app_near':>8s} {'pos_near':>8s} {'both':>6s} {'chance':>6s} {'n_near':>6s}")
    for m, r in out.items():
        for gap, v in r.items():
            print(f"{m:10s} {gap:4d} | {v['app_all']:7.1f} {v['pos_all']:7.1f} {v['chance_all']:6.1f} | "
                  f"{v['app_near']:8.1f} {v['pos_near']:8.1f} {v['both_near']:6.1f} {v['chance_near']:6.1f} {v['n_near']:6d}")


if __name__ == "__main__":
    main()
