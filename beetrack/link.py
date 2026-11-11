"""TrackletLink: offline second pass that joins the tracklets of a conservative first pass into whole tracks.
A tracker that must decide in every frame either drops a bee that hops, dashes or takes off (missed count) or keeps
its ID alive on a neighbour (false count). Here the first pass (track.PASS1) only joins boxes that overlap almost
perfectly and keeps every box, and which tracklet end continues as which tracklet start is decided afterwards, for the
whole sequence at once, by a small network that sees the geometry of both sides of the gap. It is fitted on the
tracklets of sequences with ground truth:
  uv run python -m beetrack.link --dets y11s_640 --fit train val      (writes beetrack/link_model.json)
  uv run python -m beetrack.track --tracker link --split test          (first pass + linking)
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

MODEL = Path(__file__).with_name("link_model.json")
MAX_GAP, MAX_STEP, CAP = 25, 3.0, 6.0   # candidate links: <= 25 frames (the counting protocol's own limit), <= 3 body lengths per frame, <= 6 in all
F0, F1, X0, Y0, X1, Y1, W0, H0, W1, H1, VX0, VY0, VX1, VY1, N, SC, SC0, SC1 = range(18)   # columns of summarize()


def summarize(rows, k=3):
    """One line per tracklet of `rows` (frame, id, x, y, w, h, score; sorted by id, then frame): first and last frame,
    first and last centre, first and last box size, velocity per frame at both ends (over up to k boxes; nan for a
    single box), number of boxes, mean, first and last score."""
    ids, start = np.unique(rows[:, 1], return_index=True)
    c = rows[:, 2:4] + rows[:, 4:6] / 2
    T = np.full((len(ids), 18), np.nan)
    for t, (a, b) in enumerate(zip(start, [*start[1:], len(rows)])):
        m = min(b - a, k) - 1
        T[t, :10] = rows[a, 0], rows[b - 1, 0], *c[a], *c[b - 1], *rows[a, 4:6], *rows[b - 1, 4:6]
        if m:
            T[t, VX0:VY0 + 1] = (c[a + m] - c[a]) / (rows[a + m, 0] - rows[a, 0])
            T[t, VX1:VY1 + 1] = (c[b - 1] - c[b - 1 - m]) / (rows[b - 1, 0] - rows[b - 1 - m, 0])
        T[t, N:] = b - a, rows[a:b, 6].mean(), rows[a, 6], rows[b - 1, 6]
    return ids, T


def candidates(T):
    """Pairs (i, j): tracklet j starts 1..MAX_GAP frames after tracklet i ends, close enough to be the same bee."""
    order = np.argsort(T[:, F0], kind="stable")
    f0 = T[order, F0]
    I, J = [], []
    for i in range(len(T)):
        js = order[np.searchsorted(f0, T[i, F1] + 1):np.searchsorted(f0, T[i, F1] + MAX_GAP, side="right")]
        d = np.hypot(T[js, X0] - T[i, X1], T[js, Y0] - T[i, Y1]) / (T[i, W1] * T[i, H1] * T[js, W0] * T[js, H0]) ** 0.25
        js = js[d <= np.minimum(MAX_STEP * (T[js, F0] - T[i, F1]) + 1, CAP)]
        I += [i] * len(js)
        J += list(js)
    return np.array(I, int), np.array(J, int)


def _extend(key, at, n, keep):
    """For every pair p: the pairs q of `keep` whose `key` tracklet is tracklet at[p]. Returns index arrays p, q."""
    qi = np.flatnonzero(keep)
    qi = qi[np.argsort(key[qi], kind="stable")]
    cnt = np.bincount(key[qi], minlength=n)
    off, c = np.r_[0, np.cumsum(cnt)], cnt[at]
    rep = np.repeat(np.arange(len(at)), c)
    return rep, qi[off[at[rep]] + np.arange(len(rep)) - np.repeat(np.cumsum(c) - c, c)]


def features(T, I, J, near=3):
    """What the network sees of a candidate link i -> j, lengths in body lengths (sqrt of the box area):
    gap; straight distance (also per frame); miss of i's velocity extrapolated forward and of j's backward, and the
    smallest of the three; whether each velocity exists; cosine of each velocity with the jump; change of box width and
    height, IoU of the two boxes; both speeds, lengths and scores; collinearity: when the jump is extended at
    constant velocity beyond a one- or two-box tracklet, how far the nearest third tracklet within `near` frames is from
    the extrapolated point (the three-points-on-a-line test of a flying bee); and whether each box is elongated along
    the jump (a bee moves, and blurs, along its body)."""
    a, b = T[I], T[J]
    g = b[:, F0] - a[:, F1]
    s = (a[:, W1] * a[:, H1] * b[:, W0] * b[:, H0]) ** 0.25
    p, q = a[:, [X1, Y1]], b[:, [X0, Y0]]
    norm = lambda v: np.hypot(v[:, 0], v[:, 1])
    d = q - p
    dist = norm(d) / s
    ha, hb = ~np.isnan(a[:, VX1]), ~np.isnan(b[:, VX0])
    va, vb = np.nan_to_num(a[:, [VX1, VY1]]), np.nan_to_num(b[:, [VX0, VY0]])
    fwd = np.where(ha, norm(p + va * g[:, None] - q) / s, dist)
    bwd = np.where(hb, norm(q - vb * g[:, None] - p) / s, dist)
    cos = lambda v: (v * d).sum(1) / (norm(v) * norm(d) + 1e-9)
    lo, hi = np.maximum(p - a[:, [W1, H1]] / 2, q - b[:, [W0, H0]] / 2), np.minimum(p + a[:, [W1, H1]] / 2, q + b[:, [W0, H0]] / 2)
    inter = np.prod(np.clip(hi - lo, 0, None), axis=1)
    iou = inter / (a[:, W1] * a[:, H1] + b[:, W0] * b[:, H0] - inter)
    v, sup = d / g[:, None], np.full((2, len(I)), np.inf)
    # forward: j is short, pairs q = (j -> k) leave j;  backward: i is short, pairs q = (h -> i) arrive at i
    for out, key, at, short, base, sign, far in ((sup[0], I, J, b[:, N] <= 2, b[:, [X1, Y1]], 1, X0), (sup[1], J, I, a[:, N] <= 2, a[:, [X0, Y0]], -1, X1)):
        sel = np.flatnonzero(short)
        rep, qq = _extend(key, at[sel], len(T), g <= near)
        pp = sel[rep]
        third = T[(J if sign > 0 else I)[qq]][:, [far, far + 1]]
        np.minimum.at(out, pp, norm(base[pp] + sign * v[pp] * g[qq][:, None] - third) / s[pp])
    Z = np.c_[np.log(g), dist, dist / g, fwd, bwd, np.minimum(dist, np.minimum(fwd, bwd)), ha, hb, cos(va), cos(vb),
              np.abs(np.log(b[:, W0] / a[:, W1])), np.abs(np.log(b[:, H0] / a[:, H1])), iou, norm(va) / s, norm(vb) / s,
              np.log(a[:, N]), np.log(b[:, N]), a[:, SC], b[:, SC], a[:, SC1], b[:, SC0], sup[0], sup[1]]
    along = (np.abs(d[:, 0]) - np.abs(d[:, 1])) / (norm(d) + 1e-9)     # +1: the jump is horizontal, -1: vertical
    Z = np.c_[Z, np.log(a[:, W1] / a[:, H1]) * along, np.log(b[:, W0] / b[:, H0]) * along]
    return np.clip(Z, -8, 8).astype(np.float32)


def model(m=None):
    """The fitted network (a dict as written by fit(), default link_model.json) as a function features -> probability."""
    m = m or json.loads(MODEL.read_text())

    def prob(Z):
        x = (Z - np.array(m["mu"])) / np.array(m["sd"])
        for k, (W, b) in enumerate(m["layers"]):
            x = x @ np.array(W) + np.array(b)
            if k < len(m["layers"]) - 1:
                x = np.maximum(x, 0)
        return 1 / (1 + np.exp(-x[:, 0]))
    return prob


def link(rows, thr=0.3, prob=None):
    """Join tracklets: most confident link first, every tracklet end and every tracklet start used once.
    rows: frame, id, x, y, w, h, score. Returns the rows with the joined IDs, sorted by id, then frame."""
    rows = np.array(rows, float)
    if len(rows) < 2:
        return rows
    rows = rows[np.lexsort((rows[:, 0], rows[:, 1]))]
    ids, T = summarize(rows)
    I, J = candidates(T)
    if not len(I):
        return rows
    p = (prob or model())(features(T, I, J))
    root, end_used, start_used = np.arange(len(ids)), set(), set()
    links = []
    for k in np.argsort(-p):
        if p[k] < thr:
            break
        if I[k] not in end_used and J[k] not in start_used:
            end_used.add(I[k])
            start_used.add(J[k])
            links.append((T[J[k], F0], I[k], J[k]))
    for _, i, j in sorted(links):          # in time order, so the earlier tracklet's root is already final
        root[j] = root[i]
    rows[:, 1] = ids[root][np.searchsorted(ids, rows[:, 1])]
    return rows[np.lexsort((rows[:, 0], rows[:, 1]))]


# --- fitting ---------------------------------------------------------------------------------------------------------
def under(gt, pr):
    """GT bee under every row of pr (both: frame, id, x, y, w, h, ...): one-to-one at IoU >= 0.5 first; a box left over
    takes the bee it overlaps most (IoU >= 0.3), so a second box on a bee counts as that bee. -1: no bee."""
    from .track import iou_matrix
    xyxy = lambda b: np.c_[b[:, 2:4], b[:, 2:4] + b[:, 4:6]]
    out = np.full(len(pr), -1.0)
    og, op = np.argsort(gt[:, 0], kind="stable"), np.argsort(pr[:, 0], kind="stable")
    fg, sg = np.unique(gt[og, 0], return_index=True)
    G = dict(zip(fg, np.split(og, sg[1:])))
    fp, sp = np.unique(pr[op, 0], return_index=True)
    for f, p in zip(fp, np.split(op, sp[1:])):
        g = G.get(f)
        if g is None:
            continue
        m = iou_matrix(xyxy(gt[g]), xyxy(pr[p]))
        i, j = linear_sum_assignment(-m)
        ok = m[i, j] >= 0.5
        out[p[j[ok]]] = gt[g[i[ok]], 1]
        rest = np.setdiff1d(np.arange(len(p)), j[ok])
        if len(rest):
            best = m[:, rest].argmax(0)
            soft = m[best, rest] >= 0.3
            out[p[rest[soft]]] = gt[g[best[soft]], 1]
    return out


def labelled(pr, gt):
    """Features and labels of one sequence's candidate links. pr: first-pass rows, gt: ground-truth rows (one ID per bee).
    1: the bee under i's last box is the bee under j's first box and j is where that bee's next tracklet starts;
    0: different bees, or one of the two boxes is on no bee; -1 (not used): the same bee but a later tracklet, or no
    annotated bee under either box."""
    pr = pr[np.lexsort((pr[:, 0], pr[:, 1]))]
    u = under(gt, pr)
    ids, T = summarize(pr)
    start = np.unique(pr[:, 1], return_index=True)[1]
    u0, u1 = u[start], u[np.r_[start[1:], len(pr)] - 1]
    I, J = candidates(T)
    nxt = np.full(len(T), np.inf)       # frame at which the bee under a tracklet's end next starts a tracklet
    for b in np.unique(u1[u1 >= 0]):
        fs = np.sort(T[u0 == b, F0])
        i = np.flatnonzero(u1 == b)
        k = np.searchsorted(fs, T[i, F1] + 1)
        nxt[i[k < len(fs)]] = fs[k[k < len(fs)]]
    same = (u1[I] == u0[J]) & (u1[I] >= 0)
    y = np.where(same & (T[J, F0] == nxt[I]), 1, np.where(same | ((u1[I] < 0) & (u0[J] < 0)), -1, 0))
    return features(T, I, J), y


def fit(Z, y, hidden=32, epochs=30, seed=0):
    """Two-hidden-layer network on standardised features, binary cross-entropy. Returns the dict model() reads."""
    import torch
    torch.manual_seed(seed)
    Z, y = Z[y >= 0], y[y >= 0]
    mu, sd = Z.mean(0), Z.std(0) + 1e-6
    x, t = torch.tensor((Z - mu) / sd), torch.tensor(y, dtype=torch.float32)
    net = torch.nn.Sequential(torch.nn.Linear(Z.shape[1], hidden), torch.nn.ReLU(), torch.nn.Linear(hidden, hidden), torch.nn.ReLU(), torch.nn.Linear(hidden, 1))
    opt = torch.optim.Adam(net.parameters(), 1e-3, weight_decay=1e-4)
    for _ in range(epochs):
        perm = torch.randperm(len(x))
        for k in range(0, len(x), 2048):
            b = perm[k:k + 2048]
            loss = torch.nn.functional.binary_cross_entropy_with_logits(net(x[b]).squeeze(1), t[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
    return dict(mu=mu.tolist(), sd=sd.tolist(), layers=[[l.weight.T.tolist(), l.bias.tolist()] for l in net if isinstance(l, torch.nn.Linear)])


def _check():
    """Two bees: one sits, hops two body lengths and sits again; one flies through as three single boxes on a line.
    A stand-in model that only looks at the distance joins the hop; the flight is joined only when collinearity counts."""
    box = lambda f, i, x, y: [f, i, x, y, 20, 20, 0.9]
    rows = [box(f, 1, 100, 100) for f in (1, 2, 3)] + [box(f, 2, 140, 100) for f in (4, 5, 6)]
    rows += [box(1, 3, 300, 300), box(2, 4, 360, 300), box(3, 5, 420, 300), box(2, 6, 300, 360)]
    near = lambda Z: (Z[:, 1] <= 2.5).astype(float)
    line = lambda Z: ((Z[:, 1] <= 2.5) | (np.minimum(Z[:, 21], Z[:, 22]) < 0.5)).astype(float)
    ids = lambda prob: [int(i) for i in link(rows, 0.5, prob)[:, 1]]
    assert ids(near) == [1] * 6 + [3, 4, 5, 6], ids(near)
    assert ids(line) == [1] * 6 + [3] * 3 + [6], ids(line)
    m = dict(mu=[0, 0], sd=[1, 1], layers=[[[[1.0], [-1.0]], [0.0]]])
    assert np.allclose(model(m)(np.array([[2.0, 0.0], [0.0, 2.0]])), [1 / (1 + np.exp(-2)), 1 / (1 + np.exp(2))])


_check()


def main():
    from .common import RUNS, split_specs
    from .evaluate import by_track, load_tracks
    from .track import PASS1, Tracker, run_sequence
    ap = argparse.ArgumentParser()
    ap.add_argument("--dets", default="y11s_640", help="detection cache tag under runs/dets/")
    ap.add_argument("--fit", nargs="+", default=["train", "val"], help="splits whose sequences the model is fitted on")
    ap.add_argument("--skip", nargs="*", default=[], help="sequences left out, by their name in the split (to score them with a model that has not seen them)")
    ap.add_argument("--out", default=str(MODEL))
    a = ap.parse_args()
    Z, y = [], []
    for split in a.fit:
        for ev, src, s0, s1 in split_specs(split):
            if ev in a.skip:
                continue
            rows, _ = run_sequence(np.load(RUNS / "dets" / a.dets / split / f"{ev}.npy"), s1 - s0 + 1, Tracker("ours", **PASS1))
            z, t = labelled(np.array(rows, float), by_track(load_tracks(RUNS / "gt" / split / ev / "gt" / "gt.txt")))
            Z.append(z)
            y.append(t)
    Z, y = np.vstack(Z), np.concatenate(y)
    Path(a.out).write_text(json.dumps(fit(Z, y)))
    print(f"{a.out}: fitted on {int((y == 1).sum())} true and {int((y == 0).sum())} false links ({int((y < 0).sum())} unknown left out)")


if __name__ == "__main__":
    main()
