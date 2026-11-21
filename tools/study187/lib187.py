"""Section 187 shared code: §186 C-arm rebuild, GT-free 2->1->2 extraction, scene rewrites, events, windows, bound.

Run every script from final_project (see README.md). Nothing in this module reads ground truth.
"""
import hashlib
import json
import os
from pathlib import Path
import sys

for _name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(_name, '1')
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
ROOT = Path.cwd()
assert (ROOT / 'beetrack' / 'p0.py').is_file(), 'Run from the final_project directory.'
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools' / 'study103'))
import numpy as np
from beetrack import p0 as P
from beetrack.link import link
from beetrack.track import drop_short
from swap_score import counts_from_events

OUT = Path('D:/beetrack_runs/t187')
SEALED = Path('D:/beetrack_runs/endpoints186')
NF = 3950
CELLS = [(h, k) for h in (33, 35) for k in (0, 1, 2)]
MAX_MID = 12      # middle frames with one box
NEAR = 2.0        # body lengths: every link inside a local component
SEP = 0.5         # two boxes are separated when their intersection is below half of the smaller box
SIDE = 2          # consecutive two-box frames needed before and after


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=1) + '\n', encoding='utf-8')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


# --- base -------------------------------------------------------------------------------------------------------------
def build(h, k):
    """Linked stage-1 rows of the §178 C arm before drop_short, in p0's order, and each row's stage-1 piece id."""
    D = np.load(P.RUNS / 'dets' / f'pl{h}_s{k}' / 'dev' / f'BEE24-{h}-head.npy')
    R1 = P.stage1(D[D[:, 5] >= 0.4], NF)
    rows = np.asarray(sorted(link(R1.tolist(), P.THR, P.link_model('m_dev_strict')).tolist()), float)
    order = lambda a: np.lexsort((a[:, 5], a[:, 4], a[:, 3], a[:, 2], a[:, 0]))
    i, j = order(R1), order(rows)
    box = [0, 2, 3, 4, 5, 6]
    assert np.array_equal(R1[i][:, box], rows[j][:, box]), 'link changed a box'
    assert len(np.unique(rows[:, [0, 2, 3, 4, 5]], axis=0)) == len(rows), 'two identical boxes in one frame'
    piece = np.empty(len(rows))
    piece[j] = R1[i, 1]
    return rows, piece


def final(rows):
    """drop_short as in p0: tracks with fewer than 3 boxes go."""
    return np.asarray(drop_short(rows[:, :7].tolist(), P.MIN_LEN, 0.0), float).reshape(-1, 7)


def events(rows, poly):
    """Official whole-track events (frame, +1 in / -1 out) of labelled rows after drop_short."""
    f = final(rows)
    return P.events(f, poly).astype(int).reshape(-1, 2) if len(f) else np.zeros((0, 2), int)


def counts(E):
    """(30 offsets x windows, 2) in / out counts, offsets stacked as in §186."""
    return counts_from_events(np.asarray(E, int).reshape(-1, 2), NF)


def W(pc, gc, n):
    """30-offset window error in percent of the n GT events (same as §186)."""
    return float(100 * np.abs(pc - gc).sum() / (30 * n))


def lower_bound(pc, gc, n, ranges):
    """Optimistic W: each group i may move every (offset, window, direction) cell by any integer in its own
    [min, max] change; the GT count's distance to the summed interval cannot be beaten by any joint choice."""
    lo = pc + sum((r[0] for r in ranges), np.zeros_like(pc))
    hi = pc + sum((r[1] for r in ranges), np.zeros_like(pc))
    return float(100 * np.maximum(0, np.maximum(lo - gc, gc - hi)).sum() / (30 * n))


# --- extraction (no GT) -----------------------------------------------------------------------------------------------
class Boxes:
    def __init__(self, rows):
        self.rows = rows
        self.c = rows[:, 2:4] + rows[:, 4:6] / 2
        self.s = np.sqrt(rows[:, 4] * rows[:, 5])
        f = rows[:, 0].astype(int)
        o = np.argsort(f, kind='stable')
        cut = np.searchsorted(f[o], np.arange(NF + 3))
        self.F = [o[cut[t]:cut[t + 1]] for t in range(NF + 2)]   # F[t]: row indices of frame t

    def frame(self, t):
        return self.F[t] if 1 <= t <= NF else self.F[0][:0]

    def near(self, t, anchors, R):
        """Rows of frame t whose centre is within R of any anchor row's centre."""
        idx = self.frame(t)
        if not len(idx) or not len(anchors):
            return []
        d = np.hypot(*(self.c[idx][:, None, :] - self.c[list(anchors)][None, :, :]).transpose(2, 0, 1))
        return [int(i) for i in idx[(d <= R).any(1)]]

    def separated(self, i, j):
        r = self.rows
        lo = np.maximum(r[i, 2:4], r[j, 2:4])
        hi = np.minimum(r[i, 2:4] + r[i, 4:6], r[j, 2:4] + r[j, 4:6])
        inter = np.prod(np.clip(hi - lo, 0, None))
        return inter < SEP * min(r[i, 4] * r[i, 5], r[j, 4] * r[j, 5])

    def dist(self, i, j):
        return float(np.hypot(*(self.c[i] - self.c[j])))

    def pair_step(self, pair, t, R):
        """The two separated boxes of frame t matched one to one to `pair` (min total distance, each <= R), else a reason."""
        N = self.near(t, pair, R)
        if len(N) < 2:
            return 'short'
        if len(N) > 2:
            return 'third'
        x, y = N
        if self.dist(pair[0], x) + self.dist(pair[1], y) > self.dist(pair[0], y) + self.dist(pair[1], x):
            x, y = y, x
        if max(self.dist(pair[0], x), self.dist(pair[1], y)) > R or not self.separated(x, y):
            return 'short'
        return [x, y]


def trace(B, t0, m, L):
    """One 2->1->2 attempt whose first one-box frame is t0 with box m, at body length L.
    Returns (status, component {frame: rows} or None, separated rows seen so far)."""
    R = NEAR * L
    prev = B.near(t0 - 1, [m], R)
    if len(prev) < 2 or B.near(t0, prev, R) != [m]:
        return 'none', None, []                  # fewer than two boxes before, or one goes on as another box: no merge
    if len(prev) > 2:
        return 'group', None, []                 # three or more boxes become one
    a1, b1 = prev
    if not B.separated(a1, b1):
        return 'none', None, []
    sep = [a1, b1]
    before = B.pair_step([a1, b1], t0 - 1 - (SIDE - 1), R) if t0 - SIDE >= 1 else 'short'
    if isinstance(before, str):
        return ('before_' + before) if before == 'short' else 'third', None, sep
    sep += before
    mid = [m]
    t = t0
    while True:
        N = B.near(t + 1, [mid[-1]], R)
        if len(N) == 1:
            if len(mid) == MAX_MID:
                return 'timeout', None, sep
            mid.append(N[0])
            t += 1
            continue
        if len(N) == 0:
            return 'merge_only', None, sep        # the one box ends: one side only
        if len(N) > 2:
            return 'group', None, sep
        if not B.separated(*N):
            return 'unseparated_split', None, sep
        break
    t1 = t
    c1, d1 = N
    sep += [c1, d1]
    if t1 + SIDE + 1 > NF:
        return 'after_short', None, sep
    after = B.pair_step([c1, d1], t1 + 2, R)
    if isinstance(after, str):
        return ('after_' + after) if after == 'short' else 'third', None, sep
    sep += after
    comp = {t0 - 2: before, t0 - 1: [a1, b1], t1 + 1: [c1, d1], t1 + 2: after}
    comp.update({t0 + i: [r] for i, r in enumerate(mid)})
    for t in range(t0 - 2, t1 + 3):                # a third box near the component in its own or a next frame
        anchors = comp[t] + comp.get(t - 1, []) + comp.get(t + 1, [])
        if set(B.near(t, anchors, R)) != set(comp[t]):
            return 'third', None, sep
    return 'ok', comp, sep


def candidate(B, t0, m, L0):
    """trace() iterated until the body length (median sqrt area of the separated boxes it found) is a fixed point."""
    L, seen = L0, None
    for _ in range(8):
        status, comp, sep = trace(B, t0, m, L)
        if not sep:
            return status, comp, L
        key = tuple(sorted(sep))
        if key == seen:
            return status, comp, L
        seen, L = key, float(np.median(B.s[sep]))
    return 'L_unstable', None, L


def extract(rows, piece):
    """All 2->1->2 candidates of one cell plus counts of the patterns that are not covered."""
    B = Boxes(rows)
    L0 = float(np.median(B.s))
    tally, found = {}, []
    for t0 in range(2, NF + 1):
        for m in B.frame(t0):
            status, comp, L = candidate(B, t0, int(m), L0)
            if status == 'none':
                continue
            tally[status] = tally.get(status, 0) + 1
            if status == 'ok':
                found.append(dict(t0=t0, t1=max(t for t, v in comp.items() if len(v) == 1), L=L,
                                  comp={int(t): [int(r) for r in v] for t, v in sorted(comp.items())}))
    # a stage-1 piece in two candidates: both abstain
    owner = {}
    for n, c in enumerate(found):
        for p in {piece[r] for v in c['comp'].values() for r in v}:
            owner.setdefault(p, set()).add(n)
    shared = set().union(*[s for s in owner.values() if len(s) > 1]) if owner else set()
    for n, c in enumerate(found):
        c['status'] = 'reuse' if n in shared else 'ok'
    tally['reuse'] = len(shared)
    tally['ok'] = len(found) - len(shared)
    tally.update(uncovered(B, L0))
    return found, tally, L0


def uncovered(B, L0):
    """Counted, not covered: 2->0->2 (both boxes vanish 1-12 frames, two come back) and one-box splits with no
    two-box side before them (split_only). Radius 2 x the video's median body length."""
    R = NEAR * L0
    n202 = split_only = 0
    for t in range(2, NF):
        idx = B.frame(t)
        d = np.hypot(*(B.c[idx][:, None, :] - B.c[idx][None, :, :]).transpose(2, 0, 1))
        for u, v in zip(*np.nonzero(np.triu(d <= R, 1))):
            i, j = int(idx[u]), int(idx[v])
            if B.separated(i, j):
                pair = [i, j]
                # 2->0->2: an isolated pair seen in t-1 and t, nothing near in t+1, two boxes again within 12 frames
                if B.near(t, pair, R) == sorted(pair) and isinstance(B.pair_step(pair, t - 1, R), list) \
                        and not B.near(t + 1, pair, R):
                    for g in range(t + 2, min(t + MAX_MID + 2, NF)):
                        N = B.near(g, pair, R)
                        if N:
                            n202 += len(N) == 2 and B.separated(*N) and isinstance(B.pair_step(N, g + 1, R), list)
                            break
                # split_only: one box m in t-1 near both, nothing else near them in t-1, m has no box before it
                M = B.near(t - 1, pair, R)
                if len(M) == 1 and B.near(t, M, R) == sorted(pair) and isinstance(B.pair_step(pair, t + 1, R), list):
                    cur = M[0]
                    for g in range(t - 2, max(t - 2 - MAX_MID, 0), -1):
                        N = B.near(g, [cur], R)
                        if len(N) != 1:
                            split_only += not N
                            break
                        cur = N[0]
    return dict(two_zero_two=n202, split_only=split_only)


# --- scenes and rewrites (rows: frame, id, x, y, w, h, score, uid; uid = row index in the frozen base rows) ---------
def labelled(rows):
    return np.c_[rows[:, :7], np.arange(len(rows), dtype=float)]


def scene(c, fresh):
    """Frozen candidate -> scene: uids of a, b (frames t0-2, t0-1), c, d (t1+1, t1+2), the middle boxes, and two
    unused track ids for the parts of a's and b's tracks cut off after the scene."""
    comp = {int(t): v for t, v in c['comp'].items()}
    t0, t1 = c['t0'], c['t1']
    return dict(cid=c['cid'], t0=t0, t1=t1, fresh=fresh,
                a=[comp[t0 - 2][0], comp[t0 - 1][0]], b=[comp[t0 - 2][1], comp[t0 - 1][1]],
                c=[comp[t1 + 1][0], comp[t1 + 2][0]], d=[comp[t1 + 1][1], comp[t1 + 2][1]],
                mid=[comp[t][0] for t in range(t0, t1 + 1)])


def options(sc):
    """None = the original, then every rewrite: pairing 0 (a-c, b-d) or 1 (a-d, b-c) times the middle: 'keep' each box
    with the chain its track joins, give all to a or to b, 'drop' them, or 'gt' (the two known bees' GT boxes)."""
    mids = ('keep', 'a', 'b', 'drop') + (('gt',) if sc.get('gt_mid') else ())
    return [None] + [(p, m) for p in (0, 1) for m in mids]


def touched(S, sc):
    """Track ids a scene's rewrites read or write."""
    uid = S[:, 7]
    return {float(S[uid == u, 1][0]) for u in [sc['a'][1], sc['b'][1], sc['c'][0], sc['d'][0], *sc['mid']]}


def apply(S, sc, opt):
    """Rows S after one scene's option; None when a track would get two boxes in one frame (not legal)."""
    if opt is None:
        return S
    pair, mid = opt
    uid, lab, f = S[:, 7], S[:, 1], S[:, 0]
    at = lambda u: lab[np.flatnonzero(uid == u)[0]]
    TA, TB, TC, TD = at(sc['a'][1]), at(sc['b'][1]), at(sc['c'][0]), at(sc['d'][0])
    X, Y = (TC, TD) if pair == 0 else (TD, TC)
    new, post = lab.copy(), f > sc['t1']
    new[post & (lab == X)] = TA
    new[post & (lab == Y)] = TB
    for T, z in zip((TA, TB), sc['fresh']):
        if T not in (TC, TD):
            new[post & (lab == T)] = z              # what a's or b's track did after the scene is cut off
    M = np.isin(uid, sc['mid'])
    if mid == 'keep':
        for i in np.flatnonzero(M):
            l = lab[i]
            new[i] = TA if l == TA else TB if l == TB else TA if l == X else TB if l == Y else l
    elif mid in ('a', 'b'):
        new[M] = TA if mid == 'a' else TB
    out = np.c_[S[:, :1], new, S[:, 2:]]
    if mid in ('drop', 'gt'):
        out = out[~M]
    if mid == 'gt':
        out = np.r_[out, np.array([[t, TA if who == 'a' else TB, x, y, w, h, 1.0, -1.0]
                                   for t, who, x, y, w, h in sc['gt_mid']], float).reshape(-1, 8)]
    key = out[:, 1] * 1e4 + out[:, 0]
    return out if len(np.unique(key)) == len(key) else None


def apply_all(S, scenes, choice):
    """Scenes in time order, each with choice[cid] (missing = original). Raises if a choice is not legal."""
    for sc in sorted(scenes, key=lambda s: s['t0']):
        S2 = apply(S, sc, choice.get(sc['cid']))
        assert S2 is not None, ('illegal choice', sc['cid'], choice.get(sc['cid']))
        S = S2
    return S


def groups(S, scenes):
    """Scenes that share any track id they touch (union-find), as lists, plus each group's id set."""
    sets = [touched(S, sc) for sc in scenes]
    parent = list(range(len(scenes)))

    def root(i):
        while parent[i] != i:
            i = parent[i]
        return i
    for i in range(len(scenes)):
        for j in range(i):
            if sets[i] & sets[j]:
                parent[root(i)] = root(j)
    out = {}
    for i in range(len(scenes)):
        out.setdefault(root(i), []).append(i)
    return [([scenes[i] for i in g], set().union(*[sets[i] for i in g])) for g in out.values()]


def local(S, ids):
    return S[np.isin(S[:, 1], list(ids))]


def group_range(S, scenes, ids, poly, cap=300000):
    """Elementwise min / max window-count change over every legal joint option of one group (dedup of equal states)."""
    S = local(S, ids)
    base = counts(events(S, poly))
    states = {0: S}
    for sc in sorted(scenes, key=lambda s: s['t0']):
        nxt = {}
        for T in states.values():
            for o in options(sc):
                U = apply(T, sc, o)
                if U is not None:
                    k = U[np.lexsort((U[:, 1], U[:, 0], U[:, 7]))].tobytes()
                    nxt.setdefault(k, U)
        assert len(nxt) <= cap, 'group too large to enumerate'
        states = nxt
    d = np.stack([counts(events(U, poly)) - base for U in states.values()])
    return d.min(0), d.max(0), len(states)
