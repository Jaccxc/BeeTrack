"""Private prospective211 joint209/210 actions; SOURCE ONLY before root gate.

No natural artifact/GT IO or new scorer. Caller injects frozen original208
rows/canon,209common-slate deltas and210active-slate deltas plus source seals.
One legal subset is a matching over ORIGINAL canonical runs across BOTH families;
empty is legal. Separate-family optima cannot be added. Full physical window
counts must equal K+sum(delta) for EVERY feasible joint subset before scoring.
Reuse runner211's peers/official graph/scorer by consuming compact state arrays.
"""
from pathlib import Path
import importlib.util
import re
import numpy as np


_spec = importlib.util.spec_from_file_location('joint211_original_base', Path(__file__).parents[1]/'study209'/'physical209.py')
_base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_base)


def _delta(values):
    a = np.asarray(values)
    if a.ndim != 2 or a.shape[1] != 2 or not np.issubdtype(a.dtype, np.integer):
        raise ValueError('frozen signed integer full30phase window delta[W,2] required')
    return a.astype(np.int64, copy=False)


def make_roster(co_pairs, common_slate, co_delta_by_pair, inter_pairs, active_slate, *, source_seals):
    """Explicit new joint roster from two ORIGINAL frozen slates, no reselection.

    co_delta_by_pair comes from independently sealed209 single-action windows;
    210 records carry delta_windows. Include209 iff count delta nonzero and210
    iff it is on the original active slate. Caller verifies parent files/seals;
    this pure function validates identifiers and records them without file IO.
    """
    if set(source_seals) != {'209', '210'} or any(not re.fullmatch('[0-9a-f]{64}', s) for s in source_seals.values()):
        raise ValueError('both original209/210 artifact SHA256 seals required')
    roster, excluded_inert_209 = [], []
    shape = None
    for family, records, slate in (('209', co_pairs, common_slate), ('210', inter_pairs, active_slate)):
        table = {p['pair_id']: p for p in records}
        if len(table) != len(records) or len(set(slate)) != len(slate) or not set(slate) <= table.keys():
            raise ValueError('unique original pair records and frozen slate required')
        for key in slate:
            p = table[key]
            ids = p['canonical_ids']
            if len(ids) != 2 or any(not isinstance(i, (int, np.integer)) or isinstance(i, (bool, np.bool_)) for i in ids) or not 0 < ids[0] < ids[1] or key != f'{ids[0]}|{ids[1]}':
                raise ValueError('original ascending canonical run pair required')
            d = _delta(co_delta_by_pair[key] if family == '209' else p['delta_windows'])
            if shape is None:
                shape = d.shape
            if d.shape != shape:
                raise ValueError('all frozen deltas must use identical full phase layout')
            if not d.any():
                if family == '210':
                    raise ValueError('original210 active slate contains count-inert pair')
                excluded_inert_209.append(key)
                continue
            roster.append(dict(action_id=f'{family}:{key}', family=family, pair_id=key,
                               canonical_ids=list(map(int, ids)), delta_windows=d.tolist(),
                               source_seal=source_seals[family], source_pair=p))
    roster.sort(key=lambda r: (r['canonical_ids'], r['family']))
    conflicts = []
    for i, first in enumerate(roster):
        for second in roster[i+1:]:
            overlap = sorted(set(first['canonical_ids']) & set(second['canonical_ids']))
            if overlap:
                conflicts.append(dict(actions=[first['action_id'], second['action_id']], shared_original_runs=overlap))
    return dict(schema='match211.joint_actions.v1', source_seals=dict(source_seals), actions=roster,
                original209_common_slate=list(common_slate), original210_active_slate=list(active_slate),
                excluded209_count_inert=excluded_inert_209,
                conflicts=conflicts, window_shape=list(shape) if shape else None,
                rule='single matching over immutable original runs across both families; empty legal')


def feasible_subsets(roster):
    """Yield deterministic feasible subsets, including empty, without multiway union."""
    actions = roster['actions']
    if len({r['action_id'] for r in actions}) != len(actions):
        raise ValueError('duplicate joint action ID')
    def visit(index, used, selected):
        if index == len(actions):
            yield tuple(selected)
            return
        yield from visit(index+1, used, selected)
        action = actions[index]
        ids = set(action['canonical_ids'])
        if not ids & used:
            yield from visit(index+1, used | ids, selected+[action['action_id']])
    yield from visit(0, set(), [])


def materialize(rows, canon, roster, selected_ids):
    """Joint immutable-run min-ID writer, fixed original row tie order.

    209 retains highest native score at EACH jointly observed date, ties lower
    ORIGINAL raw row index; all singly observed dates survive.210 retains all.
    Nonempty states initialize EVERY row to immutable canon before pair->min ID.
    Empty K preserves original raw IDs, values and row order bit-for-bit.
    """
    rows, canon, runs = _base.original_runs(rows, canon)
    table = {r['action_id']: r for r in roster['actions']}
    selected = list(selected_ids)
    if len(set(selected)) != len(selected) or not set(selected) <= table.keys():
        raise ValueError('unique frozen joint action IDs required')
    if not selected:
        return rows.copy(), dict(selected_actions=[], actions=[], output_raw_indices=list(range(len(rows))),
                                 deleted_raw_indices=[], original_K_exact=True)
    out, keep, used, proofs = rows.copy(), np.ones(len(rows), bool), set(), []
    out[:, 1] = canon
    for key in sorted(selected):
        action = table[key]
        a, b = action['canonical_ids']
        if a not in runs or b not in runs or used & {a, b}:
            raise ValueError('foreign or conflicting original run; cross-family chains forbidden')
        ia, ib = runs[a], runs[b]
        left = {int(rows[i, 0]): int(i) for i in ia}
        right = {int(rows[i, 0]): int(i) for i in ib}
        shared = sorted(left.keys() & right.keys())
        deleted, winners = [], []
        if action['family'] == '209':
            source = action['source_pair']
            actual = [[left[f], right[f]] for f in shared]
            if not shared or source['shared_dates'] != shared or source['row_indices'] != actual:
                raise ValueError('209 original shared-date/raw-index seal contract mismatch')
            for i, j in actual:
                winner, loser = ((i, j) if rows[i, 6] > rows[j, 6] or rows[i, 6] == rows[j, 6] and i < j else (j, i))
                keep[loser] = False
                winners.append(winner); deleted.append(loser)
        elif action['family'] == '210':
            if shared:
                raise ValueError('210 original interleaved runs have common observed dates')
        else:
            raise ValueError('unknown action family')
        ix = np.r_[ia, ib]
        out[ix, 1] = a
        retained_dates = np.sort(rows[ix[keep[ix]], 0])
        if (np.diff(retained_dates) <= 0).any() or (np.diff(retained_dates) > 25).any():
            raise ValueError('union violates native distinct-date/gap-run semantics')
        used.update((a, b))
        proofs.append(dict(action_id=key, family=action['family'], canonical_ids=[a, b], merged_canonical_id=a,
                           original_rows_A=ia.tolist(), original_rows_B=ib.tolist(),
                           shared_dates=shared, winner_raw_indices=winners, deleted_raw_indices=deleted))
    retained = np.flatnonzero(keep)
    order = retained[np.lexsort((retained, rows[retained, 0]))]
    physical = out[order]
    if physical[:, [0, 2, 3, 4, 5, 6]].tobytes() != rows[order][:, [0, 2, 3, 4, 5, 6]].tobytes():
        raise AssertionError('original observation lineage/precision altered')
    if len(np.unique(physical[:, :2], axis=0)) != len(physical):
        raise AssertionError('duplicate physical native ID/date')
    return physical, dict(selected_actions=sorted(selected), actions=proofs, output_raw_indices=order.tolist(),
                          deleted_raw_indices=np.flatnonzero(~keep).tolist(), original_K_exact=False,
                          date_order='frame then original raw index', ID_policy='immutable canon, selected whole-run pair->minimum canon')


def verify_all_window_deltas(rows, canon, roster, *, count_windows, on_state=None):
    """Injected native full30phase counter only, no GT or second event scorer.

    Check EVERY original singleton against its SEALED per-family delta first;
    then physically recount every legal joint subset against K+sum(delta).
    Optional on_state receives compact npz-compatible ids/output_raw_indices
    for runner211 to reuse SAME peers/official graph closure (not rebuilt here).
    """
    base = np.asarray(count_windows(rows))
    table = {r['action_id']: r for r in roster['actions']}
    for action in roster['actions']:
        physical, _ = materialize(rows, canon, roster, [action['action_id']])
        np.testing.assert_array_equal(np.asarray(count_windows(physical))-base, _delta(action['delta_windows']))
    summaries = []
    for selected in feasible_subsets(roster):
        physical, proof = materialize(rows, canon, roster, selected)
        expected = base.copy()
        for key in selected:
            expected = expected+_delta(table[key]['delta_windows'])
        actual = np.asarray(count_windows(physical))
        np.testing.assert_array_equal(actual, expected)
        compact = dict(ids=physical[:, 1].astype(np.int64), output_raw_indices=np.asarray(proof['output_raw_indices'], np.int64))
        if on_state is not None:
            on_state(selected, compact, proof)
        summaries.append(dict(selected=list(selected), windows=actual.tolist(), proof=proof))
    return dict(state_count=len(summaries), states=summaries, singleton_sealed_deltas_verified=True,
                every_feasible_subset_physically_recounted=True)


def synthetic_merge_assertions():
    """Tiny merge-only fixture; no natural IO, native graph or field execution."""
    rows = np.array([[2, 2, 8, 2, 2, 2, .8], [1, 1, 11, 2, 2, 2, .7],
                     [2, 1, 8, 2, 2, 2, .8], [3, 3, 7, 2, 2, 2, .9],
                     [1, 4, 11, 8, 2, 2, .9], [2, 5, 8, 8, 2, 2, .9]], float)
    canon = rows[:, 1].astype(np.int64)
    delta = [[-1, 0]]
    co = [dict(pair_id='1|2', canonical_ids=[1, 2], shared_dates=[2], row_indices=[[2, 0]])]
    inter = [dict(pair_id='2|3', canonical_ids=[2, 3], delta_windows=delta),
             dict(pair_id='4|5', canonical_ids=[4, 5], delta_windows=delta)]
    roster = make_roster(co, ['1|2'], {'1|2': delta}, inter, ['2|3', '4|5'], source_seals={'209': 'a'*64, '210': 'b'*64})
    states = list(feasible_subsets(roster))
    assert () in states and len(states) == 6
    assert not any('209:1|2' in s and '210:2|3' in s for s in states)
    k, _ = materialize(rows, canon, roster, [])
    assert k.tobytes() == rows.tobytes()
    physical, proof = materialize(rows, canon, roster, ['209:1|2', '210:4|5'])
    assert proof['deleted_raw_indices'] == [2]  # Equal score: original raw0 wins.
    assert proof['output_raw_indices'] == [1, 4, 0, 5, 3]
    assert physical[:, 1].tolist() == [1., 4., 1., 4., 3.]
    try:
        materialize(rows, canon, roster, ['209:1|2', '210:2|3'])
    except ValueError:
        pass
    else:
        raise AssertionError('cross-family original run conflict accepted')
    return dict(passed=True, legal_subsets=len(states), conflicts=roster['conflicts'], scope='tiny merge only; sealed delta and official graph checks injected later')


if __name__ == '__main__':
    print(synthetic_merge_assertions())
