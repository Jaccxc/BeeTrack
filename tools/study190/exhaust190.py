"""Independent exhaustive single-stream geometry-menu replay; JSON input only."""
import os
for _name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '1'
import argparse
import hashlib
from itertools import product
import json
from math import prod
from pathlib import Path

import numpy as np


def event_vector(event, nf):
    sizes = [(nf + offset * 25 + 749) // 750 for offset in range(30)]
    counts = np.zeros((sum(sizes), 2), dtype=np.int64)
    if event is not None:
        frame, direction = event
        assert type(frame) is int and 1 <= frame <= nf and direction in (-1, 1)
        first = 0
        for offset, size in enumerate(sizes):
            counts[first + (frame - 1 + offset * 25) // 750, int(direction == -1)] = 1
            first += size
    return counts.reshape(-1)


def enumerate_cell(cell):
    nf = cell['nf']
    baseline = np.asarray(cell['baseline_phase_counts'], dtype=np.int64).reshape(-1)
    truth = sum((event_vector(e, nf) for e in cell['gt_events']), np.zeros_like(baseline))
    residual = baseline - truth
    variable = []
    menu_sizes = []
    raw_sizes = []
    for menu in cell['menus']:
        assert len(menu['original']) == 1
        residual -= event_vector(menu['original'][0], nf)
        unique = {}
        for e in menu['geometry']:
            key = None if e is None else ((e[0] - 1) // 25, e[1])
            unique.setdefault(key, e)
        options = [event_vector(e, nf) for e in unique.values()]
        assert len(options)
        menu_sizes.append(len(options)); raw_sizes.append(len(menu['geometry']))
        if len(options) == 1:
            residual += options[0]
        else:
            variable.append(options)
    combinations = prod(menu_sizes)
    if combinations > 147456:
        raise ValueError('Unexpected enumeration above authorized design limit')
    best, checked = None, 0
    # Direct Cartesian enumeration, no interval relaxation or optimizer.
    for chosen in product(*variable):
        counts = residual.copy()
        for vector in chosen:
            counts += vector
        numerator = int(np.abs(counts).sum())
        if best is None or numerator < best:
            best = numerator
        checked += 1
    assert checked == combinations
    return dict(best_numerator=best, combinations=checked,
                raw_date_combinations=prod(raw_sizes), variable_menus=len(variable),
                denominator=30 * len(cell['gt_events']),
                best_W_percent=100 * best / (30 * len(cell['gt_events'])))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('geometry', type=Path)
    args = parser.parse_args()
    raw = args.geometry.read_bytes()
    doc = json.loads(raw)
    expected = {f'h{h}_s{k}' for h in (33, 35) for k in range(3, 9)}
    if set(doc['cells']) != expected:
        raise ValueError('Require exactly the twelve registered design cells')
    results = {key: enumerate_cell(doc['cells'][key]) for key in sorted(expected)}
    print(json.dumps(dict(input_sha256=hashlib.sha256(raw).hexdigest(), cells=results), indent=2))


if __name__ == '__main__':
    main()
