"""Synthetic-ready numerical contracts; external geometry remains separate."""
import math


def equal(a, b):
    if isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a: equal(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b): equal(x, y)
    elif isinstance(a, float): assert math.isclose(a, b, rel_tol=2e-8, abs_tol=2e-9)
    else: assert a == b


def capacity_case(unit, family, answer, capacity):
    # A caller may reconstruct only this unit's dense windows for the original checker.
    # The release adapter itself stores sparse int columns and never an all-unit V.
    assert capacity.verify_answer(unit, family, answer)
    assert answer['optimal'] == (answer['lower_bound'] == answer['upper_bound'])
    return dict(lower=answer['lower_bound'], upper=answer['upper_bound'], gap=answer['gap'])


def source_fold(fold, fitter):
    h = fold['heldout']; source = fold['natural']
    assert all(c['hive'] != h for c in [*source, *fold.get('artificial', [])])
    rebuilt = {}
    for mode in ('I','L','T'):
        result = fitter.fit(source, fold['supervision'], mode,
                            artificial=fold.get('artificial', []), artificial_supervision=fold.get('artificial_supervision', {}), heldout=h)
        expected = fold['models'][mode]
        for key in ('status','mean','scale','beta','intercept','training_records','coverage','artificial_coverage','weight_audit'):
            equal(result[key], expected[key])
        equal(result['optimization']['objective'], expected['optimization']['objective'])
        rebuilt[mode] = result
    selected = fitter.deploy(fold['target'], rebuilt, fold.get('skeleton', []))
    equal(selected, fold['selected'])
    return dict(refits=3, claim='registered zero-start local numerical fit, not global loss optimality')


def physical_metric(metric, nf, signed_bins, matching):
    evidence = metric['evidence']
    equal(signed_bins(evidence['gt'], nf), metric['gt_bins'])
    equal(signed_bins(evidence['predictions'], nf), metric['pred_bins'])
    assert len(evidence['gt']) == metric['nG'] and len(evidence['predictions']) == metric['nP']
    for threshold, name in [('0.5','edges05'),('0.3','edges03')]:
        tp = matching(evidence['predictions'], evidence['gt'], evidence[name]); value = metric['iou'][threshold]
        equal(tp, value['tp']); equal(value['pred'], metric['nP']); equal(value['gt'], metric['nG'])
        equal(value['f1'], 2*tp/max(metric['nP']+metric['nG'],1))
    return True
