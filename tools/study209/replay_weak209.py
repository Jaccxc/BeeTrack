"""Registered weak-label AP and paired true/sham motion-attribution replay."""
from collections import Counter
from replay209 import _score
from verify209 import average_precision


def replay(bundle):
    targets={}
    for hive in ('35','33','W','G'):
        rows=[r for c in bundle['clips'] if c['domain']==hive for r in c['candidates']]
        models=bundle['models'][hive]
        known=[r for r in rows if r['true_valid'] and r['weak'] in ('S','D')]
        paired=[r for r in known if r['sham_valid']]
        def summary(population,arm,sham=False):
            scores=[_score(models[arm],r,arm,sham=sham) for r in population]
            assert all(s is not None for s in scores)
            labels=[int(r['weak']=='S') for r in population]
            return dict(AP=average_precision(labels,scores),S=sum(labels),D=len(labels)-sum(labels),
                rows=len(labels),raw_logit_positive=sum(s>0 for s in scores))
        targets[hive]=dict(weak_all=dict(Counter(r['weak'] for r in rows)),main_known=len(known),
            main={arm:summary(known,arm) for arm in ('G','F','X')},paired_known=len(paired),
            paired={**{arm:summary(paired,arm) for arm in ('F','X')},'shamX':summary(paired,'X',True)})
    enough=all(targets[h]['paired']['X']['S']>=5 and targets[h]['paired']['X']['D']>=5 for h in ('35','33'))
    means=None;passed=None
    if enough:
        means={arm:sum(targets[h]['paired'][arm]['AP'] for h in ('35','33'))/2 for arm in ('X','F','shamX')}
        passed=means['X']>means['F'] and means['X']>means['shamX']
    return dict(targets=targets,attribution=dict(sufficient=enough,head_equal_AP=means,passed=passed))
