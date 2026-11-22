"""Post-result integer deductions and registered E3 envelopes; no field IO."""
import argparse,gzip,json
from fractions import Fraction
from pathlib import Path
from bound214 import replay_evidence,pool_bounds


def calculate(bundle,report):
    bounds={}
    for h in ('35','33','W','G'):
        bounds[h]=pool_bounds([replay_evidence(c['bound_evidence']) for c in bundle['clips'] if c['domain']==h])
    k=report['domains']['35']['K'];n=k['numerator'];den=k['denominator']
    lower=bounds['35']['layers']['owner_side']['lower_bound_numerator']
    threshold=(100*n-den)//100
    necessary=max(0,(lower-threshold+29)//30)
    assert (n,den,lower,threshold,necessary)==(548,11610,486,431,2)
    cancellation={}
    for arm in ('K','I'):
        total_abs=ensemble=0
        for c in report['clips']:
            if c['domain']!='33':continue
            total_abs+=sum(m['states'][arm]['numerator'] for m in c['members'])
            ensemble+=c['arms'][arm]['numerator']
        cancellation[arm]=dict(member_absolute_error_sum=total_abs,ensemble_numerator=ensemble,
                              cancellation=total_abs-ensemble)
    assert cancellation==dict(K=dict(member_absolute_error_sum=3488,ensemble_numerator=3176,cancellation=312),
                             I=dict(member_absolute_error_sum=3524,ensemble_numerator=3276,cancellation=248))
    assert 3524-3488 + 312-248 == 3276-3176 == 100
    return dict(schema='center214-supplement-v1',E3_bounds=bounds,
        posthoc_outside_B_necessary=dict(K_numerator=n,denominator=den,B_lower_numerator=lower,
            one_pp_target_max_numerator=threshold,one_endpoint_max_L1=30,
            minimum_distinct_member_endpoints=necessary,necessary_not_sufficient=True,
            physical_bee_count_inference=False,unowned_count_inference=False),
        posthoc_cancellation_33=dict(arms=cancellation,member_absolute_error_increase=36,
            cancellation_loss=64,ensemble_increase=100,denominator=16560,
            causal_image_correlation_claim=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('evidence',type=Path);p.add_argument('report',type=Path);p.add_argument('--out',required=True,type=Path);a=p.parse_args()
    with gzip.open(a.evidence,'rt',encoding='utf8') as f:b=json.load(f)
    r=json.loads(a.report.read_text(encoding='utf8'));out=calculate(b,r)
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(out,sort_keys=True,indent=2)+'\n',encoding='utf8')
