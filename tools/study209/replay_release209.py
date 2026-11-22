"""One public entry: 12 logistic refits,36 states,weak AP and exact support."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
os.environ['CUDA_VISIBLE_DEVICES']='-1'
import argparse
import json
from pathlib import Path
from replay209 import replay as replay_main
from replay_support209 import read,replay as replay_support
from replay_weak209 import replay as replay_weak


def bins(events,nf):
    result=[]
    for phase in range(30):
        shift=phase*25;counts=[[0,0] for _ in range((nf+shift+749)//750)]
        for _,signed in events:
            assert type(signed)is int and 1<=abs(signed)<=nf
            counts[(abs(signed)-1+shift)//750][int(signed<0)]+=1
        result.extend(counts)
    return result


def replay(bundle_path,support_path):
    bundle=read(bundle_path);support=read(support_path)
    for clock in support['clips']:
        clip=next(c for c in bundle['clips'] if c['domain']==clock['domain'] and c['clip']==clock['clip'])
        for state in clip['arms'].values():
            for side in ('gt','pred'):assert bins(state['events'][side],clock['nframes'])==state['bins'][side]
    return dict(main=replay_main(bundle),weak=replay_weak(bundle),support=replay_support(bundle_path,support_path),
        scope='12 source-only logistic refits; all36 event clocks/windows and edge matchings; weak AP; fixed-slate count support. No pixel/LK or GT box reassociation.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('bundle',type=Path);parser.add_argument('support',type=Path);parser.add_argument('--out',type=Path)
    args=parser.parse_args();value=replay(args.bundle,args.support)
    if args.out:args.out.write_text(json.dumps(value,sort_keys=True,separators=(',',':'))+'\n',encoding='utf-8',newline='\n')
    print(json.dumps(dict(passed=True,source_logistic_refits=12,physical_states=36,
        investment_pass=value['main']['preregistered_gates']['investment_pass'],motion_attribution=value['weak']['attribution'],
        count_support={h:{k:r[k] for k in ('baseline_numerator','min_numerator','max_numerator','denominator','onepp_active_types','total_active_types')} for h,r in value['support']['pooled'].items()})))
