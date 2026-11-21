"""Synthetic bitwise contract for immutable GT caching (opens no field files)."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path.cwd()))

import argparse
from pathlib import Path
import numpy as np
import verify203 as replay


def check(repo):
    previous,_,v,_=replay.helpers(repo);count=0
    polygon=[[-1,-1],[10,-1],[10,10],[-1,10]]
    for seed in range(12):
        rng=np.random.default_rng(seed)
        gt=np.asarray([[f,p,float(rng.integers(-2,8)),0,4,4] for p in (1,2,3) for f in (1,2,30)],float)
        if seed==0:gt[0,4]=-1
        pred=np.asarray([[f,p,float(rng.integers(-2,8)),0,4,4,.9] for p in (1,2,3,4) for f in (1,2,30)],float)
        for threshold in (.5,.3):
            a=previous.union_owners_independent(gt,pred,threshold,set(range(len(pred))),v)
            b=replay.cached_owners(gt,pred,threshold,set(range(len(pred))),v)
            for key in a:
                if isinstance(a[key],np.ndarray):assert a[key].dtype==b[key].dtype and a[key].shape==b[key].shape and a[key].tobytes()==b[key].tobytes(),key
                else:assert a[key]==b[key],key
            assert v.GT_segments(a,polygon)==v.GT_segments(b,polygon);count+=1
    return dict(passed=True,synthetic_scenes=count,field_opened=False,canonical_origin_and_owner_graph_bitwise=True,GT_segments_exact=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,default=Path.cwd());args=p.parse_args();print(check(args.repo))
