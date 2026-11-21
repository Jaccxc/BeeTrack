"""Independent section191 checker: synthetic by default, explicit sealed field mode only."""
import itertools
import json
from fractions import Fraction
from pathlib import Path
import numpy as np
import cv2
import hashlib


def bins_from_rows(rows,polygon,nf):
    """Independent gap splitting and last same-direction crossing, no project event helper."""
    events=[]
    for identity in np.unique(rows[:,1]):
        track=rows[rows[:,1]==identity];track=track[np.argsort(track[:,0],kind='stable')]
        cuts=[0,*list(np.flatnonzero(np.diff(track[:,0])>25)+1),len(track)]
        for a,b in zip(cuts,cuts[1:]):
            segment=track[a:b]
            side=[cv2.pointPolygonTest(np.asarray(polygon,np.float32),
                (float(r[2]+r[4]/2),float(r[3]+r[5]/2)),False)>=0 for r in segment]
            if side[0]!=side[-1]:
                direction=1 if side[-1] else -1
                t=max(int(segment[j,0]) for j in range(1,len(segment)) if side[j]!=side[j-1] and side[j]==side[-1])
                events.append((t,direction))
    counts=[]
    for offset in range(30):
        phase=[[0,0] for _ in range((nf+25*offset+749)//750)]
        for t,d in events:phase[(t-1+25*offset)//750][int(d<0)]+=1
        counts.extend(phase)
    return np.array(counts,dtype=np.int64),events


def team_check(c,v,g):
    """Enumerate eight errors independently; normalized team W=100*numerator/(3*sum(g))."""
    errors={};member={};exposure={kind:np.zeros((3,2),int) for kind in ('pure','mix')}
    for bits in itertools.product((0,1),repeat=3):
        rows=np.stack([v[i] if bit else c[i] for i,bit in enumerate(bits)])
        errors[''.join(map(str,bits))]=int(np.abs(rows.sum(0)-3*g).sum())
        member[bits]=int(np.abs(rows-g).sum())
        for i,bit in enumerate(bits):exposure['pure' if len(set(bits))==1 else 'mix'][i,bit]+=1
    pure=['000','111'];mixed=[k for k in errors if k not in pure]
    gain=Fraction(sum(errors[k] for k in pure),2)-Fraction(sum(errors[k] for k in mixed),6)
    single_pure=sum(Fraction(member[tuple(map(int,k))],6) for k in pure)
    single_mixed=sum(Fraction(member[tuple(map(int,k))],18) for k in mixed)
    assert single_pure==single_mixed
    assert exposure['pure'].tolist()==[[1,1]]*3 and exposure['mix'].tolist()==[[3,3]]*3
    delta=v-c;S=delta.sum(0);a=np.abs(c.sum(0)+v.sum(0)-6*g)
    formula=3*np.maximum(a,np.abs(S))-sum(np.maximum(a,np.abs(S-2*delta[i])) for i in range(3))
    assert Fraction(int(formula.sum()),6)==gain
    ceiling=np.maximum(0,3*np.abs(S)-sum(np.abs(S-2*delta[i]) for i in range(3)))
    assert np.all(formula<=ceiling)
    direct_pure18=3*(np.abs(c.sum(0)-3*g)+np.abs(v.sum(0)-3*g))
    assert np.array_equal(direct_pure18,3*np.maximum(a,np.abs(S)))
    same=np.all(delta>=0,axis=0)|np.all(delta<=0,axis=0)
    assert int(formula[same].sum())+int(formula[~same].sum())==int(formula.sum())
    return dict(team_numerators=errors,gain18=int(formula.sum()),scorefree_ceiling18=int(ceiling.sum()),
        same_direction_gain18=int(formula[same].sum()),opposite_direction_gain18=int(formula[~same].sum()),
        equal_single_mean=True,gt_bin_sum=int(g.sum()))


def synthetic():
    from mix_check import compare
    rng=np.random.default_rng(191061)
    for _ in range(250):
        c,v=rng.integers(0,12,(2,3,7,2));g=rng.integers(0,12,(7,2))
        own=team_check(c,v,g);other=compare(c,v,g)
        assert own['gain18']==-other['difference_error18_total']
        for team in other['teams']:
            key=''.join(map(str,team['arms']))
            assert own['team_numerators'][key]*6==int(team['error18'].sum())
    poly=[[0,0],[10,0],[10,10],[0,10]]
    rows=np.array([[749,7,12,1,2,2,1],[750,7,1,1,2,2,1],[800,7,1,1,2,2,1],[801,7,12,1,2,2,1]],float)
    bins,events=bins_from_rows(rows,poly,820)
    assert events==[(750,1),(801,-1)] and bins.sum()==60
    return dict(random_eight_team_checks=250,independent_gap_crossing_phase_check=True,passed=True)


def field(root):
    from beetrack import p0 as P
    from tools.study103.swap_score import event_scores
    root=Path(root);score=root/'score';read=lambda p:json.loads(p.read_text())
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    assert sha(root/'manifest.json')==(root/'ALL_FROZEN.sha256').read_text().strip()
    assert sha(score/'result_hashes.json')==(score/'SEALED.sha256').read_text().strip()
    for name,digest in read(score/'result_hashes.json').items():assert sha(score/name)==digest
    summary=read(score/'summary.json');members={};bins={};checks=[]
    for h in (33,35):
        clip=f'BEE24-{h}-head';gt=P.gt('dev',clip);poly=P.ZONE[clip]
        gc,ge=bins_from_rows(gt,poly,3950)
        for seed in range(3,9):
            for arm in ('CS','VT'):
                key=f'h{h}_s{seed}_{arm}';folder=root/key;receipt=read(folder/'receipt.json')
                assert (receipt['hive'],receipt['seed'],receipt['recipe'])==(h,seed,arm)
                assert receipt['polygon']==poly
                for name,digest in receipt['files'].items():assert sha(folder/name)==digest
                assert sha(folder/'receipt.json')==(folder/'SEALED.sha256').read_text().strip()
                rows=np.load(folder/'mot.npy');pc,pe=bins_from_rows(rows,poly,3950)
                assert sorted(pe)==sorted(map(tuple,np.load(folder/'events.npy').tolist()))
                assert np.array_equal(pc,np.load(folder/'bins.npy'))
                saved=summary['members'][key]
                assert np.array_equal(gc,saved['integer_bins']['gt']) and np.array_equal(pc,saved['integer_bins']['pred'])
                n=int(np.abs(pc-gc).sum());d=int(gc.sum())
                assert n==saved['window']['numerator'] and d==saved['window']['denominator']
                assert abs(100*n/d-saved['w30_percent'])<1e-10
                official=event_scores(gt,rows,poly)
                for t in (.5,.3):
                    for k,v in official[t].items():assert v==saved['iou'][str(t)][k]
                members[h,seed,arm]=dict(W=Fraction(100*n,d),f1={str(t):Fraction(2*official[t]['tp'],max(2*official[t]['tp']+official[t]['fp']+official[t]['fn'],1)) for t in (.5,.3)})
                bins[h,seed,arm]=pc
                checks.append(dict(h=h,seed=seed,recipe=arm,events=len(pe),window_numerator=n))
                print('MEMBER_PASS',key,flush=True)
        means=[]
        for group,seeds in [('345',(3,4,5)),('678',(6,7,8))]:
            c=np.stack([bins[h,s,'CS'] for s in seeds]);v=np.stack([bins[h,s,'VT'] for s in seeds])
            own=team_check(c,v,gc);saved=summary['heads'][str(h)]['groups'][group]
            for bits,n in own['team_numerators'].items():
                name=bits.translate(str.maketrans('01','CV'));team=saved['teams'][name]
                assert n==team['E3']['numerator'] and 3*gc.sum()==team['E3']['denominator']
                assert abs(100*n/(3*gc.sum())-team['E3']['W_percent'])<1e-10
                selected=[members[h,s,'VT' if b=='1' else 'CS'] for s,b in zip(seeds,bits)]
                for metric in ('W','0.5','0.3'):
                    expected=sum((m['W'] if metric=='W' else m['f1'][metric] for m in selected),Fraction())/3
                    actual=team['mean_member_W'] if metric=='W' else team['mean_member_F1'][metric]
                    assert expected==Fraction(actual['numerator'],actual['denominator'])
            assert own['gain18']==saved['improvement']['error18']
            assert own['scorefree_ceiling18']==saved['score_free_ceiling']['error18']
            assert own['same_direction_gain18']==saved['categories']['same_direction']['gain18']
            assert own['opposite_direction_gain18']==saved['categories']['other']['gain18']
            pure18=3*(own['team_numerators']['000']+own['team_numerators']['111'])
            mixed18=sum(n for b,n in own['team_numerators'].items() if b not in ('000','111'))
            assert pure18==saved['pure']['error18'] and mixed18==saved['mixed']['error18']
            assert saved['exposure']=={kind:{f'{s}_{arm}':(1 if kind=='pure' else 3) for s in seeds for arm in ('CS','VT')} for kind in ('pure','mixed')}
            for metric in ('W','F1_05','F1_03'):
                value=sum((members[h,s,arm]['W'] if metric=='W' else members[h,s,arm]['f1']['0.5' if metric=='F1_05' else '0.3'] for s in seeds for arm in ('CS','VT')),Fraction())/6
                record=saved['exact_equal_member_means'][metric]
                assert value==Fraction(record['numerator'],record['denominator'])
            ref=summary['published_reference'][str(h)][group]
            for arm,bits in [('CS','000'),('VT','111')]:assert abs(100*own['team_numerators'][bits]/(3*gc.sum())-ref[arm])<=.0051
            means.append(dict(pure=100*pure18/(18*gc.sum()),mixed=100*mixed18/(18*gc.sum()),
                improvement_pp=100*own['gain18']/(18*gc.sum()),score_free_ceiling_pp=100*own['scorefree_ceiling18']/(18*gc.sum()),
                CS=100*own['team_numerators']['000']/(3*gc.sum()),VT=100*own['team_numerators']['111']/(3*gc.sum())))
        for key in means[0]:assert abs(sum(m[key] for m in means)/2-summary['heads'][str(h)]['mean'][key])<1e-10
        for category in ('same_direction','other'):
            for metric in ('gain_pp','score_free_ceiling_pp'):
                average=sum(summary['heads'][str(h)]['groups'][g]['categories'][category][metric] for g in ('345','678'))/2
                assert abs(average-summary['heads'][str(h)]['mean']['categories'][category][metric])<1e-10
    return dict(all_passed=True,members=checks,score_seal=sha(score/'result_hashes.json'),
        rebuilt='Native MOT gap segments, polygon sides, events,30phase integer bins,official member F1,eight teams,exact member means,exposure,closed-form totals,scorefree ceilings; historical references read only')


if __name__=='__main__':
    import argparse,sys
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--repo',type=Path,default=Path.cwd());p.add_argument('--out',type=Path);p.add_argument('--verify-sealed',action='store_true');a=p.parse_args()
    sys.path.insert(0,str(a.repo.resolve()))
    if a.verify_sealed:
        if a.out is None:p.error('--out required')
        result=field(a.out);target=a.out/'independent.json'
        if target.exists():raise FileExistsError(target)
        target.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    else:result=synthetic()
    print(json.dumps(result,indent=2))
