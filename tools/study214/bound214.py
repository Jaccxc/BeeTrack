"""Endpoint-side count envelopes, not physical repair witnesses (source/synthetic).

all_side relaxes every frozen eligible endpoint to either polygon side, even
beyond its geometry cap. owner_side allows only original or supplied O side;
it bounds that narrower family, NOT every capped image-model displacement.
Each original run chooses its endpoints jointly. Coordinates/windows/runs may
then choose mutually inconsistent options in the envelope; resulting W is a
lower bound, never an attainable witness. No optimizer, field IO or GT matcher.
"""
import argparse
import gzip,json,sys
from pathlib import Path
from itertools import product
import numpy as np

FPS=25
PERIOD=30*FPS

def _integer_array(value,name):
    a=np.asarray(value)
    if not np.issubdtype(a.dtype,np.integer) or np.issubdtype(a.dtype,np.bool_):raise ValueError(name+' requires integers')
    return a.astype(np.int64,copy=False)

def window_counts(events,nframes):
    """Official nonempty 30-phase intervals; includes final partial windows."""
    if isinstance(nframes,bool) or not isinstance(nframes,(int,np.integer)) or nframes<1:raise ValueError('positive integer nframes')
    values=list(events);E=np.asarray(values,dtype=np.int64).reshape(-1,2)
    if len(E) and ((E[:,0]<1).any() or (E[:,0]>nframes).any() or not np.isin(E[:,1],[-1,1]).all()):raise ValueError('invalid frame/sign event')
    parts=[]
    for offset in range(0,PERIOD,FPS):
        n=(int(nframes)+offset+PERIOD-1)//PERIOD;bins=(E[:,0]-1+offset)//PERIOD
        parts.append(np.bincount(2*bins+(E[:,1]<0),minlength=2*n).reshape(n,2))
    return np.vstack(parts)

def net_event(frames,sides):
    """First/last side selects direction; last same-direction crossing dates it."""
    f=_integer_array(frames,'frames');s=np.asarray(sides)
    if f.ndim!=1 or not len(f) or s.shape!=f.shape or s.dtype!=np.bool_:raise ValueError('nonempty frames and bool side sequence')
    if (f<1).any() or (np.diff(f)<=0).any() or (np.diff(f)>25).any():raise ValueError('one canonical run: increasing dates with gaps <=25')
    if s[0]==s[-1]:return None
    direction=1 if s[-1] else -1
    last=next(j for j in range(len(f)-1,0,-1) if s[j]!=s[j-1] and (1 if s[j] else -1)==direction)
    return int(f[last]),direction

def run_menu(frames,sides,eligible,*,oracle_sides=None,mode='all_side'):
    """Only first/last observations may change; singleton is ONE shared bit.

    eligible is a list of positions within this run, not raw physical indices.
    oracle_sides is the complete bool O-side sequence (only endpoints used).
    """
    f=_integer_array(frames,'frames');s=np.asarray(sides)
    original=net_event(f,s);positions=[int(i) for i in eligible]
    if len(set(positions))!=len(positions) or not set(positions)<={0,len(f)-1}:raise ValueError('unique first/last positions only')
    if mode not in ('all_side','owner_side'):raise ValueError('unknown endpoint option family')
    O=s if oracle_sides is None else np.asarray(oracle_sides)
    if O.shape!=s.shape or O.dtype!=np.bool_:raise ValueError('oracle_sides must match full bool sequence')
    positions.sort();options=[(False,True) if mode=='all_side' else tuple(sorted({bool(s[i]),bool(O[i])})) for i in positions]
    choices=[]
    for bits in product(*options):
        side=s.copy()
        for i,b in zip(positions,bits):side[i]=b
        event=net_event(f,side)
        choices.append(dict(endpoint_bits=list(bits),event=None if event is None else list(event)))
    events=sorted({None if c['event'] is None else tuple(c['event']) for c in choices},key=lambda e:(e is not None,() if e is None else e))
    return dict(original_event=None if original is None else list(original),eligible_positions=positions,
                choices=choices,events=[None if e is None else list(e) for e in events])

def envelope(runs,nframes):
    """Input runs: frames, bool sides, eligible positions, optional oracle_sides."""
    zero=window_counts([],nframes);output={mode:dict(lower=zero.copy(),upper=zero.copy(),runs=[]) for mode in ('all_side','owner_side')}
    baseline=zero.copy()
    for index,run in enumerate(runs):
        event=net_event(run['frames'],run['sides']);baseline+=window_counts([] if event is None else [event],nframes)
        for mode in output:
            menu=run_menu(run['frames'],run['sides'],run['eligible'],oracle_sides=run.get('oracle_sides'),mode=mode)
            vectors=np.stack([window_counts([] if e is None else [e],nframes) for e in menu['events']])
            output[mode]['lower']+=vectors.min(axis=0);output[mode]['upper']+=vectors.max(axis=0)
            output[mode]['runs'].append(dict(run_index=index,run_key=run.get('run_key',index),**menu))
    for mode in output:assert (output[mode]['lower']<=baseline).all() and (baseline<=output[mode]['upper']).all()
    assert (output['all_side']['lower']<=output['owner_side']['lower']).all() and (output['owner_side']['upper']<=output['all_side']['upper']).all()
    return dict(baseline=baseline,**output)

def e3_bound(member_envelopes,gt_windows=None):
    """Sum three native member envelopes, then distance of 3*GT to interval.

    Numerator is raw sum of absolute three-member residuals; percentage uses
    3*sum(GTbins)=90*GTevents. Zero-GT clips keep their numerator for pooling;
    percentage is undefined only if the pooled denominator is zero.
    """
    if len(member_envelopes)!=3:raise ValueError('exactly three fixed members required')
    result=dict(schema='endpoint-side-interval-bound-v1',witness=False,members=3,layers={})
    baseline=sum((m['baseline'] for m in member_envelopes),np.zeros_like(member_envelopes[0]['baseline']))
    G=None if gt_windows is None else _integer_array(gt_windows,'GT windows')
    if G is not None and (G.shape!=baseline.shape or (G<0).any()):raise ValueError('GT clock shape/nonnegativity mismatch')
    D=None if G is None else 3*int(G.sum());result.update(denominator=D,baseline_pred_sum=baseline.tolist(),baseline_numerator=None if G is None else int(np.abs(baseline-3*G).sum()))
    for mode in ('all_side','owner_side'):
        lo=sum((m[mode]['lower'] for m in member_envelopes),np.zeros_like(baseline));hi=sum((m[mode]['upper'] for m in member_envelopes),np.zeros_like(baseline))
        if G is None:N=None
        else:N=int((np.maximum(lo-3*G,0)+np.maximum(3*G-hi,0)).sum())
        result['layers'][mode]=dict(lower_counts=lo.tolist(),upper_counts=hi.tolist(),lower_bound_numerator=N,
            W_lower_bound_percent=None if not D else 100*N/D,
            scope='any endpoint side, geometry relaxed' if mode=='all_side' else 'each endpoint original/O-cap side only; not all pixel-model offsets')
    if G is not None:assert result['layers']['all_side']['lower_bound_numerator']<=result['layers']['owner_side']['lower_bound_numerator']<=result['baseline_numerator']
    return result

def physical_envelope(rows,endpoints,polygon,nframes,*,oracle_rows=None):
    """Native Nx7 adapter: only reads supplied arrays; no GT/provider/file IO.

    endpoints are frozen, unique {'raw_index':int} records; caller decides
    near-boundary eligibility. O rows are eval214.materialize's full result.
    Noneligible rows must be unchanged in O; no owner means pass original row.
    """
    from beetrack.evaluate import by_track,inside
    R=np.asarray(rows,dtype=float);O=R if oracle_rows is None else np.asarray(oracle_rows,dtype=float)
    if R.ndim!=2 or R.shape[1]!=7 or O.shape!=R.shape or not np.isfinite(R).all() or not np.isfinite(O).all() or (R[:,4:6]<=0).any():raise ValueError('finite same-shaped native Nx7 positive boxes')
    selected=[int(e['raw_index']) for e in endpoints]
    if len(selected)!=len(set(selected)) or any(i<0 or i>=len(R) for i in selected):raise ValueError('unique valid frozen raw endpoints')
    if not np.array_equal(R[:,[0,1,4,5,6]],O[:,[0,1,4,5,6]]):raise ValueError('O must retain frames/IDs/size/score/order')
    if np.any(np.abs((O[:,2:4]-R[:,2:4])/R[:,4:6])>.25+1e-12):raise ValueError('O exceeds registered .25 width/height displacement cap')
    other=np.ones(len(R),bool);other[selected]=False
    if not np.array_equal(R[other],O[other]):raise ValueError('O changed a noneligible observation')
    tagged=by_track(np.c_[R,np.arange(len(R))]);raw=tagged[:,-1].astype(np.int64);canonical=tagged[:,:7]
    sides=inside(R,polygon);osides=inside(O,polygon);runs=[];used=set();eligible=set(selected)
    for pid in sorted(set(canonical[:,1])):
        ix=raw[canonical[:,1]==pid];positions=[j for j,i in enumerate(ix) if int(i) in eligible]
        if not set(positions)<={0,len(ix)-1}:raise ValueError('frozen eligible row is internal to original canonical run')
        used.update(int(ix[j]) for j in positions)
        runs.append(dict(run_key=int(pid),frames=R[ix,0].astype(np.int64),sides=sides[ix],oracle_sides=osides[ix],eligible=positions))
    assert used==eligible
    result=envelope(runs,nframes);result['eligible_endpoints']=len(selected);result['canonical_runs']=len(runs)
    fixed=window_counts([],nframes);mutable=[]
    for run in runs:
        if run['eligible']:
            mutable.append(dict(frames=run['frames'].tolist(),sides=run['sides'].tolist(),eligible=run['eligible'],oracle_sides=run['oracle_sides'].tolist()))
        else:
            event=net_event(run['frames'],run['sides']);fixed+=window_counts([] if event is None else [event],nframes)
    result['replay_member']=dict(fixed_windows=fixed.tolist(),runs=mutable)
    return result

def bound_scene(members,polygon,nframes,gt_windows=None):
    env=[physical_envelope(m['rows'],m['endpoints'],polygon,nframes,oracle_rows=m.get('oracle_rows')) for m in members]
    evidence=dict(schema='endpoint-side-bound-evidence-v1',nframes=int(nframes),gt_windows=None if gt_windows is None else np.asarray(gt_windows).tolist(),members=[m['replay_member'] for m in env])
    return dict(**e3_bound(env,gt_windows),replay_evidence=evidence,member_menus=[dict(eligible_endpoints=m['eligible_endpoints'],canonical_runs=m['canonical_runs'],all_side=m['all_side']['runs'],owner_side=m['owner_side']['runs']) for m in env])

def replay_evidence(evidence):
    """Anonymous count replay; no repository/native/physical dependencies."""
    if evidence['schema']!='endpoint-side-bound-evidence-v1':raise ValueError('unknown evidence version')
    members=[];nf=evidence['nframes'];shape=window_counts([],nf).shape
    for member in evidence['members']:
        fixed=_integer_array(member['fixed_windows'],'fixed windows')
        if fixed.shape!=shape or (fixed<0).any():raise ValueError('invalid fixed clock')
        env=envelope(member['runs'],nf)
        env['baseline']+=fixed
        for mode in ('all_side','owner_side'):
            env[mode]['lower']+=fixed;env[mode]['upper']+=fixed
        members.append(env)
    return e3_bound(members,evidence['gt_windows'])

def pool_bounds(clips):
    """Pool clip integer numerators/denominators before W (e.g. W/G groups)."""
    if not clips or any(c['denominator'] is None for c in clips):raise ValueError('nonempty scored clips required')
    D=sum(c['denominator'] for c in clips)
    return dict(denominator=D,layers={mode:dict(lower_bound_numerator=sum(c['layers'][mode]['lower_bound_numerator'] for c in clips),W_lower_bound_percent=None if not D else 100*sum(c['layers'][mode]['lower_bound_numerator'] for c in clips)/D) for mode in ('all_side','owner_side')})

def selftest(official=False):
    # Both endpoints compose, with fixed internal observations and real dates.
    run=dict(frames=np.array([1,26,51]),sides=np.array([False,True,False]),eligible=[0,2],oracle_sides=np.array([True,True,True]))
    menu=run_menu(**run);assert {None if e is None else tuple(e) for e in menu['events']}=={None,(26,1),(51,-1)} and len(menu['choices'])==4
    singleton=dict(frames=np.array([2]),sides=np.array([False]),eligible=[0],oracle_sides=np.array([True]))
    sm=run_menu(**singleton);assert len(sm['choices'])==2 and sm['events']==[None]
    # One endpoint chooses an early or late in-event; mutually exclusive clocks.
    e=envelope([run],800);fixed=envelope([],800);gt=window_counts([(26,1),(51,-1)],800)
    lower=e3_bound([e,fixed,fixed],gt);assert lower['layers']['all_side']['lower_bound_numerator']<=lower['baseline_numerator']
    # An impossible target takes BOTH opposite net directions from one run.
    # Envelope includes it coordinatewise, although no one run can emit both.
    target=window_counts([(26,1),(51,-1)],800)
    lo=np.asarray(e['all_side']['lower']);hi=np.asarray(e['all_side']['upper'])
    assert ((lo<=target)&(target<=hi)).all()
    exact=min(int(np.abs(window_counts([] if event is None else [event],800)-target).sum()) for event in menu['events'])
    relaxed=int((np.maximum(lo-target,0)+np.maximum(target-hi,0)).sum());assert relaxed==0 and exact==30
    three=e3_bound([e,e,e],target)
    vectors=[window_counts([] if event is None else [event],800) for event in menu['events']]
    exact3=min(int(np.abs(sum(vectors[i] for i in selection)-3*target).sum()) for selection in product(range(len(vectors)),repeat=3))
    assert three['layers']['all_side']['lower_bound_numerator']==0 and exact3==90
    # Exhaust all length 1..5 side sequences and all legal endpoint bit edits.
    for n in range(1,6):
        frames=np.arange(n,dtype=np.int64)*25+1
        for bits in product((False,True),repeat=n):
            sides=np.asarray(bits,bool);eligible=sorted({0,n-1});m=run_menu(frames,sides,eligible)
            if official:
                from tools.study197.candidates import event_times,window_counts as native_counts
                polygon=[[0,0],[10,0],[10,10],[0,10]]
                for choice in m['choices']:
                    side=sides.copy()
                    for i,b in zip(eligible,choice['endpoint_bits']):side[i]=b
                    R=np.asarray([[int(f),1,3 if b else 13,3,2,2,1] for f,b in zip(frames,side)],float)
                    expected=[] if choice['event'] is None else [choice['event']]
                    assert np.array_equal(window_counts(expected,int(frames[-1])+7),native_counts(event_times(R,polygon),int(frames[-1])+7))
    if official:
        R=np.array([[1,1,-1.2,3,2,2,1],[26,1,3,3,2,2,1],[51,1,-1.2,3,2,2,1]],float);O=R.copy();O[[0,2],2]=-.8
        p=physical_envelope(R,[dict(raw_index=0),dict(raw_index=2)],polygon,800,oracle_rows=O)
        assert p['all_side']['runs'][0]['events']==menu['events']
        bound=bound_scene([dict(rows=R,endpoints=[dict(raw_index=0),dict(raw_index=2)],oracle_rows=O)]*3,polygon,800,gt)
        assert bound['denominator']==180 and bound['layers']['all_side']['lower_bound_numerator']==0
        assert replay_evidence(bound['replay_evidence'])=={k:v for k,v in bound.items() if k not in ('replay_evidence','member_menus')}
    print('PASS both-end joint; singleton shared bit; incompatible-window zero bound vs exact30; exhaustive side sequences;'+(' native clock/physical adapter' if official else ' pure synthetic'))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--selftest',action='store_true');p.add_argument('--official',action='store_true');p.add_argument('--repo',type=Path,default=Path.cwd());p.add_argument('--replay',type=Path);p.add_argument('--out',type=Path);args=p.parse_args()
    if args.selftest:
        if args.official:sys.path.insert(0,str(args.repo))
        selftest(args.official)
    elif args.replay:
        opener=gzip.open if args.replay.suffix=='.gz' else open
        with opener(args.replay,'rt',encoding='utf8') as f:evidence=json.load(f)
        result=replay_evidence(evidence);text=json.dumps(result,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n'
        if args.out:args.out.write_text(text,encoding='utf8')
        else:print(text,end='')
    else:p.error('--selftest or --replay required')
