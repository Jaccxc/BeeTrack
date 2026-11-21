"""Private §190 endpoint-edit envelope, CPU; selfcheck uses synthetic rows only.

One optional physical row per eligible end, no other row/identity edits.
Menus relax cross-track conflicts. Interval bounds relax coupling across windows.
Neither bound applies to arbitrary models or edits outside this family.
"""
from itertools import product
import json
import math
from pathlib import Path
import sys

import cv2
import numpy as np


def _integer(value, name):
    if isinstance(value,(bool,np.bool_)) or not math.isfinite(float(value)) or int(value)!=value:
        raise ValueError(f'{name} must be a finite integer')
    return int(value)


def _inside(row, polygon):
    return int(cv2.pointPolygonTest(np.asarray(polygon,np.float32),
               (float(row[2]+row[4]/2),float(row[3]+row[5]/2)),False)>=0)


def _events(frames,sides):
    direction=sides[-1]-sides[0]
    if not direction:
        return None
    return (int(next(frames[k] for k in range(len(frames)-1,0,-1)
                     if sides[k]-sides[k-1]==direction)),int(direction))


def segment_menus(canonical,candidates,polygon,nf,endpoint_rows=None):
    """Return {canonical_id: [None or (event_frame,+1/-1), ...]}.

    Candidates require B (canonical receiving track), kind=start/end,
    permitted_frames. Envelope (endpoint_rows=None) permits either side at
    every allowed date. Constrained mode takes {(B,kind): [actual MOT rows]};
    missing keys mean no admissible rows, never hallucinated GT coordinates.
    Original rows remain exactly present, including when both ends are edited.
    """
    nf=_integer(nf,'nf')
    if nf<1:
        raise ValueError('nf must be positive')
    rows=np.asarray(canonical,float)
    if rows.ndim!=2 or rows.shape[1]<6 or not np.isfinite(rows).all():
        raise ValueError('canonical must be finite MOT rows with at least6 columns')
    tracks={}
    for identity in np.unique(rows[:,1]):
        key=_integer(identity,'canonical ID')
        track=rows[rows[:,1]==identity]
        track=track[np.argsort(track[:,0],kind='stable')]
        frames=np.array([_integer(f,'frame') for f in track[:,0]])
        if np.any(frames<1) or np.any(frames>nf) or np.any(np.diff(frames)<=0) or np.any(np.diff(frames)>25):
            raise ValueError('canonical track must have unique ordered frames, within clip, gaps<=25')
        tracks[key]=(track,frames.tolist(),[_inside(r,polygon) for r in track])
    dates={}
    for candidate in candidates:
        identity=_integer(candidate['B'],'candidate B');kind=candidate['kind']
        if identity not in tracks or kind not in ('start','end'):
            raise ValueError('candidate must address an existing canonical endpoint')
        boundary=tracks[identity][1][0 if kind=='start' else -1]
        allowed=dates.setdefault((identity,kind),set())
        for value in candidate['permitted_frames']:
            frame=_integer(value,'permitted frame')
            gap=boundary-frame if kind=='start' else frame-boundary
            if not 1<=frame<=nf or not 1<=gap<=12:
                raise ValueError('candidate insertion must be within12 physical frames on the proper side')
            allowed.add(frame)
    menus={}
    for identity,(track,frames,sides) in tracks.items():
        choices=[]
        for kind in ('start','end'):
            allowed=dates.get((identity,kind),set())
            options={None}
            if endpoint_rows is None:
                options.update(product(sorted(allowed),(0,1)))
            else:
                for raw in endpoint_rows.get((identity,kind),[]):
                    row=np.asarray(raw,float)
                    if row.ndim!=1 or len(row)<6 or not np.isfinite(row).all():
                        raise ValueError('constrained endpoint rows must be actual finite MOT rows')
                    frame=_integer(row[0],'endpoint row frame')
                    if frame not in allowed:
                        raise ValueError('constrained row date is not eligible for this endpoint')
                    options.add((frame,_inside(row,polygon)))
            choices.append(options)
        outcomes=set()
        for before,after in product(*choices):
            ff=([before[0]] if before else [])+frames+([after[0]] if after else [])
            ss=([before[1]] if before else [])+sides+([after[1]] if after else [])
            outcomes.add(_events(ff,ss))
        menus[identity]=sorted(outcomes,key=lambda e:(e is not None,e or (0,0)))
    return menus


def phase_counts(events,nf):
    """Official30 offsets, partial head/tail windows, columns in/out."""
    nf=_integer(nf,'nf')
    if nf<1:
        raise ValueError('nf must be positive')
    valid=[]
    for event in events:
        frame,direction=(_integer(v,'event') for v in event)
        if not 1<=frame<=nf or direction not in (-1,1):
            raise ValueError('event outside clip or invalid direction')
        valid.append((frame,direction))
    result=[]
    for offset in range(30):
        shift=25*offset
        counts=np.zeros(((nf+shift+749)//750,2),dtype=np.int64)
        for frame,direction in valid:
            counts[(frame-1+shift)//750,0 if direction>0 else 1]+=1
        result.extend(counts.tolist())
    return np.asarray(result,dtype=np.int64)


def menu_extrema(menus,nf):
    """Sum each segment's0/1 per-cell minimum/maximum; at most1event/segment."""
    shape=phase_counts([],nf).shape
    lower=np.zeros(shape,dtype=np.int64);upper=lower.copy()
    for choices in menus.values():
        if not choices:
            raise ValueError('each segment menu must include at least the unmodified outcome')
        vectors=np.stack([phase_counts([] if event is None else [event],nf) for event in choices])
        lower+=vectors.min(axis=0);upper+=vectors.max(axis=0)
    return lower,upper


def interval_bound(lower,upper,gt_events,nf,streams=1):
    """For E3 pass sums of3 seeds' extrema, streams=3 (counts beforeL1).

    Missing GT annotation must not shrink all-side menus. If GT itself is not
    supplied, no error bound can be evaluated; envelope/extrema remain usable.
    Zero GT yields a raw numerator but undefined normalized percentage.
    """
    streams=_integer(streams,'streams')
    if streams<1:
        raise ValueError('streams must be positive')
    gt_events=list(gt_events);target=streams*phase_counts(gt_events,nf)
    lower=np.asarray(lower);upper=np.asarray(upper)
    if lower.shape!=target.shape or upper.shape!=target.shape or not np.isfinite(lower).all() or not np.isfinite(upper).all():
        raise ValueError('extrema shape/finite mismatch')
    if np.any(lower<0) or np.any(upper<lower) or np.any(lower!=np.floor(lower)) or np.any(upper!=np.floor(upper)):
        raise ValueError('extrema must be ordered nonnegative integers')
    distance=np.maximum(np.maximum(lower-target,target-upper),0).astype(np.int64)
    numerator=int(distance.sum());denominator=30*streams*len(gt_events)
    return dict(numerator=numerator,denominator=denominator,
                W_lower_percent=100*numerator/denominator if denominator else None,
                raw_mean_error_lower=numerator/(30*streams),streams=streams)


def selfcheck():
    sys.path.insert(0,str(Path.cwd()))
    from beetrack.p0 import events,window_err
    polygon=[[0,0],[100,0],[100,100],[0,100]]
    row=lambda f,s:[f,1,45 if s else 115,45,10,10,.9]
    checked=0
    for n in (1,2,3):
        for sides in product((0,1),repeat=n):
            frames=list(range(13,13+n));rows=np.array([row(f,s) for f,s in zip(frames,sides)],float)
            candidates=[dict(B=1,kind='start',permitted_frames=[1,12]),
                        dict(B=1,kind='end',permitted_frames=[frames[-1]+1,frames[-1]+12])]
            menu=segment_menus(rows,candidates,polygon,50)[1]
            brute=set()
            for before,after in product([None,(1,0),(1,1),(12,0),(12,1)],
                                         [None,(frames[-1]+1,0),(frames[-1]+1,1),(frames[-1]+12,0),(frames[-1]+12,1)]):
                physical=([row(*before)] if before else [])+rows.tolist()+([row(*after)] if after else [])
                E=events(physical,polygon)
                assert len(E)<=1
                brute.add(tuple(map(int,E[0])) if len(E) else None)
                checked+=1
            assert set(menu)==brute
            constrained={(1,'start'):[row(12,1)],(1,'end'):[row(frames[-1]+1,0)]}
            cmenu=segment_menus(rows,candidates,polygon,50,constrained)[1]
            assert set(cmenu)<=set(menu)
            native=events(rows,polygon)
            original=tuple(map(int,native[0])) if len(native) else None
            assert segment_menus(rows,[],polygon,50)[1]==[original]
            assert segment_menus(rows,candidates,polygon,50,{})[1]==[original]
    # Exhaust every global choice of two menus, verify per-cell enclosure andW.
    menus={1:[None,(25,1),(26,-1)],2:[(1,1),(50,-1)]};nf=51
    lo,hi=menu_extrema(menus,nf);G=[(20,1),(50,-1)]
    bound=interval_bound(lo,hi,G,nf)
    combinations=[]
    for options in product(*menus.values()):
        E=[e for e in options if e is not None];C=phase_counts(E,nf)
        assert (lo<=C).all() and (C<=hi).all()
        assert bound['raw_mean_error_lower']<=window_err(np.array(E).reshape(-1,2),np.array(G),nf)+1e-10
        combinations.append(C)
    e3=interval_bound(3*lo,3*hi,G,nf,streams=3)
    for a,b,c in product(combinations,repeat=3):
        actual=float(np.abs((a+b+c)/3-phase_counts(G,nf)).sum()/30)
        assert e3['raw_mean_error_lower']<=actual+1e-10
    for nf in (1,24,25,26,749,750,751,1600):
        E=np.array([(1,1),(nf,-1),(nf,-1)])
        assert np.isclose(phase_counts(E,nf).sum()/30,window_err(E,np.empty((0,2)),nf))
    print(json.dumps(dict(official_endpoint_compositions_checked=checked,global_menu_combinations=len(combinations),
                          E3_combinations=len(combinations)**3,all_passed=True)))


if __name__=='__main__':
    selfcheck()
