"""§114 prediction-only native and raw crossing enumeration. Run --self-check without hive data.

observations: [frame, x100, y100, w100, h100, [[global_raw,within_frame_raw],...]].
candidates: [pre_obs,post_obs,direction,native_split_ids,[[global_pre,global_post],...]].
native_events: [split_track_id,signed_last_crossing_frame,candidate_index].
Raw and table indices are zero-based. Frames are one-based. Direction is +1 in, -1 out.
"""
import argparse
from collections import defaultdict
from math import hypot, sqrt

import cv2
import numpy as np

from beetrack.evaluate import by_track
from beetrack.track import nms


def centi(value):
    # Mirrors write_mot: direct decimal formatting after five-column float32 tracker input.
    return int(f"{float(value):.2f}".replace(".",""))


def check_rows(values, nframes, width):
    a = np.asarray(values)
    if a.ndim!=2 or a.shape[1]<width or not np.isfinite(a[:,:width]).all():
        raise ValueError(f"Expected finite Nx{width} leading fields")
    frames = a[:,0]
    if np.any(frames!=np.floor(frames)) or np.any(frames<1) or np.any(frames>nframes):
        raise ValueError("Invalid frame number")
    return a


def enumerate_candidates(dets, mot, polygon, nframes):
    """Build compact crossing proposals from one detector and its byte-pinned native MOT."""
    if nframes<1:
        raise ValueError("Expected positive frame count")
    dets = check_rows(dets,nframes,6)
    mot = check_rows(mot,nframes,6)
    if np.any(mot[:,1]!=np.floor(mot[:,1])):
        raise ValueError("Invalid native track ID")
    poly = np.asarray(polygon,dtype=np.float32)
    if poly.ndim!=2 or poly.shape[1]!=2 or len(poly)<3 or not np.isfinite(poly).all():
        raise ValueError("Expected a finite entrance polygon")

    refs = defaultdict(list)
    high = defaultdict(list)
    geometry = {}

    def properties(key):
        if key not in geometry:
            _,x,y,w,h = key
            if min(w,h)<0:
                raise ValueError(f"Inverted serialized box {key}")
            # Match zone_tracks's MOT-loaded pixel arithmetic before OpenCV's float point test.
            center = x/100+(w/100)/2, y/100+(h/100)/2
            side = cv2.pointPolygonTest(poly,center,False)>=0
            geometry[key] = (2*x+w,2*y+h,sqrt(w*h),side)
        return geometry[key]

    frame_rows = defaultdict(list)
    for i,f in enumerate(dets[:,0]):
        frame_rows[int(f)].append(i)
    for f,indices in frame_rows.items():
        raw = np.asarray(dets[indices,1:6],dtype=np.float32)
        for local in nms(raw,.5):
            i = indices[local]
            x,y,x2,y2,score = raw[local]
            key = (f,centi(x),centi(y),centi(x2-x),centi(y2-y))
            refs[key].append([i,int(local)])
            if score>=.25:
                if properties(key)[2]<=0:
                    raise ValueError(f"Zero-area high-score raw endpoint {key}")
                high[f].append((i,key))

    # Suppressed raw rows never participate in lineage or raw generation. Multiple
    # matching survivors retain every reference; the lowest raw ID is canonical.
    native = defaultdict(set)
    native_events = []
    tracks = by_track(mot[:,:6])
    starts = np.unique(tracks[:,1],return_index=True)[1]
    for start,stop in zip(starts,[*starts[1:],len(tracks)]):
        segment = tracks[start:stop]
        track_id = int(segment[0,1])
        keys = [(int(f),centi(x),centi(y),centi(w),centi(h)) for f,_,x,y,w,h in segment]
        changes = []
        for a,b in zip(keys,keys[1:]):
            if b[0]<=a[0] or b[0]-a[0]>25:
                raise ValueError(f"Duplicated native track frame or incorrect gap split: {track_id}")
            if properties(a)[3]!=properties(b)[3]:
                if properties(a)[2]<=0 or properties(b)[2]<=0:
                    raise ValueError(f"Zero-area native crossing endpoint: {track_id} {a} {b}")
                if a not in refs or b not in refs:
                    raise ValueError(f"Native crossing has no canonical raw survivor: {track_id} {a} {b}")
                native[a,b].add(track_id)
                changes.append((a,b))
        if len(changes)%2:
            a,b = changes[-1]
            sign = 1 if properties(b)[3] else -1
            native_events.append((track_id,sign*b[0],(a,b)))

    raw_pairs = defaultdict(list)
    for f,starts_at in high.items():
        for dt in (1,2,3):
            others = high.get(f+dt,())
            for i,a in starts_at:
                x,y,size,inside = geometry[a]
                matches = []
                for j,b in others:
                    xb,yb,sizeb,insideb = geometry[b]
                    if inside==insideb:
                        continue
                    d = hypot(xb-x,yb-y)/(size+sizeb)
                    if d<=2*dt:
                        matches.append((d,j,b))
                for _,j,b in sorted(matches)[:2]:
                    raw_pairs[a,b].append([i,j])

    pairs = sorted(native.keys()|raw_pairs.keys())
    endpoints = sorted({k for pair in pairs for k in pair})
    obs_index = {k:i for i,k in enumerate(endpoints)}
    pair_index = {p:i for i,p in enumerate(pairs)}
    return {
        "observations":[[*k,sorted(refs[k])] for k in endpoints],
        "candidates":[[obs_index[a],obs_index[b],1 if geometry[b][3] else -1,
                       sorted(native.get((a,b),())),sorted(raw_pairs.get((a,b),()))]
                      for a,b in pairs],
        "native_events":[[t,f,pair_index[p]] for t,f,p in sorted(native_events)],
    }


def validate_108(result, predictions):
    """Cross-check §108 canonical event descriptors when supplied a record or its prediction table."""
    if isinstance(predictions,dict):
        predictions = predictions["predictions"]
    actual = []
    for tid,f,cid in result["native_events"]:
        pre,post,sign,native_ids,_ = result["candidates"][cid]
        frame,x,y,w,h,_ = result["observations"][post]
        if tid not in native_ids or f!=sign*frame or pre>=post:
            raise ValueError("Native event lineage or endpoint order changed")
        actual.append([tid,f,x/100,y/100,round(x/100+w/100,2),round(y/100+h/100,2)])
    if actual!=predictions:
        raise ValueError("§108 native split identity, canonical crossing time, direction, or box differs")


def self_check():
    poly = [(0,-100),(100,-100),(100,100),(0,100)]
    def det(f,cx,cy=0,w=2,h=2,p=.8):
        return [f,cx-w/2,cy-h/2,cx+w/2,cy+h/2,p]
    def rows(dets,ids):
        return np.asarray([[d[0],tid,d[1],d[2],d[3]-d[1],d[4]-d[2]]
                           for d,tid in zip(dets,ids)],dtype=float).reshape(-1,6)

    # Includes a boundary-centred box, an inside-outside-inside reversal (no event
    # for this segment), a 25-frame low-score native crossing, and ID reuse after 26.
    raw=[det(1,-1),det(2,0),det(3,-1),det(4,1),det(5,-1),det(30,1,p=.22),
         det(56,1),det(58,-1)]
    tracks=rows(raw,[4,4,4,7,7,7,7,7])
    result=enumerate_candidates(np.array(raw),tracks,poly,60)
    validate_108(result,[[100007,-58,-2.,-1.,0.,1.]])
    assert sum(4 in candidate[3] for candidate in result["candidates"])==2
    assert sum(7 in candidate[3] for candidate in result["candidates"])==2
    assert any(row[3]==[7] and row[4]==[] for row in result["candidates"])

    # Three equally normalized candidates. The first input row in the arrival
    # frame wins the tie; a suppressed box and a below-.25 target cannot compete.
    raw=[det(2,1,3),det(1,-1),det(2,1,-3),det(2,1,0),det(2,1,0,p=.3),det(2,.1,6,p=.249)]
    result=enumerate_candidates(np.array(raw),np.empty((0,6)),poly,3)
    links=[p for c in result["candidates"] for p in c[4]]
    assert sorted(links)==[[1,0],[1,3]]
    assert all(i not in (4,5) for o in result["observations"] for i,_ in o[-1])
    assert next(o[-1] for o in result["observations"] if o[0]==1)==[[1,0]]

    # Geometric distance threshold is inclusive; a positive-area box whose centre
    # rounds onto the boundary has the same side as when reloaded from write_mot.
    raw=[det(1,-2),det(2,2),det(1,-.004,p=.25),det(3,-1),det(4,1)]
    result=enumerate_candidates(np.array(raw),np.empty((0,6)),poly,4)
    links=[p for c in result["candidates"] for p in c[4]]
    assert [0,1] in links and [2,1] not in links and [2,3] in links
    assert enumerate_candidates(np.empty((0,6)),np.empty((0,6)),poly,2)=={
        "observations":[],"candidates":[],"native_events":[]}
    # Native caches can contain clipped zero-height survivors below .25. They must
    # see the same NMS, retain their provenance, and generate no high-score pairs.
    base=[det(1,-1),det(2,1)]
    after=enumerate_candidates(np.array(base+[det(1,0,h=0,p=.07)]),np.empty((0,6)),poly,2)
    assert after==enumerate_candidates(np.array(base),np.empty((0,6)),poly,2)
    print("Synthetic NMS, raw lineage, geometry, tie, reversal, gap, and native descriptor checks passed.")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-check",required=True,action="store_true")
    parser.parse_args()
    self_check()
