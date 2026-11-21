"""Independent selection/event arithmetic; no field IO or producer imports."""
import math
import hashlib
import numpy as np
import cv2
MARKER="paired-rgb-v1"

def split_original(raw,nf=None):
    """Invalid rectangles retain their observation/date and may not bridge gaps."""
    a=np.asarray(raw,float)
    if a.ndim!=2 or a.shape[1]<6:raise ValueError('expected original Nx6+ GT')
    if not np.isfinite(a[:,:2]).all() or np.any(a[:,:2]!=np.floor(a[:,:2])) or np.any(a[:,0]<1) or np.any(a[:,1]<0) or np.any(a[:,1]>2**53-1):raise ValueError('invalid date/ID')
    if nf is not None and np.any(a[:,0]>nf):raise ValueError('invalid clock')
    if len(set(map(tuple,a[:,:2])))!=len(a):raise ValueError('duplicate(frame,id)')
    result=[];aliases=set()
    for identity in sorted(set(int(i) for i in a[:,1])):
        indices=np.flatnonzero(a[:,1]==identity);indices=indices[np.argsort(a[indices,0],kind='stable')];rows=a[indices]
        cuts=[0,*list(np.flatnonzero(np.diff(rows[:,0])>25)+1),len(rows)]
        for run,(lo,hi) in enumerate(zip(cuts,cuts[1:])):
            alias=identity+100000*run
            if alias in aliases or alias>2**53-1:raise ValueError('official alias collision')
            aliases.add(alias);rr=rows[lo:hi,:6]
            result.append(dict(original_id=identity,run=run,official_alias=alias,original_row_indices=indices[lo:hi].tolist(),rows={int(r[0]):r for r in rr},
                valid={int(r[0]):bool(np.isfinite(r[2:6]).all() and (r[4:6]>0).all()) for r in rr}))
    return result

def selected_pairs(raw,clip,nf=None):
    runs=split_original(raw,nf);close=[];eligible=[];allpairs=[]
    for i,left in enumerate(runs):
        for j in range(i+1,len(runs)):
            right=runs[j];best=None
            for t in sorted(set(left['rows'])&set(right['rows'])):
                if not left['valid'][t] or not right['valid'][t]:continue
                a,b=left['rows'][t],right['rows'][t]
                dx=float(a[2]+a[4]/2-b[2]-b[4]/2);dy=float(a[3]+a[5]/2-b[3]-b[5]/2)
                scale=(math.sqrt(float(a[4]*a[5]))+math.sqrt(float(b[4]*b[5])))/2
                distance=math.sqrt(dx*dx+dy*dy)/scale
                if best is None or distance<best[0]:best=(distance,t)
            base=dict(left=i,right=j,original_ids=[left['original_id'],right['original_id']],runs=[left['run'],right['run']])
            if best is None:
                allpairs.append(dict(base,state='no_valid_covisible_date'));continue
            if best[0]>2:
                allpairs.append(dict(base,state='closest_distance_over2',frame=best[1],distance=best[0]));continue
            distance,t=best;pair=dict(left=i,right=j,frame=t,distance=distance,
                original_ids=[left['original_id'],right['original_id']],runs=[left['run'],right['run']])
            close.append(pair)
            if not all(date in r['rows'] and r['valid'][date] for r in (left,right) for date in (t-4,t+4)):
                allpairs.append(dict(pair,state='fixed_nearest_date_endpoint_attrition'));continue
            string=f"{MARKER}|{clip}|{left['original_id']}|{left['run']}|{right['original_id']}|{right['run']}|{t}"
            pair=dict(pair,sha256=hashlib.sha256(string.encode('ascii')).hexdigest(),hash_input=string)
            eligible.append(pair)
            allpairs.append(dict(pair,state='eligible'))
    eligible.sort(key=lambda p:p['sha256'])
    return dict(runs=runs,pairs=allpairs,close=close,eligible=eligible,selected=eligible[:4])

def near_boundary(pair,runs,polygon):
    if polygon is None:return None
    middle=pair['frame'];poly=np.asarray(polygon,np.float32)
    for index in (pair['left'],pair['right']):
        run=runs[index];dates=sorted(run['rows'])
        for a,b in zip(dates,dates[1:]):
            if not middle-4<=a<b<=middle+4 or not run['valid'][a] or not run['valid'][b]:continue
            sides=[]
            for t in (a,b):
                row=run['rows'][t];sides.append(cv2.pointPolygonTest(poly,(float(row[2]+row[4]/2),float(row[3]+row[5]/2)),False)>=0)
            if sides[0]!=sides[1]:return True
    return False

def events_from_native(rows,polygon):
    """Original ID gap runs; boundary included; last matching-direction crossing."""
    rows=np.asarray(rows,float);poly=np.asarray(polygon,np.float32);events={}
    for native_id in sorted(set(rows[:,1])):
        track=rows[rows[:,1]==native_id];track=track[np.argsort(track[:,0],kind='stable')]
        cuts=[0,*list(np.flatnonzero(np.diff(track[:,0])>25)+1),len(track)]
        for run,(a,b) in enumerate(zip(cuts,cuts[1:])):
            r=track[a:b];sides=[]
            for x in r:sides.append(cv2.pointPolygonTest(poly,(float(x[2]+x[4]/2),float(x[3]+x[5]/2)),False)>=0)
            if sides[0]==sides[-1]:continue
            direction=1 if sides[-1] else -1
            dates=[int(r[j,0]) for j in range(1,len(r)) if sides[j]!=sides[j-1] and sides[j]==sides[-1]]
            identity=int(native_id+100000*run)
            assert identity not in events,'official gap alias collision'
            events[identity]=(dates[-1],direction)
    return events

def bins(events,nf):
    out=[]
    for offset in range(30):
        counts=[[0,0] for _ in range((nf+25*offset+749)//750)]
        for frame,direction in events:counts[(int(frame)-1+25*offset)//750][int(direction<0)]+=1
        out.extend(counts)
    return np.asarray(out,np.int64)

def maximum_matching(kept,edges):
    owner={}
    def augment(p,seen):
        for g in sorted(edges.get(p,())):
            if g in seen:continue
            seen.add(g)
            if g not in owner or augment(owner[g],seen):owner[g]=p;return True
        return False
    for p in sorted(kept):augment(p,set())
    return len(owner)
