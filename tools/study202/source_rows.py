"""Synthetic-only bridge to native linked-before-drop measurements.

No paths/data loaded on import. An eighth column carries identity, never bbox
matching. Raw detector provenance follows tracker output's actual detection index.
"""
import os
for _k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_k]='1'
os.environ['CUDA_VISIBLE_DEVICES']='-1'
import numpy as np


def linked_partition(s1,prob,s1_sources=None,expected_k=None):
    """Re-run actual link/THR, sorting and drop_short; preserve exact s1 indices.

    s1_sources optionally contains (source_sha256, original_D_row_index) per row.
    Returned source_s1_index is aligned with linked_rows; retained/orphans each
    have their own corresponding index arrays. No raw source is invented.
    """
    from beetrack.link import link
    from beetrack.p0 import THR,MIN_LEN,p0
    from beetrack.track import drop_short
    a=np.asarray(s1,float).reshape(-1,7)
    if not np.isfinite(a).all() or (a[:,4:6]<=0).any():raise ValueError('invalid s1')
    if len(np.unique(a[:,:2],axis=0))!=len(a):raise ValueError('duplicate s1 date/ID')
    if s1_sources is not None:
        if len(s1_sources)!=len(a) or len(set(map(tuple,s1_sources)))!=len(a):raise ValueError('missing/reused source identity')
    tagged=np.column_stack((a,np.arange(len(a))))
    linked=np.array(sorted(link(tagged.tolist(),THR,prob).tolist()),float).reshape(-1,8)
    native=np.array(sorted(link(a.tolist(),THR,prob).tolist()),float).reshape(-1,7)
    if not np.array_equal(linked[:,:7],native):raise ValueError('tag alters native link behavior')
    indices=linked[:,7].astype(int)
    if not np.array_equal(np.sort(indices),np.arange(len(a))):raise ValueError('link is not source-bijective')
    columns=[0,2,3,4,5,6]
    if not np.array_equal(linked[:,columns],a[indices][:,columns]):raise ValueError('link changes physical measurements')
    ids,counts=np.unique(native[:,1],return_counts=True)
    keep=np.isin(native[:,1],ids[counts>=MIN_LEN])
    retained=native[keep]
    official=np.asarray(drop_short(native.tolist(),MIN_LEN,0.0),float).reshape(-1,7)
    if not np.array_equal(retained,official) or not np.array_equal(official,np.asarray(p0(a,prob)).reshape(-1,7)):
        raise ValueError('retained is not native P0')
    if expected_k is not None and not np.array_equal(retained,np.asarray(expected_k).reshape(-1,7)):
        raise ValueError('sealed final K differs')
    orphans=[]
    for pid in ids[counts<MIN_LEN]:
        mask=native[:,1]==pid;ix=indices[mask]
        orphans.append(dict(rows=native[mask],source_s1_index=ix,
                            sources=None if s1_sources is None else [tuple(s1_sources[j]) for j in ix]))
    return dict(linked_rows=native,source_s1_index=indices,retained=retained,
                retained_s1_index=indices[keep],
                retained_sources=None if s1_sources is None else [tuple(s1_sources[j]) for j in indices[keep]],
                orphans=orphans,source_coverage='complete bijection; retained/orphans disjoint')


def stage1_sources(D,nframes,source_sha256,threshold=.4,expected_s1=None):
    """Mirror official PASS1 without modifying Tracker, tracing NMS and out[:,7].

    Full original D indices are retained before thresholding. Reject unsupported
    retro/late outputs, nonunique detection claims, or any native stage1 mismatch.
    """
    from beetrack.track import Tracker,Dets,nms
    from beetrack.p0 import stage1
    d=np.asarray(D,float)
    if d.ndim!=2 or d.shape[1]!=6 or not np.isfinite(d).all() or not source_sha256:
        raise ValueError('finite native D and source hash required')
    if (d[:,0]!=np.floor(d[:,0])).any() or (d[:,0]<1).any() or (d[:,0]>nframes).any():raise ValueError('invalid native clock')
    kept=np.flatnonzero(d[:,5]>=threshold)
    selected=d[kept]
    if (selected[:,3:5]<=selected[:,1:3]).any():raise ValueError('nonpositive retained detection')
    tracker=Tracker('link',link=0,min_len=0)
    if (not tracker.args.raw or not tracker.args.all or not tracker.args.obs or tracker.args.flight
            or any(getattr(tracker.args,k,0) for k in ('retro','hop','dup','interp'))):
        raise ValueError('unsupported PASS1 source semantics')
    rows=[];indices=[]
    for f in range(1,int(nframes)+1):
        local=np.flatnonzero(selected[:,0]==f)
        boxes=np.asarray(selected[local,1:6],np.float32).reshape(-1,5)
        order=nms(boxes,tracker.args.nms) if tracker.args.nms else np.arange(len(boxes))
        boxes=boxes[order];original=kept[local[order]]
        output=tracker.t.update(Dets(boxes))
        if len(output):
            output=np.asarray(output).copy()
            rawindex=output[:,7]
            if (rawindex!=np.floor(rawindex)).any() or (rawindex<0).any() or (rawindex>=len(boxes)).any():
                raise ValueError('untraceable tracker detection index')
            idx=rawindex.astype(int)
            if len(set(idx))!=len(idx):raise ValueError('tracker reuses current detection')
            output[:,:4]=boxes[idx,:4]
            for row,j in zip(output,idx):
                x1,y1,x2,y2,tid,score=row[:6]
                rows.append([f,tid,x1,y1,x2-x1,y2-y1,score]);indices.append(int(original[j]))
        if getattr(tracker.t,'late',[]):raise ValueError('late/retro output lacks exact original source index')
    a=np.asarray(rows,float).reshape(-1,7)
    reference=stage1(selected,int(nframes))
    if not np.array_equal(a,reference):raise ValueError('source-traced stage1 differs from official')
    if expected_s1 is not None and not np.array_equal(a,np.asarray(expected_s1).reshape(-1,7)):
        raise ValueError('sealed s1 differs')
    if len(set(indices))!=len(indices):raise ValueError('detector source reused over frames')
    return dict(rows=a,sources=[(source_sha256,j) for j in indices],
                raw_row_indices=np.asarray(indices,int),selected_indices=kept,
                native_stage1_exact=True,late_rows=0)


def self_test():
    # Exact duplicate geometry/score is deliberately NMS-suppressed: mapping is
    # decided by native NMS indices, never by searching identical measurements.
    D=np.array([[f,x,0,x+4,4,.9] for f in range(1,6) for x in [0,0]]+
               [[1,100,0,104,4,.9],[2,100,0,104,4,.9],[4,200,0,204,4,.9]],float)
    traced=stage1_sources(D,5,'SYNTHETIC-D')
    assert len(traced['rows'])==8 and sum(traced['rows'][:,0]==1)==2
    assert all(i%2==0 for i in traced['raw_row_indices'] if i<10)
    prob=lambda X:np.zeros(len(X))
    part=linked_partition(traced['rows'],prob,traced['sources'])
    assert len(part['retained'])==5 and sorted(len(o['rows']) for o in part['orphans'])==[1,2]
    assert len(set(part['source_s1_index']))==8
    # Two physically identical rows with different IDs remain distinguishable
    # through the link bridge, even when they cannot co-survive detector NMS.
    s1=np.array([[1,1,0,0,4,4,.9],[1,2,0,0,4,4,.9],[2,2,0,0,4,4,.9]],float)
    p=linked_partition(s1,prob,[('same',i) for i in range(3)])
    assert sorted(p['source_s1_index'])==[0,1,2]
    # Actual positive link changes IDs but never source identity.
    s=np.array([[1,1,0,0,4,4,.9],[2,2,0,0,4,4,.9],[3,2,0,0,4,4,.9]],float)
    q=linked_partition(s,lambda X:np.ones(len(X)))
    assert len(q['retained'])==3 and not q['orphans']
    try:linked_partition(s1,prob,[('same',0)]*3)
    except ValueError:pass
    else:raise AssertionError('duplicate source accepted')
    return 'PASS: real Tracker/NMS first-frame output; exact native stage1 and K; 1/2-row orphans; duplicate boxes and source-index bijection; positive-link ID change'


if __name__=='__main__':
    import argparse,sys
    from pathlib import Path
    p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);p.add_argument('--self-test',action='store_true',required=True)
    a=p.parse_args();sys.path.insert(0,str(a.repo.resolve()));print(self_test())
