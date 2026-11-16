"""Official event F1 and thirty-offset window counts for one complete native MOT arm."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from beetrack.evaluate import f1
from tools.study103.swap_score import counts_from_events, window_error
from tools.study104.event_overlap import archive_arm, replay_arm, score_arm


FORMAT = 'temporal117-official-graph-v1'
OFFSETS = list(range(0,750,25))


def evidence_sha256(evidence):
    wire = (json.dumps(evidence,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
    return hashlib.sha256(wire).hexdigest()


def array_hash(array):
    a = np.ascontiguousarray(array,dtype='<f8')
    h = hashlib.sha256(json.dumps(list(a.shape),separators=(',',':')).encode()+b'\n')
    h.update(a.tobytes())
    return h.hexdigest()


def mot_array(rows,nframes):
    data = np.asarray(rows,dtype=float)
    if data.shape==(0,):
        data = data.reshape(0,6)
    if data.ndim!=2 or data.shape[1]<6 or not np.isfinite(data).all():
        raise ValueError('Expected finite MOT rows: frame,id,x,y,w,h[,additional columns]')
    if (np.any(data[:,0]<1) or np.any(data[:,0]>nframes)
            or not np.array_equal(data[:,:2],np.rint(data[:,:2])) or np.any(data[:,4:6]<0)):
        raise ValueError('Invalid MOT frame, ID or box extent')
    return data


def event_counts(table,nframes,width):
    ids, signed = set(), []
    for row in table:
        if (len(row)!=width or type(row[0]) is not int or row[0] in ids
                or type(row[1]) is not int or not 1<=abs(row[1])<=nframes):
            raise ValueError('Invalid signed event descriptor or repeated resolved event ID')
        if width==6 and (not np.isfinite(row[2:]).all() or row[4]<row[2] or row[5]<row[3]):
            raise ValueError('Invalid event box in archived predictions')
        ids.add(row[0])
        signed.append(row[1])
    events = np.array([[abs(t),1 if t>0 else -1] for t in signed],dtype=int).reshape(-1,2)
    counts = {'in':sum(t>0 for t in signed),'out':sum(t<0 for t in signed)}
    return counts, counts_from_events(events,nframes)


def replay_case(evidence):
    """Recalculate metrics from signed clips and all candidate edges with no frames or detector data."""
    n = evidence['nframes']
    if evidence['format']!=FORMAT or type(n) is not int or n<1:
        raise ValueError('Unknown evidence format or invalid clip length')
    truth, preds = evidence['gt'],evidence['predictions']
    gt_count,gt_bins = event_counts(truth,n,2)
    pr_count,pr_bins = event_counts(preds,n,6)
    gt_sign = {r[0]:r[1]>0 for r in truth}
    pr_sign = {r[0]:r[1]>0 for r in preds}
    scores = {}
    at05 = set()
    for threshold,key in (('0.5','edges05'),('0.3','edges03')):
        edges = evidence[key]
        pairs = {tuple(e) for e in edges}
        if (len(pairs)!=len(edges) or any(len(e)!=2 for e in pairs)
                or any(p not in pr_sign or g not in gt_sign or pr_sign[p]!=gt_sign[g] for p,g in pairs)):
            raise ValueError('Malformed, duplicate or direction-mismatched event edge')
        if threshold=='0.5':
            at05 = pairs
        elif not at05.issubset(pairs):
            raise ValueError('The strict graph contains an edge outside the relaxed graph')
        arm = replay_arm(dict(gt=truth,predictions=preds,edges=edges))
        tp = len(arm.matches)
        directions = {}
        for d in ('in','out'):
            hits = sum(arm.gt[g]==d for g in arm.matches.values())
            directions[d] = dict(tp=hits,fp=pr_count[d]-hits,fn=gt_count[d]-hits,
                                 f1=f1(hits,pr_count[d],gt_count[d]))
        scores[threshold] = dict(tp=tp,fp=len(preds)-tp,fn=len(truth)-tp,f1=f1(tp,len(preds),len(truth)),
                                 by_direction=directions,matching=[[p,g] for p,g in sorted(arm.matches.items())])
    diff = np.abs(pr_bins-gt_bins)
    sizes = [(n+shift+749)//750 for shift in OFFSETS]
    per_offset = [chunk.sum(axis=0).tolist() for chunk in np.split(diff,np.cumsum(sizes)[:-1])]
    numerator,denominator = int(diff.sum()),int(gt_bins.sum())
    if denominator!=len(OFFSETS)*len(truth):
        raise AssertionError('Window denominator differs from 30 times GT event count')
    return dict(w30_percent=window_error(gt_bins,pr_bins) if denominator else None,
                window=dict(width_frames=750,fps=25,offsets_frames=list(OFFSETS),bins_per_offset=sizes,
                            numerator=numerator,denominator=denominator,l1_by_offset_in_out=per_offset),
                gt_counts=gt_count,pred_counts=pr_count,bias={d:pr_count[d]-gt_count[d] for d in gt_count},
                iou=scores,evidence=evidence,evidence_sha256=evidence_sha256(evidence))


def score_case(gt,loaded_mot,polygon,nframes=1000,*,cached=None,cached_sha256=None):
    """Score repaired GT and serialized native MOT on the (1-based) local clip clock.

    All real boxes, including non-event tracks, compete for GT within every frame. The threshold-specific
    sparse graphs consider every native crossing in the track's final direction. Counts use the last one.
    Cache reuse requires a trusted SHA of the entire evidence and exact array/clock provenance.
    """
    if type(nframes) is not int or nframes<1:
        raise ValueError('Expected a positive integer frame count')
    gt,mot = mot_array(gt,nframes),mot_array(loaded_mot,nframes)
    poly = np.asarray(polygon,dtype=float)
    if poly.ndim!=2 or poly.shape[1]!=2 or len(poly)<3 or not np.isfinite(poly).all():
        raise ValueError('Expected a finite xy polygon with at least three vertices')
    fingerprints = dict(gt=array_hash(gt),mot=array_hash(mot),polygon=array_hash(poly))
    if cached is not None or cached_sha256 is not None:
        if (not isinstance(cached,dict) or not cached_sha256 or evidence_sha256(cached)!=cached_sha256
                or cached.get('inputs_sha256')!=fingerprints or cached.get('nframes')!=nframes):
            raise ValueError('Cached event graph does not match its pinned digest and all current inputs')
        return replay_case(cached)
    evidence = dict(format=FORMAT,nframes=nframes,inputs_sha256=fingerprints)
    for threshold,key in ((.5,'edges05'),(.3,'edges03')):
        archived = archive_arm(score_arm(gt[:,:6],mot[:,:6],poly,threshold))
        for name in ('gt','predictions'):
            if name in evidence and evidence[name]!=archived[name]:
                raise AssertionError('Event stream changed across frame IoU thresholds')
            evidence[name] = archived[name]
        evidence[key] = archived['edges']
    return replay_case(evidence)


def self_check():
    from tools.study103.swap_score import event_scores, vector_of_counts

    poly = [[0,0],[10,0],[10,30],[0,30]]

    def bee(tid,frames,xs,y=4):
        return [[f,tid,x,y,2,4] for f,x in zip(frames,xs,strict=True)]

    def check(gt,pr):
        result = score_case(gt,pr,poly)
        # Official independent raw-MOT scorer and the archived graph must agree.
        gt,pr = mot_array(gt,1000),mot_array(pr,1000)
        counts = vector_of_counts(gt,poly,1000)
        if counts.sum():
            assert result['w30_percent']==window_error(counts,vector_of_counts(pr,poly,1000))
        if len(gt) and len(pr):
            for thr,official in event_scores(gt,pr,poly).items():
                assert all(result['iou'][str(thr)][k]==v for k,v in official.items())
        assert replay_case(json.loads(json.dumps(result['evidence'])))==result
        return result

    gt = bee(7,[10,11],[11,8])
    aligned = bee(8,[10,11],[11,8])
    # A missed bee and a spatially disjoint extra bee in the same window cancel in W.
    mislocalized = check(gt,bee(8,[10,11],[11,8],y=20))
    assert mislocalized['w30_percent']==0
    assert all((s['tp'],s['fp'],s['fn'])==(0,1,1) for s in mislocalized['iou'].values())
    wrong = check(gt,bee(8,[10,11],[8,11]))
    assert wrong['w30_percent']==200 and wrong['bias']=={'in':-1,'out':1}
    boundary = check(bee(7,[749,750],[11,8]),bee(8,[750,751],[11,8]))
    assert boundary['window']['l1_by_offset_in_out']==[[2,0]]+[[0,0]]*29
    assert (boundary['window']['numerator'],boundary['window']['denominator'])==(2,30)

    repeated = check(bee(7,[1,2,3,12],[11,8,11,8]),bee(8,[1,2,3,23],[11,8,11,8]))
    assert repeated['evidence']['gt']==[[7,12]] and repeated['evidence']['predictions'][0][:2]==[8,23]
    assert repeated['iou']['0.5']['matching']==[[8,7]]
    relaxed = check(gt,bee(8,[10,11],[11,8],y=5.8))
    assert [relaxed['iou'][t]['tp'] for t in ('0.5','0.3')]==[0,1]

    # Non-event track 9 owns both boxed endpoints of 7, then returns to its starting side.
    competitor = aligned+bee(9,[10,11,12],[11,8,11])
    competitor[0][3] += .5
    competitor[1][3] += .5
    occluded = check(gt,competitor)
    assert occluded['iou']['0.5']['tp']==0 and occluded['evidence']['predictions'][0][0]==8
    split = check(gt+bee(7,[100,101],[8,11]),aligned+bee(8,[100,101],[8,11]))
    assert split['evidence']['gt']==[[7,11],[100007,-101]]
    assert split['iou']['0.5']['by_direction']['out']['tp']==1
    assert check([],[])['w30_percent'] is None
    assert check(gt,[])['w30_percent']==100
    assert check([],aligned)['window']['denominator']==0

    exact = check(gt,aligned)
    opts = dict(cached=exact['evidence'],cached_sha256=exact['evidence_sha256'])
    assert score_case(gt,aligned,poly,**opts)==exact
    changed_graph = {**exact['evidence'],'edges05':[]}
    for rows,options in ((competitor,opts),(aligned,{**opts,'cached':changed_graph})):
        try:
            score_case(gt,rows,poly,**options)
        except ValueError:
            pass
        else:
            raise AssertionError('Cache did not reject a changed MOT or graph')
    print('PASS: official F1 and W30, cancellation, offset boundary, direction, crossings, thresholds, '
          'non-event ownership, gap splits, empty arms, JSON replay and pinned cache.')


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command',required=True)
    sub.add_parser('self-check')
    replay = sub.add_parser('replay')
    replay.add_argument('evidence',type=Path,help='Single score_case output or its nested evidence object')
    args = parser.parse_args()
    if args.command=='self-check':
        self_check()
    else:
        record = json.loads(args.evidence.read_bytes())
        res = replay_case(record.get('evidence',record))
        if 'evidence_sha256' in record and record['evidence_sha256']!=res['evidence_sha256']:
            raise ValueError('Stored evidence digest does not match the replayed graph')
        print(json.dumps({k:v for k,v in res.items() if k!='evidence'},allow_nan=False))
