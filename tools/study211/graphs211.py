"""Pure peer-reference event agreement; never true-GT supervision.

Whole native scenes compete in official frame Hungarian matching. Event edges
use the official final direction, all crossings, tol5 and frame IoU .5.
Single-left deletion is a fixed-graph counterfactual, not removal of MOT rows.
"""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']='-1'
import numpy as np


def maximum_matching(pred_ids,reference_ids,edges,deleted=()):
    deleted=set(deleted);right=set(reference_ids);all_pred=set(pred_ids);neighbors={p:[] for p in pred_ids if p not in deleted}
    for p,r in sorted(set(map(tuple,edges))):
        if p not in all_pred or r not in right:raise ValueError('edge outside event vertices')
        if p in neighbors:neighbors[p].append(r)
    owner={}
    def visit(p,seen):
        for r in neighbors[p]:
            if r in seen:continue
            seen.add(r)
            if r not in owner or visit(owner[r],seen):owner[r]=p;return True
        return False
    for p in sorted(neighbors):visit(p,set())
    return [[p,r] for r,p in sorted(owner.items(),key=lambda x:(x[1],x[0]))]


def summarize_graph(pred_ids,reference_ids,edges,deletion=False):
    pred_ids=list(map(int,pred_ids));reference_ids=list(map(int,reference_ids));edges=sorted(set(map(tuple,edges)))
    matching=maximum_matching(pred_ids,reference_ids,edges);R=len(matching)
    linked={p for p,r in edges};flags=[]
    for p in sorted(pred_ids):
        flag=dict(canonical_id=p,has_neighbor=p in linked)
        if deletion:
            drop=R-len(maximum_matching(pred_ids,reference_ids,edges,[p]))
            if drop not in (0,1):raise AssertionError('single-left marginal outside0/1')
            flag['drop_left_marginal']=drop
        flags.append(flag)
    return dict(nPred=len(pred_ids),nReference=len(reference_ids),J=len(linked),R=R,
                matching=matching,flags=flags,edges=[list(e) for e in edges])


def event_vertices(rows,polygon):
    """Canonical vertices, including all crossings used by official legal edges."""
    from beetrack.evaluate import by_track,zone_tracks
    R=by_track(np.asarray(rows,float));zones=zone_tracks(R,polygon);vertices=[]
    for pid,(direction,start,end,cross) in zones.items():
        if not direction:continue
        indices=[j for j,d in cross if d==direction];last=indices[-1] if indices else end-1
        sign=1 if direction=='in' else -1
        vertices.append(dict(canonical_id=int(pid),frame=int(R[last,0]),direction=sign,
            signed_frame=sign*int(R[last,0]),box=R[last,2:6].tolist(),start_row=int(start),end_row=int(end),
            crossings=[dict(canonical_row=int(j),frame=int(R[j,0]),direction=1 if d=='in' else -1) for j,d in cross]))
    return R,zones,vertices


def peer_graph(reference,prediction,polygon,nframes,*,deletion=False,verify_official=True):
    """Reference is another prediction member, NOT actual annotation.

    Graph's GT-like right side is explicitly named reference throughout output.
    K calls deletion=True; state calls need only exact whole-graph J/R.
    """
    from beetrack.evaluate import frame_match,match_events
    from tools.study197.candidates import window_counts
    G,gz,reference_vertices=event_vertices(reference,polygon)
    P,pz,prediction_vertices=event_vertices(prediction,polygon)
    owners=frame_match(G,P,thr=.5);edges=[]
    for pv in prediction_vertices:
        p=pv['canonical_id'];direction,start,end,cross=pz[p]
        for j,d in cross:
            if d!=direction:continue
            for r in {int(owners[j-1]),int(owners[j])}-{-1}:
                if gz[r][0]==direction and any(rd==direction and abs(G[i,0]-P[j,0])<=5 for i,rd in gz[r][3]):
                    edges.append((p,r))
    summary=summarize_graph([v['canonical_id'] for v in prediction_vertices],
                            [v['canonical_id'] for v in reference_vertices],edges,deletion)
    if verify_official:
        official=match_events(np.asarray(reference,float),np.asarray(prediction,float),polygon,tol=5)
        if (summary['nPred'],summary['nReference'],summary['R'])!=(official['pred'],official['gt'],official['tp']):
            raise AssertionError('peer graph differs from actual official matcher')
    def bins(vertices):return window_counts([dict(frame=v['frame'],direction=v['direction']) for v in vertices],nframes).tolist()
    return dict(schema='match211-peer-graph-v1',**summary,prediction_vertices=prediction_vertices,
        reference_vertices=reference_vertices,pred_windows=bins(prediction_vertices),reference_windows=bins(reference_vertices),
        frame_iou=.5,tolerance_frames=5,all_crossings_in_final_direction=True,whole_scene_frame_competition=True,
        reference_is_prediction_not_true_GT=True,deletion_counterfactual='fixed legal graph left vertex only')


def k_event_features(graphs):
    if len(graphs)!=2:raise ValueError('two complete peer references required')
    tables=[{f['canonical_id']:f for f in g['flags']} for g in graphs]
    if tables[0].keys()!=tables[1].keys():raise ValueError('K event vertex identities differ between peer references')
    result=[]
    for p in sorted(tables[0]):
        left=sum(int(t[p]['has_neighbor']) for t in tables)/2
        exact=sum(t[p]['drop_left_marginal'] for t in tables)/2
        if not 0<=exact<=left<=1:raise AssertionError('peer marginal exceeds neighborhood')
        result.append(dict(canonical_id=p,x_L=left,x_X=exact))
    return result


def self_test():
    # Interactions cannot be expressed by adding single-vertex marginals.
    g=summarize_graph([1,2],[10],[(1,10),(2,10)],True)
    assert g['J']==2 and g['R']==1 and all(f['drop_left_marginal']==0 for f in g['flags'])
    assert g['R']-len(maximum_matching([1,2],[10],g['edges'],[1,2]))==1
    def row(f,p,x):return [f,p,x,0,2,2,.9]
    ref=np.array([row(1,10,-3),row(2,10,1)],float)
    pred=np.array([row(1,1,-3),row(2,1,1),row(1,2,30),row(2,2,31)],float)
    poly=[(0,-10),(50,-10),(50,10),(0,10)]
    value=peer_graph(ref,pred,poly,30,deletion=True)
    assert value['R']==1 and value['J']==1 and value['flags'][0]['drop_left_marginal']==1
    f=k_event_features([value,value]);assert f[0]['x_L']==f[0]['x_X']==1
    double_ref=np.array([row(f,10,x) for f,x in [(1,-3),(2,1),(3,-3),(4,1)]],float)
    double_pred=double_ref.copy();double_pred[:2,1]=1;double_pred[2:,1]=2
    interaction=peer_graph(double_ref,double_pred,poly,30,deletion=True)
    assert interaction['J']==2 and interaction['R']==1 and all(f['drop_left_marginal']==0 for f in interaction['flags'])
    empty=peer_graph(np.empty((0,7)),pred,poly,30,deletion=True)
    assert empty['R']==empty['J']==0
    return dict(passed=True,tests=['2left/1right interaction','whole native official graph agreement','K canonicalfeatures','empty reference'])


if __name__=='__main__':
    import argparse,json,sys
    from pathlib import Path
    p=argparse.ArgumentParser();p.add_argument('--repo',required=True,type=Path);p.add_argument('--self-test',required=True,action='store_true')
    args=p.parse_args();sys.path.insert(0,str(args.repo.resolve()));print(json.dumps(self_test()))
