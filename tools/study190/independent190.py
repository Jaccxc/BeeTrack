"""Independent section190 synthetic and sealed-output checks; never runs field generation."""
import itertools
import json
import numpy as np
import cv2
from collections import Counter,defaultdict
from pathlib import Path
import hashlib


def physical_event(rows, polygon):
    rows=sorted(rows,key=lambda r:r[0])
    side=[cv2.pointPolygonTest(np.asarray(polygon,np.float32),
          (float(r[2]+r[4]/2),float(r[3]+r[5]/2)),False)>=0 for r in rows]
    if side[0]==side[-1]:return None
    direction=1 if side[-1] else -1
    dates=[int(rows[j][0]) for j in range(1,len(rows)) if side[j]!=side[j-1] and side[j]==side[-1]]
    return dates[-1],direction


def windows(events,nf):
    out=[]
    for o in range(30):
        nb=(nf+25*o+749)//750
        phase=[[0,0] for _ in range(nb)]
        for f,d in events:phase[int((f+25*o-1)//750)][int(d<0)]+=1
        out.extend(phase)
    return np.array(out,dtype=np.int64)


def exhaustive_menus(rows,candidates,polygon,nf,endpoint_rows=None):
    """Physically enumerate both optional endpoint observations, without production menu helpers."""
    result={}
    for identity in np.unique(rows[:,1]):
        track=rows[rows[:,1]==identity]
        records=[(int(r[0]),cv2.pointPolygonTest(np.asarray(polygon,np.float32),
                 (float(r[2]+r[4]/2),float(r[3]+r[5]/2)),False)>=0) for r in track]
        dates={'start':set(),'end':set()}
        for c in candidates:
            if c['B']==identity:dates[c['kind']].update(c['permitted_frames'])
        choices={}
        for kind in dates:
            choices[kind]=[None]+[(f,inside)
                for f in sorted(dates[kind]) for inside in (False,True)]
            if endpoint_rows is not None:
                choices[kind]=[None]+[(int(r[0]),cv2.pointPolygonTest(np.asarray(polygon,np.float32),
                    (float(r[2]+r[4]/2),float(r[3]+r[5]/2)),False)>=0) for r in endpoint_rows.get((int(identity),kind),[])]
        result[int(identity)]=set()
        for pre,post in itertools.product(choices['start'],choices['end']):
            physical=sorted(records+([] if pre is None else [pre])+([] if post is None else [post]))
            if physical[0][1]==physical[-1][1]:
                result[int(identity)].add(None)
            else:
                d=1 if physical[-1][1] else -1
                crossings=[b[0] for a,b in zip(physical,physical[1:]) if b[1]!=a[1] and b[1]==physical[-1][1]]
                result[int(identity)].add((crossings[-1],d))
    return result


def compare_envelope(rows,candidates,polygon,nf,endpoint_rows=None):
    import bounds190 as B
    brute=exhaustive_menus(rows,candidates,polygon,nf,endpoint_rows)
    production=B.segment_menus(rows,candidates,polygon,nf,endpoint_rows)
    assert all(brute[i]==set(production[i]) for i in brute)
    lower=windows([],nf);upper=lower.copy()
    for options in brute.values():
        vectors=[windows([] if e is None else [e],nf) for e in options]
        lower+=np.minimum.reduce(vectors);upper+=np.maximum.reduce(vectors)
    lo,hi=B.menu_extrema(production,nf)
    assert np.array_equal(lower,lo) and np.array_equal(upper,hi)
    return brute,lower,upper


def audit_support(raw,canonical,dets,gt,source,candidates,polygon,label_report):
    """Recompute source ambiguity, official-order anchors, missed boxes and earliest/latest choices."""
    from beetrack.evaluate import by_track,frame_match
    G=by_track(gt)
    key=lambda r:(int(r[0]),*(f'{v:.2f}' for v in r[2:6]))
    physical=defaultdict(list)
    for i,r in enumerate(source):physical[key(r)].append(i)
    counts=Counter((int(r[0]),int(r[1])) for r in source)
    uncertain=set();source_row={}
    for i,r in enumerate(G):
        matches=physical[key(r)];pair=(int(r[0]),int(r[1]))
        if len(matches)!=1 or counts[(int(source[matches[0],0]),int(source[matches[0],1]))]!=1:uncertain.add(pair)
        else:source_row[i]=matches[0]
    index=defaultdict(list)
    for i,r in enumerate(G):index[(int(r[0]),int(r[1]))].append(i)
    uncertain|={p for p,ii in index.items() if len(ii)!=1}
    ordered=by_track(np.column_stack((raw,np.arange(len(raw)))))
    owners=np.empty(len(raw),int)
    owners[ordered[:,7].astype(int)]=frame_match(G,ordered)
    DD=np.column_stack((dets[:,0],np.arange(len(dets)),dets[:,1:3],dets[:,3:5]-dets[:,1:3]))
    observed={(int(r[0]),int(g)) for r,g in zip(dets,frame_match(G,DD)) if g>=0}
    inside=lambda r:cv2.pointPolygonTest(np.asarray(polygon,np.float32),(float(r[2]+r[4]/2),float(r[3]+r[5]/2)),False)>=0
    actual={r['candidate_id']:r for r in label_report['candidates']};proposals=[];checked=0
    for c in candidates:
        owners_ab={};ambiguous=False
        for who in ('A','B'):
            ii=[int(i) for i in c['row_indices'][who] if canonical[int(i),0] in c['anchor_frames']]
            oo=[int(owners[i]) for i in ii]
            if len(ii)!=2 or len(set(oo))!=1 or -1 in oo or any((int(canonical[i,0]),g) in uncertain for i,g in zip(ii,oo)):ambiguous=True;break
            owners_ab[who]=oo[0]
        r=actual[c['candidate_id']]
        if ambiguous:
            assert r['status'] in ('unknown','excluded');continue
        if owners_ab['A']==owners_ab['B']:
            assert r['status']=='same_bee';continue
        track=canonical[canonical[:,1]==c['B']]
        endpoint=track[np.argmin(track[:,0]) if c['kind']=='start' else np.argmax(track[:,0])]
        eligible=[];all_qualified={}
        for f in c['permitted_frames']:
            pair=(int(f),owners_ab['B']);ii=index[pair]
            if len(ii)!=1 or pair in uncertain or ii[0] not in source_row:continue
            sr=source_row[ii[0]];row=source[sr]
            if np.linalg.norm(row[2:4]+row[4:6]/2-np.array(c['center']))>c['radius'] or pair in observed:continue
            checked+=1
            all_qualified[int(f)]=(sr,row)
            if inside(row)!=inside(endpoint):eligible.append((int(f),sr,row))
        confirmed={s['frame']:s for s in r['support'] if s['status']=='confirmed'}
        assert set(confirmed)==set(all_qualified)
        for f,(sr,row) in all_qualified.items():
            assert confirmed[f]['source_row']==sr
            assert np.array_equal(np.array(confirmed[f]['row'])[2:6],row[2:6])
        if eligible:
            chosen=min(eligible,key=lambda z:(z[0] if c['kind']=='start' else -z[0],int(z[2][1]),z[1]))
            assert r['chosen']['source_row']==chosen[1] and r['chosen']['frame']==chosen[0]
            assert r['chosen']['gt_id']==owners_ab['B']
            assert np.array_equal(np.array(r['chosen']['row'])[2:6],chosen[2][2:6])
            proposals.append((c['candidate_id'],int(c['B']),chosen[0],owners_ab['B']))
        else:assert r['chosen'] is None
    usage=Counter((f,g) for _,b,f,g in proposals);dest=Counter((b,f) for _,b,f,g in proposals)
    for cid,b,f,g in proposals:
        conflict=usage[f,g]>1 or dest[b,f]>1 or np.any((canonical[:,0]==f)&(canonical[:,1]==b))
        assert actual[cid]['witness_status']==('conflict_dropped' if conflict else 'added')
    return dict(qualified_support_rows=checked,proposals=len(proposals),source_ambiguous_gt_pairs=len(uncertain))


def audit_candidates(rows,candidates,nf=3950):
    """Independently check every frozen candidate's geometric/date conditions (not search completeness)."""
    centers=rows[:,2:4]+rows[:,4:6]/2
    byframe={int(f):np.flatnonzero(rows[:,0]==f) for f in np.unique(rows[:,0])}
    for c in candidates:
        sign=1 if c['kind']=='start' else -1;f=int(c['boundary_frame'])
        B=rows[rows[:,1]==c['B']]
        assert f==(int(B[:,0].min()) if sign==1 else int(B[:,0].max()))
        aa=[int(i) for i in c['row_indices']['A']];bb=[int(i) for i in c['row_indices']['B']]
        assert rows[aa,0].tolist()==[f-sign,f,f+sign] and rows[bb,0].tolist()==[f,f+sign]
        assert np.all(rows[aa,1]==c['A']) and np.all(rows[bb,1]==c['B']) and c['A']!=c['B']
        body=float(np.median(np.sqrt(rows[[aa[1],aa[2],*bb],4]*rows[[aa[1],aa[2],*bb],5])))
        assert abs(body-c['L'])<1e-10 and abs(2*body-c['radius'])<1e-10
        for a,b in zip(aa[1:],bb):
            dist=np.linalg.norm(centers[a]-centers[b]);assert .5*body<=dist<=2*body
            lo=np.maximum(rows[a,2:4],rows[b,2:4]);hi=np.minimum(rows[a,2:4]+rows[a,4:6],rows[b,2:4]+rows[b,4:6])
            inter=np.maximum(hi-lo,0).prod();assert inter/(rows[a,4]*rows[a,5]+rows[b,4]*rows[b,5]-inter)<.5
            others=[i for i in byframe[int(rows[a,0])] if rows[i,1] not in (c['A'],c['B'])]
            assert all(min(np.linalg.norm(centers[i]-centers[a]),np.linalg.norm(centers[i]-centers[b]))>2*body for i in others)
        prev=[int(i) for i in byframe[f-sign] if min(np.linalg.norm(centers[i]-centers[aa[1]]),np.linalg.norm(centers[i]-centers[bb[0]]))<=2*body]
        assert prev==[aa[0]]
        assert c['permitted_frames']==[f-sign*d for d in range(1,13) if 1<=f-sign*d<=nf]
        assert np.array_equal(np.array(c['center']),centers[bb[0]])
    return len(candidates)


def synthetic():
    polygon=[[0,0],[100,0],[100,100],[0,100]]
    rng=np.random.default_rng(190061)
    cases=0
    for n in (3,4,8):
        for _ in range(12):
            frames=13+np.cumsum(rng.integers(1,20,n))
            nf=int(frames[-1]+12)
            rows=np.array([[f,1,45 if s else 115,45,10,10,.9] for f,s in zip(frames,rng.integers(0,2,n))],float)
            c=[dict(B=1,kind='start',permitted_frames=list(range(int(frames[0])-12,int(frames[0])))),
               dict(B=1,kind='end',permitted_frames=list(range(int(frames[-1])+1,nf+1)))]
            compare_envelope(rows,c,polygon,nf);cases+=1
    # Ambiguous two-end edit: either endpoint alone can miss a joint event removal/change.
    rows=np.array([[20,1,115,45,10,10,1],[21,1,45,45,10,10,1],[22,1,115,45,10,10,1]],float)
    c=[dict(B=1,kind='start',permitted_frames=[8,19]),dict(B=1,kind='end',permitted_frames=[23,34])]
    menus,lo,hi=compare_envelope(rows,c,polygon,34)
    return dict(random_physical_track_envelopes=cases,joint_both_end_example=sorted([str(e) for e in menus[1]]),passed=True)


def field(out):
    """Explicitly called only after root authorizes sealed-score inspection."""
    from beetrack import p0 as P
    from beetrack.evaluate import by_track,match_events
    from beetrack.common import gt_slice,split_specs
    from tools.study103.swap_score import event_scores
    out=Path(out);score=out/'score';read=lambda p:json.loads(p.read_text())
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    assert sha(score/'result_hashes.json')==(score/'SEALED.sha256').read_text().strip()
    for name,digest in read(score/'result_hashes.json').items():assert sha(score/name)==digest,name
    assert sha(out/'manifest.json')==(out/'ALL_FROZEN.sha256').read_text().strip()
    summary=read(score/'summary.json');cells={};gtids={33:set(),35:set()};checks=[]
    for h in (33,35):
        clip=f'BEE24-{h}-head';G=P.gt('dev',clip);poly=P.ZONE[clip]
        src,first,last=next((s,a,b) for e,s,a,b in split_specs('dev') if e==clip)
        source=gt_slice(src,first,last)[:,:6]
        for seed in range(3,9):
            folder=out/f'h{h}_s{seed}';dest=score/f'h{h}_s{seed}';receipt=read(folder/'receipt.json')
            for name,digest in receipt['files'].items():assert sha(folder/name)==digest
            raw=np.load(folder/'mot.npy');canonical=np.load(folder/'canonical.npy');dets=np.load(folder/'dets.npy')
            cs=read(folder/'candidates.json')['candidates'];labels=read(dest/'labels.json');saved=read(dest/'scores.json');bounds=read(dest/'bounds.json')
            audit_candidates(canonical,cs)
            witness=np.load(dest/'witness.npy')
            assert np.array_equal(witness[:len(canonical)],canonical)
            assert np.array_equal(raw[:,[0,2,3,4,5,6]],canonical[:,[0,2,3,4,5,6]])
            added=witness[len(canonical):]
            assert len(set(map(tuple,witness[:,:2])))==len(witness)
            assert len(added)==sum(r.get('witness_status')=='added' for r in labels['candidates'])
            aud=audit_support(raw,canonical,dets,G,source,cs,poly,labels)
            menus,lo,hi=compare_envelope(canonical,cs,poly,3950)
            assert all(set(menus[int(k)])=={None if e is None else tuple(e) for e in v} for k,v in bounds['universal_all_side']['menus'].items())
            assert np.array_equal(lo,bounds['universal_all_side']['lower']) and np.array_equal(hi,bounds['universal_all_side']['upper'])
            supports=defaultdict(list)
            for item in labels['constrained_menu']:supports[item['B'],item['kind']].extend(item['rows'])
            limited,llo,lhi=compare_envelope(canonical,cs,poly,3950,supports)
            assert all(set(limited[int(k)])=={None if e is None else tuple(e) for e in v} for k,v in bounds['gt_evidenced']['menus'].items())
            assert np.array_equal(llo,bounds['gt_evidenced']['lower']) and np.array_equal(lhi,bounds['gt_evidenced']['upper'])
            gt_events=P.events(G,poly).astype(int).tolist();gc=windows(gt_events,3950)
            count={}
            for arm in ('base','witness'):
                rows=np.loadtxt(dest/f'{arm}.txt',delimiter=',',ndmin=2)
                E=P.events(rows,poly).astype(int).tolist();pc=windows(E,3950);count[arm]=pc
                assert np.array_equal(pc,saved[arm]['integer_bins']['pred']) and np.array_equal(gc,saved[arm]['integer_bins']['gt'])
                W=100*int(np.abs(pc-gc).sum())/(30*len(gt_events))
                assert abs(W-saved[arm]['w30_percent'])<1e-10
                official=match_events(G,rows,poly)
                scores=event_scores(G,rows,poly)
                assert official['tp']==saved[arm]['iou']['0.5']['tp']
                for t in (.5,.3):
                    for k,v in scores[t].items():assert abs(v-saved[arm]['iou'][str(t)][k])<1e-10
            # Native baseline must have the identical official graph metrics.
            native=event_scores(G,raw,poly)
            assert all(native[t][k]==saved['base']['iou'][str(t)][k] for t in (.5,.3) for k in native[t])
            assert np.array_equal(windows(P.events(raw,poly).astype(int),3950),count['base'])
            changed={}
            for identity in np.unique(canonical[:,1]):
                before=physical_event(canonical[canonical[:,1]==identity].tolist(),poly)
                after=physical_event(witness[witness[:,1]==identity].tolist(),poly)
                if before!=after:changed[int(identity)]=(before,after)
            assert set(changed)==set(labels['event_changed_segments'])
            these={r['query_gt_id'] for r in labels['candidates'] if r.get('witness_status')=='added' and r['B'] in changed}
            assert these==set(labels['event_changed_query_gt_ids']);gtids[h].update(these)
            for family in bounds:
                l=np.asarray(bounds[family]['lower']);u=np.asarray(bounds[family]['upper'])
                num=int(np.maximum(0,np.maximum(l-gc,gc-u)).sum())
                assert num==bounds[family]['bound']['numerator']
            cells[h,seed]=(count,gc,lo,hi,bounds)
            checks.append(dict(h=h,seed=seed,candidates=len(cs),additions=len(added),changed=len(changed),**aud))
            print('INDEPENDENT_PASS',h,seed,flush=True)
    for h in (33,35):
        assert sorted(gtids[h])==summary['gate'][str(h)]['changed_gt_segment_ids']
        for group,seeds in [('345',(3,4,5)),('678',(6,7,8))]:
            gc=cells[h,seeds[0]][1]
            for arm in ('base','witness'):
                total=sum(cells[h,s][0][arm] for s in seeds)
                n=int(np.abs(total-3*gc).sum());d=int(3*gc.sum())
                record=summary['aggregation'][str(h)][arm]['E3'][group]
                assert n==record['numerator'] and d==record['denominator']
            for family in ('universal_all_side','gt_evidenced'):
                l=sum(np.array(cells[h,s][4][family]['lower']) for s in seeds)
                u=sum(np.array(cells[h,s][4][family]['upper']) for s in seeds)
                n=int(np.maximum(0,np.maximum(l-3*gc,3*gc-u)).sum())
                record=summary['bound_maximum_gains'][str(h)][family]['E3'][group]
                assert n==record['numerator'] and int(3*gc.sum())==record['denominator']
    return dict(cells=checks,all_passed=True,score_seal=sha(score/'result_hashes.json'))


if __name__=='__main__':
    import argparse,sys
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo',type=Path,default=Path.cwd())
    parser.add_argument('--out',type=Path)
    parser.add_argument('--verify-sealed',action='store_true')
    args=parser.parse_args()
    sys.path.insert(0,str(args.repo.resolve()))
    if args.verify_sealed:
        if args.out is None:parser.error('--verify-sealed requires --out')
        result=field(args.out)
        report=args.out/'independent190.json'
        if report.exists():raise FileExistsError(report)
        report.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    else:result=synthetic()
    print(json.dumps(result,indent=2))
