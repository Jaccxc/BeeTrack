"""Independent pair-texture checks. Default synthetic; no field IO at import."""
import os
for _name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[_name]='1'
os.environ['CUDA_VISIBLE_DEVICES']='-1'
import hashlib
import json
from pathlib import Path
import numpy as np
import cv2
cv2.setNumThreads(1)


def crop_bounds(image_shape,box,inset=.25):
    """Select pixel centres in half-open physical rectangle, reject overflow."""
    x,y,w,h=map(float,box)
    if not np.isfinite([x,y,w,h]).all() or min(w,h)<=0:return None
    # Independent arithmetic formulation, no production crop helper.
    left=int(np.ceil(x+inset*w-.5));right=int(np.ceil(x+(1-inset)*w-.5))
    top=int(np.ceil(y+inset*h-.5));bottom=int(np.ceil(y+(1-inset)*h-.5))
    if not (0<=left<right<=image_shape[1] and 0<=top<bottom<=image_shape[0]):return None
    return left,top,right,bottom


def central_crop(rgb,box):
    bounds=crop_bounds(rgb.shape,box)
    if bounds is None:return None
    l,t,r,b=bounds
    return np.asarray(rgb[t:b,l:r],float)/255


def color_six(crop):
    """Population moments, independently summed over original RGB pixels."""
    pixels=crop.reshape(-1,3)
    mean=np.sum(pixels,axis=0)/len(pixels)
    std=np.sqrt(np.sum((pixels-mean)**2,axis=0)/len(pixels))
    return np.r_[mean,std]


def paired_costs(four):
    p,q,r,s=np.asarray(four,float)
    distance=lambda a,b:float(sum(abs(a-b))/len(a))
    aligned=(distance(p,r)+distance(q,s))/2
    exchanged=(distance(p,s)+distance(q,r))/2
    return dict(aligned=aligned,exchanged=exchanged,margin=exchanged-aligned)


def sample_grid(crop):
    ys=[int((i+.5)*len(crop)//16) for i in range(16)]
    xs=[int((j+.5)*crop.shape[1]//16) for j in range(16)]
    return np.array([[crop[y,x] for x in xs] for y in ys])


def descriptor12(grid):
    assert np.asarray(grid).shape==(16,16,3)
    color=color_six(grid)
    y=(grid[:,:,0]+2*grid[:,:,1]+grid[:,:,2])/4
    mu=float(y.sum()/256);sd=float(np.sqrt(((y-mu)**2).sum()/256))
    z=(y-mu)/sd if sd>1e-12 else np.zeros((16,16))
    texture=[]
    for radius in (1,2,4):
        h=sum(z[i,j]*z[i,j+radius] for i in range(16) for j in range(16-radius))/(16*(16-radius))
        v=sum(z[i,j]*z[i+radius,j] for i in range(16-radius) for j in range(16))/(16*(16-radius))
        texture.extend(((h+v)/2,abs(h-v)))
    return np.r_[color,texture]


def shuffled_grid(grid,clip,actual_frame,bounds,rep):
    l,t,r,b=bounds
    prefix=f'texture-null-v1|{clip}|{actual_frame}|{l}|{t}|{r}|{b}|{rep}|'
    indices=sorted(range(256),key=lambda i:(hashlib.sha256((prefix+str(i)).encode('utf-8')).digest(),i))
    return grid.reshape(256,3)[indices].reshape(16,16,3)


MODEL_COLUMNS={'P':list(range(3)),'C':list(range(9)),'T':list(range(15)),'V':list(range(9,15))}


def difference_vector(boxes,descriptors):
    boxes=np.asarray(boxes,float);descriptors=np.asarray(descriptors,float)
    assert boxes.shape==(4,4) and descriptors.shape==(4,12)
    scale=float(np.sqrt(boxes[:,2]*boxes[:,3]).mean())
    def edge(i,j):
        a,b=boxes[i],boxes[j]
        ca=a[:2]+a[2:]/2;cb=b[:2]+b[2:]/2
        geometry=[float(np.linalg.norm(ca-cb))/scale,
                  abs(float(np.log(a[2]*a[3])-np.log(b[2]*b[3]))),
                  abs(float(np.log(a[2]/a[3])-np.log(b[2]/b[3])))]
        return np.r_[geometry,abs(descriptors[i]-descriptors[j])]
    return edge(0,3)+edge(1,2)-edge(0,2)-edge(1,3)


def ridge_by_video(records,columns):
    videos=sorted({r['clip'] for r in records})
    assert videos
    x=np.array([r['difference'] for r in records])[:,columns]
    counts={v:sum(r['clip']==v for r in records) for v in videos}
    weight=np.array([1/len(videos)/counts[r['clip']] for r in records])
    rms=np.sqrt((weight[:,None]*x*x).sum(axis=0));rms[rms==0]=1
    z=x/rms
    coef=np.linalg.solve(z.T@(weight[:,None]*z)+.2*np.eye(len(columns)),z.T@weight)
    gradient=2*(z.T@(weight*(z@coef-1))+.2*coef)
    assert abs(weight.sum()-1)<1e-12 and max(abs(gradient),default=0)<1e-10
    return dict(scale=rms,coef=coef,gradient=gradient,columns=columns)


def blind_decision(difference,digest,fitted):
    flip=hashlib.sha256(('texture-order-v1|'+digest).encode('utf-8')).digest()[0]&1
    sign=-1 if flip else 1
    score=float((sign*np.asarray(difference)[fitted['columns']]/fitted['scale'])@fitted['coef'])
    blind_vote=1 if score>1e-10 else (-1 if score < -1e-10 else 0)
    # Return semantic vote and actual exchanged-after-f choice separately.
    return dict(flip=flip,score=score,vote=blind_vote*sign,
                exchanged=bool(flip ^ (blind_vote<0)))


def physical_score(truth,prediction,polygon,nframes):
    """Independent crossings/graph/augmenting paths; shared official frame IoU."""
    from beetrack.evaluate import by_track,frame_match,match_events
    own=load_own_verifier('independent_whole_events',Path(__file__).with_name('independent_support.py'))
    def structure(raw):
        rows=by_track(raw);tracks={};poly=np.asarray(polygon,np.float32)
        for pid in sorted(set(rows[:,1])):
            idx=np.flatnonzero(rows[:,1]==pid)
            sides=[cv2.pointPolygonTest(poly,(float(rows[i,2]+rows[i,4]/2),float(rows[i,3]+rows[i,5]/2)),False)>=0 for i in idx]
            d=0 if sides[0]==sides[-1] else (1 if sides[-1] else -1)
            crossing=[(int(idx[j]),1 if sides[j] else -1) for j in range(1,len(idx)) if sides[j]!=sides[j-1]]
            tracks[int(pid)]=(d,idx,crossing)
        return rows,tracks
    G,gs=structure(truth);P,ps=structure(prediction)
    ge=own.events_from_native(truth,polygon);pe=own.events_from_native(prediction,polygon)
    gb=own.bins(ge.values(),nframes);pb=own.bins(pe.values(),nframes)
    N=int(abs(pb-gb).sum());answer=dict(N=N,nGT=len(ge),nPred=len(pe),iou={})
    for threshold in (.5,.3):
        owners=frame_match(G,P,thr=threshold);edges={}
        for pid,(d,indices,crosses) in ps.items():
            if not d:continue
            for j,sign in crosses:
                if sign!=d:continue
                for target in {int(owners[j-1]),int(owners[j])}-{-1}:
                    if gs[target][0]==d and any(s==d and abs(G[k,0]-P[j,0])<=5 for k,s in gs[target][2]):edges.setdefault(pid,set()).add(target)
        tp=own.maximum_matching(pe,edges)
        answer['iou'][str(threshold)]=dict(tp=tp,fp=len(pe)-tp,fn=len(ge)-tp,f1=2*tp/max(1,len(pe)+len(ge)))
    answer['TP']=answer['iou']['0.5']['tp']
    actual=match_events(truth,prediction,polygon)
    assert (actual['gt'],actual['pred'],actual['tp'])==(len(ge),len(pe),answer['TP'])
    return answer


def aggregate_physical(records):
    totals={key:sum(r[key] for r in records) for key in ('N','TP','nGT','nPred')}
    totals['W_percent']=100*totals['N']/(30*totals['nGT']) if totals['nGT'] else None
    denom=totals['nGT']+totals['nPred'];totals['F1']=2*totals['TP']/denom if denom else None
    return totals


def verify_seal(root,manifest_name='manifest.json',seal_name='SEALED.sha256'):
    root=Path(root).resolve()
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    digest=sha(root/manifest_name)
    assert (root/seal_name).read_text().strip()==digest
    for name,expected in json.loads((root/manifest_name).read_text()).items():
        path=(root/name).resolve()
        assert path.is_relative_to(root) and sha(path)==expected
    return digest


def synthetic():
    rgb=np.full((20,20,3),[12,90,241],np.uint8)
    crop=central_crop(rgb,[2,2,12,12])
    assert crop.shape==(6,6,3)
    np.testing.assert_allclose(color_six(crop),np.r_[np.array([12,90,241])/255,np.zeros(3)],atol=1e-14)
    assert central_crop(rgb,[-50,0,10,10]) is None
    assert crop_bounds(rgb.shape,[.5,.5,2,2],inset=0)==(0,0,2,2)
    a=np.zeros(12);b=np.ones(12)
    costs=paired_costs([a,b,a,b]);assert costs==dict(aligned=0.,exchanged=1.,margin=1.)
    assert paired_costs([a,b,b,a])['margin']==-1
    grid=sample_grid(crop);assert np.all(descriptor12(grid)[6:]==0)
    striped=np.zeros((16,16,3));striped[:,1::2]=1
    f=descriptor12(striped)
    assert np.allclose(f[6:8],[0,2]) and np.allclose(f[8:],[1,0,1,0])
    for rep in range(4):
        shuffled=shuffled_grid(striped,'synthetic',12,(1,2,7,8),rep)
        np.testing.assert_allclose(descriptor12(shuffled)[:6],f[:6],atol=1e-14)
        assert np.array_equal(shuffled,shuffled_grid(striped,'synthetic',12,(1,2,7,8),rep))
    boxes=np.array([[0,0,4,4],[10,0,4,4],[1,0,4,4],[11,0,4,4]],float)
    descriptors=np.array([a,b,a,b]);difference=difference_vector(boxes,descriptors)
    reversed_difference=difference_vector(boxes[[0,1,3,2]],descriptors[[0,1,3,2]])
    np.testing.assert_allclose(reversed_difference,-difference,atol=1e-14)
    records=[dict(clip='source_a',difference=difference),dict(clip='source_b',difference=2*difference)]
    for columns in MODEL_COLUMNS.values():
        fit=ridge_by_video(records,columns)
        assert blind_decision(difference,'fixture',fit)['vote']==1
    zero_fit=ridge_by_video([dict(clip='constant',difference=np.zeros(15))],MODEL_COLUMNS['T'])
    zero=blind_decision(np.zeros(15),'fixture',zero_fit)
    assert zero['vote']==0 and zero['exchanged']==bool(zero['flip']) and np.all(zero_fit['scale']==1)
    physical_checked=False
    if (Path.cwd()/'beetrack').is_dir():
        import sys
        sys.path.insert(0,str(Path.cwd()))
        truth=np.array([[1,1,12,1,2,2],[2,1,2,1,2,2],[1,2,2,3,2,2],[2,2,12,3,2,2]],float)
        cross=truth.copy();cross[cross[:,0]>1,1]=3-cross[cross[:,0]>1,1]
        poly=[[0,0],[10,0],[10,10],[0,10]]
        original=physical_score(truth,truth,poly,50);exchanged=physical_score(truth,cross,poly,50)
        assert original['N']==0 and original['TP']==2 and exchanged['N']==60 and exchanged['nPred']==0
        assert aggregate_physical([original,exchanged])['W_percent']==50
        physical_checked=True
    return dict(passed=True,original_pixel_centres=True,population_RGB_moments=True,physical_pair_cost_sign=True,
                texture_lags=True,deterministic_RGB_shuffle=True,
                reverse_pair_antisymmetry=True,weighted_ridge=True,blind_deployment=True,
                physical_integer_windows_and_TP=physical_checked,
                scope='synthetic only; field schema/release pending')


def load_own_verifier(name,path):
    import importlib.util
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def rebuild_features(repo,root,selection_path,prereg):
    """Called only under coordinator field GO; no production extractor imported."""
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    seal=verify_seal(root/'features');roster=json.loads((root/'features/roster.json').read_text());features=json.loads((root/'features/features.json').read_text())
    assert roster['prereg']==features['prereg']==prereg
    assert sha(Path(__file__).with_name('features.py'))==features['code_sha256']==roster['code_sha256']
    assert sha(selection_path)=='027e127674826303042df7074b3b855bb282b7caaf2e7205da3030cb6a4943a4'
    assert sha(root/'features/roster.json')==features['roster_sha256']==(root/'features/ROSTER.sha256').read_text().strip()
    own=load_own_verifier('previous_independent_source',Path(__file__).with_name('independent_support.py'))
    import sys
    sys.path.insert(0,str(repo));sys.path.insert(0,str(repo/'tools'))
    from whiteboard_fold import POLY
    from gray_fold import POLY_G
    official_polygons=POLY|POLY_G
    old=json.loads(selection_path.read_text());byclip={s['clip']:s for s in old['sources']};allowed={'W':{f'BEE24-{k:02d}' for k in (*range(1,10),11,14)},'G':{f'BEE24-{k:02d}' for k in range(17,33) if k not in (24,28)}}
    assert len(roster['sources'])==25 and {s['clip'] for s in roster['sources']}==set.union(*allowed.values())
    raw={};runs={};expected=[];reuse={};near={};newcounts={};oldcounts={}
    for source in roster['sources']:
        clip=source['clip'];h=source['hive'];assert clip in allowed[h]
        assert source['gt_path']==f'data/BEE24/train/{clip}/gt/gt.txt' and source['polygon']==official_polygons[clip]
        files=sorted((repo/'data/BEE24/train'/clip/'img1').glob('*.jpg'));assert [p.name for p in files]==[f'{i:06d}.jpg' for i in range(1,source['nframes']+1)]
        original=repo/source['gt_path'];assert sha(original)==source['gt_sha256']==byclip[clip]['gt_sha256']
        gt=np.loadtxt(original,delimiter=',',usecols=range(6),ndmin=2);raw[clip]=gt
        try:reconstruction=own.selected_pairs(gt,clip,source['nframes'])
        except ValueError:
            assert source['clip_failure'];continue
        assert not source['clip_failure'];runs[clip]=reconstruction['runs']
        eligible=reconstruction['eligible'];assert len(eligible)==source['eligible_count']
        oldp=eligible[:4];oldids={x for p in oldp for x in p['original_ids']};oldruns={(i,r) for p in oldp for i,r in zip(p['original_ids'],p['runs'])}
        for cohort,pool in (('old',oldp),('new',eligible[4:16])):
            for p in pool:
                expected.append((clip,p['sha256'],cohort));i,j=p['left'],p['right'];a,b=runs[clip][i],runs[clip][j];frame=p['frame']
                saved=next(q for q in roster['pairs'] if q['clip']==clip and q['digest']==p['sha256'])
                assert saved['cohort']==cohort and saved['hash_input']==p['hash_input']
                assert [saved['id1'],saved['id2']]==p['original_ids'] and [saved['run1'],saved['run2']]==p['runs']
                endpoints=np.array([a['rows'][frame-4],b['rows'][frame-4],a['rows'][frame+4],b['rows'][frame+4]])
                endpoints[:,1]=[0,1,0,1];assert np.array_equal(endpoints,saved['endpoint_rows'])
                assert saved['near_boundary']==own.near_boundary(p,runs[clip],source['polygon'])
                shift=25 if frame+4+25<=source['nframes'] else (-25 if frame-4-25>=1 else None)
                assert saved['frame_shift']==shift
                if cohort=='new':
                    newcounts[h]=newcounts.get(h,0)+1;near[h]=near.get(h,0)+int(saved['near_boundary'])
                    r=reuse.setdefault(h,dict(pairs_with_old_native_id=0,pairs_with_old_run=0))
                    r['pairs_with_old_native_id']+=any(k in oldids for k in p['original_ids'])
                    r['pairs_with_old_run']+=any(k in oldruns for k in zip(p['original_ids'],p['runs']))
                else:oldcounts[h]=oldcounts.get(h,0)+1
    assert expected==[(p['clip'],p['digest'],p['cohort']) for p in roster['pairs']]
    assert len(features['pairs'])==len(expected)
    rebuilt=[];seen_images={};maxerror=0.
    for frozen,saved in zip(roster['pairs'],features['pairs']):
        assert all(saved[k]==v for k,v in frozen.items())
        pair=dict(frozen);grids=[];arms={}
        for armname,shift in (('primary',0),('shifted',frozen['frame_shift'])):
            record=saved[armname];crops=[];armgrids=[]
            for i,row in enumerate(frozen['endpoint_rows']):
                date=int(row[0])+shift if shift is not None else None;assert record['actual_frames'][i]==date
                rel=f"data/BEE24/train/{frozen['clip']}/img1/{date:06d}.jpg" if date is not None else None
                path=repo/rel if rel else None
                if path is None or not path.is_file():
                    grid=None;bounds=None
                    if rel:seen_images[rel]=None
                else:
                    contents=path.read_bytes();seen_images[rel]=hashlib.sha256(contents).hexdigest();image=cv2.imdecode(np.frombuffer(contents,np.uint8),cv2.IMREAD_COLOR)
                    if image is None:grid=None;bounds=None
                    else:
                        rgb=image[:,:,::-1];x,y,w,h=map(float,row[2:6]);bounds=[int(np.ceil(x+.25*w-.5)),int(np.ceil(y+.25*h-.5)),int(np.ceil(x+.75*w-.5)),int(np.ceil(y+.75*h-.5))]
                        crop=central_crop(rgb,row[2:6]);grid=None if crop is None else sample_grid(crop)
                assert record['bounds'][i]==bounds
                actual=None if grid is None else descriptor12(grid)
                assert (actual is None)==(record['crops'][i] is None)
                if actual is not None:
                    error=float(np.max(abs(actual-record['crops'][i])));maxerror=max(maxerror,error);assert error<1e-11
                crops.append(None if actual is None else actual.tolist());armgrids.append(grid)
            assert record['valid']==all(g is not None for g in armgrids)
            arms[armname]=dict(valid=record['valid'],crops=crops)
            if armname=='primary':grids=armgrids
        arms['shuffles']=[]
        for rep,record in enumerate(saved['shuffles']):
            crops=[]
            for i,grid in enumerate(grids):
                actual=None if grid is None else descriptor12(shuffled_grid(grid,frozen['clip'],saved['primary']['actual_frames'][i],saved['primary']['bounds'][i],rep))
                assert (actual is None)==(record['crops'][i] is None)
                if actual is not None:
                    np.testing.assert_allclose(actual,record['crops'][i],atol=1e-11,rtol=0);np.testing.assert_allclose(actual[:6],arms['primary']['crops'][i][:6],atol=1e-12,rtol=0)
                crops.append(None if actual is None else actual.tolist())
            arms['shuffles'].append(dict(valid=all(c is not None for c in crops),crops=crops))
        pair.update(arms);rebuilt.append(pair)
    assert seen_images==features['image_sha256']
    return dict(features_manifest=seal,pairs=len(rebuilt),old_counts=oldcounts,fresh_counts=newcounts,old_identity_reuse=reuse,near_boundary=near,image_count=len(seen_images),max_descriptor_error=maxerror),rebuilt,raw,runs,roster


def predictions_and_physical(repo,root,pairs,raw,runs,roster):
    import sys
    sys.path.insert(0,str(repo))
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    seals={k:verify_seal(root/k) for k in ('predictions','physical')}
    prediction=json.loads((root/'predictions/predictions.json').read_text());physical=json.loads((root/'physical/physical.json').read_text())
    receipt=json.loads((root/'predictions/receipt.json').read_text());assert receipt['model_code_sha256']==sha(Path(__file__).with_name('model.py'))
    for fn,digest in physical['code_sha256'].items():assert sha(Path(__file__).with_name(fn))==digest
    assert prediction['features_sha256']==sha(root/'features/features.json') and prediction['roster_sha256']==sha(root/'features/roster.json')
    assert physical['input_receipts']['predictions_manifest_sha256']==seals['predictions']
    oldids={(p['clip'],p[k]) for p in pairs if p['cohort']=='old' for k in ('id1','id2')}
    oldruns={(p['clip'],p['id'+str(i)],p['run'+str(i)]) for p in pairs if p['cohort']=='old' for i in (1,2)}
    bykey={p['clip']+'|'+p['digest']:p for p in pairs};new={k:p for k,p in bykey.items() if p['cohort']=='new'}
    assert set(new)==set(physical['results'])=={r['key'] for r in prediction['pairs']}
    fitted={};train_audit={}
    for target,source in (('W','G'),('G','W')):
        model=json.loads((root/'predictions'/f'model_for_{target}.json').read_text());assert model['source_hive']==source
        sourcepairs=sorted([p for p in pairs if p['hive']==source and p['cohort']=='old' and p['primary']['valid']],key=lambda p:p['clip']+'|'+p['digest'])
        records=[dict(clip=p['clip'],difference=difference_vector(np.array(p['endpoint_rows'])[:,2:6],p['primary']['crops'])) for p in sourcepairs]
        assert model['source_pairs']==len(records) and model['training_clips']==sorted({p['clip'] for p in sourcepairs})
        for p,r,a in zip(sourcepairs,records,model['training_records']):
            assert a['key']==p['clip']+'|'+p['digest'];np.testing.assert_allclose(r['difference'],a['d'],atol=1e-12,rtol=0)
            want=1/len(model['training_clips'])/sum(q['clip']==p['clip'] for q in sourcepairs);assert abs(a['weight']-want)<1e-15
        fitted[target]={}
        for method,columns in MODEL_COLUMNS.items():
            fit=ridge_by_video(records,columns);h=model['heads'][method]
            assert h['columns']==columns and h['ridge']==.2 and h['intercept']==0
            np.testing.assert_allclose(fit['scale'],h['scale'],atol=1e-11,rtol=1e-11);np.testing.assert_allclose(fit['coef'],h['beta'],atol=1e-10,rtol=1e-10)
            fitted[target][method]=fit
        train_audit[target]=dict(source=source,pairs=len(sourcepairs),clips=len(model['training_clips']))
    decisions={};votes={};actual={};all_physical={};metadata={s['clip']:s for s in roster['sources']}
    for saved in prediction['pairs']:
        key=saved['key'];pair=new[key];boxes=np.array(pair['endpoint_rows'])[:,2:6];bit=hashlib.sha256(('texture-order-v1|'+pair['digest']).encode()).digest()[0]&1
        assert saved['order_bit']==bit and saved['permutation']==[0,1,2+bit,3-bit]
        arms=[pair['primary'],pair['shifted'],*pair['shuffles']];votes[key]={};decisions[key]={}
        for variant,arm in zip(('R','Z','Q0','Q1','Q2','Q3'),arms):
            crops=arm['crops'] if arm['valid'] else np.zeros((4,12));d=difference_vector(boxes,crops)
            votes[key][variant]={};decisions[key][variant]={}
            for method in MODEL_COLUMNS:
                result=blind_decision(d,pair['digest'],fitted[pair['hive']][method])
                valid=method=='P' or arm['valid']
                if not valid:result.update(score=0.,vote=0,exchanged=bool(bit))
                record=saved['variants'][variant][method]
                assert record['valid']==valid and abs(record['score']-result['score'])<1e-9
                expected='tie' if result['vote']==0 else ('given' if result['vote']==(-1 if bit else 1) else 'cross')
                assert record['decision']==expected
                assert record['physical_choice']==('cross' if result['exchanged']!=bool(bit) else 'given')
                votes[key][variant][method]=int(result['vote']);decisions[key][variant][method]=int(result['exchanged'])
        selected=[];positions={}
        for target,(native,run) in enumerate(((pair['id1'],pair['run1']),(pair['id2'],pair['run2'])),1):
            r=next(r for r in runs[pair['clip']] if r['original_id']==native and r['run']==run)
            for index in r['original_row_indices']:assert index not in positions;positions[index]=target
        indices=sorted(positions);truth=raw[pair['clip']][indices,:6].copy();truth[:,1]=[positions[i] for i in indices]
        switched=truth.copy();later=switched[:,0]>pair['frame'];switched[later,1]=3-switched[later,1]
        original=physical['results'][key];assert indices==original['provenance']['original_row_indices']
        outcomes={}
        def array_sha(a):
            a=np.ascontiguousarray(a,dtype='<f8');return hashlib.sha256(json.dumps(list(a.shape),separators=(',',':')).encode()+b'\n'+a.tobytes()).hexdigest()
        assert original['provenance']['identity_rows_sha256']==array_sha(truth) and original['provenance']['cross_rows_sha256']==array_sha(switched)
        assert original['provenance']['original_gt_sha256']==array_sha(raw[pair['clip']])
        assert np.array_equal(truth[:,[0,2,3,4,5]],switched[:,[0,2,3,4,5]])
        for armname,array in (('identity',truth),('cross',switched)):
            value=physical_score(truth,array,metadata[pair['clip']]['polygon'],metadata[pair['clip']]['nframes']);published=original[armname]
            assert all(value[k]==published[k] for k in ('N','nGT','nPred','TP'))
            for t in ('0.5','0.3'):
                assert all(value['iou'][t][k]==published['iou'][t][k] for k in ('tp','fp','fn'))
                assert abs(value['iou'][t]['f1']-published['iou'][t]['f1'])<1e-14
            outcomes[armname]=value
        assert outcomes['identity']['N']==0 and outcomes['identity']['TP']==outcomes['identity']['nGT']
        all_physical[key]=outcomes
    def identity_summary(keys,variant,method):
        byclip={}
        for key in keys:byclip.setdefault(new[key]['clip'],[]).append(votes[key][variant][method])
        array=[v for vs in byclip.values() for v in vs]
        missing=0
        for key in keys:
            arms=[new[key]['primary'],new[key]['shifted'],*new[key]['shuffles']]
            index=('R','Z','Q0','Q1','Q2','Q3').index(variant)
            missing+=int(method!='P' and not arms[index]['valid'])
        return dict(S=float(np.mean([np.mean(v) for v in byclip.values()])) if byclip else None,pairs=len(array),clips=len(byclip),positive=array.count(1),negative=array.count(-1),abstain=array.count(0),missing=missing,per_clip={c:dict(N=len(v),sum_vote=sum(v),S=float(np.mean(v))) for c,v in sorted(byclip.items())})
    identity={};physical_totals={};mechanism={};physical_gate={}
    for hive in ('W','G'):
        keys=[k for k,p in new.items() if p['hive']==hive];identity[hive]={}
        subsets={'all':keys,'near_boundary':[k for k in keys if new[k]['near_boundary']],
                 'P_wrong':[k for k in keys if votes[k]['R']['P']<0],'P_tie':[k for k in keys if votes[k]['R']['P']==0],
                 'P_wrong_or_tie':[k for k in keys if votes[k]['R']['P']<=0],
                 'old_id_any_reuse':[k for k in keys if any((new[k]['clip'],new[k]['id'+str(i)]) in oldids for i in (1,2))],
                 'old_run_any_reuse':[k for k in keys if any((new[k]['clip'],new[k]['id'+str(i)],new[k]['run'+str(i)]) in oldruns for i in (1,2))]}
        for subset,subsetkeys in subsets.items():
            identity[hive][subset]={}
            for method in MODEL_COLUMNS:
                variants={variant:identity_summary(subsetkeys,variant,method) for variant in ('R','Z','Q0','Q1','Q2','Q3')}
                variants['Q_mean_S']=float(np.mean([variants['Q'+str(i)]['S'] for i in range(4)])) if subsetkeys else None
                identity[hive][subset][method]=variants
        rows={method:[all_physical[k]['cross' if decisions[k]['R'][method] else 'identity'] for k in keys] for method in MODEL_COLUMNS}
        rows['identity']=[all_physical[k]['identity'] for k in keys]
        physical_totals[hive]={method:aggregate_physical(rs) for method,rs in rows.items()}
        for method,rs in rows.items():
            tp03=sum(r['iou']['0.3']['tp'] for r in rs);den=sum(r['nGT']+r['nPred'] for r in rs)
            physical_totals[hive][method]['TP_03']=tp03
            physical_totals[hive][method]['F1_03']=2*tp03/den if den else None
        m=identity[hive]['all'];mechanism[hive]=(m['T']['R']['pairs']>=16 and m['T']['R']['clips']>=3 and
          m['T']['R']['S']>=m['C']['R']['S']+.05-1e-15 and m['T']['R']['S']>=m['T']['Z']['S']+.05-1e-15 and
          m['T']['R']['S']>=m['T']['Q_mean_S']+.05-1e-15 and m['V']['R']['S']>m['V']['Q_mean_S'])
        c,t=physical_totals[hive]['C'],physical_totals[hive]['T']
        physical_gate[hive]=dict(events_at_least_10=t['nGT']>=10,window_nonincrease=t['N']<=c['N'],window_strict=t['N']<c['N'],F1_guard=t['F1']>=c['F1']-.005-1e-15 if c['F1'] is not None and t['F1'] is not None else False)
    full_physical=all(g['events_at_least_10'] and g['window_nonincrease'] and g['F1_guard'] for g in physical_gate.values()) and any(g['window_strict'] for g in physical_gate.values())
    physical_changes={}
    for hive in ('W','G'):
        selected=[v for k,v in all_physical.items() if new[k]['hive']==hive]
        delta=[dict(window=v['identity']['N']!=v['cross']['N'],TP05=v['identity']['TP']!=v['cross']['TP'],TP03=v['identity']['iou']['0.3']['tp']!=v['cross']['iou']['0.3']['tp'],predicted_count=v['identity']['nPred']!=v['cross']['nPred']) for v in selected]
        physical_changes[hive]=dict(pairs=len(selected),zero_GT=sum(v['identity']['nGT']==0 for v in selected),**{k:sum(d[k] for d in delta) for k in ('window','TP05','TP03','predicted_count')},any=sum(any(d.values()) for d in delta))
    return dict(seals=seals,training=train_audit,physical_changes=physical_changes,identity=identity,physical_totals=physical_totals,mechanism_by_hive=mechanism,mechanism_pass=all(mechanism.values()),physical_by_hive=physical_gate,physical_pass=full_physical,combined_pass=all(mechanism.values()) and full_physical,
                physical_pairs=len(all_physical),all_predictions=len(decisions),limitations=['Controlled full-truth two-bee trials; reused bees/frames are repeated observational units.'])


def compare_score(root,report):
    seal=verify_seal(root/'score');published=json.loads((root/'score/score.json').read_text())
    checked=0
    for hive,head in published['domains'].items():
        assert head['pairs']==report['fresh_counts'][hive] and head['clips']==report['identity'][hive]['all']['T']['R']['clips']
        for subset,methods in report['identity'][hive].items():
            for method,variants in methods.items():
                for variant,stats in variants.items():
                    if variant=='Q_mean_S':
                        other=head['S'][subset]['Q'][method]['S'];assert other==stats or other is not None and stats is not None and abs(other-stats)<1e-14
                    else:
                        other=head['S'][subset][variant][method]
                        for key,value in stats.items():
                            target='abstained' if key=='abstain' else key
                            if key=='S':assert other[target]==value or value is not None and other[target] is not None and abs(other[target]-value)<1e-14
                            elif key=='per_clip':assert other[target]==value
                            else:assert other[target]==value
                        checked+=1
        for method,stats in report['physical_totals'][hive].items():
            other=head['physical'][method]
            for key in ('N','TP','nGT','nPred','W_percent','F1'):assert other[key]==stats[key]
            assert other['iou']['0.3']['TP']==stats['TP_03'] and other['iou']['0.3']['F1']==stats['F1_03']
        changes=report['physical_changes'][hive]
        for key,value in head['physical_changed'].items():assert changes[key]==value
        assert not head['identity_official_mismatch_keys']
        g=published['gates']['physical']['by_hive'][hive];own=report['physical_by_hive'][hive]
        for a,b in (('GT_at_least10','events_at_least_10'),('W_nonincrease','window_nonincrease'),('W_strict_decrease','window_strict'),('F1_guard','F1_guard')):assert g[a]==own[b]
        m=report['identity'][hive]['all'];g=published['gates']['mechanism']['by_hive'][hive]
        for key,want in [('T_R_minus_C_R',m['T']['R']['S']-m['C']['R']['S']),('T_R_minus_T_Z',m['T']['R']['S']-m['T']['Z']['S']),('T_R_minus_T_Q',m['T']['R']['S']-m['T']['Q_mean_S']),('V_R_minus_V_Q',m['V']['R']['S']-m['V']['Q_mean_S'])]:assert abs(g[key]-want)<1e-14
    assert published['gates']['mechanism']['passed']==report['mechanism_pass'] and published['gates']['physical']['passed']==report['physical_pass'] and published['gates']['both_passed']==report['combined_pass']
    return dict(score_manifest=seal,identity_cells=checked,all_physical_totals=True,all_gates=True)


def field(repo,root,selection,prereg):
    # Complete entry point is deliberately gated by CLI; coordinator owns GO.
    report,pairs,raw,runs,roster=rebuild_features(repo,root,selection,prereg)
    report.update(predictions_and_physical(repo,root,pairs,raw,runs,roster))
    report['official_score_comparison']=compare_score(root,report)
    report.update(stage='complete_independent',passed=True)
    report['verifier_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    report['support_sha256']=hashlib.sha256(Path(__file__).with_name('independent_support.py').read_bytes()).hexdigest()
    output=root/'independent.json';output.write_text(json.dumps(report,indent=2)+'\n')
    print(str(output),hashlib.sha256(output.read_bytes()).hexdigest())


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--repo',type=Path);parser.add_argument('--root',type=Path);parser.add_argument('--selection',type=Path);parser.add_argument('--prereg');parser.add_argument('--field-go',action='store_true')
    args=parser.parse_args()
    if args.field_go:
        if not all((args.repo,args.root,args.selection,args.prereg)):parser.error('field requires repo/root/selection/prereg and coordinator GO')
        field(args.repo.resolve(),args.root.resolve(),args.selection.resolve(),args.prereg)
    elif any((args.repo,args.root,args.selection,args.prereg)):parser.error('field is disabled until explicit --field-go after coordinator authorization')
    else:print(json.dumps(synthetic(),indent=2))
