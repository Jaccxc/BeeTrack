"""Independent199 identity/capacity audit. Synthetic-only until coordinator GO."""
import os
for _name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[_name]='1'
os.environ['CUDA_VISIBLE_DEVICES']='-1'
import hashlib
import itertools
import json
from fractions import Fraction
from pathlib import Path
import numpy as np
import cv2
from scipy.optimize import linear_sum_assignment
cv2.setNumThreads(1)


def canonical_with_origin(raw):
    """Independent canonical gap segments and explicit original-row provenance."""
    raw=np.asarray(raw,float)
    if not len(raw):return raw.copy(),np.empty(0,int)
    origin=np.lexsort((np.arange(len(raw)),raw[:,0],raw[:,1]));rows=raw[origin].copy()
    boundary=np.r_[True,(rows[1:,1]!=rows[:-1,1])|(np.diff(rows[:,0])>25)]
    starts=np.flatnonzero(boundary);ends=[*starts[1:],len(rows)];runs={}
    for lo,hi in zip(starts,ends):
        native_id=int(rows[lo,1]);run=runs.get(native_id,0);runs[native_id]=run+1
        rows[lo:hi,1]=native_id+100000*run
    order=np.lexsort((origin,rows[:,0],rows[:,1]));return rows[order],origin[order]


def official_iou_independent(gt_boxes,pred_boxes):
    """Official XYWH geometry, including its1e-9 union term and all GT rows."""
    G=np.asarray(gt_boxes,float);P=np.asarray(pred_boxes,float)
    Gcorner=G[:,:2]+G[:,2:];Pcorner=P[:,:2]+P[:,2:]
    left=np.maximum(G[:,None,:2],P[None,:,:2])
    right=np.minimum(Gcorner[:,None],Pcorner[None])
    inter=np.maximum(0,right-left).prod(axis=2)
    # Official area is computed after XYWH→XYXY, preserving its roundoff too.
    Gsize=Gcorner-G[:,:2];Psize=Pcorner-P[:,:2]
    return inter/(Gsize[:,0,None]*Gsize[:,1,None]+Psize[None,:,0]*Psize[None,:,1]-inter+1e-9)


def full_frame_owners(gt_raw,pred_raw,threshold,requested_indices=None):
    """Own full-frame Hungarian and full bipartite threshold degrees."""
    G,gi=canonical_with_origin(gt_raw);P,pi=canonical_with_origin(pred_raw)
    ids,starts=np.unique(G[:,1],return_index=True)
    original_ids={int(pid):set(map(int,gt_raw[gi[lo:hi],1])) for pid,lo,hi in zip(ids,starts,[*starts[1:],len(G)])}
    requested=sorted(set(range(len(P))) if requested_indices is None else set(requested_indices));rows={}
    for i in requested:
        p=P[i];rows[i]=dict(prediction_index=i,raw_prediction_index=int(pi[i]),frame=int(p[0]),pred_id=int(p[1]),owner=None,owner_row=None,original_gt_id=None,pred_degree=0,gt_degree=0,iou=None,isolated=False,graph_gt=[])
    for frame in sorted({P[i,0] for i in requested}):
        pred=np.flatnonzero(P[:,0]==frame);truth=np.flatnonzero(G[:,0]==frame)
        if not len(truth):continue
        matrix=official_iou_independent(G[truth,2:6],P[pred,2:6]);edges=matrix>=threshold
        a,b=linear_sum_assignment(-matrix)
        for local,row in enumerate(pred):
            if row not in rows:continue
            rows[row]['pred_degree']=int(edges[:,local].sum())
            rows[row]['graph_gt']=[dict(owner=int(G[truth[j],1]),original_gt_id=next(iter(original_ids[int(G[truth[j],1])])) if len(original_ids[int(G[truth[j],1])])==1 else None,iou=float(matrix[j,local]),gt_degree=int(edges[j].sum())) for j in np.flatnonzero(edges[:,local])]
        for ia,ib in zip(a,b):
            if not edges[ia,ib]:continue
            index=int(pred[ib]);owner_row=int(truth[ia]);degree=int(edges[ia].sum())
            if index not in rows:continue
            originals=original_ids[int(G[owner_row,1])];orig=next(iter(originals)) if len(originals)==1 else None
            rows[index].update(owner=int(G[owner_row,1]),owner_row=owner_row,original_gt_id=orig,
                               gt_degree=degree,iou=float(matrix[ia,ib]),isolated=bool(edges[:,ib].sum()==1 and degree==1 and orig is not None))
    output=rows if requested_indices is not None else [rows[i] for i in range(len(P))]
    return dict(G=G,P=P,gt_origin=gi,pred_origin=pi,rows=output,threshold=threshold,original_ids=original_ids)


def classify_cut(owners,left,right):
    a,b=owners['rows'][left],owners['rows'][right]
    if not a['isolated'] or not b['isolated']:return dict(label='U',reason='unmatched_or_non_singleton',left=a,right=b)
    if a['owner']==b['owner'] and a['original_gt_id']==b['original_gt_id']:
        return dict(label='S',reason='same_canonical_GT_run',left=a,right=b)
    if a['original_gt_id']!=b['original_gt_id']:
        return dict(label='D',reason='distinct_original_GT_IDs',left=a,right=b)
    return dict(label='U',reason='same_original_ID_different_gap_runs',left=a,right=b)


def censoring_anchors(owners,parent_start,parent_end,left,right,polygon):
    """Compatibility evidence at endpoints/cut; not identity of every row."""
    poly=np.asarray(polygon,np.float32)
    def side(row):return cv2.pointPolygonTest(poly,(float(row[2]+row[4]/2),float(row[3]+row[5]/2)),False)>=0
    positions=[parent_start,parent_end-1,left,right];anchors=[owners['rows'][i] for i in positions]
    if not all(a['isolated'] for a in anchors):return dict(compatible=False,reason='anchor_unknown',anchor_indices=positions)
    if len({(a['owner'],a['original_gt_id']) for a in anchors})!=1:return dict(compatible=False,reason='anchors_different_GT_runs',anchor_indices=positions)
    G,P=owners['G'],owners['P'];gt_id=anchors[0]['owner'];gt_rows=G[G[:,1]==gt_id]
    pred_first,pred_last=P[parent_start],P[parent_end-1]
    pred_sides=[side(pred_first),side(pred_last)]
    matched_sides=[side(G[anchors[i]['owner_row']]) for i in (0,1)]
    gt_sides=[side(gt_rows[0]),side(gt_rows[-1])]
    event=pred_sides[0]!=pred_sides[1]
    compatible=event and pred_sides==matched_sides and gt_sides[0]==gt_sides[1]
    later=bool(compatible and gt_sides[-1]!=pred_sides[-1] and gt_rows[-1,0]>pred_last[0])
    earlier=bool(compatible and gt_sides[0]!=pred_sides[0] and gt_rows[0,0]<pred_first[0])
    return dict(compatible=bool(compatible),reason='endpoint_censoring_compatible' if compatible else 'endpoint_sides_not_censoring',
                anchor_indices=positions,GT_run=gt_id,GT_original_ID=anchors[0]['original_gt_id'],
                prediction_dates=[int(pred_first[0]),int(pred_last[0])],GT_dates=[int(gt_rows[0,0]),int(gt_rows[-1,0])],
                prediction_sides=pred_sides,matched_GT_sides=matched_sides,GT_endpoint_sides=gt_sides,
                later_return=later,earlier_missed=earlier,
                scope='Four anchor rows may overlap; compatibility does not prove the intervening full track identity or causal censoring.')


def array_digest(value):
    a=np.ascontiguousarray(value,dtype='<f8')
    return hashlib.sha256(json.dumps(list(a.shape),separators=(',',':')).encode()+b'\n'+a.tobytes()).hexdigest()


def GT_segments(owners,polygon):
    G=owners['G'];poly=np.asarray(polygon,np.float32);segments={};original={}
    for pid in sorted(set(G[:,1])):
        ix=np.flatnonzero(G[:,1]==pid);ids=sorted(owners['original_ids'][int(pid)]);native=ids[0] if len(ids)==1 else None
        side=lambda row:cv2.pointPolygonTest(poly,(float(row[2]+row[4]/2),float(row[3]+row[5]/2)),False)>=0
        first,last=side(G[ix[0]]),side(G[ix[-1]])
        info=dict(canonical_gt_id=int(pid),original_gt_id=native,original_ids=ids,start=int(ix[0]),end=int(ix[-1]+1),
                  first_frame=int(G[ix[0],0]),last_frame=int(G[ix[-1],0]),start_inside=first,end_inside=last,
                  event=None if first==last else ('in' if last else 'out'))
        segments[int(pid)]=info
        if native is not None:original.setdefault(native,[]).append(info)
    for pieces in original.values():
        for run,segment in enumerate(sorted(pieces,key=lambda r:r['first_frame'])):segment['run']=run
    return segments


def independent_row_record(owners,index,source_canonical,source_index,polygon,segments=None):
    row=owners['rows'][index];P,G=owners['P'],owners['G'];poly=np.asarray(polygon,np.float32)
    side=lambda value:cv2.pointPolygonTest(poly,(float(value[2]+value[4]/2),float(value[3]+value[5]/2)),False)>=0
    segments=GT_segments(owners,polygon) if segments is None else segments
    owner=row['owner'];position=row['owner_row'];segment=segments.get(owner);gt=G[position] if position is not None else None
    return dict(canonical_row=index,pred_key=[row['frame'],row['pred_id']],row_sha256=array_digest(P[index]),
                official_owner=owner,owner_gt_row=position,owner_gt_original_id=row['original_gt_id'],
                owner_gt_run=segment.get('run') if segment else None,
                owner_gt_raw_index=int(owners['gt_origin'][position]) if position is not None else None,
                owner_gt_key=[row['frame'],owner] if owner is not None else None,owner_gt_row_sha256=array_digest(gt) if gt is not None else None,
                iou=row['iou'],pred_degree=row['pred_degree'],gt_degree=row['gt_degree'] if owner is not None else None,
                matched=owner is not None,isolated=row['isolated'],qualified=row['isolated'],pred_inside=side(P[index]),gt_inside=side(gt) if gt is not None else None,
                graph_gt=[dict(canonical_gt_id=n['owner'],original_gt_id=n['original_gt_id'],iou=n['iou'],gt_degree=n['gt_degree']) for n in row['graph_gt']],
                source_raw_index=int(source_index[index]),source_parent_id=int(source_canonical[index,1]),source_row_sha256=array_digest(source_canonical[index]))


def independent_censor_record(candidate,block,owners,polygon,label,segments=None):
    empty=dict(supported=False,type='unconfirmed',reason='not_drop_S')
    if candidate['delta']!=-1 or label!='S':return empty
    i0,i1=block['start'],block['end']-1;j=candidate['cut_after_canonical_row'];ix=[i0,i1,j,j+1]
    primitive=censoring_anchors(owners,block['start'],block['end'],j,j+1,polygon)
    if primitive['reason'] in ('anchor_unknown','anchors_different_GT_runs'):
        return dict(empty,reason='parent_endpoints_and_cut_not_one_isolated_GT_run',row_indices=ix)
    segments=GT_segments(owners,polygon) if segments is None else segments
    owner=owners['rows'][i0]['owner'];segment=segments[owner];P=owners['P'];G=owners['G'][segment['start']:segment['end']]
    poly=np.asarray(polygon,np.float32);side=lambda r:cv2.pointPolygonTest(poly,(float(r[2]+r[4]/2),float(r[3]+r[5]/2)),False)>=0
    sides=[side(r) for r in G];pstart,pend=side(P[i0]),side(P[i1])
    info=dict(gt_run=segment,row_indices=ix,p_first_frame=int(P[i0,0]),p_last_frame=int(P[i1,0]),p_start_inside=pstart,p_end_inside=pend,K_event_direction='in' if pend else 'out')
    if primitive['matched_GT_sides']!=[pstart,pend]:return dict(empty,**info,reason='GT_at_P_endpoint_dates_side_disagrees')
    if segment['start_inside']!=segment['end_inside']:return dict(empty,**info,reason='GT_run_has_different_endpoint_sides')
    common=segment['start_inside']
    def witness(k):return dict(frame=int(G[k,0]),canonical_gt_id=owner,original_gt_id=segment['original_gt_id'],run=segment.get('run'),inside=sides[k],row_sha256=array_digest(G[k]))
    excursions=[i for i,s in enumerate(sides) if s!=common];info['excursion_observation']=witness(excursions[0]) if excursions else None
    later=[i for i,s in enumerate(sides) if G[i,0]>P[i1,0] and s==common]
    earlier=[i for i,s in enumerate(sides) if G[i,0]<P[i0,0] and s==common]
    if common==pstart and later:return dict(supported=True,type='later_return',reason='same_GT_run_returns_after_P_end',return_observation=witness(later[0]),**info)
    if common==pend and earlier:return dict(supported=True,type='earlier_part_missed',reason='same_GT_run_starts_before_P_start',earlier_observation=witness(earlier[-1]),**info)
    return dict(empty,**info,reason='no_observed_supporting_GT_extension')


def strict_raw_budget(K_sum,denominator_per_group,groups,line):
    """IntegerΣN under strict meanE3 reduction, computed without floats."""
    threshold=Fraction(K_sum)-Fraction(line)*groups*denominator_per_group/100
    # largest integer strictly below threshold
    return -((-threshold.numerator)//threshold.denominator)-1


def verify_min_S_certificate(residual,columns,S_cost,proof):
    """Weak-duality lower bound on known-S cuts under one global count budget.

    Original binary columns avoid silently mixing S costs in capacity buckets.
    """
    p=proof['p'];g=proof['g'];Q=proof['Q'];B=proof['budget'] if 'budget' in proof else proof['B']
    assert all(type(v) is int for v in [*p,g,Q,B]) and Q>0 and g>=0 and all(abs(v)<=g for v in p)
    assert len(p)==len(residual) and len(columns)==len(S_cost)
    linear=sum(int(r)*v for r,v in zip(residual,p));dots=[sum(int(v)*w for v,w in zip(c,p)) for c in columns]
    reduced=[Q*int(cost)+value for cost,value in zip(S_cost,dots)];negative=sum(min(0,v) for v in reduced)
    numerator=linear-g*B+negative;lower=-((-numerator)//Q)
    expected=dict(budget=B,linear=linear,adjusted_column_dots=reduced,negative_sum=negative,scaled_bound=numerator,lower_bound=max(0,lower))
    for key,value in expected.items():
        if key in proof:assert proof[key]==value
    return dict(lower_bound=max(0,lower),raw_lower=lower,linear_term=linear,column_dot_products=dots,reduced_costs=reduced,negative_part_sum=negative,scaled_numerator=numerator)


def direct_identity_fit(records,heldout,feature_key):
    dim={'features_N11':11,'features_B15':15,'features_I22':22}[feature_key]
    known=[r for r in records if r['hive']!=heldout and r['eligible'] and r['identity_class'] in ('D','S')]
    if not known:return dict(mean=np.zeros(dim),scale=np.ones(dim),coef=np.zeros(dim),intercept=.5,empty=True,weights=[])
    domains={r['hive'] for r in known};weights=[]
    for r in known:
        groups={a['group'] for a in known if a['hive']==r['hive']}
        clips={a['clip'] for a in known if (a['hive'],a['group'])==(r['hive'],r['group'])}
        signs={a['delta'] for a in known if (a['hive'],a['group'],a['clip'])==(r['hive'],r['group'],r['clip'])}
        count=sum((a['hive'],a['group'],a['clip'],a['delta'])==(r['hive'],r['group'],r['clip'],r['delta']) for a in known)
        weights.append(1/len(domains)/len(groups)/len(clips)/len(signs)/count)
    w=np.asarray(weights);X=np.asarray([r[feature_key] for r in known],float);y=np.array([r['identity_class']=='D' for r in known],float)
    assert abs(w.sum()-1)<1e-12
    mean=(w[:,None]*X).sum(axis=0);constant=np.all(X==X[0],axis=0);mean[constant]=X[0,constant]
    scale=np.sqrt((w[:,None]*(X-mean)**2).sum(axis=0));scale[(scale==0)|constant]=1
    Z=(X-mean)/scale;intercept=float(w@y)
    coef=np.linalg.solve(Z.T@(w[:,None]*Z)+.2*np.eye(dim),Z.T@(w*(y-intercept)))
    if np.all(y==y[0]):coef[:]=0;intercept=float(y[0])
    assert np.max(abs(2*(Z.T@(w*(intercept+Z@coef-y))+.2*coef)),initial=0)<1e-10
    return dict(mean=mean,scale=scale,coef=coef,intercept=intercept,empty=False,weights=w)


def predict_identity(record,fitted,feature_key):
    raw=float(fitted['intercept']+((np.asarray(record[feature_key])-fitted['mean'])/fitted['scale'])@fitted['coef'])
    p=float(np.clip(raw,0,1));return dict(raw=raw,pD=p,abstain=fitted['empty'],cut=bool(not fitted['empty'] and p>.5))


def auc_pairwise(labels,scores):
    positive=[s for y,s in zip(labels,scores) if y==1];negative=[s for y,s in zip(labels,scores) if y==0]
    if not positive or not negative:return None
    twice=sum(2*int(p>n)+int(p==n) for p in positive for n in negative)
    return float(Fraction(twice,2*len(positive)*len(negative)))


def classification_stats(records,selected,scores=None):
    chosen=set(selected);counts={k:sum(r['class']==k for r in records) for k in ('D','S','U')}
    assert chosen<={r['candidate_id'] for r in records}
    cuts={k:sum(r['class']==k and r['candidate_id'] in chosen for r in records) for k in counts}
    recall=cuts['D']/counts['D'] if counts['D'] else None;false_split=cuts['S']/counts['S'] if counts['S'] else None
    labels=[];values=[];abstained=0
    for r in records:
        if r['class']=='U':continue
        item=dict(pD=float(r['candidate_id'] in chosen),abstain=False) if scores is None else scores.get(r['candidate_id'])
        if item is None or item['abstain'] or item['pD'] is None:abstained+=1;continue
        labels.append(int(r['class']=='D'));values.append(float(item['pD']))
    by_direction={}
    for delta,direction in itertools.product((-1,1),repeat=2):
        rows=[r for r in records if r['delta']==delta and r['direction']==direction]
        by_direction[f'{delta}/{direction}']=dict(total=len(rows),cuts=sum(r['candidate_id'] in chosen for r in rows),classes={c:sum(r['class']==c for r in rows) for c in counts},cut_classes={c:sum(r['class']==c and r['candidate_id'] in chosen for r in rows) for c in counts})
    return dict(total=len(records),class_denominators=counts,cut_classes=cuts,cut_count=len(chosen),D_recall=recall,S_false_split_rate=false_split,
                U_cut_count=cuts['U'],U_cut_rate=cuts['U']/counts['U'] if counts['U'] else None,
                balanced_accuracy=(recall+1-false_split)/2 if recall is not None and false_split is not None else None,
                AUROC=None if abstained else auc_pairwise(labels,values),known_score_count=len(values),known_abstentions=abstained,
                AUROC_score_type='continuous_pD' if scores is not None else 'binary_mask',by_delta_direction=by_direction)


def matched_identity_null(records,selected):
    selected=set(selected);buckets={}
    for r in records:
        if r['eligible']:buckets.setdefault((r['unit'],r['member_index'],r['delta'],r['direction']),[]).append(r)
    result=[]
    for pool in buckets.values():
        n=sum(r['candidate_id'] in selected for r in pool)
        key=lambda r:(hashlib.sha256(('endpoint-identity-null-v1|'+r['candidate_id']).encode()).digest(),r['candidate_id'])
        result.extend(r['candidate_id'] for r in sorted(pool,key=key)[:n])
    return sorted(result)


def shared(repo):
    import sys,importlib.util
    sys.path.insert(0,str(repo));path=repo/'tools/study198/verify198.py'
    spec=importlib.util.spec_from_file_location('independent198_support199',path);v=importlib.util.module_from_spec(spec);spec.loader.exec_module(v)
    return v,v.own197_module(repo)


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def compare(a,b,where='record'):
    if isinstance(a,dict):
        for k,v in a.items():assert k in b,where+'/'+k;compare(v,b[k],where+'/'+k)
    elif isinstance(a,list):
        assert len(a)==len(b),where
        for i,(x,y) in enumerate(zip(a,b)):compare(x,y,where+'/'+str(i))
    elif isinstance(a,(float,np.floating)):assert b is not None and abs(a-b)<2e-12,(where,a,b)
    else:assert a==b,(where,a,b)


CODE_DIR=Path(__file__).resolve().parent
AUDIT_DIR=None
REQUIRE_ORIGINAL_SEALS=False
def audit_path(root,name):return Path(AUDIT_DIR or root)/name

def save_receipt(path,value):
    if path.exists() or path.with_suffix('.sha256').exists():raise FileExistsError('fresh audit destination required')
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,ensure_ascii=True)+'\n',encoding='utf-8')
    path.with_suffix('.sha256').write_text(sha(path)+'\n')


def de_duplicate_censors(records):
    out={}
    for hive in ('35','33','W','G'):
        specimens={};opportunities=0
        for r in records.values():
            c=r['labels']['0.5']['censor']
            if r['hive']!=hive or not c['supported']:continue
            opportunities+=1;gt=c['gt_run'];key=(r['clip'],gt['canonical_gt_id'])
            item=specimens.setdefault(key,dict(clip=key[0],canonical_GT_run=key[1],original_GT_ID=gt['original_gt_id'],run=gt.get('run'),types=set(),predicted_directions=set(),groups=set(),members=set(),opportunities=0,candidate_ids=[]))
            item['types'].add(c['type']);item['predicted_directions'].add('in' if r['direction']==1 else 'out');item['groups'].add(r['group']);item['members'].add(str(r['member']));item['opportunities']+=1;item['candidate_ids'].append(r['candidate_id'])
        rows=[]
        for item in specimens.values():
            for k in ('types','predicted_directions','groups','members'):item[k]=sorted(item[k])
            rows.append(item)
        out[hive]=dict(opportunities=opportunities,unique_GT_runs=len(rows),later_unique=sum('later_return' in r['types'] for r in rows),earlier_unique=sum('earlier_part_missed' in r['types'] for r in rows),both_types_unique=sum(len(r['types'])>1 for r in rows),specimens=rows)
    return out


def context(repo,source197,source198):
    v,u=shared(repo);_,peers,features,families,bundles,old=v.stage2_context(repo,source198,source197)
    return v,u,peers,features,families,bundles,old


def audit_owners(repo,root,source197,source198,prereg):
    start_hash=sha(__file__);v,u,peers,features,families,bundles,old=context(repo,source197,source198)
    owner_seal=u.verify_sealed(root/'owners');assert not REQUIRE_ORIGINAL_SEALS or owner_seal=='0343fa154949939f59c55d7f0a350110b9f302bff9d7b5afdfd62debf13e0c9d'
    summary=read(root/'owners/summary.json');assert summary['prereg']==prereg and summary['candidates']==1196 and summary['members']==60
    assert summary['code_sha256']['owners199']==sha(CODE_DIR/'owners199.py')
    records={};GT={};counts={};checked_rows=0;checked_frames=0
    for hive in ('35','33','W','G'):
        shard=read(root/'owners'/f'{hive}.json');assert shard['domain']==hive and shard['prereg']==prereg;counts[hive]={t:{} for t in ('0.5','0.3')}
        for g in shard['units']:
            if g['video'] not in GT:
                path=repo/'runs/gt'/g['split']/g['video']/'gt/gt.txt';assert sha(path)==summary['GT_sha256'][path.relative_to(repo).as_posix()]
                GT[g['video']]=np.loadtxt(path,delimiter=',',usecols=range(6),ndmin=2)
            for m in g['members']:
                key=(g['unit'],str(m['member']));raw,C,index,blocks,original,sm=bundles[key];cands=sm['bundle']['candidates'];bm={b['parent_id']:b for b in blocks}
                K=u.realize_independent(raw,C,index,blocks,cands,[]);needed=set()
                for c in cands:
                    b=bm[c['parent_id']];j=c['cut_after_canonical_row'];needed.update((b['start'],b['end']-1,j,j+1))
                owners={};ev={};segments={}
                for threshold in (.5,.3):
                    t=str(threshold);owners[t]=full_frame_owners(GT[g['video']],K,threshold,needed);segments[t]=GT_segments(owners[t],g['polygon'])
                    ev[t]={i:independent_row_record(owners[t],i,C,index,g['polygon'],segments[t]) for i in needed}
                    checked_rows+=len(needed);checked_frames+=len({int(owners[t]['P'][i,0]) for i in needed})
                    assert array_digest(owners[t]['P'])==m['K_rows_sha256'] and array_digest(owners[t]['G'])==m['GT_rows_sha256']
                saved={r['candidate_id']:r for r in m['records']};assert set(saved)=={c['candidate_id'] for c in cands}
                for c in cands:
                    cid=c['candidate_id'];r=saved[cid];j=c['cut_after_canonical_row'];b=bm[c['parent_id']];labels={}
                    for t in ('0.5','0.3'):
                        label=classify_cut(owners[t],j,j+1)['label'];assert label==r['labels'][t]['class']
                        compare(ev[t][j],r['labels'][t]['left']);compare(ev[t][j+1],r['labels'][t]['right'])
                        censor=independent_censor_record(c,b,owners[t],g['polygon'],label,segments[t]);other=r['labels'][t]['censor']
                        if 'rows' in other:censor['rows']=[ev[t][i] for i in (b['start'],b['end']-1,j,j+1)]
                        compare(censor,other);labels[t]=dict(**{'class':label},left=ev[t][j],right=ev[t][j+1],censor=censor)
                        cc=counts[hive][t];cc[label]=cc.get(label,0)+1
                        if censor['supported']:cc[censor['type']]=cc.get(censor['type'],0)+1
                    records[cid]=dict(features[cid],labels=labels,identity_class=labels['0.5']['class'],**{'class':labels['0.5']['class']},class03=labels['0.3']['class'],censor_supported=labels['0.5']['censor']['supported'],censor_type=labels['0.5']['censor']['type'],censor_reason=labels['0.5']['censor']['reason'])
                assert np.array_equal(m['pred_bins'],v.replay_mask(families[key],[])['pred_bins'])
            print('INDEPENDENT199_OWNERS '+g['unit'],flush=True)
        for t in counts[hive]:
            for k in set(counts[hive][t])|set(summary['counts'][hive][t]):assert counts[hive][t].get(k,0)==summary['counts'][hive][t].get(k,0)
    assert len(records)==1196
    ledger=audit_path(root,'independent_owner_ledger.json');save_receipt(ledger,records)
    report=dict(passed=True,owners_manifest=owner_seal,counts=counts,requested_prediction_rows=checked_rows,full_scene_Hungarian_frames=checked_frames,censor_unique_specimens=de_duplicate_censors(records),owner_ledger_sha256=sha(ledger),verifier_sha256=start_hash,scope='Independent full-frame IoU/Hungarian and both-side uniqueness at all requested cut/endpoint dates; all GT/P frame competitors retained. Labels describe adjacent rows, not whole-track identity.')
    save_receipt(audit_path(root,'independent_owners.json'),report);print('INDEPENDENT199_OWNERS_PASS '+sha(audit_path(root,'independent_owners.json')),flush=True)
    return report


def audit_capacity(root,source198,records,families,bundles,v,u):
    seal=u.verify_sealed(root/'capacity_v2');assert not REQUIRE_ORIGINAL_SEALS or seal=='47b704962aac6ae50043d1e277499549eda69ed9731072b3c4033012b968b02c'
    cap=read(root/'capacity_v2/capacity199.json');units={b[4]['unit']:b[4] for b in bundles.values()}
    assert cap['input_receipt']['owners_seal']==sha(root/'owners/manifest.json') and cap['input_receipt']['graphs_sha']==sha(source198/'stage1/graph_data.json')
    masks={};proofs=[]
    for unit,g in units.items():
        candidates=[r for r in records.values() if r['unit']==unit]
        pred=sum(v.replay_mask(families[unit,str(m['member'])],[])['pred_bins'] for m in g['members'])
        gt=v.bin_events([r[1] for r in families[unit,str(g['members'][0]['member'])]['gt']],g['nf'])
        for arm,allowed in (('O_D',{'D'}),('O_DU',{'D','U'})):
            pool=[r for r in candidates if r['class'] in allowed];saved=cap['units'][unit]['families'][arm];answer=saved['oracle']
            check=u.verify_oracle_exact(pred,gt,pool,g['nf'],answer);proofs.append(dict(unit=unit,arm=arm,**check))
            for m in g['members']:
                ids=[cid for cid in answer['selected_ids'] if records[cid]['member_index']==m['member_index']]
                assert sorted(ids)==sorted(saved['member_masks'][str(m['member_index'])]);masks[unit,str(m['member']),arm]=ids
                graph=v.replay_mask(families[unit,str(m['member'])],ids)
                assert np.array_equal(graph['pred_bins'],saved['graph_witness']['members'][str(m['member_index'])]['pred_bins'])
    minima={};head_masks={}
    for h,line in (('35',Fraction(14,10)),('33',Fraction(13,10))):
        gs=[g for g in units.values() if g['domain']==h];lengths=[v.bin_events([],g['nf']).size for g in gs]
        residual=[];columns=[];cost=[];candidates=[];denominators=[];base=0;offset=0;total=sum(lengths)
        for g,length in zip(gs,lengths):
            gt=v.bin_events([r[1] for r in families[g['unit'],str(g['members'][0]['member'])]['gt']],g['nf'])
            pred=sum(v.replay_mask(families[g['unit'],str(m['member'])],[])['pred_bins'] for m in g['members'])
            r=(pred-3*gt).ravel();residual.extend(r.tolist());base+=int(abs(r).sum());denominators.append(3*int(gt.sum()))
            for c in [r for r in records.values() if r['unit']==g['unit']]:
                column=np.zeros(total,np.int64);column[offset:offset+length]=int(c['delta'])*v.bin_events([c['frame']*c['direction']],g['nf']).ravel()
                columns.append(column);cost.append(int(c['class']=='S'));candidates.append(c)
            offset+=length
        assert len(gs)==4 and len(set(denominators))==1
        result=cap['minimum_S'][h];B=strict_raw_budget(base,denominators[0],4,line);assert B==result['budget']
        exact=verify_min_S_certificate(residual,columns,cost,result['certificate']);assert exact['lower_bound']==result['protected_lower']
        x=result['x'];assert len(x)==len(candidates) and all(type(bit) is int and bit in (0,1) for bit in x)
        changed=np.asarray(residual,np.int64)+sum((column*bit for column,bit in zip(columns,x)),np.zeros(total,np.int64))
        N=int(abs(changed).sum());S=sum(a*b for a,b in zip(cost,x));assert N<=B and N==result['count_numerator'] and S==result['protected_upper']
        gain=Fraction(100*(base-N),sum(denominators));assert gain>line and result['attained_gain_fraction']==[gain.numerator,gain.denominator]
        assert result['optimal']==(exact['lower_bound']==S)
        selected=[c for c,bit in zip(candidates,x) if bit];same=[c for c in selected if c['class']=='S'];specimens={}
        for c in same:
            left=c['labels']['0.5']['left'];key=(c['clip'],left['official_owner'])
            specimens.setdefault(key,[]).append(dict(candidate_id=c['candidate_id'],unit=c['unit'],group=c['group'],member=str(c['member']),member_index=c['member_index'],delta=c['delta'],direction=c['direction'],GT_original_ID=left['owner_gt_original_id'],GT_run=left['owner_gt_run'],end=c['end'],k=c['k'],cut_dates=[c['labels']['0.5'][side]['pred_key'][0] for side in ('left','right')],event_frame=c['frame'],censor=c['labels']['0.5']['censor']))
        for g in gs:
            ids=[c['candidate_id'] for c in selected if c['unit']==g['unit']];assert sorted(ids)==sorted(result['unit_selected_ids'][g['unit']])
            for m in g['members']:
                own=[cid for cid in ids if str(records[cid]['member'])==str(m['member'])]
                assert sorted(own)==sorted(result['unit_member_masks'][g['unit']][str(m['member_index'])]);head_masks[g['unit'],str(m['member']),h]=own
        minima[h]=dict(budget=B,K_numerator=base,count_numerator=N,S_lower=exact['lower_bound'],S_upper=S,optimal=result['optimal'],gain=float(gain),S_operations=len(same),S_witness_unique_GT_runs=len(specimens),S_specimens=[dict(clip=clip,canonical_GT_run=pid,operations=ops) for (clip,pid),ops in specimens.items()],scope='Minimum member-candidate S operations. Dedup describes this witness only, not minimum unique bees.')
    report=dict(passed=True,capacity_manifest=seal,restricted_certificates=proofs,minimum_S=minima,verifier_sha256=sha(__file__))
    save_receipt(audit_path(root,'independent_capacity.json'),report)
    return report,masks,head_masks,cap


def audit_models(root,source198,records,bundles,u):
    seal=u.verify_sealed(root/'predictions');assert not REQUIRE_ORIGINAL_SEALS or seal=='c184d0643d9993dde81fc9afb1d31cd7f990cb60a5d0e4842bbf9e612b53dced'
    receipt=read(root/'predictions/receipt.json');assert receipt['model_code_sha256']==sha(CODE_DIR/'model199.py')
    assert receipt['capacity_gate']=='none; always trained' and receipt['owners_receipt']['manifest_sha256']==sha(root/'owners/manifest.json')
    source198_masks={}
    for h in ('35','33','W','G'):
        for m in read(source198/'predictions'/f'{h}.masks.json')['member_clips']:source198_masks[m['unit'],str(m['member'])]=m['selected']['I']
    allrows=list(records.values());decisions={};probabilities={h:{} for h in ('H_N','H_B','H_I')};audit={}
    for target in ('35','33','W','G'):
        model=read(root/'predictions'/f'{target}.model.json');fold=read(root/'predictions'/f'{target}.masks.json')
        assert model['heldout']==fold['heldout']==target and set(model['source_domains'])=={'35','33','W','G'}-{target}
        source=[r for r in allrows if r['hive']!=target and r['eligible'] and r['identity_class'] in ('D','S')]
        assert {r['candidate_id'] for r in source}=={r['key'] for r in model['training_records']} and model['known_source_rows']==len(source)
        fitted={name:direct_identity_fit(allrows,target,key) for name,key in (('H_N','features_N11'),('H_B','features_B15'),('H_I','features_I22'))}
        for arm,fit in fitted.items():
            head=model['heads'][arm[-1]]
            for a,b in (('mean','mean'),('scale','scale'),('coef','beta')):np.testing.assert_allclose(fit[a],head[b],atol=3e-9,rtol=1e-8)
            assert abs(fit['intercept']-head['intercept'])<1e-10 and head['ridge']==.2
        weights={r['candidate_id']:float(w) for r,w in zip(source,fitted['H_I']['weights'])}
        for r in model['training_records']:
            original=records[r['key']];assert original['hive']!=target and original['eligible'] and original['class'] in ('S','D')
            assert r['identity_class']==original['class'] and r['y']==int(original['class']=='D') and abs(r['weight']-weights[r['key']])<1e-15
            for feature in ('features_N11','features_B15','features_I22'):compare(original[feature],r[feature])
        sources=model['source_receipts'];assert sources['owner_manifest_sha256']==sha(root/'owners/manifest.json') and sources['label_threshold']=='0.5'
        assert set(sources['source_shards'])=={'35','33','W','G'}-{target}
        for h,r in sources['source_shards'].items():assert h!=target and r['file']==h+'.json' and sha(root/'owners'/r['file'])==r['sha256']
        for member in fold['member_clips']:
            key=(member['unit'],str(member['member']));rows=sorted([r for r in allrows if (r['unit'],str(r['member']))==key],key=lambda r:r['candidate_id'])
            assert member['candidate_ids']==[r['candidate_id'] for r in rows] and member['eligible']==[r['eligible'] for r in rows]
            masks=dict(K=[],I198=list(source198_masks[key]),H_N=[],H_B=[],H_I=[],M=[])
            for i,r in enumerate(rows):
                for arm,feature in (('H_N','features_N11'),('H_B','features_B15'),('H_I','features_I22')):
                    saved=member['scores'][arm][i];cid=r['candidate_id']
                    if not r['eligible']:assert saved is None;probabilities[arm][cid]=None;continue
                    pred=predict_identity(r,fitted[arm],feature);assert pred['abstain']==saved['abstain'] and abs(pred['pD']-saved['pD'])<3e-9 and abs(pred['raw']-saved['raw'])<3e-9
                    if pred['cut']:masks[arm].append(cid)
                    probabilities[arm][cid]=saved
            masks['M']=matched_identity_null(rows,masks['H_I'])
            for arm,ids in masks.items():assert sorted(ids)==sorted(member['selected'][arm]) and member['masks'][arm]==[r['candidate_id'] in ids for r in rows]
            decisions[key]=masks
        audit[target]=dict(known_D_S_sources=len(source),eligible_sources=model['eligible_source_rows'],source_domains=model['source_domains'],known_labels={c:sum(r['class']==c for r in source) for c in ('D','S')})
    assert len(decisions)==60
    report=dict(passed=True,prediction_manifest=seal,source_support=audit,all_mean_std_beta_intercepts=True,all_weights_and_masks=True,unknown_not_used_for_fit=True,capacity_gate='none; always trained',verifier_sha256=sha(__file__))
    save_receipt(audit_path(root,'independent_models.json'),report)
    return report,decisions,probabilities


def aggregate_graphs(units,scores,arms):
    result={};exact={};grouped={}
    for h in ('35','33','W','G'):
        result[h]={};exact[h]={}
        for arm in arms:
            ee=[];ff={t:[] for t in ('0.5','0.3')};rows=[]
            for gid in sorted({g['group'] for g in units.values() if g['domain']==h}):
                clips=[g for g in units.values() if (g['domain'],g['group'])==(h,gid)];N=D=0;member_metrics={t:[] for t in ff}
                for g in clips:
                    mm=[scores[g['unit'],str(m['member']),arm] for m in g['members']];gt=mm[0]['gt_bins']
                    N+=int(abs(sum(m['pred_bins'] for m in mm)-3*gt).sum());D+=3*int(gt.sum())
                for t in ff:
                    f=[]
                    for i in range(3):
                        rr=[scores[g['unit'],str(g['members'][i]['member']),arm]['iou'][t] for g in clips]
                        totals={k:sum(r[k] for r in rr) for k in ('tp','pred','gt')};member_metrics[t].append(totals);f.append(Fraction(2*totals['tp'],max(1,totals['pred']+totals['gt'])))
                    ff[t].append(sum(f)/3)
                ee.append(Fraction(100*N,D));rows.append(dict(group=gid,numerator=N,denominator=D,members=member_metrics))
            exact[h][arm]=dict(E3=sum(ee)/len(ee),F1={t:sum(v)/len(v) for t,v in ff.items()})
            result[h][arm]=dict(E3=float(exact[h][arm]['E3']),F1={t:float(x) for t,x in exact[h][arm]['F1'].items()});grouped[h,arm]=rows
    return result,exact,grouped


def audit_scores(repo,root,records,families,bundles,GT,v,u,decisions,probabilities,oracle_masks,head_masks,capacity_report):
    score=root/'score';seal=u.verify_sealed(score);assert not REQUIRE_ORIGINAL_SEALS or seal=='ae37297a0080a55d09b4379bf0fde5713ba2e24be255665cfe22c3b3085226bc'
    native=read(score/'native_scores.json');summary=read(score/'summary.json');units={b[4]['unit']:b[4] for b in bundles.values()}
    arms=('K','I198','O_D','O_DU','H_N','H_B','H_I','M');scores={};graph_count=0
    for key,ids in oracle_masks.items():decisions[key[:2]][key[2]]=ids
    replay=read(score/'classification_inputs.json');assert len(replay)==len({r['candidate_id'] for r in replay})==1196
    for r in replay:
        own=records[r['candidate_id']];key=(own['unit'],str(own['member']))
        for name in ('candidate_id','unit','domain','group','video','member','member_index','eligible','class','class03','delta','direction','frame','censor_supported','censor_type','censor_reason'):compare(own[name],r[name])
        assert r['selected']=={arm:r['candidate_id'] in decisions[key][arm] for arm in arms}
        compare({arm:probabilities[arm][r['candidate_id']] for arm in probabilities},r['probabilities'])
    for arm in arms:
        for video in native[arm]:
            for m in video['members']:
                key=(video['unit'],str(m['member']));ids=decisions[key][arm];assert sorted(ids)==sorted(m['selected_ids'])
                if arm in ('H_N','H_B','H_I','M','I198'):assert all(records[cid]['eligible'] for cid in ids)
                if arm=='O_D':assert all(records[cid]['class']=='D' for cid in ids)
                if arm=='O_DU':assert all(records[cid]['class']!='S' for cid in ids)
                own=v.replay_mask(families[key],ids);own['gt_bins']=v.bin_events([r[1] for r in families[key]['gt']],families[key]['nf']);other=m['metrics']
                assert (own['nG'],own['nP'])==(other['nG'],other['nP']) and np.array_equal(own['pred_bins'],other['pred_bins']) and np.array_equal(own['gt_bins'],other['gt_bins'])
                for t in ('0.5','0.3'):assert all(own['iou'][t][k]==other['iou'][t][k] for k in ('tp','pred','gt'))
                scores[*key,arm]=own;graph_count+=1
    assert graph_count==480
    heads={}
    for h in ('35','33'):
        for key,ids in head_masks.items():
            if key[2]!=h:continue
            result=v.replay_mask(families[key[:2]],ids);result['gt_bins']=v.bin_events([r[1] for r in families[key[:2]]['gt']],families[key[:2]]['nf']);heads[key]=result
    calls=read(score/'physical_receipts.json');cache={};role_counts={}
    for call in calls:
        key=(call['unit'],str(call['member']));arm=call['arm'];ids=head_masks[*key,arm.split('_')[-1]] if arm.startswith('min_S_') else decisions[key][arm]
        assert sorted(ids)==sorted(call['selected_ids']);role_counts[arm]=role_counts.get(arm,0)+1
        token=(key,tuple(sorted(ids)))
        expected_token=hashlib.sha256(json.dumps([key[0],key[1],sorted(ids)],sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest();assert call['path']==expected_token
        if token in cache:assert cache[token]==call['path'];continue
        directory=score/'physical'/call['path'];assert u.verify_sealed(directory)==call['seal'];raw,C,index,blocks,g,sm=bundles[key]
        actual=np.loadtxt(directory/'mot.txt',delimiter=',',usecols=range(7),ndmin=2);expected=u.realize_independent(raw,C,index,blocks,sm['bundle']['candidates'],ids);assert np.array_equal(actual,expected)
        structure=u.track_structure(actual,g['polygon']);base=u.track_structure(raw,g['polygon']);assert np.array_equal(structure['frame_order'],base['frame_order'])
        receipt=read(directory/'receipt.json')['identity'];metric=read(directory/'score.json');graph=v.replay_mask(families[key],ids)
        if g['video'] not in GT:
            path=repo/'runs/gt'/g['split']/g['video']/'gt/gt.txt';GT[g['video']]=np.loadtxt(path,delimiter=',',usecols=range(6),ndmin=2)
        assert receipt['raw_sha256']==array_digest(raw) and receipt['rows_sha256']==array_digest(actual) and receipt['gt_sha256']==array_digest(GT[g['video']]) and receipt['polygon_sha256']==array_digest(g['polygon']) and receipt['selected_ids']==sorted(ids)
        assert np.array_equal(v.bin_events([f*s for f,s in structure['events'].values()],g['nf']),graph['pred_bins'])
        assert (graph['nG'],graph['nP'])==(metric['nG'],metric['nP'])
        for t,ek in (('0.5','edges05'),('0.3','edges03')):
            assert sorted([p,gid] for p,gids in graph['edges'][t].items() for gid in gids)==metric['evidence'][ek]
            assert all(graph['iou'][t][k]==metric['iou'][t][k] for k in ('tp','pred','gt'))
        cache[token]=call['path']
    assert len(calls)==216 and role_counts=={'O_D':60,'O_DU':60,'H_I':60,'H_N':4,'H_B':4,'M':4,'min_S_35':12,'min_S_33':12}
    means,exact,groups=aggregate_graphs(units,scores,arms)
    for h in means:
        for arm in arms:
            other=summary['domains'][h]['methods'][arm];compare(means[h][arm],other)
            for row in groups[h,arm]:
                saved=next(r for r in other['groups'] if r['group']==row['group']);assert (row['numerator'],row['denominator'])==(saved['numerator'],saved['denominator'])
                for t in ('0.5','0.3'):
                    for a,b in zip(row['members'][t],saved['iou'][t]['members']):compare(a,b)
    classification_cells=0
    for h in ('35','33','W','G'):
        allrows=[r for r in records.values() if r['hive']==h]
        for scope,rows in (('all_candidates',allrows),('deployment_eligible',[r for r in allrows if r['eligible']])):
            ids={r['candidate_id'] for r in rows}
            for arm in arms:
                selected=[cid for key,masks in decisions.items() if bundles[key][4]['domain']==h for cid in masks[arm] if cid in ids]
                normalized=[dict(r,**{'class':r['class']}) for r in rows]
                own=classification_stats(normalized,selected,probabilities.get(arm));compare(own,summary['classification'][h][arm][scope]);classification_cells+=1
    head_proofs={}
    for h in ('35','33'):
        gs=[g for g in units.values() if g['domain']==h];ff={t:[] for t in ('0.5','0.3')};NN=0;EE=[]
        for g in gs:
            mm=[heads[g['unit'],str(m['member']),h] for m in g['members']];N=int(abs(sum(m['pred_bins'] for m in mm)-3*mm[0]['gt_bins']).sum());D=3*int(mm[0]['gt_bins'].sum());NN+=N;EE.append(Fraction(100*N,D))
            for t in ff:ff[t].append(sum(Fraction(2*m['iou'][t]['tp'],max(1,m['iou'][t]['pred']+m['iou'][t]['gt'])) for m in mm)/3)
        mean_E=sum(EE)/4;F={t:sum(vals)/4 for t,vals in ff.items()};saved=summary['head_min_S'][h]
        assert NN==capacity_report['minimum_S'][h]['count_numerator']==saved['raw_numerator'] and abs(float(mean_E)-saved['mean_E3'])<1e-12
        assert all(abs(float(F[t])-saved['F1'][t])<1e-14 for t in ff)
        assert mean_E<exact[h]['K']['E3']-Fraction(14 if h=='35' else 13,10)
        no_F1_loss=F['0.5']>=exact[h]['K']['F1']['0.5'];assert no_F1_loss
        S=capacity_report['minimum_S'][h]['S_upper'];unique=capacity_report['minimum_S'][h]['S_witness_unique_GT_runs']
        assert unique==(1 if S>0 else 0)
        head_proofs[h]=dict(count_witness_E3=float(mean_E),K_F1_05=float(exact[h]['K']['F1']['0.5']),witness_F1_05=float(F['0.5']),F1_05_no_loss=no_F1_loss,
            minimum_S_with_extra_head_mean_F1_at_least_K=S,minimum_distinct_affected_GT_runs=unique,
            proof='Adding a mean-F1 constraint preserves the existing lower bound; the same physical witness retains its S upper bound. Positive S minimum implies distinct-run cost>=1, and this witness supplies distinct cost1 (or zeroS gives zero).',
            limits='No per-seed F1 constraint, no.3-balanced/F1 maximum claim; no assertion this particular GT run occurs in every feasible solution or biological identity count.')
    investment=(exact['35']['K']['E3']-exact['35']['H_I']['E3']>Fraction(14,10) and exact['33']['K']['E3']-exact['33']['H_I']['E3']>Fraction(13,10) and exact['W']['H_I']['E3']-exact['W']['K']['E3']<=Fraction(14,10) and all(exact[h]['H_I']['F1']['0.5']>=exact[h]['K']['F1']['0.5']-Fraction(5,1000) for h in ('35','33','W')))
    attribution=all(exact[h]['H_I']['E3']<exact[h][arm]['E3'] for h in ('35','33') for arm in ('H_B','M'))
    assert investment==summary['investment_pass'] and attribution==summary['identity_attribution_pass']
    return dict(passed=True,score_manifest=seal,graph_streams=graph_count,classification_cells=classification_cells,physical_requested=len(calls),physical_unique=len(cache),physical_roles=role_counts,means=means,posthoc_head_F1_and_distinct_bounds=head_proofs,investment_pass=investment,identity_attribution_pass=attribution,
                scope='Exact native matrices/hashes/frame ranks/events and independent full K/A graphs validate sealed official native scores; no duplicate per-arm native IoU rematching; TP_ID not inferred.')


def synthetic():
    rows=lambda pid,x=0:np.array([[f,pid,x,0,4,4] for f in (1,2)],float)
    gt=rows(7);P=rows(1)
    for threshold in (.5,.3):
        a=full_frame_owners(gt,P,threshold);assert classify_cut(a,0,1)['label']=='S'
        changed=gt.copy();changed[1,1]=9;a=full_frame_owners(changed,P,threshold);assert classify_cut(a,0,1)['label']=='D'
        duplicate=np.r_[P,rows(2)];a=full_frame_owners(gt,duplicate,threshold);assert all(not r['isolated'] for r in a['rows'])
        missing=gt[:1];a=full_frame_owners(missing,P,threshold);assert classify_cut(a,0,1)['label']=='U'
    # Same original ID separated by the official gap is explicitly unknown.
    gt=np.array([[1,7,0,0,4,4],[40,7,0,0,4,4]],float);P=np.array([[1,1,0,0,4,4],[40,1,0,0,4,4]],float)
    a=full_frame_owners(gt,P,.5);assert classify_cut(a,0,1)['label']=='U'
    # False count after a correct local join; later return lies beyond P.
    polygon=[[0,0],[10,0],[10,10],[0,10]]
    truth=np.array([[f,7,x-1,4,2,2] for f,x in enumerate([12,8,7,6,12],1)],float)
    predicted=truth[:4].copy();predicted[:,1]=1
    owned=full_frame_owners(truth,predicted,.5)
    assert classify_cut(owned,0,1)['label']=='S'
    evidence=censoring_anchors(owned,0,4,0,1,polygon);assert evidence['compatible'] and evidence['later_return'] and not evidence['earlier_missed']
    assert strict_raw_budget(2014,90*129,4,Fraction(14,10))==1363
    # One required S deletion. Count residual1, cut−1, strict budget0.
    proof=dict(p=[1],g=1,Q=1,B=0,lower_bound=1)
    result=verify_min_S_certificate([1],[[-1]],[1],proof);assert result['lower_bound']==1
    brute=min(cost for x,cost in ((0,0),(1,1)) if abs(1-x)<=0);assert brute==result['lower_bound']
    rng=np.random.default_rng(199);training=[]
    for h in ('35','33','W','G'):
        for group in (1,2):
            for delta in (-1,1):
                for i,label in enumerate(('D','S','U','S')):
                    x=rng.normal(size=22);training.append(dict(hive=h,group=group,clip='a',delta=delta,eligible=True,identity_class=label,features_N11=x[:11],features_B15=x[:15],features_I22=x))
    for key in ('features_N11','features_B15','features_I22'):
        fit=direct_identity_fit(training,'35',key)
        changed=[dict(r,**{key:np.ones(len(r[key]))*1e9}) if r['identity_class']=='U' or r['hive']=='35' else r for r in training]
        again=direct_identity_fit(changed,'35',key);assert np.array_equal(fit['coef'],again['coef'])
        assert predict_identity(training[0],direct_identity_fit([],'35',key),key)['abstain']
    assert auc_pairwise([1,0],[.5,.5])==.5 and auc_pairwise([1,1],[.1,.8]) is None
    return dict(passed=True,full_frame_assignment=True,mutual_singleton=True,S_D_U=True,official_gap_same_native_unknown=True,
                censoring_anchors_not_whole_identity=True,strict_integer_budget=True,min_S_integer_dual=True,
                direct_D_ridge_no_U_or_target_normalization=True,conditional_AUROC_ties=True,scope='synthetic only; no field input opened')


def field(repo,root,source197,source198,prereg,reuse_verified_owners=False):
    synthetic()
    if not reuse_verified_owners:audit_owners(repo,root,source197,source198,prereg)
    owners=read(audit_path(root,'independent_owners.json'))
    assert owners['passed'] and owners['owner_ledger_sha256']==sha(audit_path(root,'independent_owner_ledger.json'))
    records=read(audit_path(root,'independent_owner_ledger.json'))
    v,u,peer,features,families,bundles,old=context(repo,source197,source198)
    capacity,om,hm,cap=audit_capacity(root,source198,records,families,bundles,v,u)
    models,masks,probs=audit_models(root,source198,records,bundles,u)
    scores=audit_scores(repo,root,records,families,bundles,{},v,u,masks,probs,om,hm,capacity)
    save_receipt(audit_path(root,'independent_scores.json'),scores)
    return assemble(root)


def assemble(root):
    names=['owners','capacity','models','scores']
    components={name:dict(path='independent_'+name+'.json',sha256=sha(audit_path(root,'independent_'+name+'.json'))) for name in names}
    reports={name:read(audit_path(root,entry['path'])) for name,entry in components.items()}
    assert all(r['passed'] for r in reports.values())
    result=dict(schema='endpoint-identity199-independent-v1',passed=True,components=components,
        verifier_sha256=sha(__file__),input_seals={name:sha(root/name/'manifest.json') for name in ('owners','capacity_v2','predictions','score')},
        minimum_S=reports['capacity']['minimum_S'],posthoc_head_F1_and_distinct_bounds=reports['scores']['posthoc_head_F1_and_distinct_bounds'],
        censor_unique_specimens=reports['owners']['censor_unique_specimens'],scope=reports['scores']['scope'],
        limitations='No fresh per-arm official rematching; sealed native matrices and legal event graphs independently rebuilt. Adjacent-row labels and annotated GT-run dedup are not biological identity. No new optimization of target identities.')
    save_receipt(audit_path(root,'independent.json'),result)
    return result


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--field-go',action='store_true')
    parser.add_argument('--repo',type=Path,default=Path.cwd())
    parser.add_argument('--root',type=Path)
    parser.add_argument('--source197',type=Path)
    parser.add_argument('--source198',type=Path)
    parser.add_argument('--prereg',default='530ee05ab8401e4313a4fc53e50318d2a37fe74e')
    parser.add_argument('--reuse-verified-owners',action='store_true')
    parser.add_argument('--source-code-dir',type=Path)
    parser.add_argument('--audit-out',type=Path)
    parser.add_argument('--require-original-seals',action='store_true')
    args=parser.parse_args()
    if args.source_code_dir:CODE_DIR=args.source_code_dir.resolve()
    if args.audit_out:AUDIT_DIR=args.audit_out.resolve()
    REQUIRE_ORIGINAL_SEALS=args.require_original_seals
    if args.field_go and any(v is None for v in (args.root,args.source197,args.source198)):parser.error('explicit root/source197/source198 required')
    if args.field_go and AUDIT_DIR and args.reuse_verified_owners:
        import shutil
        AUDIT_DIR.mkdir(parents=True,exist_ok=True)
        for name in ('independent_owners.json','independent_owner_ledger.json'):
            src=args.root/name;assert sha(src)==src.with_suffix('.sha256').read_text().strip()
            for original in (src,src.with_suffix('.sha256')):
                target=AUDIT_DIR/original.name
                if target.exists():raise FileExistsError('fresh audit directory required')
                shutil.copy2(original,target)
    result=field(args.repo,args.root,args.source197,args.source198,args.prereg,args.reuse_verified_owners) if args.field_go else synthetic()
    print(json.dumps(result,indent=2))
