"""Incremental206 numeric archive replay; no raw MOT, pixels or GT-file reads."""
import argparse,json,gzip,math,sys
from pathlib import Path
from collections import Counter,defaultdict
from fractions import Fraction
sys.path.insert(0,str(Path.cwd()));sys.path.insert(0,str(Path.cwd()/'tools/study203'))
from archive203 import Archive203,open_archive
from replay203_core import equal,physical_metric
from tools.study203.probes203 import earliest_return
from tools.study202.replay202 import signed_bins,matching
from tools.study202.stream202 import verified_path
from replay206_core import bins,flatten,bound,aggregate_bound,replay_greedy,self_test

class Archive:
 def __init__(self,root):
  self.root=Path(root);self.manifest=json.loads((self.root/'manifest.json').read_text());assert self.manifest['schema']=='return-query206-public-increment-v1'
  self.entries={(e['role'],e['key']):e for e in self.manifest['shards']};assert len(self.entries)==len(self.manifest['shards'])
  for e in self.entries.values():verified_path(self.root,e)
 def get(self,role,key='all'):
  with gzip.open(verified_path(self.root,self.entries[role,key]),'rt',encoding='utf-8')as f:return json.load(f)
 def keys(self,role):return [key for rr,key in self.entries if rr==role]

def qlist(unit):return [(m,q)for m in unit['members']for q in m['queries']]
def f1(metric,t):
 v=metric['iou'][t];return Fraction(2*v['tp'],max(v['pred']+v['gt'],1))
def aggregate(rows):
 result={}
 for h in sorted({r['domain']for r in rows}):
  gg=[]
  for group in sorted({r['group']for r in rows if r['domain']==h}):
   clips=[r for r in rows if r['domain']==h and r['group']==group];num=den=0;members=defaultdict(lambda:{'Wnum':0,'Wden':0,'0.5':[0,0,0],'0.3':[0,0,0]})
   for c in clips:
    gt=flatten(c['members'][0]['metrics']['gt_bins']);summed=[0]*len(gt);den+=3*sum(gt)
    for m in c['members']:
     x=m['metrics'];p=flatten(x['pred_bins']);assert flatten(x['gt_bins'])==gt;summed=[a+b for a,b in zip(summed,p)]
     v=members[str(m['member'])];v['Wnum']+=sum(abs(a-b)for a,b in zip(p,gt));v['Wden']+=30*x['nG']
     for t in ('0.5','0.3'):
      z=x['iou'][t]
      for i,key in enumerate(('tp','pred','gt')):v[t][i]+=z[key]
    num+=sum(abs(p-3*g)for p,g in zip(summed,gt))
   assert len(members)==3 and den>0
   gg.append((Fraction(100*num,den),sum((Fraction(100*v['Wnum'],v['Wden'])for v in members.values()),Fraction())/3,{t:sum((Fraction(2*v[t][0],max(v[t][1]+v[t][2],1))for v in members.values()),Fraction())/3 for t in ('0.5','0.3')}))
  result[h]={'E3':sum((g[0]for g in gg),Fraction())/len(gg),'W':sum((g[1]for g in gg),Fraction())/len(gg),'F1':{t:sum((g[2][t]for g in gg),Fraction())/len(gg) for t in ('0.5','0.3')}}
 return result

def iou(a,b):
 ax,ay,aw,ah=a[2:6];bx,by,bw,bh=b[2:6];inter=max(0,min(ax+aw,bx+bw)-max(ax,bx))*max(0,min(ay+ah,by+bh)-max(ay,by));return inter/(aw*ah+bw*bh-inter)

def run(root,reference,reference202,work_dir):
 a=Archive(root);summary=a.get('summary');numeric={u['unit']:u for u in a.get('numeric_inputs')['units']};units={k:a.get('owner',k)for k in a.keys('owner')};assert len(units)==20 and len(numeric)==20
 classes={h:{t:Counter()for t in ('0.5','0.3')}for h in ('35','33','W','G')};runs={};probe_counts={};baselines={};queries={};frozen_queries={};nf={}
 assert Path(reference).suffix=='.json' and json.loads(Path(reference).read_text())['zip_sha256']=='ef84315ba8611c7685c899714d38e9d35c43838825ae4de1a6f027bf119d59dd'
 with open_archive(reference,work_dir)as previous_root:
  previous=Archive203(previous_root,reference202)
  assert a.manifest['references']['study203']['zip_sha256']=='ef84315ba8611c7685c899714d38e9d35c43838825ae4de1a6f027bf119d59dd'
  for video in previous.keys('GT_universe_clip_meta'):
   for rr in previous.array('GT_universe_runs',video):
    obs=rr['observations'];probe=earliest_return(obs);assert(probe is None)==(rr['eligible_probe_or_null']is None)
    runs[video,rr['canonical_gt_id']]=dict(original=rr['original_gt_id'],run=rr.get('run'),first=obs[0]['frame'],last=obs[-1]['frame'],gaps=obs[-1]['frame']-obs[0]['frame']+1-len(obs),positive=all(o['positive_extent']for o in obs),zero=obs[0]['inside']==obs[-1]['inside'],side_last={side:max((o['frame']for o in obs if bool(o['inside'])==side),default=-1)for side in (False,True)},probe=probe)
  for key,u in units.items():
   nf[key]=u['nf'];roster=a.get('roster',key);old=previous.get('native_clip','K|'+key);byold={str(m['member']):m for m in old['members']};byroster={str(m['member']):m for m in roster['members']}
   for m in u['members']:
    member=str(m['member']);base=previous.metric(byold[member]['physical_key']);baselines[key,member]=base;r=byroster[member];frozen={q['query_id']:q for q in r['queries']};assert len(frozen)==len(m['queries'])
    assert sorted((q['canonical_id'],q['event']['frame']*q['event']['direction'])for q in m['queries'])==sorted(tuple(p[:2])for p in base['evidence']['predictions'])
    for q in m['queries']:
     fq=frozen[q['query_id']];frozen_queries[q['query_id']]=fq;queries[q['query_id']]=(key,member,q);equal(q['eligible'],fq['eligible']);equal(q['event'],fq['event']);assert fq['boundary_inclusive']and fq['immutable_native_ID']
     anchors=fq['anchor_rows'];last=anchors[-1];L=sum(math.hypot(row[4],row[5])for row in anchors[-2:])/2;equal(L,fq['L']);equal(4*L,fq['radius']);equal([last[2]+last[4]/2,last[3]+last[5]/2],fq['center']);assert fq['search_frames']==list(range(fq['endpoint_frame']+1,min(fq['endpoint_frame']+25,u['nf'])+1))
     for t,label in q['labels'].items():
      classes[u['domain']][t][label['classification']]+=1
      if 'canonical_gt_id'in label:
       gt=runs[u['video'],label['canonical_gt_id']];assert label['GT_run_key']==[u['video'],gt['original'],gt['run']];assert label['gt_run_first']==gt['first']and label['gt_run_last']==gt['last']and label['missing_internal_frames']==gt['gaps']
      anchor_same=len(label['anchors'])==3 and all(x['qualified']for x in label['anchors'])and len({x['official_owner']for x in label['anchors']})==1
      if not q['eligible']:expected_class='ineligible'
      elif not anchor_same:expected_class='unknown'
      elif any(x['pred_inside']!=x['gt_inside']for x in label['anchors'])or gt['first']==1 or gt['last']==u['nf']or not gt['positive']:expected_class='unknown'
      elif label['GT_events']:expected_class='interpretable_nonzero_GT_event'
      elif gt['side_last'][bool(fq['starting_side'])]<=fq['endpoint_frame']:expected_class='interpretable_no_return'
      else:expected_class='confirmed_legal_return'if label['confirmed']else'return_evidence_no_legal_candidate'
      assert label['classification']==expected_class
      if label['confirmed']:
       c=label['candidate'];c['event_delete_only']=True;assert gt['zero']and gt['positive']and 1<gt['first']and gt['last']<u['nf']and q['eligible'];assert not label['GT_events']
       aa=label['anchors']+[c['added_annotation']];assert all(x['qualified']and x['official_owner']==label['canonical_gt_id']and x['pred_inside']==x['gt_inside']for x in aa)
       assert label['enumerated'][-1]['reason']=='first_legal_confirmed_return'and label['enumerated'][-1]['frame']==c['frame'];assert c['frame']in fq['search_frames'];row=c['added_row'];assert math.dist((row[2]+row[4]/2,row[3]+row[5]/2),fq['center'])<=fq['radius'];assert c['deleted_event']==q['event'];assert flatten(c['delta_windows'])==[-v for v in flatten(bins([(q['event']['frame'],q['event']['direction'])],u['nf']))]
    mini={q['query_id']:q for q in numeric[key]['members'][m['member_index']]['queries']}
    for q in m['queries']:
     for t,l in q['labels'].items():assert mini[q['query_id']]['labels'][t]['confirmed']==l['confirmed']
  # Historical leaves are references, never duplicated graphs.
  calls=a.get('physical_receipts');metrics={}
  for token in a.keys('physical_receipt'):
   receipt=a.get('physical_receipt',token);ident=receipt['identity'];key,member=ident['unit'],str(ident['member'])
   if receipt['origin']=='historical':
    hist=receipt['historical'];assert hist['source_score_seal']==a.get('input_receipt')['source203_score_seal'];metric=previous.metric(hist['source_leaf']);equal(metric['pred_bins'],baselines[key,member]['pred_bins'])
   else:metric=a.get('physical',token)
   physical_metric(metric,nf[key],signed_bins,matching);metrics[token]=metric
  allbounds=a.get('bounds');reports={}
  for name,t,eligible in [('all_queries',None,False),('eligible_queries',None,True),('0.5','0.5',False),('0.3','0.3',False)]:
   rr={}
   for key,u in units.items():
    rebuilt=bound(u,t,eligible);target=allbounds['units'][key]if name=='all_queries'else allbounds['eligible'][key]if name=='eligible_queries'else allbounds['confirmed'][t][key]
    for k in rebuilt:equal(rebuilt[k],target[k])
    rr[key]=rebuilt
   reports[name]=aggregate_bound(units.values(),rr)
   for h,v in reports[name].items():target=summary['bounds'][name][h]if t is None else summary['bounds']['confirmed'][t][h];assert Fraction(*target['fraction'])==v
  greedy=a.get('greedy');selected={};prefix_states={};step_count=0
  for group in greedy:
   uu=[u for u in units.values()if u['domain']==group['domain']and u['group']==group['group']]
   for t,saved in group['thresholds'].items():
    audits={};state={u['unit']:[]for u in uu}
    for step in saved['steps']:
     unit=step['unit'];state[unit].append(step['query_id']);audits[unit,tuple(sorted(state[unit]))]=step['joint_audit'];prefix_states[unit,'prefix:'+t+':'+str(step['step'])]=dict((k,list(v))for k,v in state.items())
    for rejection in saved['rejected']:
     # This experiment's audited traces currently have no unsafe retries;
     # a rejection ledger must carry the proposed full selection for replay.
     raise AssertionError('explicit rejection selection adapter required')
    replay_greedy(uu,t,saved,audits);step_count+=len(saved['steps'])
    prefix_selected={u['unit']:[]for u in uu}
    for step in saved['steps']:
     prefix_selected[step['unit']].append(step['query_id']);physical=step['physical_prefix'];byunit=defaultdict(list)
     for req in physical['requests']:
      uk,member=req['unit'],str(req['member']);metric=metrics[req['physical']['path']]
      ids=sorted(q for q in prefix_selected[uk]if queries[q][1]==member);assert ids==req['query_ids']
      expected=flatten(baselines[uk,member]['pred_bins'])
      for qid in ids:
       event=queries[qid][2]['event'];expected=[p-d for p,d in zip(expected,flatten(bins([(event['frame'],event['direction'])],nf[uk])))]
      assert expected==flatten(metric['pred_bins']);byunit[uk].append(metric)
      for threshold in ('0.5','0.3'):
       v=metric['iou'][threshold];assert req['TP_FP_FN'][threshold]==dict(TP=v['tp'],FP=v['pred']-v['tp'],FN=v['gt']-v['tp'])
     numerator=0
     for uk,vv in byunit.items():
      assert len(vv)==3;gt=flatten(vv[0]['gt_bins']);summed=[sum(v)for v in zip(*(flatten(m['pred_bins'])for m in vv))];numerator+=sum(abs(p-3*g)for p,g in zip(summed,gt))
     assert numerator==physical['group_numerator'] and physical['raw_count_closure_exact']

    for key,ids in saved['selected'].items():selected[key,'R05'if t=='0.5'else'R03']=ids
  scored=a.get('native_scores');callmap={(c['unit'],str(c['member']),c['arm']):c for c in calls}
  for arm,clips in scored.items():
   for clip in clips:
    for m in clip['members']:
     key,member=clip['unit'],str(m['member']);c=callmap[key,member,arm];metric=metrics[c['path']];equal(m['metrics']['pred_bins'],metric['pred_bins']);equal(m['metrics']['iou'],{t:{k:metric['iou'][t][k]for k in ('tp','pred','gt','f1')}for t in ('0.5','0.3')})
     ids=[]if arm=='K'else sorted(q for q in selected[key,arm]if queries[q][1]==member);assert m['selected_query_ids']==ids
     expected=flatten(baselines[key,member]['pred_bins'])
     for query in ids:
      event=queries[query][2]['event'];expected=[p-d for p,d in zip(expected,flatten(bins([(event['frame'],event['direction'])],nf[key])))]
     assert flatten(metric['pred_bins'])==expected
  calculated={arm:aggregate(clips)for arm,clips in scored.items()}
  for arm,domains in calculated.items():
   for h,v in domains.items():
    target=summary['domains'][h]['methods'][arm];assert Fraction(*target['E3_fraction'])==v['E3']and Fraction(*target['W_mean_member_fraction'])==v['W']
    for t in ('0.5','0.3'):assert Fraction(*target['F1_fraction'][t])==v['F1'][t]
  # Every GT-selected probe, including joint zero hits, must be present.
  for key,u in units.items():
   coverage=a.get('coverage',key);expected={pid for (video,pid),r in runs.items()if video==u['video']and r['probe']is not None};assert{r['canonical_gt_id']for r in coverage['runs']}==expected
   for probe in coverage['runs']:
    gt=runs[u['video'],probe['canonical_gt_id']];assert probe['probe']['return_frame']==gt['probe']['return_frame']
    for t,v in probe['thresholds'].items():
     geo=v['geometry'];known=all(geo['GT_row'][i]>0 for i in (4,5))
     hits=[]
     for m in geo['members']:
      rows=m['frame_rows'];values=[iou(geo['GT_row'],r['row'])for r in rows]if known else []
      if known:
       for r,iv in zip(rows,values):equal(iv,r['iou'])
       assert any(iv>=float(t)for iv in values)==m['has_threshold_box']
      hits.append(m['has_threshold_box'])
     assert geo['geometric_members']==sum(hits)
     if v['time_space_known']:
      row=geo['GT_row'];space=[];eligible=[];anchors=[];confirmed=[];exact=[]
      for member,q in qlist(u):
       fq=frozen_queries[q['query_id']];center=(row[2]+row[4]/2,row[3]+row[5]/2)
       if int(row[0])not in fq['search_frames']or math.dist(center,fq['center'])>fq['radius']:continue
       space.append(q['query_id'])
       if q['eligible']:eligible.append(q['query_id'])
       label=q['labels'][t]
       if label.get('canonical_gt_id')==probe['canonical_gt_id']:anchors.append(q['query_id'])
       if label['confirmed']and label.get('canonical_gt_id')==probe['canonical_gt_id']:
        confirmed.append(q['query_id'])
        if label['candidate']['GT_canonical_index']==geo['GT_canonical_index']:exact.append(q['query_id'])
      for field,actual in [('all_query_time_space_ids',space),('eligible_query_time_space_ids',eligible),('same_run_anchor_query_ids',anchors),('same_run_confirmed_query_ids',confirmed),('exact_probe_first_candidate_query_ids',exact)]:assert v[field]==actual

   probe_counts[key]=len(expected)
  distinct={h:{t:dict(confirmed=set(),selected=set())for t in ('0.5','0.3')}for h in classes}
  for key,u in units.items():
   for member,q in qlist(u):
    for t,label in q['labels'].items():
     if label['confirmed']:
      run=tuple(label['GT_run_key']);distinct[u['domain']][t]['confirmed'].add(run)
      if q['query_id']in selected[key,'R05'if t=='0.5'else'R03']:distinct[u['domain']][t]['selected'].add(run)
  distinct={h:{t:{k:len(v)for k,v in d.items()}for t,d in dd.items()}for h,dd in distinct.items()}
  assert len(queries)==4169
  pooled_den=sum(3*sum(flatten(u['gt_bins']))for u in units.values()if u['domain']=='35');required=(pooled_den+99)//100
  assert required==summary['necessary_head35']['required_integer_gain']and (required+29)//30==summary['necessary_head35']['necessary_member_event_edits']
  edge_audits=a.get('selected_original_edges')
  for key,arm_audits in edge_audits.items():
   for arm,audit in arm_audits.items():
    ids=selected[key,arm];assert audit['selected_count']==audit['denominator']==len(ids)
    for r in audit['records']:
     qid=r['query_id'];_,member,q=queries[qid];assert qid in ids and r['canonical_id']==q['canonical_id']
     evidence=baselines[key,member]['evidence']
     for t,field in [('0.5','edges05'),('0.3','edges03')]:
      neighbors=sorted({int(g)for p,g in evidence[field]if int(p)==q['canonical_id']});assert r['legal_GT_neighbors'][t]==neighbors and r['has_any_legal_edge'][t]==bool(neighbors)
  posthoc=a.get('verification','posthoc_identity');post_counts={h:Counter()for h in classes};post_den=Counter()
  for u in units.values():
   h=u['domain']
   for member,q in qlist(u):
    post_den[h+':all_native_events']+=1
    if not q['eligible']:continue
    post_den[h+':eligible_queries']+=1;label=q['labels']['0.5'];aa=label['anchors'];qq=[x for x in aa if x['qualified']];ids={x['original_gt_id']for x in qq};owners={x['official_owner']for x in qq}
    category='observed_distinct_original_GT_IDs'if len(ids)>=2 else 'same_original_ID_distinct_GT_gap_runs'if len(ids)==1 and len(owners)>=2 else ('three_same_run_clip_or_side_problem'if any(x['pred_inside']!=x['gt_inside']for x in aa)or label.get('at_clip_start',False)or label.get('at_clip_end',False)else'three_same_run_no_listed_problem')if len(qq)==3 and len(owners)==1 else'insufficient_anchor_ownership'
    post_counts[h][category]+=1
  assert dict(post_den)==posthoc['denominators']and {h:dict(c)for h,c in post_counts.items()}==posthoc['counts']
  gate35=calculated['K']['35']['E3']-calculated['R05']['35']['E3']>=1 and calculated['R05']['35']['F1']['0.5']>=calculated['K']['35']['F1']['0.5']-Fraction(5,1000)
  return dict(distinct_GT_runs=distinct,queries=len(queries),units=len(units),unique_physical_graphs=len(metrics),fresh_graphs=len(a.keys('physical')),greedy_steps=step_count,main35_investment=gate35,classes={h:{t:dict(v)for t,v in d.items()}for h,d in classes.items()},probe_group_observations=sum(probe_counts.values()),scope='numeric query/certificate/window/annotated graph replay; external full-frame/native geometry audit remains separate')

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('evidence',nargs='?',type=Path);p.add_argument('--self-test',action='store_true');p.add_argument('--reference203',type=Path,default=Path('tools/study203/evidence.parts.json'));p.add_argument('--reference202',type=Path,default=Path('tools/study202/evidence.zip'));p.add_argument('--work-dir',type=Path);args=p.parse_args()
 if args.self_test:print(self_test())
 else:
  if args.evidence is None:p.error('explicit evidence required')
  with open_archive(args.evidence,args.work_dir)as root:print(json.dumps(run(root,args.reference203,args.reference202,args.work_dir),indent=2))
