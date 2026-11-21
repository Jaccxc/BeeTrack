"""206 incremental finalized numeric build; explicit root only, one unit at a time."""
import argparse,json,sys,hashlib
from pathlib import Path
sys.path.insert(0,str(Path.cwd()))
from tools.study202.stream202 import Writer,file_sha
from build206 import SOURCE203_ZIP,PREREG,SOURCE203_SCORE,project_query
PINS={'frozen':'066aaa245d78ee5d926cafeb709d200746e9ea9f736f820606270c35a2188d5d','owners':'02820baa4689ee0318b4e53227450cb8d9200a8d9a83e116f356db05229f514d','score':'cbedb72a4e2c833203d6ea2fa929bdd9989acb0cdc06a1b54ae7c598d1cb709a'}
COMPLETE='813abd16dd9abd730ae2879a551406dca40d8c5b0a813d05621ffcdf91139803'
def build(root,out):
 root=Path(root);sources={};w=Writer(out)
 def read(relative,expected=None):
  p=(root/relative).resolve();assert p.is_relative_to(root.resolve());digest=file_sha(p)
  if expected is not None:assert digest==expected
  sources[relative]=digest;return json.loads(p.read_text(encoding='utf-8'))
 manifests={}
 for stage,pin in PINS.items():
  manifests[stage]=read(stage+'/manifest.json',pin);assert(root/stage/'SEALED.sha256').read_text().strip()==pin
 completion=read('verification/completion.json',COMPLETE);assert completion['passed']
 def stage_read(stage,name):return read(stage+'/'+name,manifests[stage][name])
 frozen=stage_read('frozen','deployment.json');assert frozen['prereg']==PREREG and len(frozen['groups'])==20
 for ref in frozen['groups']:
  rel=ref['path']+'/queries.json';header=stage_read('frozen',rel)
  roster={k:header[k]for k in ('unit','domain','group','video','nf','polygon','split')};roster['members']=[]
  for m in header['members']:
   assert 0<=m['seed']<=14 and header['split']!='test'
   r={k:m[k]for k in ('member','member_index','seed','tag','source203_receipt')};r['roster']={k:v for k,v in m['roster'].items()if k!='queries'}
   r['queries']=[project_query(q)for q in m['roster']['queries']];roster['members'].append(r)
  w.write('roster',ref['unit'],roster)
  for role,directory in [('owner','owner_compact'),('probe','probe_units')]:
   relative='verification/'+directory+'/'+ref['path']+'.json';p=root/relative;value=read(relative,p.with_suffix('.sha256').read_text().strip());w.write(role,ref['unit'],value)
  owner_ref=next(r for r in stage_read('owners','deployment.json')['groups']if r['unit']==ref['unit'])
  coverage=stage_read('owners',owner_ref['coverage_path']);coverage['runs']=[r for r in coverage['runs']if r['probe']is not None]
  w.write('coverage',ref['unit'],coverage)
 for name in ('input_receipt','numeric_inputs','bounds','greedy','native_scores','selected_original_edges','physical_receipts','summary'):
  w.write(name,'all',stage_read('score',name+'.json'))
 calls=stage_read('score','physical_receipts.json');seen=set()
 for call in calls:
  token=call['path'];assert len(token)==64 and all(c in '0123456789abcdef'for c in token)
  if token in seen:continue
  seen.add(token);prefix='physical/'+token+'/'
  receipt=stage_read('score',prefix+'receipt.json');assert receipt['origin']==call['origin']
  w.write('physical_receipt',token,receipt)
  if call['origin']=='fresh':w.write('physical',token,stage_read('score',prefix+'score.json'))
  else:assert receipt['historical']['source_score_seal']==SOURCE203_SCORE
 for name in ('completion','independent','supply','owners','counts','scores','probe_coverage','state_links','posthoc_identity'):
  relative='verification/'+name+'.json';p=root/relative;w.write('verification',name,read(relative,p.with_suffix('.sha256').read_text().strip()))
 m=w.finish(sources,{'study203':{'commit':'3f61fabd31382cb075299735adc731422bb65558','zip_sha256':SOURCE203_ZIP,'GT_universe_sha256':'47e684d5117b7e16c8eba9176dc62ad8d237998c559fab5e45652f47acdc20ac'},'prereg':PREREG})
 m['schema']='return-query206-public-increment-v1';(Path(out)/'manifest.json').write_text(json.dumps(m,sort_keys=True,indent=2)+'\n',encoding='utf-8',newline='\n');print(json.dumps({'shards':len(m['shards']),'unique_geometry':len(seen),'gzip_bytes':sum((Path(out)/e['file']).stat().st_size for e in m['shards'])}))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--finalized-json-go',action='store_true',required=True);a=p.parse_args();build(a.root,a.out)
