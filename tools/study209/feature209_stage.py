"""Thin GT-free209 native-gray stage, no execution before published root GO.

Six old clips only. Four-frame grayscale LRU, no saved images. All point trace,
true/sham abstention reasons and image SHA receipts are preserved numerically.
"""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']='-1'
from collections import OrderedDict
import argparse
import json
import hashlib
import gzip
import io
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import sys
import cv2
import numpy as np
import catalog209 as C
import features209 as F
cv2.setNumThreads(1)


def save(path,obj):
    with Path(path).open('w',encoding='utf8',newline='\n') as stream:
        json.dump(obj,stream,sort_keys=True,separators=(',',':'),allow_nan=False);stream.write('\n')


class GrayReader:
    def __init__(self,repo,capture,nf,expected,allowed):
        self.repo=Path(repo).resolve();self.root=(self.repo/'data/BEE24/train'/capture/'img1').resolve()
        if self.root not in allowed:raise ValueError('unapproved native clip')
        self.nf=nf;self.expected=expected;self.cache=OrderedDict();self.receipts={}

    def frame(self,date):
        if int(date)!=date or not 1<=date<=self.nf:raise ValueError('native date outside approved clip')
        date=int(date)
        if date in self.cache:self.cache.move_to_end(date);return self.cache[date]
        path=self.root/f'{date:06d}.jpg';content=path.read_bytes();digest=hashlib.sha256(content).hexdigest()
        if date in self.expected and digest!=self.expected[date]['sha256']:raise ValueError('208image content drift')
        if date in self.receipts and digest!=self.receipts[date]['sha256']:raise ValueError('image changed during stage')
        image=cv2.imdecode(np.frombuffer(content,np.uint8),cv2.IMREAD_GRAYSCALE)
        if image is None:raise ValueError('unreadable native gray image')
        self.receipts[date]=dict(frame=date,path=path.relative_to(self.repo).as_posix(),sha256=digest,
                                 height=image.shape[0],width=image.shape[1])
        self.cache[date]=image
        while len(self.cache)>4:self.cache.popitem(last=False)
        return image

    def __call__(self,date,roi):
        image=self.frame(date);x0,y0,x1,y1=roi
        if not 0<=x0<x1<=image.shape[1] or not 0<=y0<y1<=image.shape[0]:raise ValueError('ROI outside native image')
        return image[y0:y1,x0:x1].copy()


def freeze_clip(catalog,reader,identity,dest,extract=F.extract_pair):
    dest=Path(dest)
    if dest.exists():
        C.verify(dest)
        if json.loads((dest/'receipt.json').read_text())['identity']!=identity:raise ValueError('resume code/input drift')
        return
    R=np.asarray(catalog['rows']);order=np.argsort(R[:,0],kind='stable');frames,start=np.unique(R[order,0],return_index=True)
    boxes={int(f):R[order[a:b],2:6] for f,a,b in zip(frames,start,[*start[1:],len(R)])}
    native_boxes=lambda f:boxes.get(int(f),np.empty((0,4)))
    nf=catalog['nframes'];records=[]
    candidates=sorted(catalog['pairs'],key=lambda p:((nf+1) if p['representative'] is None else p['representative']['frame'],p['canonical_ids']))
    dest.parent.mkdir(parents=True,exist_ok=True);temp=Path(tempfile.mkdtemp(prefix='.'+dest.name+'-',dir=dest.parent))
    try:
        with (temp/'point_evidence.jsonl.gz').open('wb') as stream:
            with gzip.GzipFile(filename='',fileobj=stream,mode='wb',mtime=0) as gz:
                with io.TextIOWrapper(gz,encoding='utf8',newline='\n') as evidence:
                    for n,p in enumerate(candidates,1):
                        rep=p['representative']
                        if rep is None:
                            arm=dict(valid=False,reason='NO_DATE',frame=None,next_frame=None,features=None,feature_vector=None,traces={})
                            value=dict(schema=F.SCHEMA,feature_names=list(F.FEATURE_NAMES),true=dict(arm),sham=dict(arm),
                                       main_valid=False,sham_valid=False,paired_valid=False,fixed_boxes_xywh=None)
                        else:
                            shape=reader.frame(rep['frame']).shape
                            value=extract(reader,native_boxes,np.asarray(rep['boxes'],float),rep['frame'],nf,shape)
                        json.dump(dict(pair_id=p['pair_id'],extract=value),evidence,sort_keys=True,separators=(',',':'),allow_nan=False);evidence.write('\n')
                        compact=dict(value)
                        for arm in ('true','sham'):compact[arm]={k:v for k,v in value[arm].items() if k!='traces'}
                        records.append(dict(compact,pair_id=p['pair_id'],canonical_ids=p['canonical_ids'],geometry4=p['geometry4'],
                            representative_frame=None if rep is None else rep['frame'],point_evidence_file='point_evidence.jsonl.gz'))
                        if n%500==0:print('FEATURE_PAIRS',catalog['metadata']['domain'],catalog['metadata']['clip'],n,'/',len(candidates),flush=True)
        records.sort(key=lambda p:tuple(p['canonical_ids']))
        save(temp/'features.json',dict(schema='common209-features-clip-v1',prereg=identity['prereg'],
            candidate_catalog_manifest_sha256=identity['candidate_catalog_manifest_sha256'],
            domain=catalog['metadata']['domain'],clip=catalog['metadata']['clip'],feature_names=list(F.FEATURE_NAMES),pairs=records))
        save(temp/'image_receipts.json',[reader.receipts[k] for k in sorted(reader.receipts)])
        save(temp/'receipt.json',dict(identity=identity,pairs=len(records),main_valid=sum(p['main_valid'] for p in records),
            sham_valid=sum(p['sham_valid'] for p in records),paired_valid=sum(p['paired_valid'] for p in records),
            GT_access=False,model_access=False,decoded_unique_frames=len(reader.receipts),gray_frame_LRU_bound=4,
            support='main=true.valid only; sham missingness never filters primary'))
        save(temp/'manifest.json',{p.name:C.sha(p) for p in temp.iterdir()});(temp/'SEALED.sha256').write_text(C.sha(temp/'manifest.json')+'\n')
        temp.rename(dest)
    finally:
        if temp.exists():shutil.rmtree(temp)


def run(args):
    if not re.fullmatch('[0-9a-f]{40}',args.prereg or ''):raise ValueError('published prereg SHA required')
    subprocess.run(['git','merge-base','--is-ancestor',args.prereg,'origin/main'],cwd=args.repo,check=True,capture_output=True)
    log=subprocess.run(['git','show',args.prereg+':docs/experiment_log.md'],cwd=args.repo,check=True,capture_output=True).stdout.decode('utf8')
    if log.count(C.MARKER)!=1:raise ValueError('unique209marker required')
    if args.out.resolve().drive.upper()!='D:':raise ValueError('D: output required')
    C.verify(args.catalog);catalogseal=C.sha(args.catalog/'manifest.json')
    deployment=json.loads((args.catalog/'deployment.json').read_text())
    if deployment['source208']!=C.SOURCE208 or deployment['prereg']!=args.prereg:raise ValueError('wrong catalog parent/prereg')
    if C.sha(args.inputs/'manifest.json')!=C.SOURCE208:raise ValueError('208parent mismatch')
    inputs_manifest=json.loads((args.inputs/'manifest.json').read_text())
    if {(c['domain'],c['clip'],c['seed']) for c in deployment['clips']}!=set(C.SPECS):raise ValueError('sixclip roster mismatch')
    allowed={(args.repo/'data/BEE24/train'/(clip[:-5] if clip.endswith('-head') else clip)/'img1').resolve() for domain,clip,seed in C.SPECS}
    records=[]
    for record in deployment['clips']:
        catalog=C.load_clip(args.catalog/record['path'],args.inputs)
        metadata=catalog['metadata'];receipt_path=record['path']+'/image_receipts.json'
        if C.sha(args.inputs/receipt_path)!=inputs_manifest[receipt_path]:raise ValueError('208image receipts drift')
        expected={r['frame']:r for r in json.loads((args.inputs/receipt_path).read_text())}
        reader=GrayReader(args.repo,metadata['capture'],metadata['nf'],expected,allowed)
        identity=dict(prereg=args.prereg,candidate_catalog_manifest_sha256=catalogseal,catalog_clip_seal=record['seal'],
            source208=C.SOURCE208,code_sha256={p.name:C.sha(p) for p in (Path(__file__),Path(F.__file__),Path(C.__file__))})
        dest=args.out/record['path'];freeze_clip(catalog,reader,identity,dest)
        records.append(dict(domain=record['domain'],clip=record['clip'],seed=record['seed'],path=record['path'],
                            features_file=record['path']+'/features.json',seal=(dest/'SEALED.sha256').read_text().strip()))
        print('FEATURE_SEALED',record['domain'],record['clip'],flush=True)
    top=dict(schema='common209-features-v1',prereg=args.prereg,candidate_catalog_manifest_sha256=catalogseal,
             source208=C.SOURCE208,clips=records)
    if (args.out/'deployment.json').exists() and json.loads((args.out/'deployment.json').read_text())!=top:raise ValueError('top resume drift')
    save(args.out/'deployment.json',top)
    files={p.relative_to(args.out).as_posix():C.sha(p) for p in args.out.rglob('*') if p.is_file() and p not in (args.out/'manifest.json',args.out/'SEALED.sha256')}
    if (args.out/'manifest.json').exists() and json.loads((args.out/'manifest.json').read_text())!=files:raise ValueError('top file drift')
    save(args.out/'manifest.json',files);(args.out/'SEALED.sha256').write_text(C.sha(args.out/'manifest.json')+'\n')
    print('FEATURES6',C.sha(args.out/'manifest.json'))


def synthetic_self_test():
    class Reader:
        def __init__(self):
            self.image=np.random.default_rng(209).integers(0,256,(96,128),dtype=np.uint8);self.receipts={}
        def frame(self,date):
            self.receipts[int(date)]=dict(frame=int(date),synthetic=True);return self.image
        def __call__(self,date,roi):
            x0,y0,x1,y1=roi;return self.frame(date)[y0:y1,x0:x1].copy()
    rows=np.array([[1,1,20,20,30,30,.9],[1,2,35,20,30,30,.9]],float)
    pair=dict(pair_id='1|2',canonical_ids=[1,2],geometry4=[1/3,1/3,np.log(2),1.],
              representative=dict(frame=1,boxes=rows[:,2:6].tolist()))
    catalog=dict(rows=rows,pairs=[pair],nframes=80,metadata=dict(domain='synthetic',clip='synthetic'))
    with tempfile.TemporaryDirectory() as tmp:
        dest=Path(tmp)/'out';reader=Reader();identity=dict(prereg='synthetic',candidate_catalog_manifest_sha256='synthetic')
        freeze_clip(catalog,reader,identity,dest);C.verify(dest)
        value=json.loads((dest/'features.json').read_text())['pairs'][0]
        assert value['pair_id']=='1|2' and value['main_valid']==value['true']['valid']
        freeze_clip(catalog,reader,identity,dest)
    return dict(passed=True,scope='synthetic real extract API/trace JSON/full manifest/resume only')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repo',required=True,type=Path);p.add_argument('--inputs',required=True,type=Path)
    p.add_argument('--catalog',required=True,type=Path);p.add_argument('--out',required=True,type=Path);p.add_argument('--prereg',required=True)
    args=p.parse_args();sys.path.insert(0,str(args.repo.resolve()));run(args)
