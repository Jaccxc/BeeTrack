"""Bounded203 directory/ZIP reader with pinned202 baseline references."""
from contextlib import contextmanager
from functools import lru_cache
import gzip
import json
import hashlib
from pathlib import Path
import shutil
import tempfile
import zipfile
import re
from tools.study202.stream202 import Reader,verified_path
from reference202 import Reference202,SHA202


def allowed_files(manifest):
    names=['manifest.json']+[e['file'] for e in manifest['shards']]
    assert len(names)==len(set(names)) and all(re.fullmatch('[0-9a-f]{64}\\.json\\.gz',v) for v in names[1:])
    return sorted(names)


@contextmanager
def open_archive(path,work_dir=None):
    path=Path(path)
    if path.is_dir():yield path;return
    if path.suffix.lower()=='.json':
        descriptor=json.loads(path.read_text(encoding='utf-8'));assert descriptor['schema']=='peer-suffix203-split-zip-v1'
        assert descriptor['parts'] and len({p['file'] for p in descriptor['parts']})==len(descriptor['parts'])
        with tempfile.TemporaryDirectory(dir=work_dir,prefix='numeric203_parts_') as temporary:
            combined=Path(temporary)/'evidence.zip';whole=hashlib.sha256();size=0
            with combined.open('xb') as out:
                for part in descriptor['parts']:
                    name=part['file'];assert re.fullmatch(r'evidence\.zip\.[0-9]{3}',name)
                    source=(path.parent/name).resolve();assert source.is_relative_to(path.parent.resolve())
                    digest=hashlib.sha256();count=0
                    with source.open('rb') as stream:
                        for chunk in iter(lambda:stream.read(1024*1024),b''):
                            digest.update(chunk);whole.update(chunk);count+=len(chunk);size+=len(chunk);out.write(chunk)
                    assert count==part['bytes'] and digest.hexdigest()==part['sha256']
                    assert count<=80*1024*1024
            assert whole.hexdigest()==descriptor['zip_sha256'] and size==descriptor['zip_bytes']
            with open_archive(combined,work_dir) as root:
                assert hashlib.sha256((root/'manifest.json').read_bytes()).hexdigest()==descriptor['manifest_sha256']
                assert len(allowed_files(json.loads((root/'manifest.json').read_text())))==descriptor['zip_entries']
                yield root
        return
    with zipfile.ZipFile(path) as archive:
        infos=archive.infolist();assert len({x.filename for x in infos})==len(infos)
        assert archive.getinfo('manifest.json').file_size<16*1024*1024
        manifest=json.loads(archive.read('manifest.json'));names=allowed_files(manifest)
        assert sorted(archive.namelist())==names
        assert all(x.compress_type==zipfile.ZIP_STORED and not x.is_dir() and not(x.flag_bits&1) for x in infos)
        with tempfile.TemporaryDirectory(dir=work_dir,prefix='numeric203_zip_') as temp:
            root=Path(temp)
            for name in names:
                with archive.open(name) as source,(root/name).open('xb') as out:shutil.copyfileobj(source,out,65536)
            yield root


def dense(sparse):
    shape=sparse['shape'];assert len(shape)in(1,2) and all(type(n)is int and n>=0 for n in shape)
    size=shape[0] if len(shape)==1 else shape[0]*shape[1];out=[0]*size
    assert len(sparse['indices'])==len(sparse['values']) and len(set(sparse['indices']))==len(sparse['indices'])
    for index,value in zip(sparse['indices'],sparse['values']):
        assert type(index)is int and 0<=index<size and type(value)is int
        out[index]=value
    return out if len(shape)==1 else [out[i:i+shape[1]] for i in range(0,size,shape[1])]


class Archive203:
    def __init__(self,root,reference202):
        self.root=Path(root);self.manifest=json.loads((self.root/'manifest.json').read_text())
        assert self.manifest['schema']=='peer-suffix203-public-sharded-v1'
        self.entries={(e['role'],e['key']):e for e in self.manifest['shards']}
        assert len(self.entries)==len(self.manifest['shards'])
        assert self.manifest['references']['study202']['sha256']==SHA202
        self.reference=Reference202(reference202)
        self.option_index=self.get('option_index','shared')
    def keys(self,role):return [key for rr,key in self.entries if rr==role]
    def get(self,role,key):
        path=verified_path(self.root,self.entries[role,key])
        with gzip.open(path,'rt',encoding='utf-8') as stream:return json.load(stream)
    def array(self,role,key):
        path=verified_path(self.root,self.entries[role,key])
        with gzip.open(path,'rt',encoding='utf-8') as stream:
            reader=Reader(stream)
            for _ in reader.array():yield reader.value()
            assert not reader.peek()
    @lru_cache(maxsize=4)
    def option_block(self,key):return self.get('option_block',key)
    @lru_cache(maxsize=64)
    def count(self,key):return dense(self.get('count_vector',key))
    def resolve(self,value,rows=None):
        if isinstance(value,list):return [self.resolve(v,rows) for v in value]
        if not isinstance(value,dict):return value
        if set(value)=={'counts_ref'}:return self.count(value['counts_ref'])
        if set(value)=={'option_ref'}:
            key=value['option_ref'];return self.resolve(self.option_block(self.option_index[key])[key],rows)
        if set(value)=={'owner_row_ref'}:
            assert rows is not None;return rows[value['owner_row_ref']]
        out={}
        for k,v in value.items():
            if k in ('gt_run_rows_ref','full_GT_run_rows_ref'):out[k.removesuffix('_ref')]=self.get('GT_run_numeric',v)
            else:out[k]=self.resolve(v,rows)
        if 'eligible_cutoff_frames' in out:
            unit,slot,parent=out['host_key'].rsplit('|',2)
            frames=out.pop('eligible_cutoff_frames')
            out['eligible_cutoffs']=[dict(frame=frame,sha256=hashlib.sha256(f'suffix-mask-v1|{unit}|{slot}|{parent}|{frame}'.encode()).hexdigest()) for frame in frames]
            assert len(frames)==out['eligible_cutoff_count']
            selected=min(out['eligible_cutoffs'],key=lambda r:(r['sha256'],r['frame']))
            assert selected['frame']==out['cutoff_frame'] and selected['sha256']==out['cutoff_sha256']
        return out
    def metric(self,key):
        if ('physical_metric',key)in self.entries:return self.get('physical_metric',key)
        ref=self.get('physical_metric_ref202',key);old,value=self.reference.baseline(ref['unit'],ref['member'])
        assert old['physical_key']==ref['physical_key']
        old_sha=self.reference.sources['score/physical/'+old['physical_key']+'/score.json']
        assert ref['identical_original_score_sha256']==old_sha==self.manifest['sources']['score/physical/'+key+'/score.json']
        return value
