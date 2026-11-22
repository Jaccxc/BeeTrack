"""Seal completed independent214 stage receipts without new scoring."""
import argparse,shutil
from pathlib import Path
import field214 as f
p=f.p

def run(a):
    assert a.field_go
    # Bind the original18 carrier files and source-key arrays to the historical
    # manifest, not only to the new producer's copy of their claimed hashes.
    import numpy as np
    original=Path(a.source212);assert p.sha(original/'manifest.json')=='61774c5f725073b1d887fdc06a8dd3a390a7566406eef12fe31c3fc859fd3b2b'
    manifest=p.read(original/'manifest.json')
    for ref in p.read(a.root/'frozen/deployment.json')['clips']:
        folder=a.root/'frozen'/ref['path'];meta=p.read(folder/'roster.json.gz');path=meta['source212_path'];assert path==ref['path']
        for name in ('native.npz','sources.json.gz'):assert p.sha(original/path/name)==manifest[path+'/'+name]
        assert p.read(folder/'sources.json.gz')==p.read(original/path/'sources.json.gz')
        with np.load(folder/'native.npz') as z:
            for i in range(3):assert np.array_equal(z['canon'+str(i)],p.canonical(z['receiver' if i==0 else 'peer'+str(i)]))
    reports={}
    for name in ('pixels.json','labels.json','models.json','physical.json'):
        assert p.read(a.out/name)['passed'];reports[name]=p.sha(a.out/name)
    photo=p.read(a.out/'pixels.json');physics=p.read(a.out/'physical.json')
    source=Path(__file__).parent;snapshot=a.out/'source_snapshot';snapshot.mkdir(exist_ok=True)
    files=('audit214.py','field214.py','physical_audit214.py','finalize214.py')
    for name in files:shutil.copyfile(source/name,snapshot/name)
    p.save(a.out/'independent.json',dict(schema='center214-independent-v1',passed=True,prereg=f.PREREG,supplement=f.SUPP,input_seals=dict(f.PINS,score=a.score_sha),reports=reports,
        source_sha256={name:p.sha(source/name) for name in files},
        scope=dict(original_K=18,physical_states=90,fullscene_geometry_IoUs=[.5,.3],pixels='All represented endpoint-date original JPEG hashes and decoded images, every eligible16x16 RGB patch/G6/odd24, no endpoint sampling',
            pixel_images=sum(c['images'] for c in photo['clips']),eligible_crops=sum(c['eligible_features'] for c in photo['clips']),all_endpoint_denominator=sum(c['all_endpoints'] for c in photo['clips']),
            original_GT_targets='All18 fullscene original Hungarian assignments, actual row for repeated GT canonical ID/date, no isolated gate',models='Four source-only receiverK folds,8 independent ridge fits, all18 K/G/I/Z shifts; O regenerated late from original actual GT row',
            scoring='90 MOT exact row order/date/ID/width/height/score and endpoint-only geometry; fresh fullscene Hungarian geometry at bothIoUs, independent DFS cardinality, independent native dates/sides/30phase windows',
            shared='cv2 native remap/polygon primitives, SciPy Hungarian solver, public197 event legal-edge/TP_ID semantic kernel; independent loops and formulas elsewhere; no production214 fitter/scorer import',
            envelopes='All full joint endpoint bit menus and A/B minmax count intervals; independent memberW and WG member-pooling supplement',
            limitations='A is relaxed and need not be physically attainable; B only bounds original/O-side family, not arbitrary learned displacement; O is reference not optimized upper bound. Matched endpoint spatial error does not establish physical bee identity or unbiased external generalization.'),
        primary_checks=physics['primary_checks'],mechanism=physics['mechanism']))
    p.save(a.out/'manifest.json',{path.relative_to(a.out).as_posix():p.sha(path) for path in a.out.rglob('*') if path.is_file() and path.name not in ('manifest.json','SEALED.sha256','manifest.json.sha256')})
    (a.out/'SEALED.sha256').write_text(p.sha(a.out/'manifest.json')+'\n')
    print('AUDIT214_FINAL_PASS',p.sha(a.out/'independent.json'),'SEAL',p.sha(a.out/'manifest.json'),flush=True)

if __name__=='__main__':
    q=argparse.ArgumentParser();q.add_argument('--source212',type=Path,required=True);q.add_argument('--root',type=Path,required=True);q.add_argument('--out',type=Path,required=True);q.add_argument('--score-sha',required=True);q.add_argument('--field-go',action='store_true');run(q.parse_args())
