"""Private small-table projection, gated after numeric packaging CPU handoff."""
import argparse
import hashlib
import json
from pathlib import Path
from replay_support209 import read


def project(root,bundle,out):
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    exact_path=root/'exact_window_support/exact.json';triangle_path=root/'attainable_diagnostic.json'
    summary_path=root/'verification/window_support_summary.json'
    assert sha(exact_path)=='c5e7801f4ed00a37f93dd7a59911141d702dba52a689c45491b247bbbec9ae4c'
    assert sha(triangle_path)=='3c3a27465e49cf1291f4c9446bdc98a7babb96d080988a6c84ee96842f31c9da'
    assert sha(summary_path)=='5f8ab26dd235c6b831ab008c4214513c3ec114adc9bb807a5ff20a5587f18a02'
    originals=read(root/'verification/numeric_candidates.json')['clips'];main=read(bundle)
    catalog_manifest=read(root/'catalog/manifest.json');catalog_top=read(root/'catalog/deployment.json')
    assert sha(root/'catalog/manifest.json')==main['receipt']['verified_stage_seals']['catalog']
    exact=read(exact_path);triangle=read(triangle_path);indexes={};clips=[]
    for clip in sorted(originals,key=lambda c:c['clip']):
        h,name=clip['clip'].split('/');indexes[h]=indexes.get(h,0)+1;anon='c'+format(indexes[h],'02d')
        ids=sorted({int(v) for r in clip['rows'] for v in r['pair_id'].split('|')});remap={v:i+1 for i,v in enumerate(ids)}
        mapping={r['pair_id']:'|'.join(str(remap[int(v)]) for v in r['pair_id'].split('|')) for r in clip['rows']}
        target=next(c for c in main['clips'] if c['domain']==h and c['clip']==anon)
        assert set(mapping.values())==set(r['pair_id'] for r in target['candidates'])
        diagnostic=next(c for c in triangle['clips'] if c['domain']==h and c['clip']==name)
        saved=next(c['exact'] for c in exact['clips'] if c['domain']==h and c['clip']==name)
        active=[]
        for action in diagnostic['pairs']:
            if action['pair_id'] not in diagnostic['common_slate']:continue
            sparse=[[2*i+j,int(v)] for i,row in enumerate(action['delta_windows']) for j,v in enumerate(row) if v]
            if sparse:active.append(dict(pair_id=mapping[action['pair_id']],delta=sparse))
        declared={key:saved[key] for key in ('n','k','active_count','baseline_numerator','denominator','feasible_active_subsets','distinct_window_vectors')}
        declared.update(minimum_numerator=saved['minimum']['numerator'],maximum_numerator=saved['maximum']['numerator'],
            minimum_vector_sha256=saved['minimum']['vector_sha256'],maximum_vector_sha256=saved['maximum']['vector_sha256'])
        ref=next(c for c in catalog_top['clips'] if c['domain']==h and c['clip']==name)
        relative=ref['path']+'/catalog.json.gz';catalog_path=root/'catalog'/relative
        assert sha(catalog_path)==catalog_manifest[relative]
        nf=read(catalog_path)['metadata']['nf']
        clips.append(dict(domain=h,clip=anon,nframes=nf,catalog_metadata_sha256=sha(catalog_path),
            slate=[mapping[k] for k in diagnostic['common_slate']],
            k=diagnostic['declared_k'],active_columns=active,declared=declared))
    value=dict(schema='common209-public-window-support-v1',main_bundle_sha256=sha(bundle),
        registration='ec0c0ef4d761b950ad551e00941bd2974a13b716',exact_json_sha256=sha(exact_path),
        independent_minimum_summary_sha256=sha(summary_path),clips=clips,
        scope='K and GT windows reused from main numeric bundle; sparse deltas only. Independent physical minimum check covers both heads; no new F1.')
    out.write_text(json.dumps(value,sort_keys=True,separators=(',',':'))+'\n',encoding='utf-8',newline='\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True);parser.add_argument('--bundle',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--go',action='store_true',required=True)
    args=parser.parse_args();project(args.root,args.bundle,args.out)
