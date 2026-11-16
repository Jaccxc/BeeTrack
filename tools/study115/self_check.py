"""Exercise the deployable exporter, frozen benchmark stages and replay with invented MOT data."""

import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from core import write_export
from export_e3 import export
from freeze_e3 import freeze_group, group_key, inventory, read_checked, specs, verify_pack


def fixture(seeds=(0,1,2)):
    poly = [[0,0],[20,0],[20,30],[0,30]]
    streams = {}
    for rank,seed in enumerate(seeds):
        rows = []
        for t,f in enumerate((24,26,115,750,752)):
            # Five reportable native tracks per detector, including same-frame duplicates and bin edges.
            y = t+2 if rank!=1 else t+3
            destination = f+(rank if t in (0,2) else 0)
            xy = (-3,-1) if t%2==0 else (-1,-3)
            rows.extend([[destination-1,t+1,xy[0],y,4,2,.95], [destination,t+1,xy[1],y,4,2,.95]])
        rows.extend([[100,18,5,10,2,2,.4],[101,18,6,10,2,2,.4]])
        streams[seed] = np.asarray(rows,float)
    return streams,poly


def main():
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        streams,poly = fixture()
        paths = [root/f'mot{s}.txt' for s in streams]
        for p,a in zip(paths,streams.values()):
            write_export(p,a)
        zones = root/'polygon.json'
        zones.write_text(json.dumps({'bee':poly}),encoding='utf-8')
        dest = root/'deploy'
        expected_sha = export(paths,zones,'bee',800,dest)
        assert export(paths,zones,'bee',800,dest) == expected_sha
        record = json.loads(read_checked(dest/'freeze.json',expected_sha))
        verify_pack(record)
        assert len(record['selectors']['controls'])==99 and max(record['quota']['pool_sizes'])==3
        assert len({str(x) for x in record['selectors']['controls']})>1
        for arm,want in record['mot_sha256'].items():
            rows = read_checked(dest/(arm+'.txt'),want).decode().splitlines()
            assert len(rows)==sum(map(len,streams.values())) and all(len(r.split(','))==10 for r in rows)
        tamper = copy.deepcopy(record)
        tamper['selectors']['controls'][0][0] = [17,100]
        try: verify_pack(tamper)
        except ValueError: pass
        else: raise AssertionError('Corrupted uniform-control selection accepted')
        # In a real benchmark inputs.py additionally enforces the role-specific historical fingerprints.
        prereg='1'*40
        for role in ('source','target'):
            for fold,part,base in specs(role):
                seeds=list(range(base,base+3)); group,poly=fixture(seeds)
                hive = 68-fold if role=='source' else fold
                meta=dict(role=role,fold=fold,part=part,first_seed=base,seeds=seeds,hive=hive,
                          clip=f'synthetic-{hive}-{part}',nframes=800)
                freeze_group(group,poly,meta,root/role/group_key(fold,part,base),prereg)
            idx = inventory(root,role,prereg)
            assert len(json.loads(read_checked(root/(role+'_index.json'),idx))['groups']) == len(specs(role))
        altered = root/'target'/'33_0_head'/'geometry.txt'
        altered.write_bytes(altered.read_bytes()+b'\n')
        try: inventory(root,'target',prereg)
        except ValueError: pass
        else: raise AssertionError('Changed submitted MOT bytes went undetected')
    print('Synthetic deploy + 12 source/6 target freezes passed; 99 controls, counts, immutability and MOT hashes replayed.')


if __name__=='__main__':
    main()
