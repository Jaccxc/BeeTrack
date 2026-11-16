"""Freeze native representatives, 99 same-second controls, and exact E3 count certificates."""

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
from tempfile import TemporaryDirectory

import numpy as np

from core import export_variants, integer_events, write_export
from tools.study103.swap_score import counts_from_events


FORMAT = 'e3-event-mot-v1'


def wire(obj):
    return (json.dumps(obj,allow_nan=False,separators=(',',':'),sort_keys=True)+'\n').encode()


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read_checked(path, digest):
    data = path.read_bytes()
    if sha(data) != digest:
        raise ValueError(f'File SHA-256 mismatch: {path}')
    return data


def select_controls(manifest, meta):
    signed = {}
    for item in manifest['observations']:
        s, row = item['seed'], item['row']
        signed[s,row[0]] = int(row[1])
    if manifest['seeds'] != meta['seeds'] or meta['first_seed'] != meta['seeds'][0]:
        raise ValueError('Detector seed provenance does not match the selector')
    blocks = [b for d in ('in','out') for b in manifest['blocks'][d] if b['canonical'] is not None]
    slots = []
    for b in blocks:
        center = (abs(signed[tuple(b['canonical'])])-1)//25
        eligible = [k for k in b['members'] if (abs(signed[tuple(k)])-1)//25 == center]
        if [e['member'] for e in b['evidence']] != eligible or b['geometry'] not in eligible:
            raise ValueError('The same-second candidate universe differs from the frozen representative')
        slots.append(eligible)
    role = {'source':0,'target':1,'deploy':2}[meta['role']]
    part = {'head':0,'tail':1,'custom':2}[meta['part']]
    prefix = [314159,role,meta['fold'],meta['first_seed'],part]
    controls = []
    for rep in range(99):
        rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([*prefix,rep])))
        controls.append([candidates[int(rng.integers(len(candidates)))] for candidates in slots])
    selectors = dict(canonical=[b['canonical'] for b in blocks],geometry=[b['geometry'] for b in blocks],
                     controls=controls)
    quota = dict(generator='numpy.PCG64', seed_prefix=prefix,replicates=list(range(99)),
                 draws='one uniform in original block order, in then out, including singleton choice sets',
                 pool_sizes=[len(s) for s in slots], numpy=np.__version__)
    return selectors, quota, signed


def count_grid(times, nframes):
    events = np.array([[abs(t),1 if t>0 else -1] for t in times],dtype=np.int64).reshape(-1,2)
    return counts_from_events(events,nframes)


def count_proof(selectors, signed, nframes):
    native = list(signed.values())
    for t in native:
        if not 1 <= abs(t) <= nframes:
            raise ValueError('A native event falls outside the clip')
    each = {k:[signed[tuple(e)] for e in selectors[k]] for k in ('canonical','geometry')}
    pooled, integer = count_grid(native,nframes), count_grid(each['canonical'],nframes)
    delta = 3*integer-pooled
    if np.max(abs(delta)) > 2 or not np.array_equal(integer,count_grid(each['geometry'],nframes)):
        raise ValueError('Official count agreement or the per-cell 2/3 certificate failed')
    per_second = Counter((t>0,(abs(t)-1)//25) for t in each['canonical'])
    for selector in selectors['controls']:
        # Equality of all per-second, per-direction counts implies equality on all 30 official offsets.
        if Counter((signed[tuple(k)]>0,(abs(signed[tuple(k)])-1)//25) for k in selector) != per_second:
            raise ValueError('A quota control changed the integer time histogram')
    by_direction = {}
    for name,sgn in (('in',1),('out',-1)):
        p = Counter(abs(t) for t in native if t*sgn > 0)
        measures = {}
        for arm, events in each.items():
            emitted = Counter(abs(t) for t in events if t*sgn > 0)
            prefix = [0]
            for f in sorted(p):
                prefix.append(prefix[-1]+3*emitted[f]-p[f])
            max_range = max(prefix)-min(prefix)
            if max_range > (2 if arm == 'canonical' else 4) or abs(prefix[-1]) > 1:
                raise ValueError('An arbitrary-interval or whole-direction certificate failed')
            measures[arm] = dict(total=sum(emitted.values()),max_interval_abs_thirds=max_range,
                                 clip_bias_thirds=prefix[-1])
        by_direction[name] = dict(pooled=sum(p.values()),**measures)
    return dict(nframes=nframes,by_direction=by_direction,num_time_bins=len(integer),
                pooled_window_counts=pooled.tolist(),integer_window_counts=integer.tolist(),
                delta_l1_thirds=int(abs(delta).sum()),max_cell_abs_thirds=int(abs(delta).max()),
                changed_representatives=sum(a!=b for a,b in zip(selectors['canonical'],selectors['geometry'])),
                bound_pp='100*delta_l1_thirds/(3*30*G), positive full-clip GT count G')


def verify_pack(pack):
    if pack['format'] != FORMAT:
        raise ValueError('Unknown event MOT package format')
    meta, original = pack['meta'], pack['manifest']
    streams = {s:[] for s in original['seeds']}
    for row in original['observations']:
        streams[row['seed']].append(row['row'])
    regenerated = integer_events(streams)
    if regenerated != {k:v for k,v in original.items() if k!='id_blocks'}:
        raise ValueError('Native geometric evidence, tie order or block partition differs')
    selectors,quota,signed = select_controls(regenerated,meta)
    if (selectors != pack['selectors'] or quota != pack['quota']
            or count_proof(selectors,signed,meta['nframes']) != pack['counts_proof']):
        raise ValueError('Seeded quota, selections, or count certificate differs')
    if len(original['id_blocks']) != len({(s,t) for s,t,_,_ in original['id_blocks']}):
        raise ValueError('Duplicate source ID block')
    next_id = 1
    for _,_,base,size in original['id_blocks']:
        if base != next_id or size < 1:
            raise ValueError('Missing or overlapping native ID block')
        next_id += size
    if not set(signed).issubset({(s,i) for s,i,_,_ in original['id_blocks']}):
        raise ValueError('Native event has no original MOT identity')


def freeze_group(streams,polygon,meta,destination,prereg=None):
    if meta['role']!='deploy' and (prereg is None or not re.fullmatch('[0-9a-f]{40}',prereg)):
        raise ValueError('Benchmark freeze requires a full preregistration commit')
    outputs, manifest = export_variants(streams,polygon)
    selectors,quota,signed = select_controls(manifest,meta)
    destination.parent.mkdir(parents=True,exist_ok=True)
    with TemporaryDirectory(dir=destination.parent) as temp:
        staged = Path(temp)/'group'
        staged.mkdir()
        mot_hashes = {}
        for arm,rows in outputs.items():
            p = staged/(arm+'.txt')
            write_export(p,rows)
            # Normalize MOT line endings as part of the portable package byte format.
            raw = p.read_bytes().replace(b'\r\n',b'\n')
            p.write_bytes(raw)
            mot_hashes[arm] = sha(raw)
        record = dict(format=FORMAT,prereg=prereg,meta=meta,manifest=manifest,selectors=selectors,
                      quota=quota,counts_proof=count_proof(selectors,signed,meta['nframes']),
                      mot_sha256=mot_hashes)
        verify_pack(record)
        data = wire(record)
        (staged/'freeze.json').write_bytes(data)
        if not destination.exists():
            try:
                os.rename(staged,destination)
            except OSError:
                if not destination.is_dir():
                    raise
        read_checked(destination/'freeze.json',sha(data))
        for arm,digest in mot_hashes.items():
            read_checked(destination/(arm+'.txt'),digest)
    return sha(data)


def specs(role):
    return [(fold,part,base) for fold in (33,35) for part in (('head','tail') if role=='source' else ('head',))
            for base in (0,3,6)]


def group_key(fold,part,base):
    return f'{fold}_{base}_{part}'


def inventory(root,role,prereg):
    cases = {}
    for fold,part,base in specs(role):
        key = group_key(fold,part,base)
        folder = root/role/key
        raw = (folder/'freeze.json').read_bytes()
        record = json.loads(raw)
        meta = record['meta']
        if (record['prereg'],meta['role'],meta['fold'],meta['part'],meta['first_seed']) != (prereg,role,fold,part,base):
            raise ValueError(f'Mismatched identity of {key}')
        verify_pack(record)
        for arm,digest in record['mot_sha256'].items():
            read_checked(folder/(arm+'.txt'),digest)
        cases[key] = sha(raw)
    data = wire(dict(format=FORMAT,role=role,prereg=prereg,groups=cases))
    path = root/(role+'_index.json')
    with TemporaryDirectory(dir=root) as temp:
        part = Path(temp)/'index'
        part.write_bytes(data)
        try:
            os.link(part,path)
        except FileExistsError:
            read_checked(path,sha(data))
    return sha(data)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest='mode',required=True)
    rep = modes.add_parser('replay'); rep.add_argument('manifest',type=Path); rep.add_argument('--sha',required=True)
    inv = modes.add_parser('inventory'); stage = modes.add_parser('freeze')
    for sub in (inv,stage):
        sub.add_argument('role',choices=('source','target')); sub.add_argument('output',type=Path)
        sub.add_argument('--prereg',required=True)
    stage.add_argument('--lineage-dir',required=True,type=Path)
    stage.add_argument('--runs-dir',required=True,type=Path)
    stage.add_argument('--zones',required=True,type=Path)
    a = parser.parse_args()
    if a.mode=='replay':
        verify_pack(json.loads(read_checked(a.manifest,a.sha)))
        print('Native event descriptors, representatives, controls and certificates reproduced.')
    elif a.mode=='inventory':
        print(inventory(a.output,a.role,a.prereg))
    else:
        from inputs import read_group
        for fold,part,base in specs(a.role):
            bank,poly,meta = read_group(a.role,fold,part,base,a.lineage_dir,a.runs_dir,a.zones)
            name = group_key(fold,part,base)
            print(name,freeze_group(bank,poly,meta,a.output/a.role/name,a.prereg),flush=True)
