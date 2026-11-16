"""Create two whole, integer-event MOT files from three P0 MOT files without labels."""

import argparse
import io
import json
from pathlib import Path

import numpy as np

from freeze_e3 import freeze_group, sha


def load_mot(path, nframes):
    raw = path.read_bytes()
    rows = (np.loadtxt(io.BytesIO(raw),delimiter=',',ndmin=2) if raw.strip() else np.empty((0,7)))
    if rows.ndim!=2 or rows.shape[1]<7:
        raise ValueError(f'{path}: supply frame,id,x,y,w,h,confidence')
    rows = rows[:,:7]
    if (not np.isfinite(rows).all() or not np.array_equal(rows[:,:2],np.rint(rows[:,:2]))
            or not np.all((rows[:,0]>=1)&(rows[:,0]<=nframes)) or np.any(rows[:,4:6]<=0)):
        raise ValueError(f'{path}: invalid P0 frame, ID or geometry')
    if not np.array_equal(rows[:,2:6],np.round(rows[:,2:6],2)) or not np.array_equal(rows[:,6],np.round(rows[:,6],3)):
        raise ValueError(f'{path}: native P0 writer uses two decimals for geometry and three for confidence')
    return rows,sha(raw)


def export(mots,zone_file,clip,nframes,destination):
    if len(mots)!=3 or len({p.resolve() for p in mots})!=3 or nframes<1:
        raise ValueError('Supply three distinct native MOT paths and a positive clip length')
    zone_bytes = zone_file.read_bytes()
    zone = json.loads(zone_bytes)
    if isinstance(zone,dict):
        zone = zone[clip]
    poly = np.asarray(zone,dtype=float)
    if poly.ndim!=2 or poly.shape[1]!=2 or len(poly)<3 or not np.isfinite(poly).all():
        raise ValueError('Supply a finite polygon or the matching --clip key in a polygon map')
    x,y = poly.T
    if np.dot(x,np.roll(y,-1)) == np.dot(y,np.roll(x,-1)):
        raise ValueError('The entrance polygon has zero area')
    data = [load_mot(p,nframes) for p in mots]
    meta = dict(role='deploy',part='custom',fold=0,first_seed=0,seeds=[0,1,2],clip=clip,nframes=nframes,
                fps=25,mot_sha256={str(s):h for s,(_,h) in enumerate(data)},zones_sha256=sha(zone_bytes))
    return freeze_group({s:rows for s,(rows,_) in enumerate(data)},poly,meta,destination)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mot',required=True,nargs=3,type=Path,help='three native files in deterministic seed order')
    parser.add_argument('--polygon',required=True,type=Path)
    parser.add_argument('--clip',required=True)
    parser.add_argument('--frames',required=True,type=int,help='length in 25-fps frames')
    parser.add_argument('--output',required=True,type=Path)
    a=parser.parse_args()
    print(export(a.mot,a.polygon,a.clip,a.frames,a.output))
