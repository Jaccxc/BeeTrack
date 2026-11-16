"""Measure the predeclared image sampling across the physical hive entrance."""
import argparse
import configparser
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
from entrance_scale import EntranceScale, band_dose, entrance_band


def measure(size, polygon):
    width, height = size
    samples = entrance_band(size, polygon)
    area = width * height
    variants = {}
    for name in ('L', 'C'):
        lens = EntranceScale.from_polygon(size, polygon, 1.5, placebo=(name == 'C'))
        dose = band_dose(lens, samples)
        x0, y0, x1, y1 = lens.rect
        variants[name] = dict(rect=list(lens.rect), gain=lens.gain, area_fraction=(x1-x0)*(y1-y0)/area,
                              mean_xyj=None if dose['mean_xyj'] is None else dose['mean_xyj'].tolist(),
                              rows_xyj=dose['rows'].tolist())
    a, b = (variants[k]['rect'] for k in ('L','C'))
    rect_area = (a[2]-a[0])*(a[3]-a[1])
    overlap = max(0,min(a[2],b[2])-max(a[0],b[0])) * max(0,min(a[3],b[3])-max(a[1],b[1]))
    assert np.isclose(variants['L']['gain'], variants['C']['gain']) and np.isclose(variants['L']['area_fraction'], variants['C']['area_fraction'])
    diff = (variants['L']['mean_xyj'][2] - variants['C']['mean_xyj'][2]) if len(samples) else None
    axes = {k:tuple(round(length * k / max(size))/length for length in size) for k in (640,960)}
    hx, hy = (h/u for h,u in zip(axes[960],axes[640]))
    return dict(size=list(size),polygon=polygon,samples_xy_weight=samples.tolist(),U_xyj=[1,1,1],
                H_xyj=[hx,hy,hx*hy],rect_overlap_px2=overlap,rect_iou=overlap/(2*rect_area-overlap),
                **variants,dose_difference=diff,
                dose_sufficient=bool(len(samples) and variants['L']['gain']>1 and diff>=.10))


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--root',type=Path,default=Path.cwd())
    cli.add_argument('--out',type=Path,required=True)
    args = cli.parse_args()
    zones = args.root/'beetrack/zones.json'
    data = json.loads(zones.read_bytes())
    output={'prereg':'c8fabf6c55c9737086b02050fabb3360bb5f1db2','zones_sha256':sha256(zones.read_bytes()).hexdigest(),'cases':{}}
    for hive in (33,35):
        clip = f'BEE24-{hive}-tail'
        path = args.root/'runs/gt/val'/clip/'seqinfo.ini'
        cfg = configparser.ConfigParser()
        cfg.read(path,encoding='utf8')
        info = cfg['Sequence']
        assert info['name']==clip and info.getint('seqLength')==1000 and info.getint('frameRate')==25
        row = measure((info.getint('imWidth'), info.getint('imHeight')), data[clip])
        row['seqinfo_sha256']=sha256(path.read_bytes()).hexdigest()
        output['cases'][clip]=row
        print(clip,'size',row['size'],'central gain',row['L']['gain'],'L/C mean xyJ',row['L']['mean_xyj'],row['C']['mean_xyj'],'delta',row['dose_difference'])
    blob = (json.dumps(output,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
    if args.out.exists():
        assert args.out.read_bytes()==blob, 'Existing artifact differs'
    else:
        args.out.parent.mkdir(parents=True,exist_ok=True)
        args.out.write_bytes(blob)
    print(sha256(blob).hexdigest())


if __name__=='__main__':
    main()
