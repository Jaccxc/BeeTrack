"""Independent §116 geometric dose verification in original image edge coordinates."""
import argparse
import hashlib
import json
from math import ceil, dist, fsum, isfinite, isclose
from pathlib import Path

import numpy as np


def require(ok, message):
    if not ok:
        raise ValueError(message)


def close(actual, expected, name):
    if expected is None:
        require(actual is None, name)
        return
    a,b = np.asarray(actual,dtype=float),np.asarray(expected,dtype=float)
    require(a.shape==b.shape and np.all(np.isfinite(a)) and np.allclose(a,b,atol=1e-10,rtol=2e-12),name)


def rectangles(width,height,polygon):
    x,y = zip(*polygon)
    padding = min(width,height)/20
    left,top = max(0.,min(x)-padding),max(0.,min(y)-padding)
    right,bottom = min(float(width),max(x)+padding),min(float(height),max(y)+padding)
    rw,rh = right-left,bottom-top
    require(rw>0 and rh>0,'Polygon has no nonempty clipped padded rectangle')
    # The corner distance is separable by axis. The low corner wins an exact tie.
    cx = width-rw if left < (width-rw)/2 else 0.
    cy = height-rh if top < (height-rh)/2 else 0.
    local=(left,top,right,bottom)
    control=(cx,cy,cx+rw,cy+rh)
    # Retaining at least half of each outside axis forces g <= (L+span)/(2*span).
    gain=min(1.5,(width+rw)/(2*rw),(height+rh)/(2*rh))
    area=rw*rh
    overlap=max(0.,min(right,control[2])-max(left,cx))*max(0.,min(bottom,control[3])-max(top,cy))
    return local,control,gain,area/(width*height),overlap,overlap/(2*area-overlap)


def axis_parts(limit,lo,hi,gain):
    outer=(limit-gain*(hi-lo))/(limit-(hi-lo)) if hi-lo < limit else 1.
    # Remove empty intervals. This also chooses the right terminal slope for extrapolation.
    parts=[]
    dst=0.
    for a,b,m in ((0.,lo,outer),(lo,hi,gain),(hi,float(limit),outer)):
        if b>a:
            parts.append((a,b,dst,m))
            dst+=(b-a)*m
    return parts


def slope(parts,coord):
    for _,b,_,scale in parts[:-1]:
        if coord < b:
            return scale
    return parts[-1][3]


def axis_map(parts,coord,inverse=False):
    """Use the terminal affine segment, including negative and beyond-canvas boxes."""
    for a,b,dst,scale in parts:
        if coord < (dst+(b-a)*scale if inverse else b):
            break
    return a+(coord-dst)/scale if inverse else dst+(coord-a)*scale


def samples(width,height,polygon):
    normal_step=min(width,height)/40
    result=[]
    for a,b in zip(polygon,polygon[1:]+polygon[:1]):
        length=dist(a,b)
        if not length:
            continue
        n=ceil(length)
        dx,dy=b[0]-a[0],b[1]-a[1]
        for j in range(n):
            fraction=(j+0.5)/n
            midx,midy=a[0]+fraction*dx,a[1]+fraction*dy
            for k in (-1,0,1):
                x=midx+k*normal_step*(-dy/length)
                y=midy+k*normal_step*(dx/length)
                if 0<=x<=width and 0<=y<=height:
                    result.append((x,y,length/n))
    return result


def dose_rows(size,rect,gain,points):
    w,h=size
    xs=axis_parts(w,rect[0],rect[2],gain)
    ys=axis_parts(h,rect[1],rect[3],gain)
    result=[]
    for x,y,_ in points:
        sx,sy=slope(xs,x),slope(ys,y)
        result.append((sx,sy,sx*sy))
    if not points:
        return result,None
    weight=fsum(point[2] for point in points)
    return result,[fsum(p[2]*v[j] for p,v in zip(points,result))/weight for j in range(3)]


def tensor_gains(size, receipts):
    """Receipts for one frame/seed, keyed U,L,C,H; gains include raster rounding."""
    w,h=size
    require(set(receipts)==set('ULCH'),'Missing or extra inference arm')
    shapes={}
    gains={}
    for arm,row in receipts.items():
        require(row['source_hw']==[h,w] and row['forward_count']==1,f'{arm} source dimensions or forward count')
        batch,channels,oh,ow=row['tensor_bchw']
        require(batch==1 and channels==3 and type(oh) is int and type(ow) is int
                and min(oh,ow)>0,f'{arm} tensor shape')
        require(max(oh,ow)==(960 if arm=='H' else 640),f'{arm} actual tensor long side')
        ratio=min(ow/w,oh/h)
        nw,nh=max(1,round(w*ratio)),max(1,round(h*ratio))
        lb=row['letterbox']
        require(lb['resized_hw']==[nh,nw] and lb['pad_xy']==[(ow-nw)//2,(oh-nh)//2],
                f'{arm} rounded resize and padding')
        close(lb['gain_xy'],[nw/w,nh/h],f'{arm} horizontal or vertical resize')
        shapes[arm]=(oh,ow)
        gains[arm]=(nw/w,nh/h)
    require(shapes['U']==shapes['L']==shapes['C'],'640 arm shape changed')
    rx,ry=(gains['H'][j]/gains['U'][j] for j in range(2))
    return [rx,ry,rx*ry]


def verify_dose(capsule, receipts=None):
    """Verify geometry.json['cases'][clip]. Optional receipts are one matched U/L/C/H quartet."""
    w,h=capsule['size']
    require(type(w) is int and type(h) is int and min(w,h)>0,'Invalid original image dimensions')
    polygon=[tuple(p) for p in capsule['polygon']]
    require(len(polygon)>=3 and all(len(p)==2 and all(map(isfinite,p)) for p in polygon),
            'Invalid original entrance polygon')
    local,control,gain,fraction,overlap,iou=rectangles(w,h,polygon)
    recomputed=samples(w,h,polygon)
    stored=capsule['samples_xy_weight']
    close(stored,recomputed,'Original coordinates or sample arc weights differ')
    close(capsule['rect_overlap_px2'],overlap,'Rectangle overlap area differs')
    close(capsule['rect_iou'],iou,'Rectangle IoU differs')
    means={}
    for arm,rect in (('L',local),('C',control)):
        rec=capsule[arm]
        close(rec['rect'],rect,f'{arm} rectangle differs')
        close(rec['gain'],gain,f'{arm} common gain differs')
        close(rec['area_fraction'],fraction,f'{arm} fraction of original area differs')
        # Evaluate the actual validated sample coordinates at piecewise discontinuities.
        rows,mean=dose_rows((w,h),rect,gain,stored)
        close(rec['rows_xyj'],rows,f'{arm} per-sample x/y/J factors differ')
        close(rec['mean_xyj'],mean,f'{arm} arc-weighted means differ')
        means[arm]=mean
    contrast=means['L'][2]-means['C'][2] if recomputed else None
    close(capsule['dose_difference'],contrast,'Unrounded J dose contrast differs')
    passed=gain!=1 and contrast is not None and contrast>=.10
    require(capsule['dose_sufficient'] is passed,'Dose threshold decision differs')
    close(capsule['U_xyj'],[1,1,1],'Uniform-dose reference differs')
    # Pre-inference check from 640/960 targets, including separate rounded raster axis scales.
    dims=[]
    for length in (640,960):
        ratio=length/max(w,h)
        dims.append([max(1,round(w*ratio)),max(1,round(h*ratio))])
    hx,hy=[dims[1][k]/dims[0][k] for k in range(2)]
    high=[hx,hy,hx*hy]
    close(capsule['H_xyj'],high,'High scale dose differs after raster rounding')
    if receipts is not None:
        close(tensor_gains((w,h),receipts),high,'Actual tensor-derived high dose differs')
    return dict(samples=len(stored),arc_weight=fsum(p[2] for p in stored),gain=gain,
                L_mean_xyj=means['L'],C_mean_xyj=means['C'],dose_difference=contrast,dose_sufficient=passed,
                overlap_px2=overlap,rect_iou=iou,H_xyj=high,receipts_checked=0 if receipts is None else 4)


def self_check():
    w,h=100,80
    p=[(35,30),(65,30),(65,50),(35,50),(35,50)]
    l,c,g,fraction,overlap,iou=rectangles(w,h,p)
    assert l==(31.,26.,69.,54.) and c==(0.,0.,38.,28.) and g==1.5
    close([fraction,overlap,iou],[38*28/8000,7*2,14/(2*38*28-14)],'Symmetric tie/corners')
    band=samples(w,h,p)
    assert len(band)==300 and isclose(fsum(s[2] for s in band),300)
    close(dose_rows((w,h),l,g,band)[1],[1.5,1.5,2.25],'Entrance band lies inside the local rectangle')
    xs=axis_parts(w,l[0],l[2],g)
    source=(l[0]-4,l[0]+4)
    mapped=[axis_map(xs,x) for x in source]
    restored=[axis_map(xs,x,True) for x in mapped]
    close(restored,source,'Inverse two edges')
    naive=axis_map(xs,sum(mapped)/2,True)
    assert abs(naive-sum(restored)/2)>.8
    for x in (-12.5,0,31,69,100,113.4):
        close(axis_map(xs,axis_map(xs,x),True),x,'Outside-canvas affine extrapolation')
    assert slope(xs,31)==g and slope(xs,69)<1
    # When the right coordinate is itself the image edge, use the last existing segment.
    assert slope(axis_parts(100,80,100,1.5),100)==1.5
    assert slope(axis_parts(100,0,20,1.5),0)==1.5
    full=[(-10,-17),(110,-17),(110,94),(-10,94)]
    assert rectangles(w,h,full)[2]==1 and not samples(w,h,full)
    clipped=[(0,0),(80,0),(80,40),(0,40)]
    lr,_,gain,*_=rectangles(w,h,clipped)
    assert gain<1.5 and isclose(axis_parts(w,lr[0],lr[2],gain)[-1][-1],.5)
    receipts={}
    for arm in 'ULCH':
        # 93×157 camera gives H/U different rounded vertical and horizontal ratios.
        size=960 if arm=='H' else 640
        target_h=576 if arm=='H' else 384
        nh=round(93*size/157)
        receipts[arm]=dict(source_hw=[93,157],tensor_bchw=[1,3,target_h,size],forward_count=1,
                           letterbox=dict(resized_hw=[nh,size],gain_xy=[size/157,nh/93],
                                          pad_xy=[0,(target_h-nh)//2]))
    actual=tensor_gains((157,93),receipts)
    assert actual[0]==1.5 and actual[1]!=actual[0]
    return dict(samples=len(band),inverse_edges=restored,inverse_center=naive,
                H_over_U_xyJ=actual)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    command=parser.add_mutually_exclusive_group(required=True)
    command.add_argument('--self-check',action='store_true')
    command.add_argument('--geometry',type=Path)
    parser.add_argument('--sha',help='Expected SHA-256 of the geometry capsule')
    args=parser.parse_args()
    if args.self_check:
        report=self_check()
    else:
        require(args.sha is not None,'Supply --sha to read frozen geometry')
        raw=args.geometry.read_bytes()
        require(hashlib.sha256(raw).hexdigest()==args.sha,'Geometry file hash mismatch')
        report={name:verify_dose(rec) for name,rec in json.loads(raw)['cases'].items()}
    print(json.dumps(report,sort_keys=True,allow_nan=False))
