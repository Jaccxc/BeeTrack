"""A fixed-camera, full-field separable warp before normal detector preprocessing."""
from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix


@dataclass
class Axis:
    src: np.ndarray
    dst: np.ndarray
    sampling: csr_matrix

    def transform(self, values, inverse=False):
        lo,hi=(self.dst,self.src) if inverse else (self.src,self.dst)
        values=np.asarray(values,dtype=np.float64)
        if np.any(~np.isfinite(values)):
            raise ValueError('Coordinates must be finite')
        # Do not clip pre-NMS detections at the image edge. Continue the terminal
        # affine segment through letterbox padding and clip at the caller's normal stage.
        index=np.searchsorted(lo,values,side='right')-1
        index=np.clip(index,0,len(lo)-2)
        return hi[index]+(values-lo[index])*(hi[index+1]-hi[index])/(lo[index+1]-lo[index])

    def slopes(self,values):
        index=np.clip(np.searchsorted(self.src,values,side='right')-1,0,len(self.src)-2)
        return np.diff(self.dst)[index]/np.diff(self.src)[index]


def build_axis(length, start, stop, gain):
    span=stop-start
    outer=(length-span*gain)/(length-span) if span<length else 1.
    src=np.array([0.,start,stop,float(length)])
    dst=np.array([0.,start*outer,start*outer+span*gain,float(length)])
    keep=np.r_[True,np.diff(src)>0]
    src,dst=src[keep],dst[keep]
    edges=np.interp(np.arange(length+1),dst,src)
    rows,cols,values=[],[],[]
    for j,(a,b) in enumerate(zip(edges[:-1],edges[1:])):
        ids=np.arange(int(np.floor(a)),min(int(np.ceil(b)),length))
        weights=(np.minimum(b,ids+1)-np.maximum(a,ids))/(b-a)
        rows.extend([j]*len(ids)); cols.extend(ids); values.extend(weights)
    grid=csr_matrix((np.asarray(values,np.float32),(rows,cols)),shape=(length,length))
    return Axis(src,dst,grid)


class EntranceScale:
    """All geometry is in original-image edge coordinates, [0,W] × [0,H]."""
    def __init__(self, size, rect, requested_gain):
        w,h=size
        x0,y0,x1,y1=rect
        if not (0<=x0<x1<=w and 0<=y0<y1<=h and np.isfinite(requested_gain) and requested_gain>0):
            raise ValueError('Expected a positive gain and a nonempty rectangle in the image')
        intervals=[((x1-x0)/w),((y1-y0)/h)]
        low=max((1-2*(1-r))/r for r in intervals)
        high=min((1-.5*(1-r))/r for r in intervals)
        self.gain=float(np.clip(requested_gain,max(low,0),high))
        self.size=w,h
        self.rect=tuple(rect)
        self.x=build_axis(w,x0,x1,self.gain)
        self.y=build_axis(h,y0,y1,self.gain)

    @classmethod
    def from_polygon(cls,size,polygon,requested_gain,placebo=False):
        w,h=size
        p=np.asarray(polygon,dtype=np.float64)
        if p.ndim!=2 or p.shape[1]!=2 or len(p)<3 or not np.isfinite(p).all():
            raise ValueError('Expected a finite entrance polygon')
        pad=.05*min(w,h)
        low=np.maximum(p.min(axis=0)-pad,0)
        high=np.minimum(p.max(axis=0)+pad,[w,h])
        if placebo:
            dims=high-low
            corners=[np.array((i,j)) for i in (0,w-dims[0]) for j in (0,h-dims[1])]
            point=max(corners,key=lambda v:(np.sum((v-low)**2),-v[0],-v[1]))
            low,high=point,point+dims
        return cls(size,(*low,*high),requested_gain)

    def boxes(self,xyxy,inverse=False):
        b=np.array(xyxy,dtype=np.float64,copy=True).reshape(-1,4)
        if np.any(b[:,2:]<b[:,:2]):
            raise ValueError('Invalid box')
        b[:,[0,2]]=self.x.transform(b[:,[0,2]],inverse)
        b[:,[1,3]]=self.y.transform(b[:,[1,3]],inverse)
        return b

    def frame(self,image):
        w,h=self.size
        if image.dtype!=np.uint8 or image.ndim!=3 or image.shape!=(h,w,3):
            raise ValueError('Expected H×W×3 uint8 frame')
        if self.gain==1:
            return image.copy()
        temp=(self.y.sampling@image.reshape(h,-1)).reshape(h,w,3)
        result=(self.x.sampling@temp.transpose(1,0,2).reshape(w,-1)).reshape(w,h,3).transpose(1,0,2)
        return np.rint(result).clip(0,255).astype(np.uint8)


class CameraInput:
    """Adapter for one pre-calibrated camera and a detector returning original-canvas xyxy."""
    def __init__(self,size,polygon,selected_size,placebo=False):
        if not isinstance(selected_size,int) or selected_size<1:
            raise ValueError('Expected an integer calibrated input length')
        self.input_size=min(640,selected_size)
        self.lens=(EntranceScale.from_polygon(size,polygon,selected_size/640,placebo)
                   if selected_size>640 else None)

    def detect(self,frame,predict):
        """predict(frame, input_length) -> Nx5 xyxy+score, after detector NMS and undoing its letterbox."""
        canvas=self.lens.frame(frame) if self.lens else frame
        rows=np.asarray(predict(canvas,self.input_size),dtype=float).reshape(-1,5).copy()
        if self.lens:
            rows[:,:4]=self.lens.boxes(rows[:,:4],inverse=True)
        return rows


def entrance_band(size,polygon):
    """Ordered original-image samples: equally spaced arc midpoints and three signed offsets."""
    w,h=size
    p=np.asarray(polygon,dtype=np.float64)
    out=[]
    for a,b in zip(p,np.roll(p,-1,axis=0)):
        v=b-a
        length=np.linalg.norm(v)
        if not length: continue
        count=int(np.ceil(length))
        normal=np.array([-v[1],v[0]])/length
        for j in range(count):
            pos=a+((j+.5)/count)*v
            for mul in (-1,0,1):
                x,y=pos+mul*.025*min(w,h)*normal
                if 0<=x<=w and 0<=y<=h:
                    out.append((x,y,length/count))
    return np.asarray(out,dtype=np.float64).reshape(-1,3)


def band_dose(lens,samples):
    """Arc-length-weighted marginal and area multipliers, not a bounding-box overlap proxy."""
    if not len(samples):
        return dict(samples=0,mean_xyj=None,rows=np.empty((0,3)))
    x=lens.x.slopes(samples[:,0])
    y=lens.y.slopes(samples[:,1])
    rows=np.column_stack((x,y,x*y))
    return dict(samples=len(rows),mean_xyj=np.average(rows,axis=0,weights=samples[:,2]),rows=rows)


def self_check():
    rng=np.random.default_rng(116)
    image=rng.integers(0,256,(40,64,3),dtype=np.uint8)
    square=[(20,10),(40,10),(40,20),(20,20)]
    pairs=np.sort(rng.uniform((0.,0.),(64.,40.),size=(256,2,2)),axis=1).reshape(-1,4)
    for requested in (.6,1.,1.5,4.):
        for placebo in (False,True):
            lens=EntranceScale.from_polygon((64,40),square,requested,placebo)
            got=lens.frame(image)
            assert got.shape==image.shape and got.dtype==image.dtype
            assert np.all(lens.frame(np.full_like(image,83))==83)
            assert np.allclose(lens.boxes(lens.boxes(pairs),True),pairs,atol=1e-11)
            padded=np.array([[-3.25,-7.,10.,56.],[-1.,14.,70.,20.]])
            assert np.allclose(lens.boxes(lens.boxes(padded),True),padded,atol=1e-11)
            assert np.array_equal(lens.boxes([[0,0,64,40]]),[[0,0,64,40]])
            if requested==1: assert np.array_equal(got,image)
            for axis in (lens.x,lens.y):
                assert np.all(np.diff(axis.dst)>0)
                assert np.allclose(axis.sampling.sum(axis=1),1,atol=1e-6)
    full=EntranceScale((64,40),(0,0,64,13),2)
    assert full.gain==1 and np.array_equal(full.frame(image),image)
    focus=EntranceScale((64,40),(20,10,40,20),1.5)
    original=np.array([[24,12,28,16],[24,14,28,18]])
    new=focus.boxes(original)
    assert np.allclose(new[:,2:]-new[:,:2],1.5*(original[:,2:]-original[:,:2]))
    # The scorer works on original-image centres, after mapping both ends back.
    assert np.array_equal(focus.boxes(new,True),original)
    # Boxes crossing a change in slope must be restored by their corners, not their centre.
    crossing=np.array([[18.,12.,28.,16.]])
    wbox=focus.boxes(crossing)[0]
    assert not np.isclose(focus.x.transform((wbox[0]+wbox[2])/2,True),(18+28)/2)
    samples=entrance_band((64,40),square)
    main=EntranceScale.from_polygon((64,40),square,1.5)
    off=EntranceScale.from_polygon((64,40),square,1.5,True)
    assert np.allclose(band_dose(main,samples)['mean_xyj'],[1.5,1.5,2.25])
    assert not np.allclose(band_dose(off,samples)['mean_xyj'],[1.5,1.5,2.25])
    for selected in (448,640,960):
        camera=CameraInput((64,40),square,selected)
        calls=[]
        def mock_detector(canvas,length):
            calls.append((canvas.shape,length))
            model_boxes=camera.lens.boxes(original) if camera.lens else original
            return np.column_stack((model_boxes,[.85,.61]))
        out=camera.detect(image,mock_detector)
        assert calls==[(image.shape,min(640,selected))]
        assert np.allclose(out[:,:4],original) and np.array_equal(out[:,4],[.85,.61])
        assert camera.detect(image,lambda canvas,length:np.empty((0,5))).shape==(0,5)
    print('Passed identity, edge-to-edge coverage, gain cap, constant images, area weights, focus/placebo band dose, box and padded-coordinate round trips, one-call resolution and empty detections.')


if __name__=='__main__':
    self_check()
