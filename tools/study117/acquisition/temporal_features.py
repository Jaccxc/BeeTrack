"""Synthetic-only causal P3 feature alignment; the original current-frame regression is untouched."""
from collections import deque

import torch


def descriptors(features):
    centered=features-features.mean(dim=1,keepdim=True)
    norm=centered.square().sum(dim=1,keepdim=True).sqrt()
    return centered/norm.clamp_min(1e-8),norm[:,0]>1e-8


def best_pair_indices(current,previous,radius=2,content=None):
    if current.ndim!=4 or current.shape[0]!=1 or current.shape!=previous.shape:
        raise ValueError('Expected matching single-camera [1,C,H,W] tensors')
    _,_,h,w=current.shape
    c,cvalid=descriptors(current)
    p,pvalid=descriptors(previous)
    if content is not None:
        if content.dtype!=torch.bool or content.shape!=(h,w):
            raise ValueError('Expected original-image content mask on the P3 grid')
        cvalid=cvalid&content
        pvalid=pvalid&content
    offsets=[(dy,dx) for dy in range(-radius,radius+1) for dx in range(-radius,radius+1)]
    forward=current.new_full((len(offsets),h,w),float('-inf'))
    reverse=forward.clone()
    for k,(dy,dx) in enumerate(offsets):
        y0,y1=max(0,-dy),min(h,h-dy)
        x0,x1=max(0,-dx),min(w,w-dx)
        if y1<=y0 or x1<=x0: continue
        ys,xs=slice(y0,y1),slice(x0,x1)
        yp,xp=slice(y0+dy,y1+dy),slice(x0+dx,x1+dx)
        sim=(c[0,:,ys,xs]*p[0,:,yp,xp]).sum(dim=0)
        sim=sim.masked_fill(~(cvalid[0,ys,xs]&pvalid[0,yp,xp]),float('-inf'))
        forward[k,ys,xs]=sim
        reverse[k,yp,xp]=sim
    fv,fi=forward.topk(2,dim=0)
    rv,ri=reverse.topk(2,dim=0)
    y,x=torch.meshgrid(torch.arange(h,device=c.device),torch.arange(w,device=c.device),indexing='ij')
    d=torch.as_tensor(offsets,device=c.device)[fi[0]]
    py=(y+d[...,0]).clamp(0,h-1)
    px=(x+d[...,1]).clamp(0,w-1)
    # reverse[k] at the donor describes the same target->donor offset k.
    valid=((fv[0]>=.8)&(fv[0]-fv[1]>=.05)&(ri[0,py,px]==fi[0])
           &(rv[0,py,px]>=.8)&(rv[0,py,px]-rv[1,py,px]>=.05))
    return valid,py,px


def classify_step(current,previous,original_logits,previous_logits,cls_head,mode='aligned',alpha=.5,content=None):
    if mode not in ('aligned','same_cell','zero_motion','past_logit','permute') or not 0<=alpha<=1:
        raise ValueError('Unknown comparison or invalid feature weight')
    if original_logits.shape!=previous_logits.shape or original_logits.shape!=(1,1,*current.shape[2:]):
        raise ValueError('Expected single-class, native P3 anchor logits')
    if mode=='zero_motion':
        c,cm=descriptors(current)
        p,pm=descriptors(previous)
        valid=cm[0]&pm[0]&((c*p)[0].sum(0)>=.8)
        if content is not None: valid=valid&content
        h,w=valid.shape
        py,px=torch.meshgrid(torch.arange(h,device=c.device),torch.arange(w,device=c.device),indexing='ij')
    else:
        valid,py,px=best_pair_indices(current,previous,content=content)
    now=original_logits[0,0].sigmoid()
    prev=previous_logits[0,0].sigmoid()
    mask=valid&(now>=.05)&(now<.25)&(prev[py,px]>=.25)
    iy,ix=torch.where(mask)
    if not len(iy) or alpha==0:
        return original_logits.clone(),mask
    jy,jx=py[iy,ix],px[iy,ix]
    if mode=='past_logit':
        output=original_logits.clone()
        prior=torch.lerp(original_logits[0,0,iy,ix],previous_logits[0,0,jy,jx],alpha)
        output[0,0,iy,ix]=torch.maximum(original_logits[0,0,iy,ix],prior)
        return output,mask
    if mode=='same_cell':
        jy,jx=iy,ix
    elif mode=='permute':
        jy,jx=jy.roll(1),jx.roll(1)
    updated=current.clone()
    updated[0,:,iy,ix]=torch.lerp(current[0,:,iy,ix],previous[0,:,jy,jx],alpha)
    proposal=cls_head(updated)
    output=original_logits.clone()
    output[0,0,iy,ix]=torch.maximum(original_logits[0,0,iy,ix],proposal[0,0,iy,ix])
    return output,mask


class NativeHistory:
    """Keep only original data from the preceding two frames; never feed enhanced features back."""
    def __init__(self):
        self.entries=deque(maxlen=2)

    def advance(self,stream_key,frame,current,logits,cls_head,mode='aligned',content=None):
        if self.entries:
            key,t,f,_=self.entries[-1]
            if key!=stream_key or t+1!=frame or f.shape!=current.shape:
                self.entries.clear()
        mask=torch.zeros(current.shape[-2:],dtype=torch.bool,device=current.device)
        if len(self.entries)==2:
            _,_,past,previous_logits=self.entries[0]
            updated,mask=classify_step(current,past,logits,previous_logits,cls_head,mode=mode,content=content)
        else:
            updated=logits.clone()
        self.entries.append((stream_key,frame,current.detach().clone(),logits.detach().clone()))
        return updated,mask


def replace_p3(raw,logits):
    """Keep the local YOLO11 Detect raw format and replace only the P3 class slice."""
    count=logits.shape[-2]*logits.shape[-1]
    out=dict(raw,scores=raw['scores'].clone())
    out['scores'][...,:count]=logits.flatten(2)
    return out


def self_check():
    torch.set_num_threads(1)
    torch.manual_seed(117)
    def cls(x): return 2*x[:,:1]-3.3
    a=torch.tensor([1.,4,-3,2,-1,0,1,-2]); b=torch.tensor([1.,-3,4,-2,1,-1,0,2])
    prev=torch.zeros(1,8,7,9); cur=torch.zeros_like(prev)
    prev[0,:,3,2]=2.5*a; cur[0,:,3,3]=.85*a
    prev[0,:,5,4]=2.5*b; cur[0,:,5,4]=.85*b
    native=cls(cur); past=cls(prev)
    def score(c,old,x=3,y=3,mode='aligned'):
        logits,mask=classify_step(c,old,cls(c),cls(old),cls,mode)
        return float(logits[0,0,y,x].sigmoid()),bool(mask[y,x])
    aligned,selected=classify_step(cur,prev,native,past,cls)
    fixed,_=classify_step(cur,prev,native,past,cls,'same_cell')
    assert selected.sum()==2
    assert native[0,0,3,3].sigmoid()<.25 and aligned[0,0,3,3].sigmoid()>=.25
    assert fixed[0,0,3,3].sigmoid()<.25 and fixed[0,0,5,4]==aligned[0,0,5,4]
    assert torch.equal(classify_step(cur,prev,native,past,cls,alpha=0)[0],native)
    z,_=classify_step(cur,prev,native,past,cls,'zero_motion')
    assert z[0,0,3,3]==native[0,0,3,3] and z[0,0,5,4]==aligned[0,0,5,4]
    dropped=torch.ones(7,9,dtype=torch.bool); dropped[3,2]=False
    assert not classify_step(cur,prev,native,past,cls,content=dropped)[1][3,3]
    # With a linear classifier, mixing features and mixing the aligned logits are identical.
    assert torch.allclose(classify_step(cur,prev,native,past,cls,'past_logit')[0],aligned,atol=1e-6)
    memory=NativeHistory()
    for t,feat in ((1,prev),(2,prev),(3,cur)):
        y,m=memory.advance('A',t,feat,cls(feat),cls)
        if t<3: assert not m.any()
    assert torch.equal(y,aligned)
    assert not memory.advance('B',4,cur,native,cls)[1].any()
    assert not memory.advance('B',6,cur,native,cls)[1].any()
    empty=torch.zeros_like(cur)
    assert not classify_step(empty,prev,cls(empty),past,cls)[1].any()
    ambiguous=cur.clone(); ambiguous[0,:,3,4]=.85*a
    assert not score(ambiguous,prev)[1]
    # No recursively strengthened history: a weak native prior cannot promote the next query.
    assert not classify_step(cur,cur,native,native,cls)[1].any()
    # Identical evidence permits a nuisance world to receive exactly the same promotion.
    # A geometric reciprocity check does not prove that the returned detection is a bee.
    nuisance=cur.clone(); nuisance_prev=prev.clone()
    assert score(nuisance,nuisance_prev)==score(cur,prev)
    fake={'scores':torch.cat((native.flatten(2),torch.ones(1,1,7)),dim=-1),
          'boxes':torch.rand(1,64,70),'feats':[cur]}
    out=replace_p3(fake,aligned)
    assert out['boxes'] is fake['boxes'] and torch.equal(out['scores'][...,-7:],fake['scores'][...,-7:])
    print('Moved/stationary bees, missing/ambiguous evidence, masks, cold state, time/clip resets, nuisance ambiguity, score interpolation and regression controls passed.')


def random_detector_check():
    from ultralytics.nn.tasks import DetectionModel
    torch.set_num_threads(1)
    model=DetectionModel('yolo11s.yaml',nc=1,verbose=False).cpu().eval()
    head=model.model[-1]
    with torch.inference_mode():
        pred,raw=model(torch.rand(1,3,128,160))
        p3=raw['feats'][0]
        base=head.cv3[0](p3)
        assert torch.equal(base.flatten(2),raw['scores'][...,:base.shape[2]*base.shape[3]])
        identity=replace_p3(raw,base)
        assert torch.equal(head._inference(identity),pred)
        changed=base.clone(); changed[...,1,1]+=2
        mod=head._inference(replace_p3(raw,changed))
        assert torch.equal(mod[:,:4],pred[:,:4]) and torch.count_nonzero(mod[:,4:]!=pred[:,4:])==1
    print('Randomly initialized single-class YOLO11s: native P3 logits, identity decode and current boxes verified.')


if __name__=='__main__':
    self_check()
    random_detector_check()
