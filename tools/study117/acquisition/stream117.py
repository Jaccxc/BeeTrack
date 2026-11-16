"""Causal P3 correction for one camera and one frozen single-class FP32 YOLO detector.

Create TemporalP3('A') for deployment or TemporalP3('research') for U/A/Z/Q/Zm/D.
On every original forward pass, extract current = raw['feats'][0], and reshape
the first H*W values of raw['scores'] into Bx1xHxW as logits. Call
advance((clip_id, model_hash, letterbox_signature), frame_id, current, logits,
head.cv3[0], content_mask). Outputs map arm names to read-only P3 logits. The
caller passes those logits and the untouched box regression into its usual
decoder and NMS. No new detector forward pass takes place in this adapter.
"""
from collections import deque
from numbers import Integral

import torch

from feature_arms import ARM_NAMES, compute_arms


class TemporalP3:
    def __init__(self, mode='A'):
        if mode not in ('A','research'):
            raise ValueError("Expected 'A' or 'research'")
        self.arms=ARM_NAMES if mode=='research' else ('A',)
        self.native=deque(maxlen=2)
        self.stream_key=None
        self.last_frame=None
        self.content=None

    def reset(self):
        self.native.clear()
        self.stream_key=None
        self.last_frame=None
        self.content=None

    @torch.inference_mode()
    def advance(self,stream_key,frame_id,current,logits,cls_head,content):
        """Return (P3 arm logits, row-major selected/donor cell IDs); frame numbers are one-based."""
        if not isinstance(frame_id,Integral) or isinstance(frame_id,bool) or frame_id<1:
            raise ValueError('Expected a positive integer native frame number')
        if current.ndim!=4 or current.shape[0]!=1 or current.dtype!=torch.float32:
            raise ValueError('Expected single-stream native FP32 P3 features')
        if logits.shape!=(1,1,*current.shape[-2:]) or logits.dtype!=current.dtype or logits.device!=current.device:
            raise ValueError('Expected matching single-class native P3 logits')
        if content.shape!=current.shape[-2:] or content.dtype!=torch.bool or content.device!=current.device:
            raise ValueError('Expected a valid bool content mask on the same P3 grid')
        if self.native:
            previous=self.native[-1][0]
            clean=(stream_key==self.stream_key and frame_id==self.last_frame+1 and
                   previous.shape==current.shape and previous.device==current.device and
                   torch.equal(content,self.content))
        else:
            clean=False
        history=self.native[0] if clean and len(self.native)==2 else (None,None)
        outputs,info=compute_arms(current,history[0],logits,history[1],cls_head,content,arms=self.arms)
        # Save originals only, after a successful computation. Clones isolate caller-owned buffers.
        original=(current.detach().clone(),logits.detach().clone())
        if not clean:
            self.native.clear()
            self.stream_key=stream_key
            self.content=content.detach().clone()
        self.native.append(original)
        self.last_frame=int(frame_id)
        return outputs,info


def check_stream():
    """One thousand synthetic forwards, plus independent donor and data-lifetime checks."""
    torch.set_num_threads(1)
    torch.manual_seed(117)
    class CountHead:
        def __init__(self): self.calls=0
        def __call__(self,features):
            self.calls+=1
            return 2*features[:,:1]-3.3

    def logits(features): return 2*features[:,:1]-3.3
    strong=torch.zeros(1,8,5,8)
    weak=torch.zeros_like(strong)
    a=torch.tensor([1.,4,-3,2,-1,0,1,-2]); b=torch.tensor([1.,-3,4,-2,1,-1,0,2])
    strong[0,:,1,2]=2.5*a; weak[0,:,1,3]=.85*a
    strong[0,:,3,6]=2.5*b; weak[0,:,3,6]=.85*b
    empty=torch.zeros_like(weak)
    keep=torch.ones((5,8),dtype=torch.bool)
    live,study=TemporalP3('A'),TemporalP3('research')
    live_head,study_head=CountHead(),CountHead()
    for frame in range(1,1001):
        # True weak detections occur two frames after the strong baseline boxes.
        # Frames 5 and 7 exercise repeated weak evidence, which cannot self-propagate.
        phase=(frame-1)%20+1
        source=strong if phase==1 else weak if phase in (3,5,7) else empty
        offered=source.clone(); native_logits=logits(offered)
        a_before,all_before=live_head.calls,study_head.calls
        solo,sinfo=live.advance(('synthetic','model1','pad1'),frame,offered,native_logits,live_head,keep)
        all_arms,ainfo=study.advance(('synthetic','model1','pad1'),frame,offered,native_logits,study_head,keep)
        assert tuple(solo)==('A',) and tuple(all_arms)==ARM_NAMES and torch.equal(solo['A'],all_arms['A'])
        assert torch.equal(sinfo['current_ids'],ainfo['current_ids']) and torch.equal(sinfo['donor_ids'],ainfo['donor_ids'])
        assert torch.equal(all_arms['U'],native_logits)
        if phase==3:
            assert ainfo['current_ids'].tolist()==[11,30] and ainfo['donor_ids'].tolist()==[10,30]
            assert ainfo['z_current_ids'].tolist()==[30]
            assert live_head.calls==a_before+1 and 1<=study_head.calls-all_before<=4
            assert float(solo['A'][0,0,1,3].sigmoid())>=.25
        else:
            assert not len(sinfo['current_ids']) and live_head.calls==a_before
            assert study_head.calls==all_before
        # A camera decoder may recycle its frame/feature buffers after this call.
        offered.fill_(99); native_logits.fill_(99)
    assert live_head.calls==50 and len(live.native)==2

    def cold_at_change(reason,new_key,new_frame,new_feat,new_mask):
        stream=TemporalP3('A'); head=CountHead()
        old_key=('synthetic','model1','pad1')
        for f in (1,2):
            stream.advance(old_key,f,strong,logits(strong),head,keep)
        pred,ids=stream.advance(new_key,new_frame,new_feat,logits(new_feat),head,new_mask)
        assert not len(ids['current_ids']) and not head.calls and torch.equal(pred['A'],logits(new_feat)),reason
    key=('synthetic','model1','pad1')
    cold_at_change('camera',('other','model1','pad1'),3,weak,keep)
    cold_at_change('new model',('synthetic','model2','pad1'),3,weak,keep)
    cold_at_change('different transform',('synthetic','model1','pad2'),3,weak,keep)
    cold_at_change('skipped frame',key,4,weak,keep)
    cold_at_change('duplicate frame',key,2,weak,keep)
    cold_at_change('grid shape',key,3,weak[:,:,:,:7],keep[:,:7])
    shifted=keep.clone(); shifted[:,0]=False
    cold_at_change('different padding mask',key,3,weak,shifted)
    live.reset()
    assert len(live.native)==0 and live.content is None and live.last_frame is None
    print('1,000-frame A-only vs research bit-identical; 50 selected frames, exactly 50 A tower calls. Camera, model, transform, shape, mask, gaps, duplicate frames, mutable buffers and no feedback passed.')


def check_correspondence():
    """Enumerate local cosine competitors independently, without using the vectorized matcher."""
    import numpy as np
    import feature_arms
    from unittest.mock import patch

    rng=np.random.default_rng(117)
    h,w=5,8
    old=rng.normal(size=(h,w,12)).astype(np.float32)
    new=np.roll(old,1,axis=1)+.01*rng.normal(size=old.shape).astype(np.float32)
    keep=rng.random((h,w))>.12
    old[1,0]=0
    new[1,0]=new[1,1]=old[1,2]  # reversal ambiguity among current cells
    native=rng.uniform(-3,-.2,size=(h,w)).astype(np.float32)
    past=rng.uniform(-2,1,size=(h,w)).astype(np.float32)
    forward={}; reverse={}
    for y,x in zip(*np.where(keep)):
        c=new[y,x].astype(float); c-=c.mean()
        for j in range(max(0,y-2),min(h,y+3)):
            for k in range(max(0,x-2),min(w,x+3)):
                if not keep[j,k]: continue
                p=old[j,k].astype(float); p-=p.mean()
                length=np.linalg.norm(c)*np.linalg.norm(p)
                if length<1e-12: continue
                similarity=float(np.dot(c,p)/length)
                forward.setdefault(y*w+x,[]).append((similarity,j*w+k))
                reverse.setdefault(j*w+k,[]).append((similarity,y*w+x))
    def unique_winner(options):
        ranked=sorted(options,reverse=True)
        if not ranked or ranked[0][0]<.8 or (len(ranked)>1 and ranked[0][0]-ranked[1][0]<.05):
            return None
        return ranked[0][1]
    expected=[]
    for target,options in sorted(forward.items()):
        donor=unique_winner(options)
        if donor is None or unique_winner(reverse[donor])!=target: continue
        pc=1/(1+np.exp(-float(native.flat[target])))
        pp=1/(1+np.exp(-float(past.flat[donor])))
        if .05<=pc<.25 and pp>=.25: expected.append((target,donor))
    assert expected
    cur=torch.from_numpy(new).permute(2,0,1).unsqueeze(0)
    previous=torch.from_numpy(old).permute(2,0,1).unsqueeze(0)
    l=torch.from_numpy(native)[None,None]; p=torch.from_numpy(past)[None,None]
    keep_t=torch.from_numpy(keep)
    with patch.object(feature_arms,'best_pair_indices',wraps=feature_arms.best_pair_indices) as matcher:
        result,ids=compute_arms(cur,previous,l,p,lambda x:x[:,:1],keep_t)
        assert matcher.call_count==1 and tuple(result)==ARM_NAMES
        assert list(zip(ids['current_ids'].tolist(),ids['donor_ids'].tolist()))==expected
    # Each non-Q arm requests the same single stationary donor, so one tower call suffices for all.
    current=torch.zeros(1,4,5,8); initial=torch.zeros_like(current)
    initial[0,:,2,3]=torch.tensor([3.,1.,-2.,-.5]); current[0,:,2,3]=initial[0,:,2,3]/4
    calls=[]
    def tower(x):
        calls.append(x.shape)
        return x[:,:1]
    all_arms,ids=compute_arms(current,initial,torch.full_like(l,-2.),torch.ones_like(p),tower,torch.ones_like(keep_t))
    assert ids['current_ids'].tolist()==ids['donor_ids'].tolist()==[19]
    assert len(calls)==1 and all(torch.equal(all_arms['A'],all_arms[a]) for a in ('Z','Zm','D'))
    print('Independent scalar competition matched all donor IDs; one correspondence search for six active arms, one shared tower call for identical stationary treatments.')


if __name__=='__main__':
    check_stream()
    check_correspondence()
