"""Synthetic-only CPU parity check. Run via the BeeTrack uv environment.

python tools/study116/check_warp.py --geometry tools/study116 --weights runs/detect/loho_no33_s0/weights/best.pt
"""

import argparse
from functools import partial
import hashlib
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
for var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[var] = "1"

import cv2
import numpy as np
import torch
from ultralytics import YOLO, __version__
from ultralytics.models.yolo.detect.predict import DetectionPredictor
from ultralytics.utils import ops, nms

from warp_predict import WarpPredictor, native_detections


def scene(h, w):
    rng = np.random.default_rng(137)
    grain = np.clip(rng.normal(185, 12, (h, w)), 0, 255).astype(np.uint8)
    bgr = np.stack([grain*.82, grain*.95, grain], axis=-1).astype(np.uint8)
    for y in range(24, h-16, 40):
        for x in range(24, w-16, 52):
            size = rng.uniform(.55, 1.8)*w/700
            def blob(dx, dy, a, b, color, angle=0):
                cv2.ellipse(bgr, (int(x+dx*size), int(y+dy*size)),
                            (max(1, int(a*size)), max(1, int(b*size))), angle, 0, 360, color, -1, cv2.LINE_AA)
            blob(1,1,12,6,(68,87,109),15)
            blob(0,0,10,5,(29,37,43),14)
            for off in (-5,-1,3):
                blob(off,0,1.2,4,(51,100,147),14)
            blob(8,0,4,4,(17,23,25))
            for sign in (-1,1):
                blob(0,sign*4,7,2.2,(133,151,165),sign*32)
    return bgr


def raw_boxes(boxes, scores):
    xywh = ops.xyxy2xywh(torch.as_tensor(np.asarray(boxes), dtype=torch.float32))
    return torch.column_stack((xywh, torch.tensor(scores, dtype=torch.float32))).T.unsqueeze(0)


def synth_nms(lens_class):
    # Unclipped border boxes share IoU .5. Early clipping would turn it into 1.0.
    boxes = [[-100,10,100,30],[0,10,100,30],[0,10,100,30],[170,30,180,40],
             [180,40,190,50],[185,70,190,80],[190,80,195,90],[10,10,30,20]]
    scores = [[.9,0],[.8,0],[0,.7],[.05,0],[.049,0],[.4,0],[.3,0],[.01,0]]
    raw = raw_boxes(boxes, scores)
    out = native_detections(raw,(128,224),(128,224),None,2)
    stock = nms.non_max_suppression(raw.clone(),.05,.7,nc=2)[0]
    ops.clip_boxes(stock[:,:4],(128,224))
    assert torch.equal(out, stock) and len(out)==5
    cap = native_detections(raw,(128,224),(128,224),None,2,max_det=2)
    assert torch.equal(cap,out[:2])
    for dtype in (torch.float32,torch.float16):
        head=raw.to(dtype)
        actual=native_detections(head,(128,224),(297,628),None,2)
        stock=nms.non_max_suppression(head.clone(),.05,.7,nc=2)[0]
        stock[:,:4]=ops.scale_boxes((128,224),stock[:,:4],(297,628))
        assert torch.equal(actual,stock)
    dropped = nms.non_max_suppression(raw_boxes(np.clip(boxes,0,None), scores),.05,.7,nc=2)[0]
    assert len(dropped) == 4

    lens = lens_class((224,128), (60,25,140,90), 1.5)
    def overlap(a,b):
        return max(0,min(a[2],b[2])-max(a[0],b[0])) / (max(a[2],b[2])-min(a[0],b[0]))
    for x0 in range(15,58):
        a=[x0,45,100,65]
        for dx in range(3,22):
            b=[x0+dx,45,100+dx,65]
            aw,bw=lens.boxes([a,b])
            orig_iou, warped_iou = overlap(a,b), overlap(aw,bw)
            if (orig_iou-.7)*(warped_iou-.7)<0 and min(abs(orig_iou-.7),abs(warped_iou-.7))>.008:
                candidates=raw_boxes([aw,bw]+[aw]*5, [[.9],[.8]]+[[0]]*5)
                native=native_detections(candidates,(128,224),(128,224),lens,1)
                wrapped=nms.non_max_suppression(candidates.clone(),.05,.7,nc=1)[0]
                assert len(native)==(1 if orig_iou>.7 else 2) and len(native)!=len(wrapped)
                return dict(native=round(orig_iou,6),warped=round(warped_iou,6),
                            native_kept=len(native),warped_kept=len(wrapped),before_clip=5,early_clip=4)
    raise AssertionError("Missing synthetic NMS order counterexample")


class Capture:
    def postprocess(self, preds, img, orig_imgs, **kwargs):
        head=preds[0] if isinstance(preds,(list,tuple)) else preds
        self.capture=dict(shape=list(img.shape),dtype=str(img.dtype),nc=len(self.model.names),
                          input=hashlib.sha256(img.contiguous().numpy().tobytes()).hexdigest(),
                          head=hashlib.sha256(head.contiguous().numpy().tobytes()).hexdigest(),
                          candidates=int((head[0,4:].amax(0)>self.args.conf).sum().item()))
        return super().postprocess(preds,img,orig_imgs,**kwargs)


class Stock(Capture,DetectionPredictor):
    pass


class Adapted(Capture,WarpPredictor):
    def postprocess(self, preds, img, orig_imgs, **kwargs):
        # Identify retained dense-head anchors on the *same forward pass* when NMS order changes.
        head=preds[0] if isinstance(preds,(list,tuple)) else preds
        anchors=torch.nonzero(head[0,4:].amax(0)>self.args.conf,as_tuple=True)[0]
        assert len(anchors)<=30000
        _,wrapped=nms.non_max_suppression(head.clone(), self.args.conf,self.args.iou,
                                          nc=len(self.model.names),max_det=self.args.max_det,return_idxs=True)
        suppress=nms.nms_boxes
        selected=[]
        def capture_kept(boxes,scores,threshold):
            idx=suppress(boxes,scores,threshold)
            selected.extend(anchors[idx[:self.args.max_det]].tolist())
            return idx
        with patch.object(nms,"nms_boxes",capture_kept):
            result=super().postprocess(preds,img,orig_imgs,**kwargs)
        wrapped_ids=set(wrapped[0].tolist())
        selected_ids=set(selected)
        self.order_check=dict(native_only=len(selected_ids-wrapped_ids),warped_only=len(wrapped_ids-selected_ids),
                              both=len(wrapped_ids&selected_ids))
        return result


def check(weights, geometry):
    if __version__!="8.4.208" or weights.resolve().parts[-4:]!=("detect","loho_no33_s0","weights","best.pt"):
        raise ValueError("Use the pinned package and the allowed loho_no33_s0 research checkpoint")
    sys.path.insert(0, str(geometry.resolve()))
    from entrance_scale import EntranceScale
    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    check_nms=synth_nms(EntranceScale)
    stock, adapted=YOLO(weights), YOLO(weights)
    opts=dict(device="cpu",imgsz=640,conf=.05,iou=.7,max_det=300,half=True,verbose=False,save=False)
    rows=[]
    for h,w in ((473,853),(833,473),(641,641)):
        im=scene(h,w)
        poly=[(w*.3,h*.3),(w*.65,h*.3),(w*.65,h*.6),(w*.3,h*.6)]
        identity=EntranceScale.from_polygon((w,h),poly,1)
        expected=stock.predict(im,predictor=Stock,**opts)[0].boxes.data
        capture=stock.predictor.capture.copy()
        if adapted.predictor is not None:
            adapted.predictor.lens=identity
        result=adapted.predict(im,predictor=partial(Adapted,lens=identity),**opts)[0].boxes.data
        assert adapted.predictor.capture==capture and torch.equal(expected,result)
        assert adapted.predictor.order_check==dict(native_only=0,warped_only=0,both=len(result))
        focus=EntranceScale.from_polygon((w,h),poly,1.5)
        adapted.predictor.lens=focus
        changed=adapted.predict(im,**opts)[0].boxes.data
        assert adapted.predictor.capture["shape"]==capture["shape"]
        assert adapted.predictor.capture["input"]!=capture["input"]
        rows.append(dict(image_hw=[h,w],batch=capture["shape"],dtype=capture["dtype"],
                         native_candidates=capture["candidates"],native_kept=len(result),
                         focused_kept=len(changed),focused_nms=adapted.predictor.order_check,
                         input_equal=True,raw_equal=True,identity_box_error=0))
    assert any(r["native_kept"] for r in rows), "Synthetic weighted run must exercise nonempty stock NMS"
    return dict(ultralytics=__version__,weights=weights.parent.parent.name,
                NMS_order=check_nms,scenes=rows)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometry",type=Path,required=True,help="Directory containing entrance_scale.py")
    parser.add_argument("--weights",type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(check(args.weights,args.geometry),indent=2))
