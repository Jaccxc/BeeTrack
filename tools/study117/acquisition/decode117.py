"""Decode the §117 six-arm dense YOLO11 head into original-camera measurements."""

from time import perf_counter

import torch
from ultralytics.utils import nms, ops


ARMS = ("U", "A", "Z", "Q", "Zm", "D")


def original_boxes(pred, input_hw, source_hw):
    # Keep out-of-image coordinates for the raw donor evidence; stock clips only after NMS.
    ih, iw = input_hw
    h, w = source_hw
    gain = min(ih/h, iw/w)
    nh, nw = max(1, round(h*gain)), max(1, round(w*gain))
    px, py = round((iw-nw)/2-.1), round((ih-nh)/2-.1)
    xyxy = ops.xywh2xyxy(pred[0, :4].T)
    xyxy[:, 0::2] = (xyxy[:, 0::2]-px) / (nw/w)
    xyxy[:, 1::2] = (xyxy[:, 1::2]-py) / (nh/h)
    return xyxy


@torch.inference_mode()
def summarize(pred, arm_logits, info, past_boxes, source_hw, input_hw, past_logits=None):
    """Return (per-frame evidence, native unbounded camera xyxy for all current P3 cells).

    The runner stores the frame identity, so an eligible row always refers to local t and t−2.
    `pred` is the original single-class dense [1,5,N] output. `past_boxes` is this same
    function's second return from t−2. Both boxes and native logits refer to the unaltered stream.
    """
    if (pred.ndim != 3 or pred.shape[:2] != (1,5) or pred.shape[-1] == 6
            or pred.device.type != "cpu" or pred.dtype != torch.float32):
        raise ValueError("Expected native single-class CPU FP32 YOLO dense head [1,5,N]")
    if set(arm_logits) != set(ARMS):
        raise ValueError("Expected scores for exactly the six registered arms")
    native = arm_logits["U"]
    if native.ndim != 4 or native.shape[:2] != (1,1) or native.numel() > pred.shape[-1]:
        raise ValueError("Mismatched native P3 logits")
    n3 = native.numel()
    raw_boxes = original_boxes(pred, input_hw, source_hw)
    clip_boxes = ops.clip_boxes(raw_boxes.clone(), source_hw)
    ids = info["current_ids"].tolist()
    donor_ids = info["donor_ids"].tolist()
    z_ids = info["z_current_ids"].tolist()
    if (len(ids) != len(donor_ids) or ids != sorted(set(ids)) or z_ids != sorted(set(z_ids))
            or any(i < 0 or i >= n3 for i in (*ids, *donor_ids, *z_ids))):
        raise ValueError("Invalid eligible P3 anchor indices")
    if ids and (past_logits is None or past_logits.shape != native.shape
                or past_boxes is None or past_boxes.shape != (n3, 4)):
        raise ValueError("Selected donors require the original t−2 P3 boxes and logits")
    native_scores = pred[0,4,:n3]
    if not torch.allclose(native.sigmoid().flatten(), native_scores, atol=1e-7, rtol=0):
        raise ValueError("Captured native P3 logits differ from decoded scores")
    started = perf_counter()
    base = nms.non_max_suppression(pred, conf_thres=.05, iou_thres=.7, max_det=300, nc=1)[0]
    base[:, :4] = ops.scale_boxes(input_hw, base[:, :4], source_hw)
    stock_ms = 1000*(perf_counter()-started)
    timing = {"stock_reference":stock_ms, "indexed":{}}
    by_arm, eligible_scores, used = {}, {}, set(ids)
    cached = []
    for name in ARMS:
        logits = arm_logits[name]
        if logits.shape != native.shape:
            raise ValueError(f"Mismatched P3 geometry for {name}")
        if torch.equal(logits, native):
            scores = native_scores
        else:
            # Always retain original probabilities for unchanged cells, including P4/P5.
            scores = torch.where(logits.flatten()!=native.flatten(), logits.sigmoid().flatten(), native_scores)
        eligible_scores[name] = scores[ids].tolist()
        started = perf_counter()
        same = next((arm for archived,arm in cached if torch.equal(archived,scores)), None)
        lookup_ms = 1000*(perf_counter()-started)
        if same is not None:
            by_arm[name] = by_arm[same]
            proxy = stock_ms if same == "U" else timing["indexed"][same]["nms_scale_ms"]
            timing["indexed"][name] = dict(cache_hit=True, reused_arm=same, lookup_ms=lookup_ms,
                                           nms_scale_ms=0., standalone_proxy_ms=proxy)
            continue
        arm = pred.clone()
        arm[0,4,:n3] = scores
        started = perf_counter()
        selected, anchors = nms.non_max_suppression(
            arm, conf_thres=.05, iou_thres=.7, max_det=300, nc=1, return_idxs=True)
        rows, indices = selected[0], anchors[0]
        scaled = ops.scale_boxes(input_hw, rows[:, :4].clone(), source_hw)
        nms_ms = 1000*(perf_counter()-started)
        timing["indexed"][name] = dict(cache_hit=False, reused_arm=None, lookup_ms=lookup_ms,
                                       nms_scale_ms=nms_ms, standalone_proxy_ms=stock_ms if name=="U" else nms_ms)
        if not torch.equal(scaled, clip_boxes[indices]) or not torch.equal(rows[:,4], arm[0,4,indices]):
            raise ValueError("Stock NMS anchor identity, coordinates, or scores differ")
        if name == "U" and (not torch.equal(base[:, :4],scaled) or not torch.equal(base[:,4:],rows[:,4:])):
            raise ValueError("Tracked native anchor IDs change official detections")
        index_list = indices.tolist()
        used.update(index_list)
        by_arm[name] = [[i, score] for i,score in zip(index_list,rows[:,4].tolist())]
        cached.append((scores,name))
    eligible = []
    if ids:
        past = past_boxes[donor_ids]
        past_clipped = ops.clip_boxes(past.clone(), source_hw).tolist()
        past_scores = past_logits.sigmoid().flatten()[donor_ids].tolist()
        current_raw = raw_boxes[ids].tolist()
        donor_raw = past.tolist()
        for k, (i,j) in enumerate(zip(ids,donor_ids)):
            eligible.append(dict(current_id=i, donor_id=j, moved=(i!=j), current_raw_box=current_raw[k],
                                 donor_raw_box=donor_raw[k], donor_box=past_clipped[k], donor_score=past_scores[k],
                                 scores={a:eligible_scores[a][k] for a in ARMS}))
    used = sorted(used)
    boxes = dict(zip(map(str,used), clip_boxes[used].tolist()))
    return dict(boxes=boxes, arms=by_arm, eligible=eligible, z_ids=z_ids,
                stock_u_exact=True, nms_timing_ms=timing), raw_boxes[:n3].clone()


def self_check():
    """Exercise strict threshold, cross-scale suppression, box clipping and same-forward order."""
    torch.set_num_threads(1)
    src_hw, model_hw = (251,401),(96,160)
    xywh = torch.tensor([[255,255,10,10],[5,5,3,3],[5,5,3,3],[35,30,5,6],
                         [60,40,5,6],[5,5,3,3],[100,25,8,8],[10,10,1,1]], dtype=torch.float32)
    logits = torch.tensor([-.4,-2.2,-1.8,-3.6,-3.5]).reshape(1,1,1,5)
    head = torch.cat((xywh.T,torch.cat((logits.flatten().sigmoid(),torch.tensor([.24,.05,.051])))[None]))[None]
    arms = {a:logits.clone() for a in ARMS}
    arms["A"].flatten()[1] = .9
    arms["Q"].flatten()[1] = .65
    empty = torch.empty(0,dtype=torch.long)
    cold, past = summarize(head,dict.fromkeys(ARMS,logits),
                          dict(current_ids=empty,donor_ids=empty,z_current_ids=empty),None,src_hw,model_hw)
    assert cold["arms"]["U"] == cold["arms"]["D"] and [r[0] for r in cold["arms"]["U"]]==[0,5,7]
    timing=cold["nms_timing_ms"]
    assert timing["indexed"]["A"]["cache_hit"] and timing["indexed"]["A"]["nms_scale_ms"]==0
    assert timing["indexed"]["A"]["standalone_proxy_ms"]==timing["stock_reference"]>0
    current, now = summarize(head,arms,dict(current_ids=torch.tensor([1]), donor_ids=torch.tensor([0]),
                              z_current_ids=empty),past,src_hw,model_hw,logits)
    assert torch.equal(now,past) and current["arms"]["A"][0][0]==1
    row, = current["eligible"]
    assert row["moved"] and row["current_id"]==1 and row["donor_raw_box"]!=row["donor_box"]
    assert row["donor_box"]==cold["boxes"]["0"] and row["scores"]["U"]<.25<row["scores"]["A"]
    boosted=torch.full((303,),.85)
    synthetic=torch.zeros(1,5,303)
    synthetic[0,0]=torch.arange(303)*2
    synthetic[0,2:4]=1.0
    synthetic[0,4]=boosted
    equal=boosted[:256].logit().reshape(1,1,16,16)
    clipped, _=summarize(synthetic,dict.fromkeys(ARMS,equal),
                         dict(current_ids=empty,donor_ids=empty,z_current_ids=empty),None,src_hw,model_hw)
    assert len(clipped["arms"]["U"])==300
    print("Stock NMS, anchor/class scores, suppression across scales, clipping, donors, ties and max_det verified.")


if __name__=="__main__":
    self_check()
