"""Full-resolution lens, stock YOLO11 preprocessing, then NMS in camera coordinates.

Use a fresh YOLO instance for this predictor so an earlier predict() call cannot reuse
a different cached predictor. Set the lens via functools.partial(WarpPredictor, lens=lens).
"""

import torch
from ultralytics.engine.results import Results
from ultralytics.models.yolo.detect.predict import DetectionPredictor
from ultralytics.utils import nms, ops


def native_detections(raw, input_hw, source_hw, lens, classes, conf=.05, iou=.7, max_det=300):
    """Decode a single dense YOLO11 detect head. Keep the original head dtype through xywh2xyxy."""
    if isinstance(raw, (tuple, list)):
        raw = raw[0]
    if raw.ndim != 3 or raw.shape[:2] != (1, 4 + classes) or raw.shape[2] == 6:
        raise ValueError("Expected a single dense, non-end-to-end YOLO11 detect output")
    candidates = raw[0].T
    score, label = candidates[:, 4:].max(dim=1)
    above = score > conf
    score, label = score[above].float(), label[above].float()
    boxes = ops.xywh2xyxy(candidates[above, :4]).float()
    if not len(boxes):
        return raw.new_empty((0, 6), dtype=torch.float32)

    # The pinned scale_boxes uses separate gains after raster rounding. It clips xyxy,
    # so undo the affine transform here and leave clipping until after suppression.
    h, w = source_hw
    oh, ow = input_hw
    r = min(oh/h, ow/w)
    nh, nw = max(1, round(h*r)), max(1, round(w*r))
    px, py = round((ow-nw)/2-.1), round((oh-nh)/2-.1)
    boxes[:, 0::2] = (boxes[:, 0::2]-px) / (nw/w)
    boxes[:, 1::2] = (boxes[:, 1::2]-py) / (nh/h)
    if lens is not None:
        transformed = lens.boxes(boxes.detach().cpu().numpy(), inverse=True)
        boxes = torch.as_tensor(transformed, dtype=boxes.dtype, device=boxes.device)
    # Pinned Ultralytics 8.4.208: class-aware offset, strict threshold and max_det.
    if len(boxes) > 30000:
        cap = score.argsort(descending=True)[:30000]
        boxes, score, label = boxes[cap], score[cap], label[cap]
    kept = nms.nms_boxes(boxes + label[:, None]*7680, score, iou)[:max_det]
    return torch.column_stack((ops.clip_boxes(boxes[kept], source_hw), score[kept], label[kept]))


class WarpPredictor(DetectionPredictor):
    def __init__(self, *args, lens=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.lens = lens

    def preprocess(self, im):
        if not isinstance(im, list) or len(im) != 1:
            raise ValueError("Use one full-resolution uint8 camera image per prediction")
        src = im[0]
        warped = self.lens.frame(src) if self.lens is not None else src
        if src.dtype.name != "uint8" or src.shape != warped.shape or warped.dtype != src.dtype:
            raise ValueError("Expected an unchanged camera canvas and uint8 BGR pixel format")
        return super().preprocess([warped])

    def postprocess(self, preds, img, orig_imgs, **kwargs):
        if (self.args.task != "detect" or getattr(self.model, "end2end", False)
                or self.args.agnostic_nms or self.args.classes is not None or self.args.augment):
            raise ValueError("Use the original single-pass class-aware YOLO11 detect protocol")
        original = orig_imgs[0]
        boxes = native_detections(preds, img.shape[-2:], original.shape[:2], self.lens,
                                  len(self.model.names), self.args.conf, self.args.iou, self.args.max_det)
        return [Results(original, path=self.batch[0][0], names=self.model.names, boxes=boxes)]
