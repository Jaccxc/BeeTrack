"""Run the §116 annotation-free detector arm on the two registered source tails."""

import argparse
from functools import partial
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter


HERE = Path(__file__).resolve().parent
DOSES = {"U":640, "L":640, "C":640, "H":960}


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(path.read_bytes())


def load_components(geometry, inspector_dir):
    sys.path[:0] = [str(geometry.resolve()), str(inspector_dir.resolve())]
    import entrance_scale
    import inspector
    import ultralytics
    import warp_predict
    from ultralytics.engine import predictor
    from ultralytics.models.yolo.detect import predict
    from ultralytics.utils import ops, nms
    import pilot_cpu

    if ultralytics.__version__ != "8.4.208":
        raise ValueError("§116 is pinned to Ultralytics 8.4.208")
    scripts = [Path(__file__), *[Path(m.__file__) for m in
               (entrance_scale, inspector, pilot_cpu, warp_predict, predictor, predict, nms, ops)]]
    hashes = {p.name:digest(p) for p in scripts}
    return entrance_scale.EntranceScale, inspector, hashes


def predictor_class():
    from ultralytics.models.yolo.detect.predict import DetectionPredictor
    from warp_predict import WarpPredictor
    import torch

    class Measured(WarpPredictor):
        def preprocess(self, images):
            start = perf_counter()
            original, = images
            canvas = self.lens.frame(original) if self.lens is not None else original
            if canvas.shape != original.shape or canvas.dtype != original.dtype:
                raise ValueError("Warp changed the input canvas")
            warped = perf_counter()
            tensor = DetectionPredictor.preprocess(self, [canvas])
            self.timing = {"warp":1000*(warped-start), "preprocess":1000*(perf_counter()-warped)}
            return tensor

        def inference(self, img, *args, **kwargs):
            start = perf_counter()
            out = super().inference(img, *args, **kwargs)
            self.timing["forward"] = 1000*(perf_counter()-start)
            return out

        def postprocess(self, preds, img, orig_imgs, **kwargs):
            start = perf_counter()
            main = super().postprocess(preds, img, orig_imgs, **kwargs)
            mid = perf_counter()
            wrapped = DetectionPredictor.postprocess(self, preds, img, orig_imgs, **kwargs)[0].boxes.data.clone()
            if self.lens is not None and len(wrapped):
                wrapped[:, :4] = torch.as_tensor(self.lens.boxes(wrapped[:, :4].numpy(),inverse=True),
                                                  dtype=wrapped.dtype)
            self.wrapped = wrapped.tolist()
            self.timing.update(native_nms=1000*(mid-start), wrapped_nms=1000*(perf_counter()-mid))
            if self.lens is None and not torch.equal(main[0].boxes.data, wrapped):
                raise ValueError("Unwarped arm differs from stock CPU boxes, scores, classes or ordering")
            return main

    return Measured


def letterbox(source_hw, bchw):
    h,w = source_hw
    oh,ow = bchw[2:]
    r = min(oh/h,ow/w)
    nh,nw = max(1,round(h*r)), max(1,round(w*r))
    return {"resized_hw":[nh,nw], "gain_xy":[nw/w,nh/h], "pad_xy":[round((ow-nw)/2-.1),round((oh-nh)/2-.1)]}


def get_inputs(repo, requests, fold, seed):
    src = f"BEE24-{68-fold}"
    clip = src+"-tail"
    p = requests/clip/"requests.json"
    req = read(p)
    frames = req.get("frames")
    if (set(req)!={"clip","frames","nframes"} or req["clip"]!=clip or req["nframes"]!=1000
            or not isinstance(frames,list) or any(type(f) is not int or not 1<=f<=1000 for f in frames)
            or frames!=sorted(set(frames)) or len(frames)>32):
        raise ValueError("Expected at most 32 requested local frames from an allowlisted 1000-frame tail")
    zone = repo/"beetrack/zones.json"
    weights = repo/f"runs/detect/loho_no{fold}_s{seed}/weights/best.pt"
    return req, repo/"data/BEE24/train"/src/"img1", weights, read(zone)[clip], {
        "requests_file_sha256":digest(p), "weights_sha256":digest(weights), "zones_sha256":digest(zone)}


def one_forward(model, cls, image, lens, size):
    received = []
    def observed(module, args):
        x = args[0]
        received.append((list(x.shape),str(x.dtype),x.device.type))
    hook = model.model.register_forward_pre_hook(observed)
    try:
        if model.predictor is not None:
            model.predictor.lens=lens
        start = perf_counter()
        result = model.predict(image,predictor=partial(cls,lens=lens),imgsz=size,rect=True,device="cpu",
                               conf=.05,iou=.7,max_det=300,half=False,batch=1,save=False,verbose=False)[0]
        elapsed = 1000*(perf_counter()-start)
    finally:
        hook.remove()
    if len(received)!=1 or received[0][1:]!=("torch.float32","cpu"):
        raise ValueError(f"Expected one actual CPU FP32 model forward, observed {received!r}")
    shape = received[0][0]
    if shape[:2]!=[1,3] or max(shape[2:])!=size:
        raise ValueError(f"Unexpected physical tensor size {shape} for imgsz {size}")
    pred = model.predictor
    return dict(predictions=result.boxes.data.tolist(),wrapped=pred.wrapped,forward_count=len(received),
                source_hw=list(image.shape[:2]),tensor_bchw=shape,letterbox=letterbox(image.shape[:2],shape),
                timing_ms=dict(pred.timing,total=elapsed))


def run_cell(args, fold, seed, components, runtime, cls):
    import cv2
    import numpy as np
    from ultralytics import YOLO
    from pilot_cpu import cpu_readback
    lens_class, inspector, code = components
    req, images, weights, poly, hashes = get_inputs(args.repo,args.requests,fold,seed)
    clip = req["clip"]
    hashes.update(code_sha256=code,prereg=args.prereg)
    paths = {}
    model, shape, lenses = None, None, {}
    output = args.output
    for f in req["frames"]:
        original = 4000+f
        jpg = images/f"{original:06d}.jpg"
        try:
            data = jpg.read_bytes()
        except FileNotFoundError as exc:
            raise ValueError(f"Requested image missing: {clip} {f} (native {original})") from exc
        image_sha = hashlib.sha256(data).hexdigest()
        im = None
        sizes = {}
        for dose, size in DOSES.items():
            receipt = output/"receipts"/clip/f"{fold}_{seed}_{dose}"/f"{f:04}.json"
            keys = dict(fold=fold, seed=seed, clip=clip, dose=dose, frame=f, original_frame=original,
                        image_sha256=image_sha, configuration_sha256=inspector.sha(hashes))
            if receipt.exists():
                row = read(receipt)
                if any(row.get(k)!=v for k,v in keys.items()) or row["forward_count"]!=1:
                    raise ValueError(f"Existing receipt has mismatched immutable inputs: {receipt}")
            else:
                if im is None:
                    start=perf_counter()
                    im=cv2.imdecode(np.frombuffer(data,dtype=np.uint8),cv2.IMREAD_COLOR)
                    decode=1000*(perf_counter()-start)
                    if im is None:
                        raise ValueError(f"Requested JPEG cannot be decoded: {clip} {f} (native {original})")
                    if shape is None:
                        shape=im.shape[:2]
                        lenses={d:lens_class.from_polygon((shape[1],shape[0]),poly,1.5,placebo=d=="C")
                                for d in ("L","C")}
                    if shape!=im.shape[:2]:
                        raise ValueError("Camera canvas changed inside the clip")
                if model is None:
                    model=YOLO(weights)
                first = model.predictor is None
                scores=one_forward(model,cls,im,lenses.get(dose),size)
                if first:
                    cpu_readback(args.threads)
                scores["timing_ms"]["decode_shared"]=decode
                scores["timing_ms"]["total_with_shared_decode"]=scores["timing_ms"]["total"]+decode
                row=dict(keys,**scores)
                inspector.write_once(receipt,row)
            sizes[dose]=row["tensor_bchw"]
            paths[dose,f]=receipt
        if not sizes["U"]==sizes["L"]==sizes["C"]:
            raise ValueError("U/L/C differ in physical tensor resolution")
    cpu_readback(args.threads)
    completed = {}
    for dose,size in DOSES.items():
        records={f:read(paths[dose,f]) for f in req["frames"]}
        frame_sha={str(f):digest(paths[dose,f]) for f in req["frames"]}
        run=dict(hashes,fold=fold,seed=seed,dose=dose,imgsz=size,requested_gain=1.5 if dose in ("C","L") else 1.,
                 cpu=runtime,frame_receipts_sha256=frame_sha,torch_dtype="float32",rect=True,
                 iou=.7,conf=.05,max_det=300,original_start=4001,original_end=5000)
        for folder,field in (("predictions","predictions"),("wrapped_predictions","wrapped")):
            result=inspector.freeze_predictions(req,dict(run,nms_space="original" if field=="predictions" else "warped"),
                                                 {f:v[field] for f,v in records.items()})
            path=output/folder/clip/f"{fold}_{seed}_{dose}.json"
            completed[path.relative_to(output).as_posix()]=inspector.write_once(path,result)
    print(f"Sealed {clip}, fold {fold} seed {seed}, {len(req['frames'])} frames × 4 arms",flush=True)
    return completed


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo",type=Path,default=Path.cwd())
    p.add_argument("--requests",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--geometry",type=Path,default=HERE)
    p.add_argument("--inspector",type=Path,default=HERE)
    p.add_argument("--prereg",required=True)
    p.add_argument("--threads",type=int,choices=(1,2),default=2)
    args=p.parse_args()
    if len(args.prereg)!=40 or any(c not in "0123456789abcdef" for c in args.prereg):
        p.error("--prereg requires the 40-character preregistration commit")
    from pilot_cpu import configure_cpu
    runtime=configure_cpu(args.threads)
    from filelock import FileLock
    components=load_components(args.geometry,args.inspector)
    cls=predictor_class()
    args.output.mkdir(parents=True,exist_ok=True)
    seals={}
    with FileLock(str(args.output/"detector.lock"),timeout=0):
        for fold in (33,35):
            for seed in range(3):
                seals.update(run_cell(args,fold,seed,components,runtime,cls))
        checksum=components[1].write_once(args.output/"detector_done.json",{
            "study":116,"prereg":args.prereg,"files":seals,"code_sha256":components[2]})
        print("Sealed all 24 primary and 24 paired same-forward stock-NMS manifests:",checksum)


if __name__=="__main__":
    main()
