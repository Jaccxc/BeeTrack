"""One native CPU YOLO11 forward per source-tail frame, with six same-forward P3 arms."""
import argparse
from collections import deque
import hashlib
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from time import perf_counter


HERE = Path(__file__).resolve().parent
ARMS = ("U","A","Z","Q","Zm","D")


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f,"sha256").hexdigest()


def write_once(path, data):
    path.parent.mkdir(parents=True,exist_ok=True)
    with TemporaryDirectory(dir=path.parent) as td:
        tmp=Path(td)/"data"
        tmp.write_bytes(data)
        try:
            os.link(tmp,path)
        except FileExistsError:
            if sha(tmp)!=sha(path):
                raise ValueError(f"Immutable output differs: {path}")


def write_json(path, value):
    data=(json.dumps(value,ensure_ascii=False,allow_nan=False,sort_keys=True,separators=(",",":"))+"\n").encode()
    write_once(path,data)


def raster_mask(source_hw,input_hw,p3_hw):
    import torch
    h,w=source_hw
    oh,ow=input_hw
    ph,pw=p3_hw
    if (8*ph,8*pw)!=(oh,ow):
        raise ValueError("Expected stride-eight P3 on stock letterbox")
    gain=min(oh/h,ow/w)
    nh,nw=max(1,round(gain*h)),max(1,round(gain*w))
    left,top=round((ow-nw)/2-.1),round((oh-nh)/2-.1)
    y,x=torch.arange(ph)*8+4,torch.arange(pw)*8+4
    valid=((y>=top)&(y<top+nh))[:,None]&((x>=left)&(x<left+nw))[None,:]
    return valid,dict(pad_xy=[left,top],resized_hw=[nh,nw],valid_cells=int(valid.sum()))


class Detector:
    def __init__(self, weights):
        import psutil
        from ultralytics.models.yolo.detect.predict import DetectionPredictor
        from ultralytics.utils.checks import check_imgsz
        self.stock=DetectionPredictor(overrides=dict(imgsz=640,rect=True,device="cpu",conf=.05,iou=.7,
                                                     batch=1,max_det=300,verbose=False,save=False))
        self.stock.setup_model(weights,verbose=False)
        self.stock.imgsz=check_imgsz(640,stride=self.stock.model.stride,min_dim=2)
        self.stock.model.warmup()
        self.root=self.stock.model.backend.model
        self.head=self.root.model[-1]
        if self.head.nc!=1 or self.head.end2end or self.stock.model.fp16:
            raise ValueError("This study requires the original FP32 single-class non-end-to-end head")
        self.calls=0
        self.process=psutil.Process()
        def count(module,args):
            self.calls+=1
        self.hook=self.root.register_forward_pre_hook(count)

    def frame(self,image,history,frame):
        import torch
        from decode117 import summarize
        from feature_arms import compute_arms

        start=perf_counter()
        tensor=self.stock.preprocess([image])
        shape=list(tensor.shape)
        if tensor.dtype!=torch.float32 or tensor.device.type!="cpu" or shape[:2]!=[1,3] or max(shape[2:])!=640:
            raise ValueError(f"Unexpected detector tensor {shape}")
        pre=perf_counter()
        before=self.calls
        dense,raw=self.stock.model(tensor)
        if self.calls!=before+1:
            raise ValueError("Expected exactly one native model forward")
        forward=perf_counter()
        p3=raw["feats"][0].contiguous(memory_format=torch.channels_last)
        ph,pw=p3.shape[-2:]
        logits=raw["scores"][:,:,:ph*pw].reshape(1,1,ph,pw)
        valid,raster=raster_mask(image.shape[:2],shape[2:],(ph,pw))
        prev=previous=old_boxes=None
        donor_frame=None
        if len(history)==2:
            donor_frame,prev,previous,old_boxes=history[0]
            if donor_frame!=frame-2 or prev.shape!=p3.shape:
                raise ValueError("Native t−2 history differs in clock or camera geometry")
            prev=prev.contiguous(memory_format=torch.channels_last)
        selected,info=compute_arms(p3,prev,logits,previous,self.head.cv3[0],valid)
        feature_end=perf_counter()
        record,boxes=summarize(dense,selected,info,old_boxes,image.shape[:2],shape[2:],previous)
        end=perf_counter()
        memory=self.process.memory_info()
        nms_timing=record.pop("nms_timing_ms",{})
        hist_native=sum(x.numel()*x.element_size() for _,f,l,b in history for x in (f,l))
        hist_boxes=sum(b.numel()*b.element_size() for _,f,l,b in history)
        record.update(frame=frame,original_frame=4000+frame,donor_frame=donor_frame,source_hw=list(image.shape[:2]),
                      input_bchw=shape,p3_hw=[ph,pw],p3_channels=p3.shape[1],raster=raster,backbone_forwards=1,
                      process_rss_bytes=memory.rss,process_peak_rss_bytes=getattr(memory,"peak_wset",None),
                      history_native_bytes=hist_native,history_diagnostic_boxes_bytes=hist_boxes,
                      timing_ms=dict(preprocess=1000*(pre-start),forward=1000*(forward-pre),
                                     features_six=1000*(feature_end-forward),nms_pack=1000*(end-feature_end),
                                     compute_total=1000*(end-start),a_path_ms=info["a_path_ms"],
                                     feature_detail=info["timing_ms"],nms_detail=nms_timing))
        return record,(frame,p3,logits,boxes)


def setup(args):
    sys.path[:0]=[str(args.helpers.resolve()),str(args.prototype.resolve())]
    from pilot_cpu import configure_cpu
    import psutil
    if args.cpus:
        p=psutil.Process()
        if len(args.cpus)!=args.threads or not set(args.cpus)<=set(p.cpu_affinity()):
            raise ValueError("Requested physical CPU affinity is not available")
        p.cpu_affinity(args.cpus)
    runtime=configure_cpu(args.threads)
    import ultralytics
    import decode117,feature_arms,pilot_cpu,store117,temporal_features
    from ultralytics.engine import predictor
    from ultralytics.utils import nms,ops
    from ultralytics.nn.modules import head
    if ultralytics.__version__!="8.4.208":
        raise ValueError("Expected locked Ultralytics 8.4.208")
    code={Path(m.__file__).name:sha(m.__file__) for m in
          (decode117,feature_arms,pilot_cpu,store117,temporal_features,predictor,nms,ops,head)}
    code[Path(__file__).name]=sha(__file__)
    return runtime,code


def export_cell(root,h,s,config,n=1000):
    import numpy as np
    from store117 import iter_records
    key=f"{h}_{s}"; db=root/"cases"/(key+".sqlite")
    arrays={a:[] for a in ARMS}
    total=0
    images=hashlib.sha256()
    for row in iter_records(db):
        total+=1
        if row["frame"]!=total or row["backbone_forwards"]!=1 or not row["stock_u_exact"]:
            raise ValueError("Incomplete original stream")
        images.update(bytes.fromhex(row["image_sha256"]))
        for a in ARMS:
            arrays[a].extend([[total,*row["boxes"][str(i)],score] for i,score in row["arms"][a]])
    if total!=n:
        raise ValueError(f"Got {total} frames; expected {n}")
    files={db.relative_to(root).as_posix():sha(db)}
    for a,v in arrays.items():
        import io
        buf=io.BytesIO()
        np.save(buf,np.asarray(v,dtype=np.float64).reshape(-1,6),allow_pickle=False)
        path=root/"detections"/key/(a+".npy")
        write_once(path,buf.getvalue())
        files[path.relative_to(root).as_posix()]=sha(path)
    meta=root/"cases"/(key+".json")
    write_json(meta,dict(config=config,frames=n,backbone_forwards=n,images_sha256=images.hexdigest(),files=files))


def run_cell(args,hive,seed,runtime,code,synthetic=None,stop_after=None):
    import cv2
    import numpy as np
    import torch
    from filelock import FileLock
    from pilot_cpu import cpu_readback
    from store117 import Store

    source=f"BEE24-{68-hive}"
    key=f"{hive}_{seed}"
    root=args.output
    weights=args.repo/f"runs/detect/loho_no{hive}_s{seed}/weights/best.pt"
    n=1000 if synthetic is None else len(synthetic[1])
    wsha=sha(weights) if synthetic is None else "synthetic_single_class"
    cfg=dict(study=117,prereg=args.prereg,fold=hive,seed=seed,clip=source+"-tail",nframes=n,
             weights_sha256=wsha,code_sha256=code,cpu=runtime,source_start=4001,arms=list(ARMS),
             ultralytics="8.4.208",conf=.05,iou=.7,max_det=300)
    db=root/"cases"/(key+".sqlite")
    db.parent.mkdir(parents=True,exist_ok=True)
    with FileLock(str(db)+".worker.lock",timeout=0),torch.inference_mode():
        # Fast resume on an already completed case without mutating its byte-level fingerprint.
        if (db.with_suffix(".json")).is_file():
            meta=json.loads(db.with_suffix(".json").read_bytes())
            if meta["config"]!=cfg or not all(sha(root/path)==digest for path,digest in meta["files"].items()):
                raise ValueError(f"Completed case mismatch {key}")
            return
        with Store(db,cfg) as store:
            previous=deque(store.history(),maxlen=2)
            detector=None
            for f in range(store.last+1,(min(n,stop_after) if stop_after else n)+1):
                start=perf_counter()
                if synthetic is None:
                    jpg=args.repo/"data/BEE24/train"/source/"img1"/f"{f+4000:06d}.jpg"
                    data=jpg.read_bytes()
                else:
                    data=synthetic[1][f-1]
                digest=hashlib.sha256(data).hexdigest()
                im=cv2.imdecode(np.frombuffer(data,dtype=np.uint8),cv2.IMREAD_COLOR)
                if im is None:
                    raise ValueError(f"Unreadable registered frame {source} {4000+f}")
                decode=1000*(perf_counter()-start)
                if detector is None:
                    detector=Detector(weights if synthetic is None else synthetic[0])
                    cpu_readback(args.threads)
                store.begin(f,f)
                record,native=detector.frame(im,previous,f)
                record["image_sha256"]=digest
                record["timing_ms"]["read_hash_decode"]=decode
                previous.append(native)
                store.append(f,record)
                store.commit(previous)
                if f%100==0:
                    print(f"Committed {key} frame {f}/{n}",flush=True)
            if store.last<n:
                return
            store.finalize(n)
        export_cell(root,hive,seed,cfg,n)
        print(f"Sealed source {source}-tail fold {hive}, seed {seed}: {n} native forwards",flush=True)


def seal(args):
    files={}
    for h in (33,35):
        for s in range(3):
            meta=args.output/"cases"/f"{h}_{s}.json"
            row=json.loads(meta.read_bytes())
            cfg=row["config"]
            if ((cfg["prereg"],cfg["fold"],cfg["seed"],row["frames"],row["backbone_forwards"])
                    !=(args.prereg,h,s,1000,1000)):
                raise ValueError(f"Incomplete source {h}_{s}")
            for path,digest in row["files"].items():
                if sha(args.output/path)!=digest:
                    raise ValueError(f"Changed file {path}")
                files[path]=digest
            files[meta.relative_to(args.output).as_posix()]=sha(meta)
    if len(files)!=48:
        raise ValueError("Expected six cases, six metadata records and 36 detection caches")
    path=args.output/"detector_done.json"
    write_json(path,dict(study=117,prereg=args.prereg,files=files,backbone_forwards=6000))
    print("detector_done.json",sha(path),flush=True)


def self_check(args,runtime,code):
    import cv2
    import numpy as np
    import torch
    from ultralytics.nn.tasks import DetectionModel
    from store117 import iter_records
    torch.manual_seed(117)
    model=DetectionModel("yolo11s.yaml",nc=1,verbose=False).eval()
    g=np.random.default_rng(117)
    ims=[cv2.imencode(".jpg",g.integers(0,256,size=(53,101,3),dtype=np.uint8))[1].tobytes() for _ in range(4)]
    with TemporaryDirectory() as td:
        out=Path(td)
        result=[]
        for label,step in (("full",False),("resume",True)):
            args.output=out/label
            if step:
                run_cell(args,33,0,runtime,code,(model,ims),stop_after=2)
            run_cell(args,33,0,runtime,code,(model,ims))
            rows=list(iter_records(args.output/"cases/33_0.sqlite"))
            for r in rows:
                del r["timing_ms"]
                del r["process_rss_bytes"],r["process_peak_rss_bytes"]
            assert [r["donor_frame"] for r in rows]==[None,None,1,2]
            assert all(r["stock_u_exact"] and r["backbone_forwards"]==1 for r in rows)
            result.append(rows)
        assert result[0]==result[1]
    print("Random native YOLO, six arms, stock equivalence, four frames and restart at t=3 passed.")


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode",choices=("run","seal","self-check"))
    ap.add_argument("--repo",type=Path,default=Path.cwd())
    ap.add_argument("--output",type=Path,default=Path("runs/study117"))
    ap.add_argument("--helpers",type=Path,default=Path.cwd()/"tools/study116")
    ap.add_argument("--prototype",type=Path,default=HERE)
    ap.add_argument("--prereg")
    ap.add_argument("--fold",type=int,choices=(33,35))
    ap.add_argument("--threads",type=int,choices=(1,2),default=2)
    ap.add_argument("--cpus",type=int,nargs="+")
    args=ap.parse_args()
    if args.mode=="self-check":
        args.prereg="0"*40
    elif not args.prereg or len(args.prereg)!=40 or any(c not in "0123456789abcdef" for c in args.prereg):
        ap.error("Expected the full preregistration commit")
    if args.mode=="seal":
        seal(args)
        return
    runtime,code=setup(args)
    if args.mode=="self-check":
        self_check(args,runtime,code)
        return
    if args.fold is None:
        ap.error("Specify one registered source with --fold")
    for s in range(3):
        run_cell(args,args.fold,s,runtime,code)


if __name__=="__main__":
    main()
