"""Re-open the original source tracks and each exported physical §119 oracle witness."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from beetrack.evaluate import by_track

from tools.study117.physical117 import load_mot, verify_mot


def bytes_sha(path, expected):
    raw=path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=expected:
        raise ValueError(f"Physical artifact fingerprint differs: {path}")
    return raw


def per_frame_owners(gt, prediction, frames):
    """Rebuild full competition on every actual frame used by any newly added ID."""
    arrays=[]
    for pool in [gt,prediction]:
        by_frame=defaultdict(list)
        for r in by_track(pool):
            if int(r[0]) in frames:
                by_frame[int(r[0])].append(r)
        arrays.append(by_frame)
    output={".5":{},".3":{}}
    for f in frames:
        actual=np.array(arrays[0][f])
        pred=np.array(arrays[1][f])
        if not len(actual) or not len(pred):
            continue
        a, b = actual[:,2:6], pred[:,2:6]
        left=np.maximum(a[:,None,:2],b[None,:,:2])
        a_hi,b_hi=a[:,:2]+a[:,2:],b[:,:2]+b[:,2:]
        end=np.minimum(a_hi[:,None,:],b_hi[None,:,:])
        overlap=np.maximum(0,end-left).prod(axis=2)
        union=(a_hi-a[:,:2]).prod(axis=1)[:,None]+(b_hi-b[:,:2]).prod(axis=1)[None,:]-overlap
        iou=overlap/(union+1e-9)
        rr,cc=linear_sum_assignment(-iou)
        for name in output:
            for i,j in zip(rr,cc):
                if iou[i,j]>=float(name):
                    output[name][f,int(pred[j,1])]=int(actual[i,1])
    return output


def compare_case(raw_gt, p0, unions, case):
    gt=load_mot(raw_gt)
    lookup={(int(r[1]),int(r[0])):r for r in by_track(gt)}
    poly=np.asarray(case["polygon"],dtype=np.float32)
    n=case["nframes"]
    for item in case["event_inventory"]:
        origin=item["t"]
        expected=[]
        for f in range(max(1,origin-8),min(n,origin+5)+1):
            r=lookup.get((item["id"],f))
            if r is None:
                continue
            centre=r[2:4]+r[4:6]/2
            side=cv2.pointPolygonTest(poly,tuple(map(float,centre)),False)>=0
            expected.append([f if side else -f,*r[2:6]])
        if expected!=item["local_observations"]:
            raise ValueError("Serialized complete local observation catalogue differs from the physical GT")
    verified={}
    summary={}
    for name,raw in [("P0",p0),*unions.items()]:
        d=hashlib.sha256(raw).hexdigest()
        if d not in verified:
            verified[d]=verify_mot(raw,gt,poly,n)
        exp=case["baseline"] if name=="P0" else case["bounds"][name]["official_f1_of_this_witness"]
        z=verified[d]
        if name!="P0" and z["window"]["numerator"]!=case["bounds"][name]["min_numerator"]:
            raise ValueError("Physical witness clocks do not attain the declared optimum")
        if name=="P0" and z["window"]["numerator"]!=case["bounds"]["canonical_no_edge"]["base_numerator"]:
            raise ValueError("Physical P0 W differs")
        ids={int(k):int(v) for k,v in exp["0.5"]["added_targets"].items()}
        physical=load_mot(raw)
        new={i:physical[physical[:,1]==i] for i in ids}
        frames={int(r[0]) for segment in new.values() for r in segment}
        owner=per_frame_owners(gt,physical,frames)
        for i,segment in new.items():
            if len(segment)!=2:
                raise ValueError("A new explicit MOT ID has the wrong number of rows")
            for r in segment:
                if not np.array_equal(r[2:6], lookup[ids[i],int(r[0])][2:6]):
                    raise ValueError("A physical witness box differs from the original GT row")
        for tau in ("0.5","0.3"):
            m=z["thresholds"][tau]
            if any(m[k]!=exp[tau][k] for k in ("tp","fp","fn","f1","edges","matching")):
                raise ValueError("Independent physical graph or official maximum matching differs")
            for i,segment in new.items():
                local=[owner[tau[1:]].get((int(r[0]),i),-1) for r in segment[np.argsort(segment[:,0])]]
                if local!=exp[tau]["added_frame_owners"][str(i)]:
                    raise ValueError("All-frame competition on one of the true witness boxes disagrees")
        summary[name]=dict(mot_sha256=d, rows=len(physical), checked_ends=2*len(ids),
                           window=z["window"]["numerator"],tp05=z["thresholds"]["0.5"]["tp"])
    return summary


def verify(root,repo,prior,expected):
    index=json.loads(bytes_sha(root/"index.json",expected))
    output={}
    for fold in (33,35):
        for seed in range(3):
            key=f"{fold}_{seed}_tail"
            clip=f"BEE24-{68-fold}-tail"
            header=index["cases"][key]
            case=json.loads(bytes_sha(root/header["file"],header["sha256"]))
            orig=case["meta"]
            if (orig["clip"],orig["fold"],orig["seed"],orig["prereg"])!=(clip,fold,seed,index["prereg"]):
                raise ValueError("Illegal source clip or detector identity")
            gt=bytes_sha(repo/"runs/gt/val"/clip/"gt/gt.txt",orig["hashes"]["gt"])
            native=bytes_sha(prior/"source/tail_mot"/(key+".txt"),orig["hashes"]["mot"])
            real={}
            for name,opt in case["bounds"].items():
                receipt=opt["physical_mot"]
                real[name]=bytes_sha(root/"physical"/key/receipt["file"],receipt["sha256"])
            output[key]=compare_case(gt,native,real,case)
            print(key,"five full MOTs checked",flush=True)
    return dict(index_sha256=expected,cases=output)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("root",type=Path)
    p.add_argument("--index-sha",required=True)
    p.add_argument("--repo",type=Path,default=Path.cwd())
    p.add_argument("--prior",type=Path,required=True,help="Root of the original study 108 source outputs")
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    a.output.write_text(json.dumps(verify(a.root,a.repo,a.prior,a.index_sha),sort_keys=True)+"\n",encoding="utf8")
