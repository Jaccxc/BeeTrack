"""Score only the six pinned §108 source tails, after the study has been registered."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import re
from tempfile import TemporaryDirectory

import numpy as np

from beetrack.evaluate import load_tracks
from tools.study108.inputs108 import digest, save
from tools.study114.sources114 import ARCHIVES
from capacity import analyze


def physical_sink(original_path, destination):
    """Preserve all native row bytes and add two ten-column observations per selected bee."""
    native=original_path.read_bytes().splitlines(keepends=True)
    destination.mkdir(parents=True,exist_ok=True)
    def save_union(arm, expected, n_new):
        lines=native.copy()
        if n_new:
            if lines and not lines[-1].endswith((b"\r",b"\n")):
                lines[-1]+=b"\n"
            for f,tid,x,y,w,h in expected[-n_new:]:
                lines.append(f"{int(f)},{int(tid)},{x!s},{y!s},{w!s},{h!s},1,-1,-1,-1\n".encode())
            lines.sort(key=lambda r:int(float(r.split(b",",1)[0])))
        path=destination/(arm+".txt")
        with TemporaryDirectory(dir=destination) as temp:
            stage=Path(temp)/"union.txt"
            stage.write_bytes(b"".join(lines))
            actual=load_tracks(stage)
            def sorted_rows(x):
                return x[np.lexsort((x[:,1],x[:,0]))]
            if not np.array_equal(sorted_rows(actual),sorted_rows(expected[:,:6])):
                raise ValueError("Physical MOT did not preserve every selected and native row")
            stage.replace(path)
        return actual,dict(file=path.name,sha256=digest(path),native_rows=len(native),added_rows=n_new)
    return save_union


def study(repo, prior, output, prereg):
    if not re.fullmatch("[0-9a-f]{40}",prereg):
        raise ValueError("Record the published study preregistration revision")
    rel="docs/study108_source.json.gz"
    archive=(repo/rel).read_bytes()
    if hashlib.sha256(archive).hexdigest()!=ARCHIVES[rel]:
        raise ValueError("Source-only §108 archive is not the published snapshot")
    ref=json.loads(gzip.decompress(archive))
    cells=[]
    for fold in (33,35):
        clip=f"BEE24-{68-fold}-tail"
        for seed in range(3):
            key=f"{fold}_{seed}_tail"
            desc=ref["masks"]["predictions"][key]
            label=ref["labels"][key]
            det=f"dets/loho_no{fold}_s{seed}/val/{clip}.npy"
            paths={"det":repo/"runs"/det,"mot":prior/"source/tail_mot"/(key+".txt"),
                   "gt":repo/"runs/gt/val"/clip/"gt/gt.txt","zone":repo/"beetrack/zones.json"}
            hashes={kind:digest(path) for kind,path in paths.items()}
            if hashes!={"det":desc["sources"][det],"gt":label["gt_mot_sha256"],"mot":desc["mot_sha256"],
                        "zone":desc["sources"]["beetrack/zones.json"]} or desc["clip"]!=clip:
                raise ValueError(f"Source-only evidence no longer matches published fingerprints: {key}")
            if label["predictions_sha256"]!=ref["masks"]["prediction_sha256"][key]:
                raise ValueError("Signed event freeze and archived GT labels disagree")
            cells.append((key,fold,seed,clip,paths,hashes,dict(label,predictions=desc["predictions"])))
    summary={}
    for key,fold,seed,clip,paths,hashes,old in cells:
        result=analyze(load_tracks(paths["gt"]),load_tracks(paths["mot"]),
                       json.loads(paths["zone"].read_bytes())[clip],1000,old,
                       physical_sink(paths["mot"],output/"physical"/key))
        result.update(meta=dict(fold=fold,seed=seed,clip=clip,hashes=hashes,prereg=prereg))
        path=output/("case_"+key+".json")
        save(path,result)
        main=result["bounds"]["canonical_no_edge"]
        strict=main["official_f1_of_this_witness"]["0.5"]
        summary[key]=dict(file=path.name,sha256=digest(path),clip=clip,w_gain_pp=main["max_gain_pp"],
                          base_numerator=main["base_numerator"],opt_numerator=main["min_numerator"],
                          denominator=main["denominator"],eligible=main["eligible_bees"],
                          strict_ids=strict["strict_same_bees"],f1=strict["f1"],base_f1=result["baseline"]["0.5"]["f1"])
        print(key,summary[key]["w_gain_pp"],len(summary[key]["strict_ids"]),flush=True)
    from fractions import Fraction
    eps=Fraction(1,10**9)
    gates={}
    for fold in (33,35):
        cases=[summary[f"{fold}_{s}_tail"] for s in range(3)]
        gains=[Fraction(100*(c["base_numerator"]-c["opt_numerator"]),c["denominator"]) for c in cases]
        mean_w=sum(gains)/3
        enough=mean_w>=2 and sum(g>eps for g in gains)>=2
        paired=sum(bool(c["strict_ids"]) for c in cases)>=2
        delta=sum(c["f1"]-c["base_f1"] for c in cases)/3
        gates[str(fold)]=dict(clip=cases[0]["clip"],mean_w_gain_pp=float(mean_w),positive_seeds=sum(g>eps for g in gains),
                             capacity=enough,strict_seeds=sum(bool(c["strict_ids"]) for c in cases),
                             delta_f1=delta,physical=paired and delta>=-float(eps),advance=enough and paired and delta>=-float(eps))
    save(output/"index.json",dict(prereg=prereg,history_sha256=ARCHIVES[rel],cases=summary,gates=gates))
    print(json.dumps(gates,sort_keys=True))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo",type=Path,default=Path.cwd())
    p.add_argument("--prior",type=Path,required=True,help="Root of the original study 108 source outputs")
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--prereg",required=True)
    a=p.parse_args()
    study(a.repo,a.prior,a.output,a.prereg)
