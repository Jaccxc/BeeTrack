"""Freeze source candidates, fit six source-head models, and evaluate the sealed source tails."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import gzip
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import re
import sys
from tempfile import TemporaryDirectory


KEYS = [f"{fold}_{seed}_{part}" for fold in (33,35) for seed in range(3) for part in ("head","tail")]


def require(valid, context):
    if not valid:
        raise ValueError(context)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def wire(record):
    return (json.dumps(record,sort_keys=True,separators=(",",":"),ensure_ascii=True,allow_nan=False)+"\n").encode()


def read(path, fingerprint=None):
    data = path.read_bytes()
    if fingerprint is not None:
        require(digest(data)==fingerprint,"File receipt changed: " + path.name)
    return json.loads(gzip.decompress(data) if path.suffix==".gz" else data)


def write(path, record):
    data = wire(record)
    if path.suffix==".gz":
        # Freeze and release receipts include gzip bytes. CPython's default OS
        # byte depends on version/platform; the original source freeze used 10.
        packed = gzip.compress(data,compresslevel=9,mtime=0)
        data = packed[:9] + bytes([10]) + packed[10:]
    path.parent.mkdir(parents=True,exist_ok=True)
    with TemporaryDirectory(dir=path.parent) as tmp:
        part=Path(tmp)/"complete"
        part.write_bytes(data)
        try:
            os.link(part,path)
        except FileExistsError:
            require(path.read_bytes()==data,"Conflicting previous receipt: "+str(path))
    return digest(data)


def source_file(root,old,desc,name):
    from sources114 import resolve_source
    path=resolve_source(root,old,desc["paths"][name])
    data=path.read_bytes()
    require(digest(data)==desc["hashes"][name], "Original input differs: "+desc["paths"][name])
    return data


def freeze_one(job):
    root, old, run, prereg, key, desc, historical = job
    sys.path.insert(0,str(root))
    import numpy as np
    from candidate114 import enumerate_candidates, validate_108

    det=np.load(BytesIO(source_file(root,old,desc,"det")),allow_pickle=False)
    mot=np.loadtxt(BytesIO(source_file(root,old,desc,"mot")),delimiter=",",usecols=range(6),ndmin=2)
    poly=json.loads(source_file(root,old,desc,"zones"))[desc["clip"]]
    geometry=enumerate_candidates(det,mot,poly,desc["nframes"])
    validate_108(geometry,historical)
    row=dict(study=114,role="source",prereg=prereg,source=desc,geometry=geometry)
    checksum=write(run/"source"/(key+".json.gz"),row)
    return key,checksum,digest(wire(row))


def inventory(root,old):
    from sources114 import preflight, verify_metadata
    record=preflight(root,old)
    verify_metadata(record,root)
    require(set(record["cases"])==set(KEYS),"The registered source set is incomplete")
    return record


def freeze(args,root,old):
    require(re.fullmatch("[0-9a-f]{40}",args.prereg),"Supply the complete preregistration revision")
    origins=inventory(root,old)
    historical=read(args.source108)["masks"]
    require(historical["role"]=="source","Expected the historical source prediction archive")
    jobs=[(root,old,args.run,args.prereg,k,origins["cases"][k],historical["predictions"][k]) for k in KEYS]
    with ProcessPoolExecutor(args.workers) as executor:
        cases,contents={},{}
        for key,checksum,content_sha in executor.map(freeze_one,jobs):
            cases[key]=checksum
            contents[key]=content_sha
            print(key,checksum,flush=True)
    index=dict(study=114,role="source",prereg=args.prereg,cases=cases,content_sha256=contents,
               source108_sha256=digest(args.source108.read_bytes()),references=origins["archives"])
    print("source_inventory.json",write(args.run/"source_inventory.json",index),flush=True)


def load_freeze(run,sha):
    index=read(run/"source_inventory.json",sha)
    require(index["study"]==114 and index["role"]=="source" and set(index["cases"])==set(KEYS),
            "A complete twelve-cell frozen universe is required")
    # Open one large source per lookup; the seal can be checked before any GT
    # via streamed fingerprints without retaining millions of candidate rows.
    for k in KEYS:
        with (run/"source"/(k+".json.gz")).open("rb") as file:
            require(hashlib.file_digest(file,"sha256").hexdigest()==index["cases"][k], "Case changed: "+k)
    def get(key):
        row=read(run/"source"/(key+".json.gz"),index["cases"][key])
        fold,seed,part=key.split("_")
        d=row["source"]
        require(digest(wire(row))==index["content_sha256"][key] and row["prereg"]==index["prereg"]
                and (d["fold"],d["seed"],d["part"])==(int(fold),int(seed),part)
                and d["hive"]==68-int(fold) and d["nframes"]==(3950 if part=="head" else 1000),
                "Source cell differs from its registration: "+key)
        return row
    return index,get


def clip_gt(root,old,desc):
    import numpy as np
    gt=np.loadtxt(BytesIO(source_file(root,old,desc,"gt")),delimiter=",",usecols=range(6),ndmin=2)
    polygon=json.loads(source_file(root,old,desc,"zones"))[desc["clip"]]
    return gt,polygon


def label_partition(root,old,run,cases,part,freeze_sha):
    from gt114 import turns,opportunity
    from score114 import geometry_labels,label_from_proof

    gt_receipts, labels, label_receipts = {}, {}, {}
    for fold in (33,35):
        reference=cases(f"{fold}_0_{part}")["source"]
        gt,polygon=clip_gt(root,old,reference)
        clip=reference["clip"]
        ledger=turns(gt,polygon)
        gt_record=dict(clip=clip,gt_sha256=reference["hashes"]["gt"],polygon=polygon,
                       turns=ledger,opportunity=opportunity(ledger))
        write(run/"gt"/(clip+".json.gz"),gt_record)
        gt_receipts[clip]=digest(wire(gt_record))
        for seed in range(3):
            key=f"{fold}_{seed}_{part}"
            frozen=cases(key)
            require(frozen["source"]["hashes"]["gt"]==reference["hashes"]["gt"],"GT differs by seed")
            proof=geometry_labels(frozen["geometry"],gt,polygon)
            labeled=label_from_proof(frozen["geometry"],proof)
            record=dict(key=key,source_inventory_sha256=freeze_sha,gt_sha256=reference["hashes"]["gt"],
                        proof=proof,labeled=labeled)
            write(run/"labels"/(key+".json.gz"),record)
            label_receipts[key]=digest(wire(record))
            labels[key]=labeled
    return gt_receipts,labels,label_receipts


def fit(args,root,old):
    from score114 import fit_ranker
    index,cases=load_freeze(args.run,args.freeze_sha)
    gt,heads,labels=label_partition(root,old,args.run,cases,"head",args.freeze_sha)
    fits={key:fit_ranker(labeled) for key,labeled in heads.items()}
    record=dict(study=114,role="source",prereg=index["prereg"],source_inventory_sha256=args.freeze_sha,
                gt=gt,labels=labels,fits=fits)
    print("models.json",write(args.run/"models.json",record))


def decision(labels,models):
    from score114 import score_ranks,seed_gate,fold_gate
    scores,gates,folds={},{},{}
    for f in (33,35):
        seeds={}
        for s in range(3):
            head,tail=f"{f}_{s}_head",f"{f}_{s}_tail"
            coefs=models["fits"][head]["coefficients"]
            for key in (head,tail):
                scores[key]=score_ranks(labels[key],coefs)
            seeds[str(s)]=seed_gate(scores[head],scores[tail])
        gates[str(f)]=seeds
        folds[str(f)]=fold_gate(seeds)
    return dict(scores=scores,seed_gates=gates,fold_gates=folds)


def score(args,root,old):
    index,cases=load_freeze(args.run,args.freeze_sha)
    models=read(args.run/"models.json",args.models_sha)
    head_keys={k for k in KEYS if k.endswith("_head")}
    require(models["prereg"]==index["prereg"] and models["source_inventory_sha256"]==args.freeze_sha
            and set(models["fits"])==set(models["labels"])==head_keys,"Six frozen head fits are required")
    labels={}
    for key,expected in models["labels"].items():
        row=read(args.run/"labels"/(key+".json.gz"))
        require(digest(wire(row))==expected,"A source-head label changed after model fit")
        labels[key]=row["labeled"]
    # Tail GT becomes accessible only after the six head records have been verified.
    gt,tail,receipts=label_partition(root,old,args.run,cases,"tail",args.freeze_sha)
    labels.update(tail)
    report=dict(study=114,role="source",prereg=index["prereg"],source_inventory_sha256=args.freeze_sha,
                models_sha256=args.models_sha,labels={**models["labels"],**receipts},gt={**models["gt"],**gt},
                **decision(labels,models))
    print("results.json",write(args.run/"results.json",report))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",default=Path.cwd(),type=Path,help="BeeTrack repository root")
    parser.add_argument("--run",type=Path,required=True)
    parser.add_argument("--study108-root",type=Path)
    parser.add_argument("--source108",type=Path,default=Path("docs/study108_source.json.gz"))
    actions=parser.add_subparsers(dest="mode",required=True)
    for mode in ("preflight","freeze","fit","score"):
        sub=actions.add_parser(mode)
        if mode=="freeze":
            sub.add_argument("--prereg",required=True)
            sub.add_argument("--workers",type=int,choices=(1,2,3),default=2)
        if mode in ("fit","score"):
            sub.add_argument("--freeze-sha",required=True)
        if mode=="score":
            sub.add_argument("--models-sha",required=True)
    args=parser.parse_args()
    root=args.root.resolve()
    sys.path.insert(0,str(root))
    args.run=args.run.resolve()
    old=(args.study108_root or args.run.parent/"study108").resolve()
    if not args.source108.is_absolute():
        args.source108=root/args.source108
    for env in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
        os.environ[env]="1"
    if args.mode=="preflight":
        print("source_inputs.json",write(args.run/"source_inputs.json",inventory(root,old)))
    elif args.mode=="freeze":
        freeze(args,root,old)
    elif args.mode=="fit":
        fit(args,root,old)
    else:
        score(args,root,old)


if __name__=="__main__":
    main()
