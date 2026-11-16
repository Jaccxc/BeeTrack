"""Rebuild source-only labels and ranking from frozen candidates, sparse overlap evidence and all GT turns."""
import argparse
import gzip
import math
from pathlib import Path
import sys

from run114 import KEYS,decision,digest,load_freeze,read,require,wire,write


SOURCE_FREEZE="691a2b14d8ce2b5f31acf8248824a92cba87c895333f33ebbe8cdb7481809521"


def compact_report(report, gt):
    scores={k:{f:v for f,v in r.items() if f!="group_ranks"} for k,r in report["scores"].items()}
    return dict(scores=scores,seed_gates=report["seed_gates"],fold_gates=report["fold_gates"],
                opportunity={clip:{k:v for k,v in item["opportunity"].items() if k!="origins"}
                             for clip,item in gt.items()})


def pack(run,path,freeze_sha,models_sha,src108):
    from array114 import pack_geometry,pack_proof
    inv,load=load_freeze(run,freeze_sha)
    require(freeze_sha==SOURCE_FREEZE and digest(src108.read_bytes())==inv["source108_sha256"],
            "Wrong source freeze or historical archive")
    models=read(run/"models.json",models_sha)
    report=read(run/"results.json")
    require(report["models_sha256"]==models_sha and report["source_inventory_sha256"]==freeze_sha,
            "Score report references a different freeze")
    gt={clip:read(run/"gt"/(clip+".json.gz")) for clip in report["gt"]}
    require(all(digest(wire(gt[k]))==v for k,v in report["gt"].items()),"GT turn receipt mismatch")
    blocks={}
    for key in KEYS:
        case=load(key)
        labels=read(run/"labels"/(key+".json.gz"))
        require(digest(wire(labels))==report["labels"][key],"Label provenance mismatch "+key)
        blocks[key]=dict(source=case["source"],geometry=pack_geometry(case["geometry"]),
                         proof=pack_proof(labels["proof"]))
        del case, labels
    payload=dict(study=114,version=1,role="source",source_inventory=inv,source_inventory_sha256=freeze_sha,
                 models=models,models_sha256=models_sha,gt=gt,cases=blocks,
                 labels_sha256=report["labels"],results_sha256=digest(wire(report)),summary=compact_report(report,gt))
    print(path.name,write(path,payload),path.stat().st_size,flush=True)


def close_fit(actual,expected):
    if isinstance(expected,dict):
        return (isinstance(actual,dict) and expected.keys()==actual.keys()
                and all(close_fit(actual[k],v) for k,v in expected.items()))
    if isinstance(expected,list):
        return isinstance(actual,list) and len(actual)==len(expected) and all(close_fit(a,b) for a,b in zip(actual,expected))
    if isinstance(expected,float):
        return isinstance(actual,(int,float)) and math.isclose(actual,expected,abs_tol=1e-9,rel_tol=1e-9)
    return actual==expected


def replay(root,bundle,src108,independent=False):
    from array114 import unpack_geometry,unpack_proof
    from candidate114 import validate_108
    from gt114 import opportunity
    from score114 import label_from_proof,fit_ranker,score_ranks
    from sources114 import verify_metadata
    require(bundle["study"]==114 and bundle["version"]==1 and bundle["role"]=="source",
            "Unexpected source archive")
    inv=bundle["source_inventory"]
    freeze_sha=digest(wire(inv))
    require(freeze_sha==bundle["source_inventory_sha256"]==SOURCE_FREEZE
            and digest(src108.read_bytes())==inv["source108_sha256"] and set(bundle["cases"])==set(KEYS),
            "Source freeze or historical source archive differs")
    refs=read(src108)["masks"]["predictions"]
    descriptors={k:bundle["cases"][k]["source"] for k in KEYS}
    verify_metadata(dict(study=114,role="source",archives=inv["references"],cases=descriptors),root)
    models=bundle["models"]
    models_sha=digest(wire(models))
    require(models_sha==bundle["models_sha256"] and models["source_inventory_sha256"]==freeze_sha
            and set(models["fits"])=={k for k in KEYS if k.endswith("_head")},
            "Models were not pinned to the complete source freeze")
    gt=bundle["gt"]
    require(set(gt)=={r["clip"] for r in descriptors.values()},"Missing source GT diagnostic")
    gsha={k:digest(wire(x)) for k,x in gt.items()}
    for clip, item in gt.items():
        require(item["clip"]==clip and opportunity(item["turns"])==item["opportunity"],
                "GT-only competing-turn diagnostic differs")
        if clip.endswith("-head"):
            require(gsha[clip]==models["gt"][clip],"Head fitting GT differs")
    labels={}
    for key in KEYS:
        item=bundle["cases"][key]
        geom=unpack_geometry(item["geometry"])
        proof=unpack_proof(item["proof"])
        desc=descriptors[key]
        frozen=dict(study=114,role="source",prereg=inv["prereg"],source=desc,geometry=geom)
        require(digest(wire(frozen))==inv["content_sha256"][key],"Candidate reconstruction differs: "+key)
        validate_108(geom,refs[key])
        clip=gt[desc["clip"]]
        require(desc["hashes"]["gt"]==clip["gt_sha256"] and proof["polygon"]==clip["polygon"] and
                proof["turns"]==sorted([g,t,d] for g,u,t,d,a,b in clip["turns"]),
                "GT change-of-side identity or polygon differs")
        labeled=label_from_proof(geom,proof)
        label_record=dict(key=key,source_inventory_sha256=freeze_sha,gt_sha256=clip["gt_sha256"],
                          proof=proof,labeled=labeled)
        label_sha=digest(wire(label_record))
        require(label_sha==bundle["labels_sha256"][key],"Reconstructed labels differ: "+key)
        if key.endswith("_head"):
            require(label_sha==models["labels"][key] and close_fit(fit_ranker(labeled),models["fits"][key]),
                    "Head model fitting checkpoint differs: "+key)
        if independent:
            from verify114 import check_case
            beta=models["fits"][key.rsplit("_",1)[0]+"_head"]["coefficients"]
            check_case(geom,proof,labeled,beta,score_ranks(labeled,beta))
        labels[key]=labeled
        del geom,proof,label_record,frozen
    report=dict(study=114,role="source",prereg=inv["prereg"],source_inventory_sha256=freeze_sha,
                models_sha256=models_sha,labels=bundle["labels_sha256"],gt=gsha,**decision(labels,models))
    require(digest(wire(report))==bundle["results_sha256"] and compact_report(report,gt)==bundle["summary"],
            "Final controls, gates or report fingerprint differ")
    if independent:
        from verify114 import check_decisions
        check_decisions(report)
    return report


def metric(value):
    return "NA" if value is None else f"{value:.4f}"


def print_report(report,table):
    for fold,row in report["fold_gates"].items():
        print(f"fold {fold}: {row['n_pass']}/3 seeds pass, fold pass {row['passes']}")
    if table:
        print("fold_seed_partition | mixed anchors/bees | model distance time CV native chance | model-distance")
        for key in KEYS:
            row=report["scores"][key]
            scores=row["summary"]["all"]
            values=[scores.get(m,{}).get("equal_bee") for m in ("model","distance","time","cv","native")]
            diff=None if values[0] is None or values[1] is None else values[0]-values[1]
            values += [row["chance_uniform_selection"]["equal_bee"],diff]
            print(f"{key} | {row['mixed_groups']}/{row['mixed_origin_bees']} | "+" ".join(map(metric,values)))


def main():
    import os
    for env in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
        os.environ[env]="1"
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=Path.cwd())
    modes=parser.add_subparsers(dest="mode",required=True)
    make=modes.add_parser("pack")
    make.add_argument("run",type=Path)
    make.add_argument("archive",type=Path)
    make.add_argument("--freeze-sha",required=True)
    make.add_argument("--models-sha",required=True)
    verify=modes.add_parser("replay")
    verify.add_argument("archive",type=Path)
    verify.add_argument("--table",action="store_true")
    verify.add_argument("--independent",action="store_true",help="Also run the independent candidate-group verifier")
    for cmd in (make,verify):
        cmd.add_argument("--source108",type=Path,default=Path("docs/study108_source.json.gz"))
    args=parser.parse_args()
    root=args.root.resolve()
    sys.path.insert(0,str(root))
    src108=args.source108 if args.source108.is_absolute() else root/args.source108
    if args.mode=="pack":
        pack(args.run,args.archive,args.freeze_sha,args.models_sha,src108)
    else:
        print_report(replay(root,read(args.archive),src108,args.independent),args.table)


if __name__=="__main__":
    main()
