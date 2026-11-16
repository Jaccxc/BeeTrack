"""Replay frozen post-NMS missing-box interventions through the native P0 tracker."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import time
from tempfile import TemporaryDirectory

import numpy as np

from beetrack.common import RUNS, write_mot
from beetrack.link import model
from beetrack.track import Tracker, nms
from tools.study104.track_reverse import STRICT_SHA256, finish, first_pass


def prepare_survivors(dets, nframes):
    """Raw-cache row IDs surviving P0's native second NMS, in exact tracking order."""
    dets = np.asarray(dets)
    if dets.ndim != 2 or dets.shape[1] != 6 or not np.isfinite(dets).all():
        raise ValueError("Expected finite [frame,x1,y1,x2,y2,score] raw-cache rows")
    frames = dets[:,0].astype(np.int64)
    if not np.array_equal(frames,dets[:,0]) or np.any(frames<1) or np.any(frames>nframes):
        raise ValueError("Invalid frame index")
    ordered = np.argsort(frames, kind="stable")
    splits = np.searchsorted(frames[ordered], np.arange(1,nframes+2))
    kept = []
    for start,stop in zip(splits,splits[1:]):
        if stop>start:
            raw_ids = ordered[start:stop]
            # Tracker.update casts the five detection fields to float32 before its NMS.
            kept.append(raw_ids[nms(np.asarray(dets[raw_ids,1:6],dtype=np.float32), .5)])
    return np.concatenate(kept) if kept else np.empty(0,dtype=np.int64)


def first_pass_survivors(dets, nframes, survivors, deleted=(), trace=False):
    """Track selected observations; optional lineage uses ByteTrack's native observation indices."""
    survivors = np.asarray(survivors, dtype=np.int64)
    deleted = np.asarray(deleted, dtype=np.int64)
    if (len(np.unique(deleted))!=len(deleted) or not np.all(np.isin(deleted,survivors))
            or np.any(survivors<0) or np.any(survivors>=len(dets))):
        raise ValueError("Every deletion must name a distinct post-NMS survivor")
    retained = survivors[~np.isin(survivors,deleted)] if len(deleted) else survivors
    frames = dets[retained,0]
    if np.any(np.diff(frames)<0):
        raise ValueError("Expected chronological post-NMS order")
    tracker = Tracker("link", link=0, min_len=0, nms=0)
    source_ids, output = [], []
    if trace:
        update = tracker.t.update
        def observed(batch):
            result = update(batch)
            if len(result):
                indices = result[:,7].astype(np.int64)
                if not np.array_equal(indices,result[:,7]) or np.any(indices<0) or np.any(indices>=len(raw_ids)):
                    raise ValueError("Native tracker returned an invalid detection index")
                source_ids.append(raw_ids[indices])
            return result
        tracker.t.update = observed
    cuts = np.searchsorted(frames,np.arange(1,nframes+2))
    for frame,(start,stop) in enumerate(zip(cuts,cuts[1:]),start=1):
        raw_ids = retained[start:stop]
        out = tracker.update(dets[raw_ids,1:6])
        if len(out):
            output.append(np.column_stack((np.full(len(out),frame),out[:,4],out[:,0],out[:,1],
                                           out[:,2]-out[:,0],out[:,3]-out[:,1],out[:,5])))
    if getattr(tracker.t,"late",[]):
        raise RuntimeError("Late activation is disabled in the registered P0 first pass")
    rows = np.concatenate(output) if output else np.empty((0,7),dtype=np.float64)
    ids = None
    if trace:
        ids = np.concatenate(source_ids) if source_ids else np.empty(0,dtype=np.int64)
        if len(rows)!=len(ids) or not np.array_equal(rows[:,0],dets[ids,0]):
            raise ValueError("Native observation lineage was lost")
    return rows, ids


def self_check():
    # The higher-score box at x=8 suppresses x=9. Its post-NMS deletion cannot resurrect x=9.
    det = np.array([[f,x,4,x+8,12,.9 if x==8 else .65] for f in (3,1,2,4,6,7,8)
                    for x in (9,8,38)],dtype=np.float64)
    keep = prepare_survivors(det,9)
    rows, ids = first_pass_survivors(det,9,keep,trace=True)
    np.testing.assert_array_equal(rows,first_pass(det,9))
    np.testing.assert_array_equal(rows[:,[0,2,3]],det[ids][:,[0,1,2]])
    ix = np.flatnonzero((det[:,0]==4)&(det[:,1]==8))
    changed, traced = first_pass_survivors(det,9,keep,ix,trace=True)
    assert not np.any((changed[:,0]==4)&(changed[:,2]!=38)) and int(ix[0]) not in traced
    before_nms=np.delete(det,ix,axis=0)
    newly_kept=before_nms[prepare_survivors(before_nms,9)]
    assert np.any((newly_kept[:,0]==4)&(newly_kept[:,1]==9))
    empty, ixempty = first_pass_survivors(det[:0],9,[],trace=True)
    assert empty.shape==(0,7) and ixempty.shape==(0,)
    row={"near_indices":list(range(8)), "far_indices":list(range(10,18)), "opposite_idx":19}
    masks={f"{family}_{phase}":[row[side+"_indices"][j] for j in offsets]
           for family,side in (("nearall","near"),("near","near"),("far","far"))
           for phase,offsets in PHASES.items()}
    selection={"rows_all":[row],"rows":[row],"masks":masks}
    assert arm_masks(selection,True)=={arm:tuple(ids) for arm,ids in masks.items()}
    assert len(arm_masks(selection,False))==4
    empty_selection={"rows_all":[],"rows":[],"masks":{arm:[] for arm in masks}}
    assert set(arm_masks(empty_selection,True).values())=={()}
    bad=dict(row,opposite_idx=3)
    try:
        arm_masks(dict(selection,rows_all=[bad]),False)
    except ValueError:
        pass
    else:
        raise AssertionError("A guard entered the deletion mask")
    print("Synthetic native MOT, lineage, NMS timing, pair gating, exposure and guard checks passed",flush=True)


def bytes_hash(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def array_hash(array,dtype):
    return hashlib.sha256(np.asarray(array,dtype=dtype,order="C").tobytes()).hexdigest()


def publish_temp(temp,path):
    """Expose a completed file, or accept identical bytes from a prior run."""
    path.parent.mkdir(parents=True,exist_ok=True)
    try:
        os.link(temp,path)
    except FileExistsError:
        if bytes_hash(temp)!=bytes_hash(path):
            raise ValueError(f"Conflicting artifact at {path}")


def write_record(path, record):
    path.parent.mkdir(parents=True,exist_ok=True)
    with TemporaryDirectory(dir=path.parent) as tmp:
        temp=Path(tmp)/path.name
        temp.write_text(json.dumps(record,sort_keys=True,allow_nan=False,separators=(",",":"))+"\n",
                        encoding="utf8",newline="\n")
        publish_temp(temp,path)


def paths(root,fold,seed):
    if fold not in (33,35) or seed not in range(9):
        raise ValueError("Expected one of the registered leave-one-hive detector folds")
    clip=f"BEE24-{fold}-head"
    tag=f"loho_no{fold}_s{seed}"
    return (clip,RUNS/"dets"/tag/"dev"/(clip+".npy"),
            RUNS/"trackers"/"dev"/("q_lk_"+tag)/"data"/(clip+".txt"),
            root/"prepared"/f"{fold}_{seed}.npz",root/"tracks"/f"{fold}_{seed}")


def verify95(path,fold,seed):
    # Import official scoring only for the clean reference, never for selecting a mask.
    from beetrack import evaluate as ref
    from tools.study103.swap_score import _reference_rows, vector_of_counts, window_error
    clip=f"BEE24-{fold}-head"
    polygon=json.loads(ref.ZONES.read_text(encoding="utf8"))[clip]
    gt=ref.load_tracks(RUNS/"gt"/"dev"/clip/"gt"/"gt.txt")
    output=ref.load_tracks(path)
    truth=vector_of_counts(gt,polygon,3950)
    estimate=vector_of_counts(output,polygon,3950)
    official=ref.match_events(gt,output,polygon)
    total_delta=int(estimate.sum()-truth.sum())
    assert total_delta%30==0
    values=(round(window_error(truth,estimate),2),total_delta//30,official["tp"],
            official["pred"]-official["tp"],official["gt"]-official["tp"])
    if values!=_reference_rows(fold)[seed]:
        raise ValueError(f"Replay differs from §95 for {fold}/{seed}: {values}")
    return list(values)


def check_prepared(prepared, baseline, native, det_path, prereg):
    with np.load(prepared,allow_pickle=False) as pack:
        md=json.loads(pack["metadata"].item())
        for key,dtype in (("survivors","<i8"),("origins","<i8"),("pass1","<f8")):
            if array_hash(pack[key],dtype)!=md[key+"_sha256"]:
                raise ValueError(f"Prepared {key} hash mismatch")
    if (md["prereg"]!=prereg or md["strict_sha256"]!=STRICT_SHA256
            or md["baseline_sha256"]!=md["native_sha256"] or bytes_hash(baseline)!=md["baseline_sha256"]
            or bytes_hash(native)!=md["native_sha256"] or bytes_hash(det_path)!=md["det_sha256"]):
        raise ValueError("Original detection cache, model, baseline, or prepared native MOT mismatch")
    return md


def prepare_case(root,fold,seed,prereg,probability):
    clip,det_path,baseline,prepared,track_dir=paths(root,fold,seed)
    native=track_dir/"native.txt"
    track_dir.mkdir(parents=True,exist_ok=True)
    prepared.parent.mkdir(parents=True,exist_ok=True)
    if prepared.exists():
        check_prepared(prepared,baseline,native,det_path,prereg)
        return
    dets=np.load(det_path,allow_pickle=False)
    survivors=prepare_survivors(dets,3950)
    rows,origins=first_pass_survivors(dets,3950,survivors,trace=True)
    with TemporaryDirectory(dir=track_dir) as tmp:
        temp=Path(tmp)/"native.txt"
        write_mot(temp,finish(rows,probability))
        checksum=bytes_hash(temp)
        if checksum!=bytes_hash(baseline):
            raise ValueError(f"Bytewise native MOT mismatch for {fold}/{seed}; replay not used")
        reference=verify95(temp,fold,seed)
        publish_temp(temp,native)
    md=dict(study=111,prereg=prereg,fold=fold,seed=seed,clip=clip,nframes=3950,
            det_sha256=bytes_hash(det_path),native_sha256=checksum,baseline_sha256=checksum,
            strict_sha256=STRICT_SHA256,p0_reference=reference,survivors_sha256=array_hash(survivors,"<i8"),
            pass1_sha256=array_hash(rows,"<f8"),origins_sha256=array_hash(origins,"<i8"))
    with TemporaryDirectory(dir=prepared.parent) as tmp:
        temp=Path(tmp)/prepared.name
        np.savez_compressed(temp,survivors=survivors,origins=origins,pass1=rows,
                            metadata=np.asarray(json.dumps(md,sort_keys=True,allow_nan=False)))
        publish_temp(temp,prepared)


PHASES={"B0":(1,2,3),"B1":(4,5,6),"S0":(1,3,5),"S1":(2,4,6)}


def sealed_case(root,fold,seed,prereg,freeze_sha):
    path=root/"freeze111.json"
    if bytes_hash(path)!=freeze_sha:
        raise ValueError("Selection inventory fingerprint mismatch")
    inventory=json.loads(path.read_bytes())
    name=f"{fold}_{seed}"
    case_path=root/"target"/(name+".json")
    case_sha=bytes_hash(case_path)
    if inventory["prereg"]!=prereg or inventory["cases"][name]!=case_sha:
        raise ValueError("Selection no longer matches its preregistration and freeze")
    case=json.loads(case_path.read_bytes())
    gate=inventory["folds"][str(fold)]
    if (case["fold"]!=fold or case["seed"]!=seed or case["clip"]!=f"BEE24-{fold}-head" or
            case["prereg"]!=prereg or len(case["rows_all"])!=gate["nearall_counts"][seed] or
            len(case["rows"])!=gate["paired_counts"][seed] or
            gate["paired_run"]!=(sum(v>=10 for v in gate["paired_counts"])>=7)):
        raise ValueError("Unexpected intervention case or paired coverage decision")
    return case,case_sha,gate["paired_run"]


def arm_masks(case, paired):
    families=[("nearall",case["rows_all"],"near")]
    if paired:
        families += [("near",case["rows"],"near"),("far",case["rows"],"far")]
    masks={}
    for family,rows,side in families:
        guards=set()
        for r in rows:
            guards.add(r["opposite_idx"])
            for bank in (["near","far"] if family!="nearall" else ["near"]):
                indices=r[bank+"_indices"]
                if len(indices)!=8 or len(set(indices))!=8:
                    raise ValueError("Each region must have eight distinct surviving observations")
                guards.update((indices[0],indices[-1]))
        for phase,offsets in PHASES.items():
            arm=f"{family}_{phase}"
            ids=sorted(r[side+"_indices"][index] for r in rows for index in offsets)
            if ids!=case["masks"][arm] or len(set(ids))!=3*len(rows) or set(ids)&guards:
                raise ValueError(f"Incorrect deletion count, phase or guard in {arm}")
            masks[arm]=tuple(ids)
    return masks


def treat_case(root,fold,seed,prereg,probability,freeze_sha):
    case,case_sha,paired=sealed_case(root,fold,seed,prereg,freeze_sha)
    clip,det_path,baseline,prepared,destination=paths(root,fold,seed)
    native=destination/"native.txt"
    metadata=check_prepared(prepared,baseline,native,det_path,prereg)
    hashes=case["hashes"]
    if (hashes["prepared"]!=bytes_hash(prepared) or hashes["p0"]!=metadata["native_sha256"] or
            hashes["det"]!=metadata["det_sha256"] or hashes["strict"]!=STRICT_SHA256):
        raise ValueError("Native inputs differ from the sealed intervention case")
    with np.load(prepared,allow_pickle=False) as pack:
        survivors=pack["survivors"]
    dets=np.load(det_path,allow_pickle=False)
    masks=arm_masks(case,paired)
    seen={():native}
    files={"native":bytes_hash(native)}
    stamps={}
    for arm,deleted in masks.items():
        path=destination/(arm+".txt")
        marker=destination/(arm+".json")
        deletion_digest=array_hash(deleted,"<i8")
        if marker.exists():
            old=json.loads(marker.read_bytes())
            if (old["freeze_sha256"]!=freeze_sha or old["case_sha256"]!=case_sha or
                    old["deleted_sha256"]!=deletion_digest or old["mot_sha256"]!=bytes_hash(path)):
                raise ValueError(f"Incomplete or inconsistent previous result at {path}")
        else:
            if deleted in seen:
                publish_temp(seen[deleted],path)
            else:
                rows,_=first_pass_survivors(dets,3950,survivors,deleted)
                with TemporaryDirectory(dir=destination) as tmp:
                    temp=Path(tmp)/path.name
                    write_mot(temp,finish(rows,probability))
                    publish_temp(temp,path)
            stamp=dict(arm=arm,freeze_sha256=freeze_sha,case_sha256=case_sha,
                       deleted_sha256=deletion_digest,deleted_count=len(deleted),
                       reused=seen[deleted].stem if deleted in seen else None,mot_sha256=bytes_hash(path))
            write_record(marker,stamp)
        files[arm]=bytes_hash(path)
        stamps[arm]=bytes_hash(marker)
        seen.setdefault(deleted,path)
        print(f"{clip} seed {seed} {arm}: {len(deleted)} post-NMS deletions",flush=True)
    complete=dict(study=111,prereg=prereg,fold=fold,seed=seed,clip=clip,freeze_sha256=freeze_sha,
                  case_sha256=case_sha,det_sha256=metadata["det_sha256"],strict_sha256=STRICT_SHA256,
                  native_p0_reference=metadata["p0_reference"],paired_run=paired,
                  arms=files,arm_records_sha256=stamps)
    write_record(destination/"done.json",complete)


def verify_completion(root,prereg,freeze_sha):
    cases={}
    for fold in (33,35):
        for seed in range(9):
            frozen,case_sha,paired=sealed_case(root,fold,seed,prereg,freeze_sha)
            destination=root/"tracks"/f"{fold}_{seed}"
            marker=destination/"done.json"
            complete=json.loads(marker.read_bytes())
            expected=set(arm_masks(frozen,paired))|{"native"}
            if (set(complete["arms"])!=expected or complete["case_sha256"]!=case_sha or
                    complete["freeze_sha256"]!=freeze_sha):
                raise ValueError(f"Missing or mismatched treatment output at {marker}")
            for arm,sha in complete["arms"].items():
                if bytes_hash(destination/(arm+".txt"))!=sha:
                    raise ValueError(f"Output differs from completed case: {fold}/{seed}/{arm}")
                if arm!="native" and bytes_hash(destination/(arm+".json"))!=complete["arm_records_sha256"][arm]:
                    raise ValueError(f"Changed deletion fingerprint: {fold}/{seed}/{arm}")
            cases[f"{fold}_{seed}"]=bytes_hash(marker)
    path=root/"tracks_done.json"
    write_record(path,dict(study=111,prereg=prereg,freeze_sha256=freeze_sha,cases=cases))
    print(f"All registered output files verified: {path} SHA256 {bytes_hash(path)}",flush=True)


def worker(job):
    mode,root,fold,seed,prereg,weights,extra=job
    if bytes_hash(weights)!=STRICT_SHA256:
        raise ValueError("Strict P0 linker fingerprint mismatch")
    probability=model(json.loads(weights.read_bytes()))
    t=time.monotonic()
    print(f"Start {mode} {fold}/{seed}",flush=True)
    if mode=="prepare":
        prepare_case(root,fold,seed,prereg,probability)
    else:
        treat_case(root,fold,seed,prereg,probability,extra)
    print(f"Done {mode} {fold}/{seed} in {time.monotonic()-t:.1f}s",flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode",choices=("synthetic","prepare","treat"))
    parser.add_argument("--root",type=Path,default=Path("D:/beetrack_runs/study111"))
    parser.add_argument("--weights",type=Path)
    parser.add_argument("--prereg")
    parser.add_argument("--folds",type=int,nargs="+",choices=(33,35),default=[33,35])
    parser.add_argument("--seeds",type=int,nargs="+",choices=range(9),default=list(range(9)))
    parser.add_argument("--workers",type=int,default=3)
    parser.add_argument("--freeze-sha")
    args=parser.parse_args()
    if args.mode=="synthetic":
        self_check()
        return
    if not args.prereg or len(args.prereg)!=40 or not args.weights or args.workers<1:
        parser.error("Supply the registered commit, strict model and a positive worker count")
    if args.mode=="treat" and not args.freeze_sha:
        parser.error("Real treatment requires the sealed selection manifest SHA-256")
    for env in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
        os.environ[env]="1"
    os.environ["CUDA_VISIBLE_DEVICES"]="-1"
    jobs=[(args.mode,args.root,h,s,args.prereg,args.weights,args.freeze_sha) for s in args.seeds for h in args.folds]
    if args.workers==1:
        for job in jobs:
            worker(job)
    else:
        with ProcessPoolExecutor(args.workers) as pool:
            futures=[pool.submit(worker,job) for job in jobs]
            for future in as_completed(futures):
                future.result()
    if args.mode=="treat" and set(args.folds)=={33,35} and set(args.seeds)==set(range(9)):
        verify_completion(args.root,args.prereg,args.freeze_sha)


if __name__=="__main__":
    main()
