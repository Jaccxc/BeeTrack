"""Build and verify the portable source-only static-landmark evidence for study 113."""
import argparse
import base64
from fractions import Fraction
import gzip
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import struct
from tempfile import TemporaryDirectory


PREREG = "34fbedbb4c82fdfce1798627becf1f42e1695d8d"
SOURCE_INVENTORY = "7d9c7ba842f21e9cc62a045b5ffa022b00bbfb266582ec96f9105be380bf6629"
CELLS = {f"{f}_{s}" for f in (33, 35) for s in range(9)}
CLIPS = {f"BEE24-{h}-head" for h in (33,35)}
F32 = struct.Struct("<5f")
I32 = struct.Struct("<11i")


def check(ok, reason):
    if not ok:
        raise ValueError(reason)


def wire(data):
    return (json.dumps(data, ensure_ascii=True, allow_nan=False, sort_keys=True, separators=(",", ":")) + "\n").encode()


def digest(data):
    return sha256(data).hexdigest()


def read_pinned(path, fingerprint):
    raw = path.read_bytes()
    check(digest(raw) == fingerprint, "Source file checksum mismatch: " + path.name)
    return json.loads(raw)


def encode(rows, fmt):
    return base64.b64encode(b"".join(fmt.pack(*row) for row in rows)).decode("ascii")


def decode(text, fmt):
    raw = base64.b64decode(text, validate=True)
    check(len(raw) % fmt.size == 0, "Truncated numerical array")
    return [list(row) for row in fmt.iter_unpack(raw)]


def scaled_cents(values):
    result = [round(x*100) for x in values]
    check([i/100 for i in result] == values, "MOT coordinate cannot be stored losslessly at hundredth pixels")
    return result


def compact(case):
    meta = {k:v for k,v in case.items() if k not in ("sample_frames","quarters","measurement")}
    packed = [encode(q,F32) for q in case["quarters"]]
    check([decode(q,F32) for q in packed] == case["quarters"], "Probe input is not lossless FP32")
    events = [[e["id"], e["f_plus"] if e["direction"] == "in" else -e["f_plus"], e["f_minus"],
               *scaled_cents(e["pre_xyxy"]), *scaled_cents(e["post_xyxy"])] for e in case["measurement"]["events"]]
    return dict(meta=meta, quarters_f32=packed, events_i32=encode(events, I32))


def overlap(a,b):
    width = max(0,min(a[2],b[2])-max(a[0],b[0]))
    height = max(0,min(a[3],b[3])-max(a[1],b[1]))
    shared = width*height
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - shared
    return Fraction(shared,union)


def pair_marks(left,right):
    edges = []
    for i,a in enumerate(left):
        for j,b in enumerate(right):
            iou = overlap(a,b)
            if iou >= Fraction(1,2):
                edges.append((iou,i,j))
    matches, used_left, used_right = [], set(), set()
    for iou,i,j in sorted(edges,key=lambda t:(-t[0],t[1],t[2])):
        if i not in used_left and j not in used_right:
            matches.append(dict(ranks=[i,j],iou=float(iou),selected_events=0,touching_events=0))
            used_left.add(i)
            used_right.add(j)
    return sorted(matches,key=lambda row:row["ranks"])


def measured(quarters, native, nframes):
    check(len(quarters)==4 and all(len(q)<=300 for q in quarters), "Expected four capped detector lists")
    ranked = []
    for q in quarters:
        keep = []
        for i,(x,y,X,Y,p) in enumerate(q):
            check(all(math.isfinite(v) for v in (x,y,X,Y,p)) and X>x and Y>y and 0 <= p <= 1,
                  "Invalid detector row")
            if p >= .25:
                keep.append(dict(raw_index=i, score=p, xyxy=[x,y,X,Y]))
        keep.sort(key=lambda r:(-r["score"],r["xyxy"],r["raw_index"]))
        ranked.append(keep)
    # Put both decimal MOT corners and exact binary FP32 corners on one integer grid.
    den = math.lcm(100,*(v.as_integer_ratio()[1] for q in ranked for r in q for v in r["xyxy"]))
    boxes = []
    for q in ranked:
        exact = []
        for r in q:
            ratios = [v.as_integer_ratio() for v in r["xyxy"]]
            exact.append([num*(den//bottom) for num,bottom in ratios])
        boxes.append(exact)
    marks = [pair_marks(boxes[i],boxes[i+1]) for i in (0,2)]
    events, seen = [], set()
    for pid, signed, preframe, *positions in native:
        after, direction = abs(signed), "in" if signed>0 else "out"
        check(pid not in seen and 1 <= preframe < after <= nframes and after-preframe <= 25,
              "Invalid native identity or crossing frames")
        seen.add(pid)
        before, post = positions[:4], positions[4:]
        check(all(b[0]<b[2] and b[1]<b[3] for b in (before,post)), "Invalid native crossing box")
        ref = None if preframe <= nframes//2 < after else int(after <= nframes//2)
        event = dict(id=pid, direction=direction, f_minus=preframe, f_plus=after,
                     pre_xyxy=[v/100 for v in before],post_xyxy=[v/100 for v in post],reference_half=ref,
                     landmarks_touched=[],selected_landmark=None,best_landmark=None,best_score=None,best_ious=None)
        if ref is not None:
            candidates = []
            for mark in marks[ref]:
                i,j = mark["ranks"]
                pair = (boxes[2*ref][i],boxes[2*ref+1][j])
                four = [overlap([x*(den//100) for x in b],geom) for b in (before,post) for geom in pair]
                candidates.append((min(four),mark["ranks"],four,mark))
            candidates.sort(key=lambda row:(-row[0],row[1]))
            if candidates:
                best, ranks, scores, landmark = candidates[0]
                event.update(best_score=float(best),best_ious=list(map(float,scores)),best_landmark=ranks)
                if best >= Fraction(3,10):
                    event["selected_landmark"] = ranks
                    landmark["selected_events"] += 1
            for score,ranks,scores,landmark in candidates:
                if score >= Fraction(3,10):
                    landmark["touching_events"] += 1
                    event["landmarks_touched"].append(dict(ranks=ranks,ious=list(map(float,scores))))
        events.append(event)
    events.sort(key=lambda r:(r["f_plus"],r["direction"]=="out",r["id"]))
    counts, halves = {}, [{},{}]
    for d in ("in","out"):
        stream = [e for e in events if e["direction"]==d]
        split = sum(e["reference_half"] is None for e in stream)
        counts[d] = dict(total=len(stream),split_excluded=split,assessable=len(stream)-split,
                         touched=sum(e["selected_landmark"] is not None for e in stream))
        for h in (0,1):
            local = [e for e in stream if e["reference_half"]==1-h]
            halves[h][d] = dict(assessable=len(local), touched=sum(e["selected_landmark"] is not None for e in local))
    return dict(quarters=ranked,landmarks=marks,events=events,counts=counts,counts_by_half=halves,
                touched_ids=[e["id"] for e in events if e["selected_landmark"] is not None])


def samples(nframes):
    seq = []
    for j in range(4):
        start, end = 1+j*nframes//4, (j+1)*nframes//4
        seq.append([start+k*(end-start)//30 for k in range(31)])
    return seq


def verify_median(clip,manifest):
    check(manifest["nframes"]==3950 and manifest["prereg"]==PREREG and manifest["study"]==113
          and manifest["clip"]==clip and manifest["decode"]=="opencv IMREAD_COLOR BGR uint8",
          "Median input or decoder differs")
    check(len(manifest["quarters"])==4 and len(manifest["shape"])==3 and manifest["shape"][2]==3,
          "Invalid median shape")
    frame_ids = samples(manifest["nframes"])
    for j,quarter in enumerate(manifest["quarters"]):
        check(quarter["quarter"]==j and quarter["file"]==f"quarter{j}.npy"
              and frame_ids[j] == [f for f,_ in quarter["samples"]], "Median sampling differs")
        for value in [quarter["npy_sha256"], quarter["bgr_sha256"], *(s for _,s in quarter["samples"])]:
            check(len(value)==64 and all(c in "0123456789abcdef" for c in value),"Invalid median receipt")


def fold_report(cases, inv, fingerprint):
    folds = {}
    for h,g in ((33,184),(35,129)):
        rows = []
        for s in range(9):
            measure = cases[f"{68-h}_{s}"]["measurement"]
            m = len(measure["touched_ids"])
            rows.append(dict(seed=s,counts=measure["counts"],m=m,cap_pp=100*m/g))
        total = sum(row["m"] for row in rows)
        distribution = sorted(row["cap_pp"] for row in rows)
        threshold = math.ceil(Fraction(2*9*g,100))
        folds[str(h)] = dict(fold=68-h,gt_count=g,seeds=rows,total_m=total,mean_m=total/9,
                             mean_m_by_direction={d:sum(row["counts"][d]["touched"] for row in rows)/9
                                                  for d in ("in","out")},
                             mean_cap_pp=100*total/(9*g), median_cap_pp=distribution[4],
                             range_cap_pp=[distribution[0],distribution[-1]],
                             seeds_ge_2pp=sum(100*r["m"] >= 2*g for r in rows),required_m=threshold,
                             eligible_for_followup=total>=threshold)
    return dict(study=113,prereg=PREREG,role="source",inventory_sha256=fingerprint,cases=inv["cases"],hives=folds)


def replay(bundle, source108):
    from lineage113 import compare108

    inv = bundle["inventory"]
    check(inv["prereg"]==PREREG and inv["study"]==113 and inv["role"]=="source"
          and set(inv["medians"])==CLIPS and set(bundle["medians"])==CLIPS
          and set(inv["cases"])==CELLS and set(bundle["cases"])==CELLS, "Incomplete source freeze")
    sha = digest(wire(inv))
    check(sha == bundle["inventory_sha256"] == SOURCE_INVENTORY, "Global inventory fingerprint mismatch")
    for clip, manifest in bundle["medians"].items():
        verify_median(clip, manifest)
        check(digest(wire(manifest))==inv["medians"][clip], "Median manifest fingerprint mismatch")
    cases = {}
    for key, payload in bundle["cases"].items():
        meta = payload["meta"]
        clip = meta["clip"]
        check(meta["prereg"]==PREREG and meta["hashes"]["medians"]==inv["medians"][clip],
              "Median or registration changed between cases")
        required = dict(device="cpu",half=False,imgsz=640,conf=.05,iou=.7,max_det=300,augment=False)
        check(all(meta["detector"][k]==v for k,v in required.items()),"Detector protocol differs")
        quarters = [decode(q,F32) for q in payload["quarters_f32"]]
        native = decode(payload["events_i32"],I32)
        measures = measured(quarters,native,meta["nframes"])
        case = dict(meta,sample_frames=samples(meta["nframes"]),quarters=quarters,measurement=measures)
        check(digest(wire(case)) == inv["cases"][key], "Independent reconstruction changed " + key)
        cases[key] = case
    compare108(cases,source108)
    report = fold_report(cases,inv,sha)
    check(digest(wire(report))==bundle["summary_sha256"], "Source summary fingerprint differs")
    return cases, report


def pack(run, path, sha, source108):
    inv = read_pinned(run/"source_inventory.json",sha)
    check(set(inv["medians"])==CLIPS and set(inv["cases"])==CELLS,"Registered inventory differs")
    medians = {c:read_pinned(run/"medians"/c/"manifest.json",v) for c,v in inv["medians"].items()}
    cases = {key:compact(read_pinned(run/"source"/(key+".json"),v)) for key,v in inv["cases"].items()}
    archive = dict(study=113,format=1,inventory=inv,inventory_sha256=sha,medians=medians,cases=cases,
                   summary_sha256=digest((run/"source_summary.json").read_bytes()))
    replay(archive,source108)
    raw = gzip.compress(wire(archive),mtime=0)
    path.parent.mkdir(parents=True,exist_ok=True)
    with TemporaryDirectory(dir=path.parent) as temp:
        part=Path(temp)/"archive"
        part.write_bytes(raw)
        try:
            os.link(part,path)
        except FileExistsError:
            check(path.read_bytes()==raw, "Different archive already exists at " + str(path))
    print(path.name, len(raw), digest(raw))


def show(cases,report,table):
    for hive,stats in report["hives"].items():
        print(f"source {hive}, fold {stats['fold']}: touched {stats['total_m']} of "
              f"{stats['required_m']} required across seeds; mean capacity {stats['mean_cap_pp']:.4f} pp; "
              f"follow-up {stats['eligible_for_followup']}")
    if table:
        print("source fold seed | quarter raw | quarter score >=.25 | landmarks by half | native in/out | "
              "split in/out | assessable in/out | touched in/out | touched half 1,2 | capacity pp")
        for hive,stats in report["hives"].items():
            for r in stats["seeds"]:
                case = cases[f"{stats['fold']}_{r['seed']}"]
                m = case["measurement"]
                cols = ["/".join(str(len(x)) for x in v) for v in (case["quarters"],m["quarters"],m["landmarks"])]
                cols += ["/".join(str(r["counts"][d][k]) for d in ("in","out"))
                         for k in ("total","split_excluded","assessable","touched")]
                halves = [sum(h[d]["touched"] for d in ("in","out")) for h in m["counts_by_half"]]
                cols += ["/".join(map(str,halves)), f"{r['cap_pp']:.4f}"]
                print(f"{hive} {stats['fold']} {r['seed']} | " + " | ".join(cols))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command",required=True)
    make=commands.add_parser("pack")
    make.add_argument("run",type=Path)
    make.add_argument("archive",type=Path)
    make.add_argument("--inventory-sha",required=True)
    verify=commands.add_parser("replay")
    verify.add_argument("archive",type=Path)
    verify.add_argument("--table",action="store_true")
    for cmd in (make,verify):
        cmd.add_argument("--source108",type=Path,required=True)
    args=parser.parse_args()
    if args.command=="pack":
        pack(args.run,args.archive,args.inventory_sha,args.source108)
    else:
        data=json.loads(gzip.decompress(args.archive.read_bytes()))
        check(data["study"]==113 and data["format"]==1,"Unexpected source archive version")
        cases, report=replay(data,args.source108)
        show(cases,report,args.table)


if __name__=="__main__":
    main()
