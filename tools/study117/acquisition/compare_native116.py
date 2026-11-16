"""Compare native §117 U with the same actual images and weights from §116.

Each comparison uses sealed U predictions only and performs no model forward.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sqlite3
import zlib


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compare(previous, current, partial=False):
    prior=json.loads((previous/"detector_done.json").read_bytes())
    results={}
    for h in (33,35):
        clip=f"BEE24-{68-h}-tail"
        for s in range(3):
            key=f"{h}_{s}"; dbpath=current/"cases"/(key+".sqlite")
            if not dbpath.exists() and partial:
                continue
            rel=f"predictions/{clip}/{key}_U.json"
            archive=previous/rel
            assert sha(archive)==prior["files"][rel]
            before=json.loads(archive.read_bytes())
            con=sqlite3.connect(dbpath.resolve().as_uri()+"?mode=ro",uri=True)
            try:
                cfg,last,sealed=con.execute("SELECT config,last,sealed FROM state WHERE id=1").fetchone()
                assert json.loads(cfg)["weights_sha256"]==before["run"]["weights_sha256"]
                assert partial or sealed==1
                frames=boxes=0
                for f in before["requested_frames"]:
                    if f>last and partial:
                        continue
                    blob,db_sha=con.execute("SELECT record,sha256 FROM frames WHERE frame=?",(f,)).fetchone()
                    raw=zlib.decompress(blob)
                    assert hashlib.sha256(raw).digest()==db_sha
                    rec=json.loads(raw)
                    legacy=previous/"receipts"/clip/(key+"_U")/f"{f:04}.json"
                    assert sha(legacy)==before["run"]["frame_receipts_sha256"][str(f)]
                    old=json.loads(legacy.read_bytes())
                    assert rec["image_sha256"]==old["image_sha256"] and rec["input_bchw"]==old["tensor_bchw"]
                    reconstructed=[[*rec["boxes"][str(i)],p,0.0] for i,p in rec["arms"]["U"]]
                    assert reconstructed==before["predictions"][str(f)]==old["wrapped"], (key,f,"stock U")
                    frames+=1; boxes+=len(reconstructed)
                results[key]=dict(identical_frames=frames,identical_ordered_boxes=boxes)
            finally:
                con.close()
    return results


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("previous",type=Path)
    p.add_argument("current",type=Path)
    p.add_argument("--partial",action="store_true")
    p.add_argument("--output",type=Path)
    args=p.parse_args()
    text=json.dumps(compare(args.previous,args.current,args.partial),sort_keys=True,indent=2)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(text+"\n",encoding="utf-8")
