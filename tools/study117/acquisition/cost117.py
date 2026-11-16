"""Read timing and memory from a sealed detector study without opening annotations."""
import argparse
import hashlib
import json
from pathlib import Path
import sqlite3
from statistics import median
import zlib


def costs(root):
    seal=json.loads((root/"detector_done.json").read_bytes())
    out={}
    for h in (33,35):
        times={name:[] for name in ("decode","preprocess","forward","features_a_prefix","features_six",
                                    "reference_nms","a_nms_proxy","compute_six","u_component_sum","a_component_sum")}
        hist=set()
        for seed in range(3):
            path=root/"cases"/f"{h}_{seed}.sqlite"
            with path.open("rb") as src:
                assert hashlib.file_digest(src,"sha256").hexdigest()==seal["files"][path.relative_to(root).as_posix()]
            con=sqlite3.connect(path.resolve().as_uri()+"?mode=ro",uri=True)
            try:
                for blob, in con.execute("SELECT record FROM frames ORDER BY frame"):
                    row=json.loads(zlib.decompress(blob)); t=row["timing_ms"]
                    nms=t["nms_detail"]
                    stock=nms["stock_reference"]
                    a=nms["indexed"]["A"]["standalone_proxy_ms"]
                    shared=t["read_hash_decode"]+t["preprocess"]+t["forward"]
                    values=(t["read_hash_decode"],t["preprocess"],t["forward"],t["a_path_ms"],t["features_six"],
                            stock,a,t["compute_total"],shared+stock,shared+t["a_path_ms"]+a)
                    for k,v in zip(times,values):
                        times[k].append(v)
                    if row["frame"]>2:
                        hist.add((row["history_native_bytes"],row["history_diagnostic_boxes_bytes"]))
            finally:
                con.close()
        assert len(times["forward"])==3000 and len(hist)==1
        native, audit=hist.pop()
        out[h]=dict(frames=3000,medians_ms={k:round(median(v),4) for k,v in times.items()},
                    two_frame_feature_logit_bytes=native,diagnostic_box_bytes=audit)
    return out


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("root",type=Path)
    ap.add_argument("--output",type=Path)
    args=ap.parse_args()
    s=json.dumps(costs(args.root),indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(s+"\n",encoding="utf-8")
    print(s)
