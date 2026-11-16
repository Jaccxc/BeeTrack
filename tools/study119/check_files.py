"""Synthetic end-to-end check of portable inputs, ten-column export and raw rescoring."""
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from capacity import analyze
from run_capacity import physical_sink
from verify_physical import compare_case, load_mot


def raw(lines):
    return "".join(",".join(map(str,[*r,1,-1,-1,-1]))+"\n" for r in lines).encode()


def check():
    poly=[[0,0],[5,0],[5,30],[0,30]]
    gt=raw([[5,1,-3,1,2,2],[6,1,1,1,2,2],[10,2,-3,16,2,2],[11,2,1,16,2,2],
            [12,3,1,24,2,2],[13,3,-3,24,2,2],[14,3,1,24,2,2]])
    p0=raw([[5,10,-3,1,2,2],[6,11,1,1,2,2],[10,12,-3,19,2,2]])
    with TemporaryDirectory() as d:
        here=Path(d)
        source=here/"raw.txt"
        source.write_bytes(p0)
        result=analyze(load_mot(gt),load_mot(p0),poly,30,physical_sink=physical_sink(source,here/"out"))
        frozen=json.loads(json.dumps(result))
        cases={}
        for name in frozen["bounds"]:
            rec=frozen["bounds"][name]["physical_mot"]
            physical=(here/"out"/rec["file"]).read_bytes()
            assert hashlib.sha256(physical).hexdigest()==rec["sha256"]
            cases[name]=physical
        checked=compare_case(gt,p0,cases,frozen)
        assert len(checked)==5 and checked["canonical_no_edge"]["tp05"]==1
        orig=raw([[5,10,-3,1,2,2],[6,11,1,1,2,2],[10,12,-3,19,2,2]])
        assert p0==orig
    print("Four augmented ten-column files preserve each native row; both frame owners, all edges and 30 origins replay.")


if __name__=="__main__":
    check()
