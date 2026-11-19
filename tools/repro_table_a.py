"""Reproduce Table A of report/RESULTS_v2.md (val + dev, the seen hives, 486 GT events) from the cached detections in
runs/dets/<det>/{val,dev}/, for the detectors y11s_640, y11s4_s1, y11s4_s2:
  P0 with the three cross-fitted link models of §74 (m_v0_val* on val, m_v0_dev* on dev; mean over the three),
  P0 with m_dev_strict.json (the test link model; report only on val + dev), default and calibrated ByteTrack.
Every number is asserted against the log. CPU only.
  PYTHONPATH=. uv run python tools/repro_table_a.py [--out DIR] [--ref DIR]
--out keeps the tracker outputs as DIR/<arm>/<split>/<video>.txt (otherwise each one is deleted once scored);
--ref asserts that every output has the same SHA-256 as the file with the same relative path under DIR."""
import argparse
import hashlib
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from beetrack.common import split_specs, write_mot
from beetrack.evaluate import load_tracks
from beetrack.p0 import bytetrack, calibrated, dets, link_model, nframes, p0, pooled, score, stage1

ROOT = Path(__file__).resolve().parents[1]
DETS = ("y11s_640", "y11s4_s1", "y11s4_s2")
VALDEV = [(sp, ev) for sp in ("val", "dev") for ev, *_ in split_specs(sp)]
SEEDS = ("", "_s1", "_s2")
NEV = 486

# logged values, per detector in DETS order
EXP = {
    # §91, §96; per link model from the Table A notes (§91)
    "P0": dict(win=(18.89, 17.34, 19.15), mean=18.46, range=1.81, bias=(-33.3, -44.0, -12.7), f1=(0.753, 0.740, 0.728),
               f1_mean=0.740, links=((18.74, 19.01, 18.93), (17.48, 16.49, 18.07), (19.25, 18.16, 20.04))),
    # §125 clarification 1 gives the mean; per detector from the §125 tool's val + dev run
    "P0 strict": dict(win=(19.53, 19.71, 19.42), mean=19.56, range=0.29, bias=(-16.0, -29.0, -5.0), f1=(0.743, 0.738, 0.724),
                      f1_mean=0.735),
    # §93, §96; F1 per detector §102
    "BT": dict(win=(41.37, 38.13, 41.04), mean=40.18, range=3.24, bias=(-193.0, -179.0, -161.0), f1=(0.560, 0.547, 0.528),
               f1_mean=0.545),
    # §93, §96: val counts x k fitted on dev, dev counts x k fitted on val
    "cal BT": dict(win=(33.78, 29.09, 41.60), mean=34.82, range=12.51, bias=(20.3, 13.9, 46.9)),
}
K_TEST = ((2.016, 1.396), (1.866, 1.364), (1.969, 1.192))     # §93 'k for test' (k_in, k_out), fitted on val + dev


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description="Reproduce Table A (val + dev) from cached detections.")
    ap.add_argument("--out", type=Path, help="keep the tracker outputs in this folder")
    ap.add_argument("--ref", type=Path, help="folder of reference tracker outputs, compared by SHA-256")
    a = ap.parse_args()
    t0 = time.time()
    tmp = None if a.out else tempfile.TemporaryDirectory()
    out = a.out or Path(tmp.name)
    links = {(sp, k): link_model(f"m_v0_{sp}{s}") for sp in ("val", "dev") for k, s in enumerate(SEEDS)}
    strict = link_model("m_dev_strict")
    sums, R = {}, {arm: [] for arm in EXP}

    def run(rows, name, sp, ev):
        """Write the MOT file and score it as written (boxes rounded to 2 decimals), as the logged numbers were."""
        f = out / name / sp / f"{ev}.txt"
        write_mot(f, rows)
        sums[f.relative_to(out).as_posix()] = sha(f)
        s = score(load_tracks(f), sp, ev)
        if not a.out:
            f.unlink()
        return s

    for d in DETS:
        tl, st, bt = {k: [] for k in range(3)}, [], {"val": [], "dev": []}
        for sp, ev in VALDEV:
            D, nf = dets(d, sp, ev), nframes(sp, ev)
            S1 = stage1(D, nf)
            for k in range(3):
                tl[k].append(run(p0(S1, links[sp, k]), f"p0_{d}_l{k}", sp, ev))
            st.append(run(p0(S1, strict), f"p0_strict_{d}", sp, ev))
            bt[sp].append(run(bytetrack(D, nf), f"bt_{d}", sp, ev))
        assert sum(s["gt"] for s in st) == NEV
        per = [pooled(tl[k]) for k in range(3)]
        R["P0"].append({**{x: float(np.mean([p[x] for p in per])) for x in per[0]}, "links": [p["win"] for p in per]})
        R["P0 strict"].append(pooled(st))
        R["BT"].append(pooled(bt["val"] + bt["dev"]))
        cal, kt = calibrated(bt)
        R["cal BT"].append({**cal, "k": kt})
        print(f"{d}: done, {(time.time() - t0) / 60:.1f} min", flush=True)

    print(f"\nTable A, val + dev ({NEV} GT events): 30 s window error % per detector ({' / '.join(DETS)}), mean [range]; "
          f"bias; event F1@0.5")
    for arm, e in EXP.items():
        r = R[arm]
        w, b = [x["win"] for x in r], [x["bias"] for x in r]
        f1 = [x["f1"] for x in r] if "f1" in e else None
        print(f"  {arm:9s} {' / '.join(f'{x:5.2f}' for x in w)}  mean {np.mean(w):5.2f} [{np.ptp(w):5.2f}]  "
              f"bias {' / '.join(f'{x:+.1f}' for x in b)}" +
              (f"  F1 {' / '.join(f'{x:.3f}' for x in f1)} (mean {np.mean(f1):.3f})" if f1 else ""))
        if "links" in e:
            print(f"  {'':9s} per link model: " + ", ".join(f"{d} {' / '.join(f'{x:.2f}' for x in q['links'])}" for d, q in zip(DETS, r)))
        assert [round(x, 2) for x in w] == list(e["win"]), (arm, w)
        assert (round(float(np.mean(w)), 2), round(float(np.ptp(w)), 2)) == (e["mean"], e["range"]), (arm, w)
        assert [round(x, 1) for x in b] == list(e["bias"]), (arm, b)
        if f1:
            assert [round(x, 3) for x in f1] == list(e["f1"]) and round(float(np.mean(f1)), 3) == e["f1_mean"], (arm, f1)
        if "links" in e:
            assert [tuple(round(x, 2) for x in q["links"]) for q in r] == list(e["links"]), (arm, [q["links"] for q in r])
    ks = [tuple(round(float(x), 3) for x in q["k"]) for q in R["cal BT"]]
    print("  cal BT k for test (val + dev fit, k_in / k_out): " + ", ".join(f"{d} {x[0]:.3f} / {x[1]:.3f}" for d, x in zip(DETS, ks)))
    assert ks == list(K_TEST), ks
    print(f"all logged numbers reproduced ({len(sums)} tracker outputs)")

    if a.ref:
        bad = [p for p, h in sums.items() if not (a.ref / p).exists() or sha(a.ref / p) != h]
        assert not bad, f"{len(bad)} outputs differ from {a.ref}, first {bad[0]}"
        print(f"all {len(sums)} tracker outputs byte-identical (SHA-256) to the files under {a.ref}")

    # every imported module comes from this repo or the Python installation, nothing from another folder on the path
    roots = [ROOT, Path(sys.prefix).resolve(), Path(sys.base_prefix).resolve()]
    files = {m.__file__ for m in list(sys.modules.values()) if getattr(m, "__file__", None)}
    stray = sorted(f for f in files if not any(Path(f).resolve().is_relative_to(r) for r in roots))
    assert not stray, stray
    print(f"import check: all {len(files)} imported module files are under the repo or the Python installation")
    if tmp:
        tmp.cleanup()
    print(f"runtime {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
