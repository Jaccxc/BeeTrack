"""Log §161 (issue #2, #3): the crossing join (CJ, beetrack.p0.crossing_join) after P0's link step, against P0 and the
logged calibrated ByteTrack on seven evaluation units. CPU only, never a test sequence, nothing trained, nothing swept.
  甲 P0 (beetrack.p0.p0).   戊 P0 + CJ: the same linked rows -> crossing_join -> drop tracks under 3 boxes -> events.
  丁 calibrated ByteTrack fit B, per seed as logged (read, not rerun).
Units, detector seeds 0..8 each (val + dev: Table A's 3 detectors):
  35-head, 33-head (dev), 35-tail, 33-tail (val): loho_no<h>_s0..8 with m_dev_strict.json (§95, §152), the clean checks;
  whiteboard (13 clips, loho_noW, m_noW.json, §157) and gray board (16 clips, loho_noG, m_noG.json, §159): the folds that
  found the cause, errors summed over the clips / all GT events of the fold (137, 152);
  val + dev: P0 = mean over the 3 cross-fitted link models m_v0_* (486 GT events, tools/repro_table_a.py).
Score: 30 s window error % (p0.window_err), bias (predicted - GT events), event F1@0.5, every output scored as read back
from its MOT file. Per join: the GT bee under a's last and under b's first box (the GT box of that frame with the highest
IoU, if >= 0.5; -1 none), same GT id = same bee; dropped = the merged track still has under 3 boxes.
Criteria (§161): seed groups 0-2 / 3-5 / 6-8 (means of single-seed errors, for 丁 too), d = arm - base, PASS if mean d < 0
and |mean d| > max(range of the group means of both); criterion 1 = 戊 vs 甲, criterion 2 = 戊 vs 丁; the 9-seed reading
is report only. Criterion (a): on val + dev 戊 is at most 2 points above 甲.
Asserts (before anything is written): the toy test next to crossing_join (on import); 甲 = the logged P0 per seed (2
decimals), its bias and F1 where logged, mean / range / group means; val + dev = Table A (per detector and link model,
mean, range, bias, F1); whiteboard, gray-board and 35-head P0 MOT files byte-identical to the §157 / §159 caches; where a
video has no join, 戊's MOT file is 甲's byte for byte; the logged 丁 values agree with a second logged source.
  PYTHONPATH=. CUDA_VISIBLE_DEVICES=-1 uv run python tools/crossing_join.py selfcheck|run [--out DIR]
selfcheck: the same on seed 0 of every unit and val + dev's first detector. Writes crossing_join[_selfcheck].txt and
.json to DIR (default D:/beetrack_runs/c161), nothing else (MOT files go to a temporary folder)."""
import argparse
import json
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from beetrack.common import write_mot
from beetrack.evaluate import load_tracks
from beetrack.link import link
from beetrack.p0 import MIN_LEN, THR, ZONE, crossing_join, dets, gt, link_model, nframes, pooled, stage1
from beetrack.track import drop_short, iou_matrix
from gray_fold import GRAY
from repro_table_a import DETS, EXP, NEV, SEEDS as LINKS, VALDEV
from whiteboard_fold import WB, dtag, line, rng, wscore
from window_lengths import EXP_HEADS, GROUPS

TEST = ("BEE24-12", "BEE24-13", "BEE24-16", "BEE24-34", "BEE24-36")
W154, T153 = Path("D:/beetrack_runs/w154"), Path("D:/beetrack_runs/t153")
ZF = SimpleNamespace(poly=ZONE)
CLEAN = ("35-head", "33-head", "35-tail", "33-tail")
TAIL33_RANGE = (19.31, 57.87)          # §152 text: calibrated ByteTrack on 33-tail per seed from 19.31 to 57.87


def units():
    """name -> videos, polygons (F.poly), link model, detector tag of (seed, split), cached P0 MOT file or None."""
    U = {}
    for part, sp in (("head", "dev"), ("tail", "val")):
        for h in ("35", "33"):
            cache = (lambda s, sp, ev: WB.out / "cache" / f"p0s_loho_no35_s{s}" / sp / f"{ev}.txt") if (h, part) == ("35", "head") else None
            U[f"{h}-{part}"] = dict(videos=[(sp, f"BEE24-{h}-{part}")], F=ZF, linker="m_dev_strict",
                                    tag=lambda s, sp, h=h: f"loho_no{h}_s{s}", cache=cache)
    for F in (WB, GRAY):
        U[F.name] = dict(videos=F.clips, F=F, linker=F.linker.stem, tag=lambda s, sp, F=F: dtag(s, sp, F=F),
                         cache=lambda s, sp, ev, F=F: F.out / "cache" / f"{F.p0c}{dtag(s, sp, F=F)}" / sp / f"{ev}.txt")
    return {k: U[k] for k in (*CLEAN, WB.name, GRAY.name)}


def grp(x):
    return [float(np.mean([x[i] for i in g])) for g in GROUPS]


def r2(x):
    return [round(float(v), 2) for v in x]


def logged(name):
    """The logged 甲 (win % per seed, bias, F1 or None, group means) and 丁 (win % per seed), 丁 checked against a second
    logged source. Returns (dict, description of the check)."""
    if name in CLEAN:
        h, part = name.split("-")
        W = json.loads((W154 / "window_lengths_heads.json").read_text(encoding="utf-8"))["folds"][h]
        if part == "head":                 # §95 values in tools/window_lengths.py; 丁 in full from §154's output
            E = EXP_HEADS[h]
            cal = [x[1] for x in W["arms"]["cal BT"]["per_seed"]]
            assert r2(cal) == list(E["cal BT"]["seed"]) and r2(grp(cal)) == list(E["cal BT"]["groups"]), (name, cal)
            assert (round(float(np.mean(cal)), 2), round(rng(cal), 2)) == (E["cal BT"]["mean"], E["cal BT"]["range"]), name
            return (dict(win=E["P0"]["seed"], bias=E["P0"]["bias"], f1=None, groups=E["P0"]["groups"], cal=cal),
                    "丁 from w154/window_lengths_heads.json = §95 (EXP_HEADS) per seed, mean, range, group means")
        R = json.loads((T153 / f"res_tail_{h}.json").read_text(encoding="utf-8"))
        assert R["target"] == ["val", name.replace(h, f"BEE24-{h}", 1)] and len(R["seeds"]) == 9, R["target"]
        cal = [x["cal_bt"] for x in R["seeds"]]
        # fit B is the head's (§152: same k), and the 33-tail range is the one in §152's text
        assert [f"{v:.3f}" for x in R["seeds"] for v in x["k"]] == [f"{v:.3f}" for k in W["k"] for v in k], name
        assert h == "35" or (round(min(cal), 2), round(max(cal), 2)) == TAIL33_RANGE, (name, cal)
        return (dict(win=[x["p0"] for x in R["seeds"]], bias=[x["events"]["640_id"] - R["gt_n"] for x in R["seeds"]], f1=None,
                     groups=[g["p0"] for g in R["groups"]], cal=cal),
                f"丁 from t153/res_tail_{h}.json; its fit B k = the head's (3 decimals)" + ("" if h == "35" else
                                                                                        f"; per-seed range {TAIL33_RANGE} as §152's text"))
    F = {WB.name: WB, GRAY.name: GRAY}[name]
    J = json.loads((F.out / f"{F.file}.json").read_text(encoding="utf-8"))[F.name]
    cal = [x["cal"] for x in J["seeds"]]
    txt = next(l for l in (F.out / f"{F.file}.txt").read_text(encoding="utf-8").splitlines() if l.strip().startswith("丁 cal BT B"))
    vals, mean = txt.split("丁 cal BT B")[1].split("mean")
    assert [float(v) for v in vals.split("/")] == r2(cal) and float(mean.split("[")[0]) == round(float(np.mean(cal)), 2), name
    assert r2(grp(cal)) == r2(J["groups"]["丁 cal BT single"]), name
    return (dict(win=[x["p0"] for x in J["seeds"]], bias=[x["p0_bias"] for x in J["seeds"]], f1=[x["p0_f1"] for x in J["seeds"]],
                 groups=J["groups"]["甲 P0 single"], cal=cal),
            f"丁 from {F.out.name}/{F.file}.json = its .txt line (2 decimals, mean) and group means")


def scored(rows, sp, ev, F, tmp):
    """wscore of the rows as read back from a MOT file, and the file's bytes."""
    f = Path(tmp) / "rows.txt"
    write_mot(f, rows)
    return wscore(load_tracks(f), sp, ev, F), f.read_bytes()


def owner(r, g):
    """GT id of the GT box with the highest IoU with box r in r's frame, if that IoU >= 0.5, else -1."""
    G = g[g[:, 0] == r[0]]
    if not len(G):
        return -1
    xyxy = lambda b: np.c_[b[:, 2:4], b[:, 2:4] + b[:, 4:6]]
    m = iou_matrix(xyxy(np.asarray(r[None, :6])), xyxy(G))[0]
    return int(G[m.argmax(), 1]) if m.max() >= 0.5 else -1


def video(R, prob, sp, ev, F, tmp, ref=None):
    """One video: 甲 and 戊 from the same linked rows (甲 exactly as p0.p0), scored; every join with its GT bees."""
    L = sorted(link(R.tolist(), THR, prob).tolist())
    rows, joins = crossing_join(L, F.poly[ev])
    (p, bp), (c, bc) = (scored(drop_short(x, MIN_LEN, 0.0), sp, ev, F, tmp) for x in (L, sorted(rows)))
    assert joins or bc == bp, ev                        # no join: 戊 is 甲, byte for byte
    assert ref is None or bp == ref.read_bytes(), ref
    A, g, out = np.array(L), gt(sp, ev), []
    for a, b, dt, d in joins:
        ta, tb = A[A[:, 1] == a], A[A[:, 1] == b]       # in frame order (L is sorted by frame)
        bee = [owner(ta[-1], g), owner(tb[0], g)]
        out.append(dict(video=ev, a=int(a), b=int(b), dt=int(dt), d=d, bee=bee, same=bee[0] == bee[1] >= 0,
                        dropped=len(ta) + len(tb) < MIN_LEN))
    return p, c, out


def seed_unit(u, s, prob, tmp):
    V = [video(stage1(dets(u["tag"](s, sp), sp, ev), nframes(sp, ev)), prob, sp, ev, u["F"], tmp, u["cache"] and u["cache"](s, sp, ev))
         for sp, ev in u["videos"]]
    return dict(n=sum(v[0]["gt"] for v in V), p0=pooled([v[0] for v in V]), cj=pooled([v[1] for v in V]),
                joins=[j for v in V for j in v[2]])


def check(name, S, L, seeds):
    """甲 = the logged P0: per seed window error (2 decimals), bias, F1 (3 decimals) where logged; with 9 seeds also the
    group means."""
    for s, r in zip(seeds, S):
        assert round(r["p0"]["win"], 2) == round(L["win"][s], 2), (name, s, r["p0"]["win"], L["win"][s])
        assert r["p0"]["bias"] == L["bias"][s], (name, s, r["p0"]["bias"], L["bias"][s])
        assert L["f1"] is None or round(r["p0"]["f1"], 3) == round(L["f1"][s], 3), (name, s, r["p0"]["f1"])
    if len(S) == 9:
        assert r2(grp([r["p0"]["win"] for r in S])) == r2(L["groups"]), name


def jstats(J):
    n, same = len(J), sum(j["same"] for j in J)
    un, dr = sum(min(j["bee"]) < 0 for j in J), sum(j["dropped"] for j in J)
    return dict(joins=n, same=same, unowned=un, dropped=dr, same_frac=same / n if n else None,
                same_frac_owned=same / (n - un) if n - un else None)


def crit(say, name, A, B, la, lb):
    d, bar = np.subtract(A, B), max(rng(A), rng(B))
    ok = bool(d.mean() < 0 and -d.mean() > bar)
    say(f"  {name}: d = {la} - {lb} [{line(d, '+.2f')}] mean {d.mean():+.2f}; bar max(range {la} {rng(A):.2f}, range {lb} "
        f"{rng(B):.2f}) = {bar:.2f}: {'PASS' if ok else 'no'} (margin {-d.mean() - bar:+.3f}); {la} lower in {(d < 0).sum()} / {len(d)}")
    return dict(d=d.tolist(), mean_d=float(d.mean()), bar=bar, passed=ok, lower=int((d < 0).sum()))


def report(say, name, u, S, L):
    vids = u["videos"][0][1] if len(u["videos"]) == 1 else f"{len(u['videos'])} clips"
    say(f"{name}: {vids}, {S[0]['n']} GT events, detectors {u['tag'](0, u['videos'][-1][0])}..s8, link model {u['linker']}.json")
    W = {"甲 P0": [r["p0"]["win"] for r in S], "戊 P0+CJ": [r["cj"]["win"] for r in S], "丁 cal BT": L["cal"]}
    say("  window error % per seed (0..8), 9-seed mean [range]; group means (0-2 / 3-5 / 6-8), their range")
    for a, x in W.items():
        say(f"  {a:9s} {line(x)}  mean {np.mean(x):6.2f} [{rng(x):6.2f}]  groups {line(grp(x))}  range {rng(grp(x)):6.2f}")
    for k, f in (("bias", "+6.0f"), ("f1", "6.3f")):
        for a, key in (("甲 P0", "p0"), ("戊 P0+CJ", "cj")):
            x = [r[key][k] for r in S]
            say(f"  {a:9s} {'bias' if k == 'bias' else 'F1  '} {line(x, f)}  mean {np.mean(x):{f}}")
    js = [jstats(r["joins"]) for r in S]
    for k in ("joins", "same", "unowned", "dropped"):
        say(f"  {k:9s} per seed {line([j[k] for j in js], '4d')}")
    t = jstats([j for r in S for j in r["joins"]])
    f = lambda x: "n/a" if x is None else f"{x:.3f}"
    say(f"  joins over the 9 seeds: {t['joins']}, same GT bee {t['same']} ({f(t['same_frac'])}; {f(t['same_frac_owned'])} of the "
        f"{t['joins'] - t['unowned']} with both ends owned), an unowned end {t['unowned']}, dropped (< 3 boxes) {t['dropped']}")
    C = {"criterion 1": crit(say, "criterion 1 (groups)", grp(W["戊 P0+CJ"]), grp(W["甲 P0"]), "戊", "甲"),
         "criterion 2": crit(say, "criterion 2 (groups)", grp(W["戊 P0+CJ"]), grp(W["丁 cal BT"]), "戊", "丁"),
         "criterion 1, 9 seeds (report)": crit(say, "criterion 1, 9 seeds (report)", W["戊 P0+CJ"], W["甲 P0"], "戊", "甲"),
         "criterion 2, 9 seeds (report)": crit(say, "criterion 2, 9 seeds (report)", W["戊 P0+CJ"], W["丁 cal BT"], "戊", "丁")}
    return dict(videos=u["videos"], linker=u["linker"], gt_events=S[0]["n"], seeds=[dict(p0=r["p0"], cj=r["cj"], cal=c, **jstats(r["joins"]))
                for r, c in zip(S, L["cal"])], mean={a: float(np.mean(x)) for a, x in W.items()}, range={a: rng(x) for a, x in W.items()},
                groups={a: grp(x) for a, x in W.items()}, joins_total=t, criteria=C, joins=[r["joins"] for r in S])


def valdev(dets_, tmp, say):
    links = {(sp, k): link_model(f"m_v0_{sp}{x}") for sp in ("val", "dev") for k, x in enumerate(LINKS)}
    out, t0 = [], time.time()
    for d in dets_:
        P, C, J = [[], [], []], [[], [], []], []
        for sp, ev in VALDEV:
            R = stage1(dets(d, sp, ev), nframes(sp, ev))
            for k in range(3):
                p, c, j = video(R, links[sp, k], sp, ev, ZF, tmp)
                P[k].append(p)
                C[k].append(c)
                J += [x | dict(link=k) for x in j]
        assert sum(s["gt"] for s in P[0]) == NEV
        pp, cc = [pooled(x) for x in P], [pooled(x) for x in C]
        mean = lambda q: {x: float(np.mean([y[x] for y in q])) for x in q[0]}
        out.append(dict(det=d, p0=mean(pp), cj=mean(cc), p0_links=[y["win"] for y in pp], cj_links=[y["win"] for y in cc], joins=J))
        print(f"val + dev {d}: done, {(time.time() - t0) / 60:.1f} min", flush=True)
    E = EXP["P0"]
    for i, r in enumerate(out):              # Table A
        assert (round(r["p0"]["win"], 2), round(r["p0"]["bias"], 1), round(r["p0"]["f1"], 3)) == (E["win"][i], E["bias"][i], E["f1"][i]), r["p0"]
        assert tuple(round(x, 2) for x in r["p0_links"]) == E["links"][i], r["p0_links"]
    if len(out) == 3:
        w = [r["p0"]["win"] for r in out]
        assert (round(float(np.mean(w)), 2), round(rng(w), 2), round(float(np.mean([r["p0"]["f1"] for r in out])), 3)) == \
            (E["mean"], E["range"], E["f1_mean"]), w
    cal = [x[1] for x in json.loads((W154 / "window_lengths.json").read_text(encoding="utf-8"))["arms"]["cal BT"]["per_det"]]
    assert r2(cal) == list(EXP["cal BT"]["win"]), cal
    say(f"val + dev ({NEV} GT events): per detector ({' / '.join(r['det'] for r in out)}), P0 = mean over the 3 cross-fitted link "
        f"models; mean [range]")
    W = {"甲 P0": [r["p0"]["win"] for r in out], "戊 P0+CJ": [r["cj"]["win"] for r in out], "丁 cal BT": cal[:len(out)]}
    for a, x in W.items():
        say(f"  {a:9s} {line(x)}  mean {np.mean(x):6.2f} [{rng(x):6.2f}]")
    d = np.subtract(W["戊 P0+CJ"], W["甲 P0"])
    say(f"  戊 - 甲    {line(d, '+6.2f')}  mean {d.mean():+6.2f}")
    for k, f in (("bias", "+6.1f"), ("f1", "6.3f")):
        for a, key in (("甲 P0", "p0"), ("戊 P0+CJ", "cj")):
            say(f"  {a:9s} {'bias' if k == 'bias' else 'F1  '} {line([r[key][k] for r in out], f)}")
    for a, key in (("甲 P0", "p0_links"), ("戊 P0+CJ", "cj_links")):
        say(f"  {a:9s} per link model: " + ", ".join(f"{r['det']} {line(r[key], '.2f')}" for r in out))
    t = [jstats(r["joins"]) for r in out]
    say(f"  joins per detector (3 link models summed): {line([x['joins'] for x in t], 'd')}; same GT bee {line([x['same'] for x in t], 'd')}; "
        f"an unowned end {line([x['unowned'] for x in t], 'd')}; dropped {line([x['dropped'] for x in t], 'd')}")
    ok = bool(d.mean() <= 2)
    say(f"  criterion (a): 戊 - 甲 mean {d.mean():+.2f} <= +2: {'PASS' if ok else 'no'}")
    say(f"  checks: 甲 = Table A per detector (window, bias, F1) and per link model{', mean, range, F1 mean' if len(out) == 3 else ''}; "
        f"丁 = Table A cal BT = w154/window_lengths.json")
    return dict(dets=[r["det"] for r in out], gt_events=NEV, per_det=[{k: v for k, v in r.items() if k != "joins"} | jstats(r["joins"]) for r in out],
                cal=W["丁 cal BT"], mean={a: float(np.mean(x)) for a, x in W.items()}, range={a: rng(x) for a, x in W.items()},
                d=d.tolist(), mean_d=float(d.mean()), criterion_a=ok, joins=[r["joins"] for r in out])


def main():
    ap = argparse.ArgumentParser(description="Crossing join after P0 (log §161).")
    ap.add_argument("mode", choices=["selfcheck", "run"])
    ap.add_argument("--out", type=Path, default=Path("D:/beetrack_runs/c161"), help="folder for the .txt and .json")
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    seeds, dets_ = (range(9), DETS) if a.mode == "run" else ((0,), DETS[:1])
    U = units()
    assert not any(ev.startswith(TEST) for u in U.values() for _, ev in u["videos"]) and not any(ev.startswith(TEST) for _, ev in VALDEV)
    t0, lines, res = time.time(), [], {}

    def say(s):
        print(s, flush=True)
        lines.append(s)

    say("self-check: crossing_join toy test passed (on import of beetrack.p0)")
    with tempfile.TemporaryDirectory() as tmp:
        for name, u in U.items():
            L, note = logged(name)
            prob = link_model(u["linker"])
            S = []
            for s in seeds:
                S.append(seed_unit(u, s, prob, tmp))
                print(f"{name} seed {s}: 甲 {S[-1]['p0']['win']:.2f} 戊 {S[-1]['cj']['win']:.2f}, {len(S[-1]['joins'])} joins; "
                      f"{(time.time() - t0) / 60:.1f} min", flush=True)
            check(name, S, L, seeds)
            say(f"self-check {name}: 甲 = logged P0 for seed(s) {list(seeds)} (window 2 decimals, bias{', F1' if L['f1'] else ''}"
                f"{', group means' if len(S) == 9 else ''}){'; P0 MOT files = cache' if u['cache'] else ''}; {note}")
            if a.mode == "run":
                res[name] = report(say, name, u, S, L)
            else:
                res[name] = dict(seed0=dict(p0=S[0]["p0"], cj=S[0]["cj"], cal=L["cal"][0], **jstats(S[0]["joins"])))
                say(f"  seed 0: 甲 {S[0]['p0']['win']:.2f} 戊 {S[0]['cj']['win']:.2f} 丁 {L['cal'][0]:.2f}; {jstats(S[0]['joins'])}")
        res["val+dev"] = valdev(dets_, tmp, say)

    if a.mode == "run":
        mean = {n: res[n]["mean"]["戊 P0+CJ"] - res[n]["mean"]["甲 P0"] for n in U}
        lower, worse = [n for n in CLEAN if mean[n] < 0], [n for n in CLEAN if mean[n] > 2]
        both = [n for n in U if res[n]["criteria"]["criterion 1"]["passed"] and res[n]["criteria"]["criterion 2"]["passed"]]
        d1 = res["val+dev"]["criterion_a"] and len(lower) >= 3 and not worse
        d2 = d1 and len(both) == len(U)
        say("summary: 戊 - 甲 9-seed mean per unit: " + ", ".join(f"{n} {mean[n]:+.2f}" for n in U) +
            f"; val + dev {res['val+dev']['mean_d']:+.2f}")
        say(f"  clean units with 戊 below 甲: {len(lower)} / 4 {lower}; above by more than 2: {worse}; units passing criteria 1 and 2: "
            f"{both}")
        say(f"  §161 decision, mechanical reading: 1 (report-only arm in §125) {'yes' if d1 else 'no'}; 2 (main setting) "
            f"{'yes' if d2 else 'no'}")
        res["summary"] = dict(d_mean=mean, clean_lower=lower, clean_worse_2=worse, pass_1_and_2=both, decision_1=d1, decision_2=d2)
    say(f"self-checks passed ({a.mode})")
    name = "crossing_join" + ("_selfcheck" if a.mode == "selfcheck" else "")
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / f"{name}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (a.out / f"{name}.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {a.out / name}.txt and .json; runtime {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
