"""Summarize the frozen §116 event endpoints, keeping physical clip and GT segment identity."""
from itertools import product
import math


ARMS = ("U","L","C","H")
SEEDS = ("0","1","2")
MODES = ("original","wrapped")
THRESHOLDS = ("0.5","0.3")
PATTERNS = tuple("".join(bits) for bits in product("01",repeat=4))
DIRECTIONS = ("all","in","out")
ENDPOINTS = ("both","u","t")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _population(events):
    directions = {}
    for e in events:
        tid, d, u, t = (e[k] for k in ("gt_tid","dir","before","arrival"))
        require(type(tid) is int and tid not in directions and d in ("in","out")
                and type(u) is int and type(t) is int and 1 <= u < t, "Invalid or repeated sampled GT segment")
        directions[tid] = d
    require(len(directions)<=16, "Frozen sample exceeds the registered per-clip maximum")
    return directions


def _witnesses(seed, directions):
    require(set(seed)==set(ARMS), "A detector arm is missing or unexpected")
    expected = {str(tid) for tid in directions}
    for arm in ARMS:
        require(set(seed[arm])==set(MODES), "Missing or unexpected NMS view")
        for mode in MODES:
            thresholds=seed[arm][mode]
            require(set(thresholds)==set(THRESHOLDS), "Missing or unexpected official IoU")
            for threshold in THRESHOLDS:
                record=thresholds[threshold]
                require(set(record["ends"])==expected and "regions" in record,
                        "Endpoint or regional witness missing from the frozen sample")
                for ends in record["ends"].values():
                    require(len(ends)==2 and all(type(x) is bool or x is None for x in ends),
                            "Each frame endpoint needs a boolean or an explicit missing observation")
            for tid in expected:
                high,low=(thresholds[t]["ends"][tid] for t in THRESHOLDS)
                require(not any(a is True and b is False for a,b in zip(high,low)),
                        "A high-IoU hit became a low-IoU miss under the same frame assignment")


def _bit(ends, endpoint):
    if endpoint=="both":
        # A witnessed miss suffices to rule out two-endpoint success.
        flag=False if False in ends else None if None in ends else True
    else:
        flag=ends[0 if endpoint=="u" else 1]
    return "?" if flag is None else str(int(flag))


def _table(rows):
    """Keep denominators, including observable U/L and U/H contrasts in incomplete rows."""
    complete={k:0 for k in PATTERNS}
    incomplete={}
    categories={k:set() for k in (*ARMS,"gain","loss","h_only","rescued_l","rescued_c")}
    # For the controls report which subset of the H-only cases each rescued.
    recover={x+y:0 for x in "01" for y in "01"}
    uncertain_recovery={}
    for tid,bits in rows.items():
        if "?" in bits:
            incomplete[bits]=incomplete.get(bits,0)+1
        else:
            complete[bits]+=1
        u,l,c,h=bits
        for arm,b in zip(ARMS,bits):
            if b=="1":
                categories[arm].add(tid)
        if u=="0" and l=="1":
            categories["gain"].add(tid)
        if u=="1" and l=="0":
            categories["loss"].add(tid)
        if u=="0" and h=="1":
            categories["h_only"].add(tid)
            if l=="1":
                categories["rescued_l"].add(tid)
            if c=="1":
                categories["rescued_c"].add(tid)
            pair=l+c
            if "?" in pair:
                uncertain_recovery[pair]=uncertain_recovery.get(pair,0)+1
            else:
                recover[pair]+=1
    result=dict(n=len(rows),known_four=sum(complete.values()),four_bits=complete,incomplete_four=incomplete,
                successes={a:len(categories[a]) for a in ARMS},
                known_by_arm={a:sum(bits[i]!="?" for bits in rows.values()) for i,a in enumerate(ARMS)},
                u_to_l=dict(gain=len(categories["gain"]),loss=len(categories["loss"]),
                            known=sum("?" not in b[:2] for b in rows.values())),
                h_only=dict(n=len(categories["h_only"]),l=len(categories["rescued_l"]),c=len(categories["rescued_c"]),
                            lc=recover,incomplete_lc=uncertain_recovery))
    return result,categories


def _overlap(groups):
    votes = set.union(*groups)
    return dict(union=sorted(votes), intersection=sorted(set.intersection(*groups)),
                at_least_two=sorted(t for t in votes if sum(t in group for group in groups)>=2))


def _seed_gate(primary_ends):
    primary=primary_ends["both"]
    s=primary["successes"]
    tests=dict(complete=all(r["known_four"]==r["n"] for r in primary_ends.values()),
               l_gt_u=s["L"]>s["U"],l_gt_c=s["L"]>s["C"],
               has_h_only=primary["h_only"]["n"]>=1,l_rescues_h_only=primary["h_only"]["l"]>=1)
    reasons=[k for k,v in tests.items() if not v]
    return dict(passes=not reasons,conditions=tests,failed=reasons,successes=s,
                h_only=primary["h_only"]["n"],rescued_l=primary["h_only"]["l"])


def _clip_gate(n, dose, seed_results):
    g,delta,points=(dose[k] for k in ("g","delta_j","n_points"))
    require(type(points) is int and points>=0 and type(g) in (float,int) and math.isfinite(g) and g>=1
            and (delta is None and points==0 or type(delta) in (float,int) and math.isfinite(delta)),
            "Invalid entrance sampling dose")
    passed=[s for s in SEEDS if seed_results[s]["gate"]["passes"]]
    tests=dict(sample_of_16=n==16,nonidentity_magnification=g!=1,has_dose_points=points>0,
               delta_j_at_least_0_10=delta is not None and delta>=0.10,at_least_two_seeds=len(passed)>=2)
    return dict(passes=all(tests.values()),conditions=tests,failed=[k for k,v in tests.items() if not v],
                passing_seeds=passed,observed_n=n,required_n=16)


def summarize(clips, observations, dose):
    """Use original-coordinate NMS at IoU 0.5 for gates; both NMS views and IoUs retain all tables."""
    require(len(clips)==2 and set(clips)==set(observations)==set(dose),"Expected two complete physical source clips")
    require({r["fold"] for r in clips.values()}=={33,35},"Expected one source clip per detector fold")
    out={}
    for clip in sorted(clips):
        info=clips[clip]
        directions=_population(info["events"])
        obs=observations[clip]
        require(set(obs)==set(SEEDS),"Expected seeds 0, 1 and 2 for the same frozen sample")
        seed_results={}
        seen={}
        for seed in SEEDS:
            _witnesses(obs[seed],directions)
            views={}
            for mode in MODES:
                views[mode]={}
                for tau in THRESHOLDS:
                    arms={a:obs[seed][a][mode][tau] for a in ARMS}
                    tables={}
                    for direction in DIRECTIONS:
                        tids=[tid for tid,d in directions.items() if direction=="all" or d==direction]
                        tables[direction]={}
                        for end in ENDPOINTS:
                            bits={t:"".join(_bit(arms[a]["ends"][str(t)],end) for a in ARMS) for t in tids}
                            table,groups=_table(bits)
                            tables[direction][end]=table
                            seen.setdefault((mode,tau,direction,end),[]).append(groups)
                    views[mode][tau]=dict(tables=tables,regions={a:arms[a]["regions"] for a in ARMS})
            primary=views["original"]["0.5"]["tables"]["all"]
            seed_results[seed]=dict(gate=_seed_gate(primary),views=views)
        across={}
        for (mode,tau,direction,end),groups in seen.items():
            target=across.setdefault(mode,{}).setdefault(tau,{}).setdefault(direction,{})
            signatures={k:_overlap([g[k] for g in groups]) for k in groups[0]}
            signatures["ever_gain_and_ever_loss"]=sorted(set(signatures["gain"]["union"])&set(signatures["loss"]["union"]))
            target[end]=signatures
        gate=_clip_gate(len(directions),dose[clip],seed_results)
        out[clip]=dict(fold=info["fold"],n=len(directions),sample_ids=sorted(directions),dose=dose[clip],
                       seeds=seed_results,across_seeds=across,gate=gate)
    advance=[c for c,v in out.items() if v["gate"]["passes"]]
    return dict(arm_order=list(ARMS),clips=out,
                decision=dict(advance_full_source=advance,stop_fixed_1_5=not advance,
                              both_clips_pass=len(advance)==len(clips)))


def self_check():
    clips,observed,doses={},{},{}
    for clip,fold in (("a",33),("b",35)):
        clips[clip]=dict(fold=fold,events=[dict(gt_tid=i,dir="in" if i<8 else "out",before=i+1,arrival=i+2)
                                          for i in range(16)])
        doses[clip]=dict(g=1.5,delta_j=0.10 if clip=="a" else math.nextafter(.10,0),n_points=7)
        observed[clip]={}
        for s in SEEDS:
            # Shared physical bee 0 gains with seeds 0/1 but loses with seed 2.
            patterns=(["0101","0111","1010"] if s!="2" else ["1010","0000","1000"])+["0000"]*13
            obs={}
            for a,arm in enumerate(ARMS):
                obs[arm]={}
                for mode in MODES:
                    obs[arm][mode]={}
                    for tau in THRESHOLDS:
                        ends={str(i):[bool(int(bits[a])),bool(int(bits[a]))] for i,bits in enumerate(patterns)}
                        # A single out-bound endpoint should not produce a paired success.
                        ends["12"]=[arm=="U",False]
                        if mode=="wrapped":
                            ends["0"]=[False,False]
                        obs[arm][mode][tau]=dict(ends=ends,regions=dict(test=3))
            observed[clip][s]=obs
    summary=summarize(clips,observed,doses)
    assert summary["decision"]["advance_full_source"]==[]  # C ties L, so no seed passes.
    for clip in clips:
        for s in ("0","1"):
            for mode in MODES:
                for tau in THRESHOLDS:
                    observed[clip][s]["C"][mode][tau]["ends"]["1"]=[False,False]
    summary=summarize(clips,observed,doses)
    assert summary["decision"]["advance_full_source"]==["a"] and summary["clips"]["a"]["gate"]["passing_seeds"]==["0","1"]
    assert summary["clips"]["b"]["gate"]["failed"]==["delta_j_at_least_0_10"]
    a=summary["clips"]["a"]
    tab=a["seeds"]["0"]["views"]["original"]["0.5"]["tables"]
    assert len(tab["all"]["both"]["four_bits"])==16 and tab["all"]["both"]["four_bits"]["0000"]==13
    assert tab["all"]["both"]["h_only"]==dict(n=2,l=2,c=0,lc={"00":0,"01":0,"10":2,"11":0},incomplete_lc={})
    assert tab["all"]["both"]["u_to_l"]==dict(gain=2,loss=1,known=16)
    assert tab["out"]["u"]["successes"]["U"]==1 and tab["out"]["both"]["successes"]["U"]==0
    groups=a["across_seeds"]["original"]["0.5"]["all"]["both"]
    assert groups["gain"]==dict(union=[0,1],intersection=[],at_least_two=[0,1])
    assert groups["ever_gain_and_ever_loss"]==[0]
    observed["a"]["1"]["U"]["original"]["0.5"]["ends"]["12"]=[False,None]
    failed=summarize(clips,observed,doses)
    assert failed["decision"]["stop_fixed_1_5"] and "complete" in failed["clips"]["a"]["seeds"]["1"]["gate"]["failed"]
    doses["a"]=dict(g=1.,delta_j=None,n_points=0)
    failed=summarize(clips,observed,doses)
    assert "has_dose_points" in failed["clips"]["a"]["gate"]["failed"]
    print("Sample IDs, strict contrasts, dose boundary, cross-seed overlap and incomplete endpoints verified.")


if __name__=="__main__":
    self_check()
