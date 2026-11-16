"""Source-only diagnostic for a centred seven-frame, two-observation crossing head."""
from collections import Counter
from itertools import product

import cv2
import numpy as np

from beetrack.evaluate import by_track, frame_match, zone_tracks
from tools.study103.swap_bound import best_disjoint
from tools.study103.swap_score import counts_from_events
from tools.study104.event_overlap import archive_arm, replay_arm, score_arm
from tools.study110.oracle110 import score_oracle


def observations(rows, polygon, nframes):
    """Catalogue adjacent truth flips, then enumerate genuine two-box output witnesses."""
    rows = by_track(rows)
    poly = np.asarray(polygon, dtype=np.float32)
    centres = rows[:, 2:4] + rows[:, 4:6]/2
    inside = np.array([cv2.pointPolygonTest(poly, tuple(map(float, c)), False) >= 0 for c in centres])
    roster, flips = {}, []
    for tid, (direction, start, end, changes) in zone_tracks(rows, poly).items():
        tr = rows[start:end]
        f = tr[:, 0].astype(int)
        if not np.array_equal(f, tr[:, 0]) or not (np.all(np.diff(f) > 0) and 1 <= f[0] <= f[-1] <= nframes):
            raise ValueError("GT segment rows need unique, ordered integral frames inside the study clip")
        final = changes[-1][0] if direction else None
        for pos,(k,d) in enumerate(changes):
            t, u = int(rows[k,0]), int(rows[k-1,0])
            later = int(rows[changes[pos+1][0],0]) if pos+1 < len(changes) else None
            labels={}
            for h in (3,25,100):
                left=np.searchsorted(f,t-h,side="left")
                right=np.searchsorted(f,t+h,side="right")-1
                labels[str(h)]=bool(f[left]<=u and inside[start+left]!=inside[k]
                                    and inside[start+right]==inside[k] and (later is None or later>t+h))
            kind=("final" if k==final else "no_net_event" if direction is None else
                  "reverse_of_net" if d!=direction else "superseded")
            flips.append(dict(id=tid, direction=d, u=u, t=t, canonical=bool(k==final),
                              complete=(4 <= t <= nframes-3), gap=t-u,
                              kind=kind, local_labels=labels, next_flip=later,
                              return_visible=(later is not None and later<=t+3)))
        if final is None:
            continue
        t0, u0 = int(rows[final,0]), int(rows[final-1,0])
        sign = 1 if direction=="in" else -1
        witnesses = []
        for j in np.flatnonzero((abs(f-t0)<=5) & (f>=4) & (f<=nframes-3)):
            if inside[start+j] != (sign == 1):
                continue
            for i in range(j-1,-1,-1):
                if f[j]-f[i] > 3:
                    break
                if inside[start+i] != inside[start+j]:
                    witnesses.append(dict(id=tid, direction=direction, u=int(f[i]), t=int(f[j]),
                                          signed_t=sign*int(f[j]), first=tr[i,:6].tolist(),
                                          last=tr[j,:6].tolist(), canonical=(f[i]==u0 and f[j]==t0)))
        for v in witnesses:
            v["canonical"]=bool(v["canonical"])
        near=[[int(f[j]) if inside[start+j] else -int(f[j]), *tr[j,2:6].tolist()]
              for j in np.flatnonzero((f>=t0-8) & (f<=t0+5))]
        roster[tid] = dict(direction=direction, signed_t=sign*t0, u=u0, t=t0,
                          local_observations=near, witnesses=witnesses)
    return roster, flips


def categories(arm):
    hits = set(arm.matches.values())
    incident = {int(g) for _,g in arm.edges}
    if not hits <= incident:
        raise AssertionError("Selected official matching contains a forbidden edge")
    return {g: ("matched" if g in hits else "edge_unmatched" if g in incident else "no_edge")
            for g in arm.gt}


def counts(signed, n):
    events=np.array([[abs(v), 1 if v>0 else -1] for v in signed],dtype=int).reshape(-1,2)
    return counts_from_events(events,n)


def options(roster, eligible, family):
    """Equal GT and arrival have the same W column. Keep one predeclared physical witness."""
    if family not in ("canonical", "tolerant"):
        raise ValueError("Unknown witness family")
    selected=[]
    for g in sorted(eligible):
        seen=set()
        for v in sorted(roster[g]["witnesses"],key=lambda x:(x["t"], -int(x["canonical"]), -x["u"])):
            if v["t"] in seen or (family=="canonical" and not v["canonical"]):
                continue
            seen.add(v["t"])
            selected.append(v)
    return selected


def minimum_error(base, truth, opts, n):
    frames = [r["signed_t"] for r in opts]
    ids = [r["id"] for r in opts]
    fixed=score_oracle(base,frames,truth,n) if len(set(ids))==len(ids) else None
    if fixed is not None and fixed["certified"]:
        proof=fixed
        picks, optimum = fixed["selected_donor_indices"],fixed["optimal_numerator"]
        method="fixed-time interval LP or attained integer floor"
    else:
        before,target=counts(base,n),counts(truth,n)
        deltas=[counts([f],n) for f in frames]
        # Each option uses one shared GT slot and an exclusive anonymous slot.
        groups={g:k for k,g in enumerate(sorted(set(ids)))}
        pairs=[(groups[g],-i-1,abs(f)) for i,(g,f) in enumerate(zip(ids,frames))]
        picks, raw = best_disjoint(before,target,pairs,deltas,np.ones(len(target)))
        optimum=int(round(raw))
        method="one-time-per-GT integer program with matched primal and solver dual"
        proof={"lower":optimum,"upper":optimum}
        if fixed is not None:
            proof["initial_lp_gap"]=fixed["integer_gap"]
    increment=[frames[k] for k in picks]
    if len({ids[k] for k in picks})!=len(picks):
        raise AssertionError("One true bee would generate multiple reports")
    native=int(abs(counts(base,n)-counts(truth,n)).sum())
    check=int(abs(counts([*base,*increment],n)-counts(truth,n)).sum())
    if check!=optimum or optimum>native:
        raise AssertionError("Physical witness clocks did not attain certified minimum")
    denominator=30*len(truth)
    return dict(method=method,proof=proof,options=len(opts),eligible_bees=len(set(ids)),selected=picks,
                base_numerator=native,min_numerator=optimum,denominator=denominator,
                max_gain_pp=100*(native-optimum)/denominator if denominator else None)


def append_witnesses(p0, chosen):
    if not chosen:
        return p0, {}
    next_id=1+int(max(by_track(p0)[:,1],default=0))
    added=[]
    mapping={}
    for i,w in enumerate(chosen,next_id):
        mapping[i]=w["id"]
        for b in (w["first"],w["last"]):
            added.append([b[0],i,*b[2:6]])
    return np.vstack((p0[:,:6],np.asarray(added,float))),mapping


def physical_metrics(gt, native, extra, mapping, poly, loose_native=None):
    if loose_native is None:
        old=extra[~np.isin(extra[:,1],list(mapping))]
        loose_native=score_arm(gt,old,poly,.3)
    after=score_arm(gt,extra,poly,.5)
    if len(after.events)!=len(native.events)+len(mapping):
        raise AssertionError("Appended two-row traces changed the native event inventory")
    expected=set(native.events)
    if any(after.events.get(i)!=native.events[i] for i in expected):
        raise AssertionError("A native event changed direction, time or geometry")
    result={}
    for tau in (.5,.3):
        arm=after if tau==.5 else score_arm(gt,extra,poly,tau)
        tp=len(arm.matches); p=len(arm.events); g=len(arm.gt)
        recovered=[]
        ownership={}
        if mapping:
            rr=by_track(extra)
            owners=frame_match(by_track(gt),rr,thr=tau)
            for pid,gid in mapping.items():
                both=owners[rr[:,1]==pid]
                if len(both)!=2:
                    raise ValueError("New event must contain exactly two real boxes")
                ownership[str(pid)]=list(map(int,both))
                if np.all(both==gid) and arm.matches.get(pid)==gid:
                    recovered.append(gid)
        result[str(tau)]=dict(tp=tp,fp=p-tp,fn=g-tp,f1=2*tp/max(1,p+g),
                             added_matched_to_own=sum(arm.matches.get(i)==j for i,j in mapping.items()),
                             added_with_own_edge=sum([i,j] in arm.edges for i,j in mapping.items()),
                             prior_matched_bees_lost=sorted(set((native if tau==.5 else loose_native).matches.values())
                                                           -set(arm.matches.values())),
                             prior_pred_ids_lost=sorted(set((native if tau==.5 else loose_native).matches)-set(arm.matches)),
                             strict_same_bees=sorted(recovered),matching=sorted(map(list,arm.matches.items())),
                             edges=arm.edges,added_targets=dict(mapping),added_frame_owners=ownership)
    return result


def analyze(gt, p0, polygon, nframes, archived=None, physical_sink=None):
    if gt.ndim!=2 or p0.ndim!=2 or gt.shape[1]<6 or p0.shape[1]<6:
        raise ValueError("Expected physical MOT rows with frame, ID and camera xywh")
    gt, p0 = gt[:,:6],p0[:,:6]
    raw, flips = observations(gt,polygon,nframes)
    base=score_arm(gt,p0,polygon,.5)
    loose=score_arm(gt,p0,polygon,.3)
    gt_times=[raw[k]["signed_t"] for k in sorted(raw)]
    signed=[t if d=="in" else -t for d,t,_ in base.events.values()]
    graph=archive_arm(base)
    if graph["gt"]!=[[g,raw[g]["signed_t"]] for g in sorted(raw)]:
        raise AssertionError("Truth and official final same-direction crossing differ")
    if archived is not None:
        for a,k in [(base,"edges05"),(loose,"edges03")]:
            z=archive_arm(a)
            predictions=[[i,t,*(round(float(c),2) for c in box)] for i,t,*box in z["predictions"]]
            if z["gt"]!=archived["gt"] or predictions!=archived["predictions"] or z["edges"]!=archived[k]:
                raise AssertionError("Wrong native P0, GT, camera zone or event-graph freeze")
            restored=replay_arm(dict(gt=archived["gt"],predictions=archived["predictions"],edges=archived[k]))
            if a.matches!=restored.matches or a.variable_gt!=restored.variable_gt:
                raise AssertionError("Restored maximum matching or tie-variability disagrees")
    primary=categories(base)
    secondary=categories(loose)
    inventory=[dict(id=g,**item,category05=primary[g],category03=secondary[g],
                    variable_match05=g in base.variable_gt,variable_match03=g in loose.variable_gt)
               for g,item in sorted(raw.items())]
    results={}
    replayed={():physical_metrics(gt,base,p0,{},polygon,loose)}
    for family, group in product(("canonical","tolerant"),("no_edge","all_unmatched")):
        eligible=[g for g,v in primary.items() if v=="no_edge" or (group=="all_unmatched" and v=="edge_unmatched")]
        opts=options(raw,eligible,family)
        opt=minimum_error(signed,gt_times,opts,nframes)
        chosen=[opts[k] for k in opt["selected"]]
        key=tuple((r["id"],r["u"],r["t"]) for r in chosen)
        extra,mapping=append_witnesses(p0,chosen)
        if physical_sink is not None:
            extra,receipt=physical_sink(f"{family}_{group}",extra,2*len(chosen))
            opt["physical_mot"]=receipt
        if key not in replayed:
            replayed[key]=physical_metrics(gt,base,extra,mapping,polygon,loose)
        opt.update(chosen=[{k:v[k] for k in ("id","direction","u","t","canonical")} for v in chosen],
                   official_f1_of_this_witness=replayed[key])
        results[f"{family}_{group}"]=opt
    return dict(nframes=nframes, polygon=np.asarray(polygon).tolist(),event_inventory=inventory, adjacent_flips=flips,
                categories05=dict(Counter(primary.values())), categories03=dict(Counter(secondary.values())),
                native_events=graph["predictions"],gt_events=graph["gt"],edges05=graph["edges"],
                edges03=loose.edges,baseline=replayed[()], bounds=results)
