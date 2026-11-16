"""Replay §115 count errors and event-matching scores from signed frames and sparse legal edges."""
from bisect import bisect_right
from collections import Counter
from fractions import Fraction

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching


ARMS = ("canonical", "geometry")
TAUS = ("0.5", "0.3")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def _events(pairs, nframes):
    out = {}
    for pid, t in pairs:
        require(type(pid) is int and pid not in out and type(t) is int and 1 <= abs(t) <= nframes,
                "Duplicate event ID or invalid signed frame")
        out[pid] = t
    return out


def _grids(events, nframes):
    """Enumerate the 30 offset grids, counting events in each physical (left,right] interval."""
    by_dir = [sorted(abs(f) for f in events if f*sign > 0) for sign in (1,-1)]
    grids = []
    for offset in range(0,750,25):
        cuts = [0, *range(750-offset, nframes, 750), nframes]
        grids.append([[bisect_right(times,hi)-bisect_right(times,lo) for times in by_dir]
                      for lo,hi in zip(cuts,cuts[1:])])
    return grids


def _window(pred, truth, scale, gt_counts):
    """Predictions use integer units of 1/scale. Accumulate absolute errors before dividing."""
    err = [[sum(abs(p[d]-scale*g[d]) for p,g in zip(ps,gs)) for d in range(2)]
           for ps,gs in zip(pred,truth)]
    ng = sum(gt_counts)
    w = Fraction(100*sum(map(sum,err)),30*scale*ng) if ng else None
    totals = [Fraction(sum(p[d] for p in pred[0]),scale) for d in range(2)]
    return (dict(w_percent=float(w) if w is not None else None, count_in_out=list(map(float,totals)),
                 bias_in_out=[float(p-g) for p,g in zip(totals,gt_counts)]),
            dict(scale=scale, denominator_per_offset=scale*ng, abs_error_in_out_by_offset=err))


class Graph:
    """A fixed set of prediction and GT events; each restricted graph is matched anew."""
    def __init__(self, predictions, truth, pairs):
        self.pids, self.gids = sorted(predictions), sorted(truth)
        self.pi = {p:i for i,p in enumerate(self.pids)}
        gi = {g:i for i,g in enumerate(self.gids)}
        self.psign = np.array([predictions[p]>0 for p in self.pids], dtype=bool)
        self.gsign = np.array([truth[g]>0 for g in self.gids], dtype=bool)
        self.edges = set(map(tuple,pairs))
        require(len(self.edges) == len(pairs), "Duplicate legal graph edge")
        rows, cols = [], []
        for p,g in self.edges:
            require(p in self.pi and g in gi and (predictions[p]>0)==(truth[g]>0),
                    "Illegal endpoint or direction in event graph")
            rows.append(self.pi[p])
            cols.append(gi[g])
        self.adj = csr_matrix((np.ones(len(rows), dtype=bool),(rows,cols)),shape=(len(self.pids),len(self.gids)))

    def score(self, chosen=None, detailed=True):
        pids = self.pids if chosen is None else sorted(chosen)
        require(len(set(pids)) == len(pids), "Repeated selected event")
        rows = [self.pi[p] for p in pids]
        graph = self.adj if chosen is None else self.adj[rows,:]
        if graph.nnz:
            partners = maximum_bipartite_matching(graph, perm_type="column")
            gt_hits = partners[partners >= 0]
        else:
            gt_hits = np.array([],dtype=int)
        tp, npred, ng = len(gt_hits), len(pids), len(self.gids)
        ans = dict(tp=tp, fp=npred-tp, fn=ng-tp, f1=2*tp/max(1,npred+ng))
        if detailed:
            p_in, g_in, tp_in = int(self.psign[rows].sum()), int(self.gsign.sum()), int(self.gsign[gt_hits].sum())
            ans["by_direction"] = {}
            for name,p,g,h in (("in",p_in,g_in,tp_in),("out",npred-p_in,ng-g_in,tp-tp_in)):
                ans["by_direction"][name] = dict(tp=h, fp=p-h, fn=g-h)
        return ans


def _blocks(pack, parents, signed):
    blocks, allowed, canon, geom = [], [], [], []
    for d,sign in (("in",1),("out",-1)):
        ordered = sorted((key for key,t in signed.items() if t*sign>0),key=lambda key:(abs(signed[key]),*key))
        declared = pack["manifest"]["blocks"][d]
        require(len(declared)==(len(ordered)+2)//3, "Wrong number of rank blocks")
        for i,b in enumerate(declared):
            members = ordered[3*i:3*i+3]
            require(list(map(tuple,b["members"]))==members, "Rank block differs from native event stream")
            if len(members)==1:
                require(b["canonical"] is None and b["geometry"] is None and b["evidence"]==[], "Invalid singleton")
                continue
            middle=members[1]
            sec=(abs(signed[middle])-1)//25
            group=[key for key in members if (abs(signed[key])-1)//25==sec]
            require(tuple(b["canonical"])==middle and [tuple(x["member"]) for x in b["evidence"]]==group
                    and tuple(b["geometry"]) in group, "Frozen block eligibility or selector differs")
            blocks.append(sign)
            allowed.append([parents[k] for k in group])
            canon.append(parents[middle])
            geom.append(parents[tuple(b["geometry"])])
    return blocks,allowed,canon,geom


def _interval_stats(pooled, chosen, sign):
    ticks = Counter(abs(t) for t in pooled if t*sign>0)
    emitted = Counter(abs(t) for t in chosen if t*sign>0)
    prefix = [0]
    for f in sorted(ticks):
        prefix.append(prefix[-1]+3*emitted[f]-ticks[f])
    return dict(total=sum(emitted.values()), max_interval_abs_thirds=max(prefix)-min(prefix),
                clip_bias_thirds=prefix[-1])


def _counts(pack, signed, selected, truth, per_seed):
    n=pack["meta"]["nframes"]
    pooled_times=list(signed.values())
    pooled,gt=_grids(pooled_times,n),_grids(truth.values(),n)
    picked = {arm:[signed[p] for p in ids] for arm,ids in selected.items()}
    grid=_grids(picked["canonical"],n)
    require(grid==_grids(picked["geometry"],n), "Canonical and geometry selectors have different window counts")
    flat_p, flat_i=[sum(m,[]) for m in (pooled,grid)]
    residual=[[3*i-p for i,p in zip(ii,pp)] for ii,pp in zip(flat_i,flat_p)]
    budget=sum(abs(x) for row in residual for x in row)
    largest=max(abs(x) for row in residual for x in row)
    require(largest<=2,"E3 per-cell 2/3 bound violated")
    by={}
    for d,sign in (("in",1),("out",-1)):
        by[d]={"pooled":sum(t*sign>0 for t in pooled_times)}
        for a in ARMS:
            rec=_interval_stats(pooled_times,picked[a],sign)
            require(abs(rec["clip_bias_thirds"])<=1 and rec["max_interval_abs_thirds"]<=(2 if a=="canonical" else 4),
                    "Whole-direction or arbitrary-interval bound violated")
            by[d][a]=rec
    expected=dict(nframes=n,pooled_window_counts=flat_p,integer_window_counts=flat_i,num_time_bins=len(flat_p),
                  max_cell_abs_thirds=largest,delta_l1_thirds=budget,by_direction=by,
                  changed_representatives=sum(a!=b for a,b in zip(selected["canonical"],selected["geometry"])))
    require(all(pack["counts_proof"][k]==v for k,v in expected.items()),"Frozen count certificate differs")
    gt_counts=[sum(t>0 for t in truth.values()),sum(t<0 for t in truth.values())]
    integer,ip=_window(grid,gt,1,gt_counts)
    e3,ep=_window(pooled,gt,3,gt_counts)
    native,proof={},{}
    for s,events in per_seed.items():
        count,proof[s]=_window(_grids(events.values(),n),gt,1,gt_counts)
        native[s]={"count":count}
    offset_resid=[[sum(abs(3*ii[d]-pp[d]) for ii,pp in zip(i,p)) for d in (0,1)] for i,p in zip(grid,pooled)]
    ng=len(truth)
    diff_num=3*sum(map(sum,ip["abs_error_in_out_by_offset"]))-sum(map(sum,ep["abs_error_in_out_by_offset"]))
    require(abs(diff_num)<=budget,"Truth-dependent difference violates the triangle bound")
    return dict(gt_in_out=gt_counts,e3=e3,integer=integer,native=native,
                w_delta_vs_e3_pp=100*diff_num/(3*30*ng) if ng else None,
                w_delta_bound_pp=100*budget/(3*30*ng) if ng else None,
                offset_proof=dict(offsets_frames=list(range(0,750,25)),native=proof,integer=ip,e3=ep,
                                  integer_vs_e3_abs_thirds_in_out=offset_resid))


def check_witness(pairs, edges, eligible_predictions, optimum):
    """Verify a reported matching attains optimum, accepting all optimal tie witnesses."""
    allowed, valid = set(eligible_predictions), set(map(tuple,edges))
    matched=list(map(tuple,pairs))
    require(len(matched)==optimum and len({p for p,_ in matched})==optimum
            and len({g for _,g in matched})==optimum
            and all(p in allowed and (p,g) in valid for p,g in matched), "Invalid claimed optimum witness")


def score_graphs(pack, graphdata):
    """Recompute primary metric fields and exact per-offset error components.

    Pools use exported physical parent IDs. Standalone graphs use original
    native IDs within one seed. An optional oracle_selections supplies a physical
    witness for each IoU, and oracle_matchings / oracle_block_matchings can
    supply its corresponding legal matched edges. Frame-level owner-change
    counts are supplied observations, not inferable from event graphs.
    """
    m,meta=pack["manifest"],pack["meta"]
    seeds,n=meta["seeds"],meta["nframes"]
    require(len(set(seeds))==3 and len(seeds)==3 and n>0,"Invalid detector bank or clip length")
    truth=_events(graphdata["gt"],n)
    parents={(s,i):base for s,i,base,_ in m["id_blocks"]}
    require(len(parents)==len(m["id_blocks"]) and len(set(parents.values()))==len(parents),
            "Duplicate physical parent mapping")
    per_seed={}
    for seed in seeds:
        rows=[o["row"][:2] for o in m["observations"] if o["seed"]==seed]
        per_seed[str(seed)]=_events(rows,n)
    signed={(int(s),p):t for s,events in per_seed.items() for p,t in events.items()}
    require(len(signed)==len(m["observations"]) and signed.keys()<=parents.keys(),"Native event parent missing")
    pooled={parents[key]:t for key,t in signed.items()}
    block_sign,allowed,canon,geom=_blocks(pack,parents,signed)
    selectors=pack["selectors"]
    require(len(selectors["controls"])==99,"Expected 99 quota controls")
    def convert(items):
        require(len(items)==len(allowed),"Selector did not choose one representative per block")
        ids=[parents[tuple(k)] for k in items]
        require(len(set(ids))==len(ids) and all(p in b for p,b in zip(ids,allowed)),
                "Selector includes an ineligible member")
        return ids
    selected={a:convert(selectors[a]) for a in ARMS}
    require(canon==selected["canonical"] and geom==selected["geometry"],"Contradictory frozen selectors")
    controls=[convert(choices) for choices in selectors["controls"]]
    if "quota" in pack:
        prefix=[314159,{"source":0,"target":1,"deploy":2}[meta["role"]],meta["fold"],
                meta["first_seed"],{"head":0,"tail":1,"custom":2}[meta["part"]]]
        quota=pack["quota"]
        require(quota["generator"]=="numpy.PCG64" and quota["seed_prefix"]==prefix
                and quota["replicates"]==list(range(99)) and quota["pool_sizes"]==list(map(len,allowed)),
                "Frozen quota generator or block sizes differ")
        for rep,actual in enumerate(controls):
            rng=np.random.Generator(np.random.PCG64([*prefix,rep]))
            require(actual==[pool[int(rng.integers(len(pool)))] for pool in allowed],
                    f"Control {rep} differs from its registered draw")
    # All grid boundaries are integer seconds. Equality in every second and
    # direction proves all 99 controls have exactly the canonical integer W.
    def seconds(ids):
        return Counter(((abs(pooled[p])-1)//25,pooled[p]>0) for p in ids)
    signature=seconds(canon)
    require(all(seconds(c)==signature for c in controls),"Quota control changed a directional time bin")
    out=_counts(pack,pooled,selected,truth,per_seed)
    out.update(canonical={},geometry={},lean={a:{} for a in ARMS},
               controls=[{} for _ in controls],selector_ceiling={})
    slot={p:b for b,ids in enumerate(allowed) for p in ids}
    require(len(slot)==sum(map(len,allowed)),"Eligible block sets intersect")
    for tau in TAUS:
        edges=graphdata["pool"][tau]
        pool=Graph(pooled,truth,edges)
        for a in ARMS:
            ids=selected[a]
            out[a][tau]=pool.score(ids)
            lean=Graph({p:pooled[p] for p in ids},truth,graphdata["lean"][a][tau])
            out["lean"][a][tau]=lean.score()
        for scores,ids in zip(out["controls"],controls):
            scores[tau]=pool.score(ids,detailed=False)
        for seed in seeds:
            s=str(seed)
            solo=Graph(per_seed[s],truth,graphdata["native"][s][tau])
            changed=graphdata["changed_box_owners"][s][tau]
            require(type(changed) is int and changed>=0,"Invalid observed frame ownership count")
            mask=[parents[seed,p] for p in per_seed[s]]
            out["native"][s][tau]=dict(standalone=solo.score(),in_full_pool=pool.score(mask),changed_box_owners=changed)
        # Disjoint blocks give an exact upper bound. Each matched block can
        # select one of its witnessing members without consuming another block's event.
        block_edges=sorted({(slot[p],g) for p,g in pool.edges if p in slot})
        oracle=Graph(dict(enumerate(block_sign)),truth,block_edges).score()
        if "oracle_selections" in graphdata:
            selected_oracle=convert(graphdata["oracle_selections"][tau])
            require(pool.score(selected_oracle)==oracle,"Reported physical selector fails to attain the exact ceiling")
            if "oracle_matchings" in graphdata:
                check_witness(graphdata["oracle_matchings"][tau],edges,selected_oracle,oracle["tp"])
            if "oracle_block_matchings" in graphdata:
                matches=graphdata["oracle_block_matchings"][tau]
                check_witness(matches,block_edges,range(len(allowed)),oracle["tp"])
                require(all((selected_oracle[b],g) in pool.edges for b,g in matches),
                        "Reported physical event does not witness its block edge")
        for field in ("tp","f1"):
            oracle[field+"_headroom"]={a:oracle[field]-out[a][tau][field] for a in ARMS}
        out["selector_ceiling"][tau]=oracle
    return out


def self_check():
    # Boundary frame 750 closes one cell at offset zero, while 751 opens its own.
    time=_grids([749,750,751,-1],751)
    assert time[0]==[[2,1],[1,0]] and time[1]==[[0,1],[3,0]]
    truth=_grids([750,-1],751)
    summary,proof=_window(time,truth,3,[1,1])
    assert proof["abs_error_in_out_by_offset"][0]==[2,2]
    assert summary["count_in_out"]==[1.0,1/3]
    p={10:7,11:7,12:8,13:-9}
    g={100:7,101:9,102:-9}
    pairs=[[10,100],[10,101],[11,100],[12,101],[13,102]]
    graph=Graph(p,g,pairs)
    assert graph.score([10,11,13])["tp"]==3 and graph.score([11,12,13])["tp"]==3
    assert graph.score([11,13])["by_direction"]["in"]==dict(tp=1,fp=0,fn=1)
    assert Graph({},g,[]).score()["f1"]==0
    check_witness([[10,101],[11,100],[13,102]],pairs,p,3)
    print("Offset boundaries, fractional counts, graph reassignment and optimal witness verified.")


if __name__=="__main__":
    self_check()
