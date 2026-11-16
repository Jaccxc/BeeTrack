"""Independent §115 verification from three native event descriptor tables; no MOT or GT I/O."""
from fractions import Fraction
from operator import index

import numpy as np


def check(condition, reason):
    if not condition:
        raise ValueError(reason)


def centibox(values):
    units = [100*Fraction(str(v)) for v in values]
    check(len(units)==4 and all(v.denominator==1 for v in units), 'Expected four exact centipixel coordinates')
    x,y,r,b = map(int,units)
    check(x<r and y<b, 'Invalid descriptor box')
    return x,y,r,b


def support_half(a,b):
    x = max(0,min(a[2],b[2])-max(a[0],b[0]))
    y = max(0,min(a[3],b[3])-max(a[1],b[1]))
    intersection = x*y
    union = (a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-intersection
    return 2*intersection>=union


def prefixes(frames, n):
    """At index i, counts both directions through the end of frame i, including ties."""
    a = np.asarray(frames,dtype=np.int64)
    check(all(1<=abs(int(f))<=n for f in a),'Frame outside the clip')
    positive = np.bincount(a[a>0],minlength=n+1)
    negative = np.bincount(-a[a<0],minlength=n+1)
    return np.column_stack((positive.cumsum(),negative.cumsum()))


def windows(prefix):
    """Compute partial end windows by subtracting counts at physical frame boundaries."""
    n = len(prefix)-1
    result = []
    for offset in range(0,750,25):
        cuts = [0,*range(750-offset,n,750),n]
        result.append(np.diff(prefix[cuts],axis=0))
    return np.concatenate(result)


def verify_counts(pack, descriptors):
    """Check selector evidence, RNG draws and every frozen count field. Return a compact audit."""
    meta,manifest=pack['meta'],pack['manifest']
    seeds=sorted(descriptors)
    check(len(seeds)==3 and seeds==manifest['seeds']==meta['seeds'] and seeds[0]==meta['first_seed'],
          'Native source seed provenance mismatch')
    n=index(meta['nframes'])
    check(n>=1,'Nonpositive clip length')
    observed,frames,boxes={},{},{}
    for seed,records in descriptors.items():
        for row in records:
            key=(seed,index(row[0]))
            f=index(row[1])
            check(key not in frames and 1<=abs(f)<=n,'Repeated native identity or invalid signed frame')
            observed[key]=list(row)
            frames[key]=f
            boxes[key]=centibox(row[2:])
    expected_observations=[{'seed':s,'row':observed[s,i]} for s,i in sorted(observed)]
    check(manifest['observations']==expected_observations, 'Native descriptor order, identity, time or box mismatch')

    canonical,chosen,pools=[],[],[]
    for direction,positive in [('in',True),('out',False)]:
        ordered=sorted((k for k,f in frames.items() if (f>0)==positive),key=lambda k:(abs(frames[k]),*k))
        block_list=manifest['blocks'][direction]
        check(len(block_list)==(len(ordered)+2)//3,'Incorrect number of rank blocks')
        for pos,stored in enumerate(block_list):
            block=ordered[3*pos:3*pos+3]
            check(stored['members']==list(map(list,block)),'Block membership differs from time-seed-ID rank')
            if len(block)==1:
                check(stored['canonical'] is None and stored['geometry'] is None and stored['evidence']==[],
                      'A final singleton cannot emit any representative')
                continue
            second=block[1]
            slot=[k for k in block if (abs(frames[k])-1)//25==(abs(frames[second])-1)//25]
            evidence,ranking=[],[]
            for k in slot:
                others=sorted(p for p in ordered if p[0]!=k[0] and abs(frames[p]-frames[k])<=5
                              and support_half(boxes[p],boxes[k]))
                votes=len({p[0] for p in others})
                evidence.append(dict(member=list(k),other_seeds=votes,supporters=list(map(list,others))))
                ranking.append((-votes,abs(block.index(k)-1),*k))
            winner=min(ranking)[2:]
            check(stored['evidence']==evidence and stored['canonical']==list(second)
                  and stored['geometry']==list(winner), 'Support, ties or representative provenance differs')
            canonical.append(second)
            chosen.append(winner)
            pools.append(slot)

    selectors=pack['selectors']
    check(selectors['canonical']==list(map(list,canonical)) and selectors['geometry']==list(map(list,chosen)),
          'Contradictory selector records')
    rng_seed=[314159, {'source':0,'target':1,'deploy':2}[meta['role']],meta['fold'],meta['first_seed'],
              {'head':0,'tail':1,'custom':2}[meta['part']]]
    check(pack['quota']==dict(generator='numpy.PCG64',seed_prefix=rng_seed,replicates=list(range(99)),
                              draws='one uniform in original block order, in then out, including singleton choice sets',
                              pool_sizes=list(map(len,pools)),numpy=np.__version__), 'Quota RNG metadata differs')
    check(len(selectors['controls'])==99,'Missing control selector')
    pooled=prefixes(list(frames.values()),n)
    pooled_windows=windows(pooled)
    reference=windows(prefixes([frames[k] for k in canonical],n))
    stats={}
    control_max=0
    for rep in range(-2,99):
        if rep==-2:
            pick=canonical
        elif rep==-1:
            pick=chosen
        else:
            rng=np.random.Generator(np.random.PCG64([*rng_seed,rep]))
            pick=[slot[rng.integers(len(slot))] for slot in pools]
            check(selectors['controls'][rep]==list(map(list,pick)),f'Quota draw differs at replicate {rep}')
        timeline=prefixes([frames[k] for k in pick],n)
        check(np.array_equal(windows(timeline),reference),f'Window counts differ for selector {rep}')
        residual=3*timeline-pooled
        span=np.ptp(residual,axis=0)
        check(np.all(span<=(2 if rep==-2 else 4)) and np.all(abs(residual[-1])<=1),
              'Arbitrary-interval or full-clip count guarantee fails')
        if rep<0:
            stats['canonical' if rep==-2 else 'geometry']=[
                dict(total=int(timeline[-1,j]),max_interval_abs_thirds=int(span[j]),clip_bias_thirds=int(residual[-1,j]))
                for j in range(2)]
        else:
            control_max=max(control_max,int(np.max(span)))

    by_dir={d:dict(pooled=int(pooled[-1,j]),**{k:v[j] for k,v in stats.items()})
            for j,d in enumerate(('in','out'))}
    delta=3*reference-pooled_windows
    biggest=int(np.abs(delta).max())
    numer=int(np.abs(delta).sum())
    changed=sum(a!=b for a,b in zip(canonical,chosen))
    check(biggest<=2,'Official window residual exceeds 2/3')
    expected=dict(nframes=n,by_direction=by_dir,num_time_bins=len(reference),pooled_window_counts=pooled_windows.tolist(),
                  integer_window_counts=reference.tolist(),delta_l1_thirds=numer,max_cell_abs_thirds=biggest,
                  changed_representatives=changed,bound_pp='100*delta_l1_thirds/(3*30*G), positive full-clip GT count G')
    check(pack['counts_proof']==expected, 'Frozen window certificate differs from direct frame-boundary counts')
    return dict(native_events=len(frames),issued=len(chosen),changed_representatives=changed,window_bins=len(reference),
                delta_l1_thirds=numer,max_cell_abs_thirds=biggest,by_direction=by_dir,controls=99,
                control_max_interval_abs_thirds=control_max)
