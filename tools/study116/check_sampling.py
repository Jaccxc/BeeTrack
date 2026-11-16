"""Run from the project environment with its root on PYTHONPATH; synthetic only."""
import hashlib
import numpy as np

from beetrack.evaluate import frame_match
from inspector import freeze_predictions
from sampling import _match_frame, prepare_sample, score_dose


POLY = [[10,0],[40,0],[40,40],[10,40]]


def main():
    def obs(f, tid, center, y=10, w=4):
        return [f,tid,center-w/2,y-2,w,4]

    rows = [obs(1,4,8),obs(2,4,15),obs(3,4,8),obs(7,4,10),
            obs(1,8,11,20),obs(7,8,6,20),obs(3,12,8.5),obs(7,12,8.5),
            obs(3,20,8,30,0),obs(7,20,11,30,0),
            obs(3,30,5,32),obs(29,30,13,32)]
    req, lab = prepare_sample(list(reversed(rows)),POLY,'mock-tail',60)
    assert req == dict(clip='mock-tail',nframes=60,frames=[1,3,7])
    ev = {r['gt_tid']:r for r in lab['events']}
    assert set(ev)=={4,8,20} and (ev[4]['before'],ev[4]['arrival'])==(3,7)
    assert ev[4]['rank'] == hashlib.sha256(b'beetrack-warp-pilot-v1|mock-tail|in|4|7').hexdigest()
    assert lab['population']['shortfall']==13
    duplicate = [obs(3,4,8),*rows]
    try:
        prepare_sample(duplicate,POLY,'mock-tail',60)
    except ValueError:
        pass
    else:
        raise AssertionError('duplicate observed GT accepted')

    def det(cx,y=10,w=4,score=.7):
        return [cx-w/2,y-2,cx+w/2,y+2,score]

    def scored(data):
        return score_dose(lab,freeze_predictions(req,dict(fold=33,seed=0,dose='U'),data))

    base = {1:[det(11,20)],3:[det(8),det(8.5)],7:[det(10),det(6,20)]}
    good = {r['gt_tid']:r for r in scored(base)['events']}
    assert good[4]['visible05'] and good[8]['visible05'] and not good[20]['visible03']
    contested = scored({**base,3:[det(8.5)]})
    assert contested['frames']['3']['predictions'][0]['gt_tid03']==12
    assert not next(x for x in contested['events'] if x['gt_tid']==4)['visible03']
    wrong = scored({**base,7:[det(9.9),det(6,20)]})
    missed = next(x for x in wrong['events'] if x['gt_tid']==4)
    assert missed['ends'][1]['match']['owner05'] and not missed['ends'][1]['match']['correct_side']
    assert not missed['visible05']
    assert not any(x['visible03'] for x in scored({**base,7:[]})['events'])
    unavailable = freeze_predictions(req,dict(fold=33,seed=0,dose='L',status='unavailable',unavailable_reason='bad frame'),None)
    assert all(x['visible05'] is None for x in score_dose(lab,unavailable)['events'])
    req0, lab0 = prepare_sample([],POLY,'empty',60)
    assert req0['frames']==[] and lab0['population']['selected']==0
    assert score_dose(lab0,freeze_predictions(req0,dict(fold=33,seed=0,dose='U'),{}))['events']==[]
    many = []
    for tid in range(30):
        cx=(8,11) if tid < 28 else (11,8)
        many.extend([obs(1,tid,cx[0]),obs(2,tid,cx[1])])
    req2, lab2 = prepare_sample(many,POLY,'fill',2)
    expected_in = sorted(range(28),key=lambda i:hashlib.sha256(f'beetrack-warp-pilot-v1|fill|in|{i}|2'.encode()).hexdigest())[:14]
    assert req2['frames']==[1,2] and {r['gt_tid'] for r in lab2['events']}=={*expected_in,28,29}
    assert lab2['population']['selected_by_direction']==dict(out=2,**{'in':14})
    assert len(lab2['gt_frames']['1'])==30
    rng = np.random.default_rng(116)
    for n,m in [(0,0),(0,5),(5,0),(7,8),(8,7),(20,27),(14,14)]:
        centers = rng.uniform(3,9,size=(max(n,m),2))
        truth = [[1,i,*centers[i],4,4] for i in range(n)]
        candidates = [[*pos,*(pos+4),.75] for pos in centers[:m] + rng.uniform(-1,1,size=(m,2))]
        truth = np.asarray(truth).reshape(-1,6)
        rows = [dict(gt_tid=int(id),xywh=[x,y,w,h],xyxy=[x,y,x+w,y+h],inside=False)
                for _,id,x,y,w,h in truth]
        new, ordered = _match_frame(1, rows, candidates, np.array(POLY,dtype=np.float32))
        pr = np.array([[1,q['index'],*q['xyxy'][:2],*(np.array(q['xyxy'][2:])-q['xyxy'][:2])]
                       for q in ordered]).reshape(-1,6)
        for tau in (.3,.5):
            strict = {v['index']:tid for tid,v in new.items() if tau==.3 or v['owner05']}
            expected = {q['index']:int(o) for q,o in zip(ordered,frame_match(truth,pr,thr=tau)) if o>=0}
            assert strict == expected
    print('Synthetic last genuine crossing, gap split, boundary, same-frame competition, zero width, empty and undersized cohorts passed.')


if __name__=='__main__':
    main()
