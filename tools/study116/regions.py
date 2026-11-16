"""Whole-frame detection recall by GT center and false positives by detection center."""
import argparse
import math


def region_counts(labels, scored, main_rect):
    """Return counts for each IoU, pooled over the requested frames, or None for an unavailable dose.

    GT-region TP and FN are by the GT center. Pred-region matched_pred and FP are by the
    predicted center, so a cross-boundary match may increment two different regions.
    IoU 0.5 uses the scorer's ownership flag. Neither threshold requires polygon-side agreement.
    """
    if scored['status']=='unavailable':
        return None
    if scored['status']!='available' or scored['clip']!=labels['clip'] or labels['gt_frames'].keys()!=scored['frames'].keys():
        raise ValueError('Labels and scored frames must refer to the same available clip and frame set')
    xmin,ymin,xmax,ymax=main_rect
    if not all(math.isfinite(x) for x in main_rect) or xmin>=xmax or ymin>=ymax:
        raise ValueError('The common main rectangle must have positive finite extents')

    def region(box):
        x1,y1,x2,y2=box
        return 'inside' if xmin <= (x1+x2)/2 <= xmax and ymin <= (y1+y2)/2 <= ymax else 'outside'

    result={tau:{name:dict(gt=0,tp=0,pred=0,matched_pred=0) for name in ('inside','outside')}
            for tau in ('0.3','0.5')}
    for f,truth in labels['gt_frames'].items():
        preds=scored['frames'][f]['predictions']
        matches={int(tid):m for tid,m in scored['frames'][f]['matches'].items()}
        gt_ids=[t['gt_tid'] for t in truth]
        by_index={p['index']:p for p in preds}
        linked={m['index']:tid for tid,m in matches.items()}
        if (len(set(gt_ids))!=len(gt_ids) or len(by_index)!=len(preds) or len(linked)!=len(matches)
                or not matches.keys()<=set(gt_ids) or not linked.keys()<=by_index.keys()
                or any(p['gt_tid03']!=linked.get(p['index']) for p in preds)
                or any(type(m['owner05']) is not bool for m in matches.values())):
            raise ValueError(f'Inconsistent within-frame GT/prediction ownership at {f}')
        for tau,groups in result.items():
            for box in truth:
                cell=groups[region(box['xyxy'])]
                cell['gt']+=1
                owner=matches.get(box['gt_tid'])
                cell['tp']+=int(owner is not None and (tau=='0.3' or owner['owner05']))
            for box in preds:
                cell=groups[region(box['xyxy'])]
                cell['pred']+=1
                tid=box['gt_tid03']
                cell['matched_pred']+=int(tid is not None and (tau=='0.3' or matches[tid]['owner05']))
    for groups in result.values():
        groups['all']={k:groups['inside'][k]+groups['outside'][k] for k in groups['inside']}
        for cell in groups.values():
            cell['fn']=cell['gt']-cell['tp']
            cell['fp']=cell['pred']-cell['matched_pred']
            cell['recall']=cell['tp']/cell['gt'] if cell['gt'] else None
            cell['precision']=cell['matched_pred']/cell['pred'] if cell['pred'] else None
    return result


def self_check():
    # GT inside, prediction outside. Another pair reverses these roles and matches only at .3.
    labels=dict(clip='mock',gt_frames={'1':[dict(gt_tid=2,xyxy=[8,2,12,4]),dict(gt_tid=4,xyxy=[10,2,14,4])],
                                        '2':[dict(gt_tid=8,xyxy=[0,1,2,3])], '3':[]})
    scored=dict(clip='mock',status='available',frames={
        '1':dict(predictions=[dict(index=5,xyxy=[9,2,13,4],gt_tid03=2),dict(index=7,xyxy=[8,2,12,4],gt_tid03=4),
                              dict(index=11,xyxy=[1,2,1,2],gt_tid03=None)],
                 matches={'2':dict(index=5,owner05=True,correct_side=False),'4':dict(index=7,owner05=False,correct_side=True)}),
        '2':dict(predictions=[],matches={}), '3':dict(predictions=[],matches={})})
    res=region_counts(labels,scored,[0,0,10,10]); a,b=res['0.3'],res['0.5']
    assert (a['all']['gt'],a['all']['tp'],a['all']['fp'],a['all']['fn'])==(3,2,1,1)
    assert (b['all']['gt'],b['all']['tp'],b['all']['fp'],b['all']['fn'])==(3,1,2,2)
    assert (b['inside']['tp'],b['inside']['matched_pred'],b['inside']['fp'])==(1,0,2)
    assert (b['outside']['tp'],b['outside']['matched_pred'],b['outside']['fp'])==(0,1,0)
    assert b['inside']['recall']==.5 and b['outside']['precision']==1
    assert region_counts(labels,{'status':'unavailable'},[0,0,10,10]) is None
    e=region_counts(dict(clip='empty',gt_frames={}),dict(status='available',clip='empty',frames={}),[0,0,2,2])
    assert e['0.5']['inside']['recall'] is None and e['0.3']['all']['fp']==0
    print('Synthetic GT/prediction-region assignment, boundary, zero area, strict ownership and empty inputs passed.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['self-check']); parser.parse_args()
    self_check()
