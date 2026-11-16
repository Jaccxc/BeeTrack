"""Cross-check detector frame assignments from serialized per-arm NPYs with stock frame_match."""
import argparse
import gzip
from pathlib import Path
import json

import cv2
import numpy as np
from beetrack.evaluate import by_track,frame_match
from physical117 import ARMS,require,save,read_checked


def check(root,repo,detector_sha,mechanism_sha):
    detector=json.loads(read_checked(root/'detector_done.json',detector_sha))
    index=json.loads(read_checked(root/'mechanism/index.json',mechanism_sha))
    require(index['detector_index_sha256']==detector_sha and len(index['cases'])==6,'Unrelated inputs')
    result={}
    for key,record in index['cases'].items():
        evidence=json.loads(gzip.decompress(read_checked(root/'mechanism'/record['file'],record['sha256'])))
        meta=evidence['meta']
        fold,seed=map(int,key.split('_')); clip=f'BEE24-{68-fold}-tail'
        require((meta['fold'],meta['seed'],meta['clip'],meta['nframes'])==(fold,seed,clip,1000),'Source differs')
        truth_path=repo/'runs/gt/val'/clip/'gt/gt.txt'
        read_checked(truth_path,meta['source_gt_sha256'])
        truth=by_track(np.loadtxt(truth_path,delimiter=',',usecols=range(6),ndmin=2))
        polygon=np.asarray(json.loads(read_checked(repo/'beetrack/zones.json',meta['zones_sha256']))[clip],np.float32)
        def inside(x,y):
            return cv2.pointPolygonTest(polygon,(float(x),float(y)),False)>=0
        gt_side={(int(r[0]),int(r[1])):inside(r[2]+r[4]/2,r[3]+r[5]/2) for r in truth}
        frames=evidence['frames']
        require([f['f'] for f in frames]==list(range(1,1001)),'Incomplete source')
        result[key]={}
        for arm in ARMS:
            name=f'detections/{key}/{arm}.npy'
            src=root/name
            read_checked(src,detector['files'][name])
            detections=np.asarray(np.load(src,allow_pickle=False),dtype=np.float64)
            rows=detections[detections[:,5]>=.25]
            order=np.lexsort((np.arange(len(rows)),rows[:,4],rows[:,3],rows[:,2],rows[:,1],-rows[:,5],rows[:,0]))
            rows=rows[order]
            anchors=[pid for f in frames for pid in f['pools'][f['arms'][arm]]['ids']]
            native=np.column_stack([rows[:,0],anchors,rows[:,1:3],rows[:,3:5]-rows[:,1:3]])
            expected={}
            for f in frames:
                pool=f['pools'][f['arms'][arm]]
                for gt,anchor,iou,side in pool['matches']:
                    expected[f['f'],anchor]=gt,iou,side
            stats={}
            for tau in (.5,.3):
                assigned=frame_match(truth,native,thr=tau)
                links={}
                for k in np.flatnonzero(assigned>=0):
                    r=rows[k]; f=int(r[0]); gid=int(assigned[k])
                    same_side=inside((r[1]+r[3])/2,(r[2]+r[4])/2)==gt_side[f,gid]
                    links[f,anchors[k]]=(gid,same_side)
                wanted={k:(gt,side) for k,(gt,iou,side) in expected.items() if iou>=tau}
                require(links==wanted,f'Official full-frame ownership or side differs for {key}/{arm}/{tau}')
                stats[str(tau)]=dict(tp=len(links),fp=len(rows)-len(links),fn=len(truth)-len(links))
            result[key][arm]=stats
        print('Full detector NPY native frame ownership verified',key,flush=True)
    out=dict(study=117,prereg=index['prereg'],mechanism_sha256=mechanism_sha,detector_sha256=detector_sha,
             case_arm_frames=36000,thresholds=[.5,.3],cases=result)
    print(save(root/'validation/native_frame_agreement.json',out),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('root',type=Path);p.add_argument('repo',type=Path)
    p.add_argument('--detector-sha',required=True);p.add_argument('--mechanism-sha',required=True)
    a=p.parse_args()
    check(a.root,a.repo,a.detector_sha,a.mechanism_sha)
