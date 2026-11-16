"""Export selected native E3 event identities while preserving every competitor's MOT boxes."""
import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from beetrack.common import write_mot
from beetrack.evaluate import by_track, frame_match, zone_tracks
from tools.study104.event_overlap import score_arm
from tools.study108.inputs108 import describe
from integer_events import integer_events


def export_mot(streams, polygon, selected):
    """Return Nx7 MOT rows plus {(seed,resolved_native_id): (first_id, block_size)}.

    Inputs are three numeric streams of 1-based frame, ID, xywh and score, plus
    the selected (seed,resolved_native_id) pairs. ID blocks are arm-independent.
    Output rows are ordered by (seed,resolved native ID,frame), which is also
    official by_track order for every variant, including inside split segments.
    """
    if len(streams) != 3:
        raise ValueError('Expected three detector streams')
    choices = {tuple(key) for key in selected}
    blocks, output, events, next_id = {}, [], set(), 1
    for seed, native in sorted(streams.items()):
        raw = np.asarray(native, dtype=float)
        if raw.shape == (0,):
            raw = raw.reshape(0,7)
        if (raw.ndim != 2 or raw.shape[1] != 7 or not np.isfinite(raw).all()
                or not np.array_equal(raw[:,:2], np.rint(raw[:,:2])) or np.any(raw[:,0]<1)
                or np.any(raw[:,4:6] <= 0)):
            raise ValueError('Expected valid [frame,id,x,y,w,h,score] MOT observations')
        rows = by_track(raw)
        if len(np.unique(rows[:,:2],axis=0)) != len(rows):
            raise ValueError('Native ID has more than one row at a frame')
        for tid, (direction, start, stop, changes) in zone_tracks(rows, polygon).items():
            key = (seed,tid)
            size = 1+len(changes)
            blocks[key] = (next_id,size)
            if direction:
                events.add(key)
            segment = rows[start:stop]
            segment[:,1] = next_id
            if direction and key not in choices:
                starts = [j for j,_ in changes]
                for offset,(j,k) in enumerate(zip(starts,starts[1:]+[stop]),start=1):
                    rows[j:k,1] = next_id+offset
            next_id += size
        output.append(rows)
    if not choices <= events:
        raise ValueError(f'Unknown or non-event selected: {sorted(choices-events)}')
    return np.concatenate(output), blocks


def write_export(path, rows):
    """Chronological MOTChallenge serialization, with stable new IDs within a frame."""
    write_mot(path, rows[np.lexsort((rows[:,1],rows[:,0]))])


def export_variants(streams, polygon):
    """Convert three native MOT streams into two count-matched, complete MOT variants and an ID manifest."""
    descriptors = {s:describe(np.asarray(rows,dtype=float).reshape(-1,7)[:,:6],polygon) for s,rows in streams.items()}
    manifest = integer_events(descriptors)
    outputs = {}
    for arm in ('canonical','geometry'):
        selected = {tuple(b[arm]) for group in manifest['blocks'].values() for b in group if b[arm] is not None}
        outputs[arm], layout = export_mot(streams,polygon,selected)
        expected = sorted([[layout[(s,r[0])][0],*r[1:]] for s,records in descriptors.items() for r in records
                           if (s,r[0]) in selected])
        if describe(outputs[arm][:,:6],polygon) != expected:
            raise ValueError('Exported MOT does not reproduce its selected native event descriptors')
    manifest['id_blocks'] = [[s,i,base,size] for (s,i),(base,size) in sorted(layout.items())]
    return outputs, manifest


def self_check():
    polygon = [[0,0],[10,0],[10,20],[0,20]]
    def bee(frames, tid, xs, y, score):
        return [[f,tid,x,y,2,2,score] for f,x in zip(frames,xs)]
    frames, xs = [2,3,4,5,6], [11,8,11,8,7]
    gt = np.array(bee(frames,99,xs,2,.95))[:,:6]
    # Identical crossing geometry: the first detector owns the GT box on all five frames.
    first, second = bee(frames,7,xs,2,.91), bee(frames,7,xs,2,.73)
    # A crossing-and-return non-event stays whole even when there are side changes.
    # After a >25-frame gap its reused native ID becomes a separate eligible out event.
    third = bee(frames,4,[11,8,8,11,12],10,.84)+bee([40,41],4,[8,11],10,.63)
    streams = {2:third[::-1],0:first[::-1],1:second}
    outputs, choices = export_variants(streams,polygon)
    assert set(outputs) == {'canonical','geometry'} and len(choices['id_blocks']) == 4
    assert len(outputs['canonical']) == len(outputs['geometry']) == sum(map(len,streams.values()))
    selected = {(1,7)}
    empty = export_mot({0:[],1:[],2:[]},polygon,[])[0]
    assert empty.shape==(0,7)
    native, blocks = export_mot(streams,polygon,{(0,7),(1,7),(2,100004)})
    kept, same_blocks = export_mot(streams,polygon,selected)
    assert same_blocks == blocks == {(0,7):(1,4),(1,7):(5,4),(2,4):(9,3),(2,100004):(12,2)}
    np.testing.assert_array_equal(kept[:,[0,2,3,4,5,6]],native[:,[0,2,3,4,5,6]])
    np.testing.assert_array_equal(by_track(kept),kept)
    assert kept[:5,1].tolist()==[1,2,3,4,4] and kept[5:10,1].tolist()==[5]*5
    assert kept[10:15,1].tolist()==[9]*5 and kept[15:,1].tolist()==[12,13]
    for threshold in (.3,.5):
        owners = frame_match(by_track(gt),by_track(kept),threshold)
        np.testing.assert_array_equal(owners,frame_match(by_track(gt),by_track(native),threshold))
        assert owners[:5].tolist()==[99]*5 and owners[5:10].tolist()==[-1]*5
        for wanted in (set(),{(0,7)},{(2,100004)},selected,{(0,7),(1,7),(2,100004)}):
            variant, layout = export_mot(streams,polygon,wanted)
            np.testing.assert_array_equal(frame_match(by_track(gt),by_track(variant),threshold),owners)
            result = score_arm(gt,variant,polygon,threshold)
            assert result.events.keys()=={layout[k][0] for k in wanted}
        with_controls = score_arm(gt,kept,polygon,threshold)
        lean = kept[np.isin(kept[:,1],[blocks[(1,7)][0],blocks[(2,4)][0]])]
        removed_competitors = score_arm(gt,lean,polygon,threshold)
        assert len(with_controls.matches)==0 and len(removed_competitors.matches)==1
        assert len(with_controls.gt)==len(removed_competitors.gt)==len(with_controls.events)==len(removed_competitors.events)==1
    with TemporaryDirectory() as tmp:
        path = Path(tmp)/'tracks.txt'
        for wanted in (set(),{(0,7)},{(2,100004)},selected,{(0,7),(1,7),(2,100004)}):
            variant,_ = export_mot(streams,polygon,wanted)
            write_export(path,variant)
            reread = np.loadtxt(path,delimiter=',',ndmin=2)[:,:7]
            np.testing.assert_array_equal(by_track(reread),variant)
            for threshold in (.3,.5):
                before = score_arm(gt,variant,polygon,threshold)
                after = score_arm(gt,reread,polygon,threshold)
                assert after.events==before.events and after.matches==before.matches
    print('MOT roundtrip, selected IDs, all native observations and frame owners agree. '
          'Synthetic F1 with original competitors: 0; deleting their boxes: 1.')


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['self-check'])
    parser.parse_args()
    self_check()
