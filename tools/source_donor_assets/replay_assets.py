"""Replay published selection and reported per-frame engineering tables, without pixels."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path

def iou(frame):
    x,y,w,h=frame['original_box_xywh'];left,top=frame['x'],frame['y'];hh,ww=frame['crop_shape']
    intersection=max(0,min(x+w,left+ww)-max(x,left))*max(0,min(y+h,top+hh)-max(y,top))
    return intersection/(w*h+ww*hh-intersection)
def replay(e):
    selection=e['selection'];tracks=selection['tracks'];donors=e['donors']
    eligible=[r for r in tracks if r['eligible']]
    assert len(eligible)==selection['eligible_count']
    for r in eligible:
        text=f"{r['hive']}|{r['direction']}|{r['seq']}|{r['track_id']}|{r['f']}"
        assert r['hash_input']==text and hashlib.sha256(text.encode('ascii')).hexdigest()==r['digest']
        assert r['first']==r['f']-8 and r['last']==r['f']+8
        assert r['frames']==list(range(r['first'],r['last']+1))
        assert not set(r['frames'])&set(r['background_frames']) and len(r['background_frames'])<=15
    selected=[]
    for name,cell in selection['cells'].items():
        qualified=sorted([r for r in eligible if f"{r['hive']}_{r['direction']}"==name],
                         key=lambda r:(r['digest'],r['seq'],r['track_id'],r['f']))
        picked=[];seen=set()
        for r in qualified:
            if r['seq'] not in seen:picked.append(r);seen.add(r['seq'])
            if len(picked)==2:break
        assert picked==cell['selected'] and cell['selected_count']==len(picked)
        assert cell['eligible_count']==len(qualified) and cell['eligible_videos']==len({r['seq'] for r in qualified})
        assert cell['no_replacement'];selected+=picked
    assert len(donors)==len(selected)
    byidentity={(r['hive'],r['seq'],r['track_id'],r['f']):r for r in selected}
    frames=masks=shams=good=touch=0
    passes={}
    for name,m in donors.items():
        r=byidentity[m['hive'],m['seq'],m['track_id'],m['f']]
        assert [f['frame'] for f in m['frames']]==r['frames']
        nmask=nsham=ngood=ntouch=0
        for f in m['frames']:
            if f['mask_status']=='ok':
                nmask+=1;qa=f['qa']
                assert qa['mask_pixels_inside_target_box']+qa['mask_pixels_outside_target_box']==qa['mask_pixels']
                collision=any(v['mask_pixels']>0 for v in qa['other_gt_box_intersections'])
                assert collision==f['intersects_other_bee']
                ntouch+=collision
                value=iou(f);assert abs(value-f['mask_bbox_iou'])<1e-12
                assert f['mechanical_frame_pass']==(value>=.5 and not collision)
                ngood+=f['mechanical_frame_pass']
            else:assert not f['mechanical_frame_pass']
            if f['sham_status']=='ok':
                nsham+=1;assert f['alpha_support_min_samples']>=3
                assert f['background_valid_alpha_pixels']==f['qa']['mask_pixels']
        assert (nmask,nsham,ngood,ntouch)==(m['mask_successes'],m['sham_successes'],m['qa_good_frames'],m['other_bee_intersection_frames'])
        assert m['engineering_pass']==(nmask==17 and nsham==17 and ngood>=15)
        passes[name]=m['engineering_pass'];frames+=17;masks+=nmask;shams+=nsham;good+=ngood;touch+=ntouch
    for name,cell in e['results']['cells'].items():
        selected_names=[r['directory'] for r in e['results']['donors'] if f"{r['hive']}_{r['direction']}"==name]
        assert cell['engineering_passes']==sum(passes[n] for n in selected_names)
        assert cell['pair_ready']==(len(selected_names)==2 and all(passes[n] for n in selected_names))
    if 'original_independent' in e:
        a=e['original_independent']['acceptance']
        assert a['selection_rebuilt_independently'] and a['all_png_and_background_crops_checked']
        assert (a['donors'],a['frames'])==(len(donors),frames)
        assert a['background_annotation_rejection_frames']+a['pixel_support_failure_frames']==frames-shams
        for name,cell in a['cells'].items():
            assert cell['pair_ready']==e['results']['cells'][name]['pair_ready']
        for audit in e['original_independent']['evidence']['donors']:
            original=next(m for m in donors.values() if (m['seq'],m['track_id'],m['f'])==(audit['seq'],audit['track_id'],audit['f']))
            assert len(audit['frames'])==len(original['frames'])==17
            for f,old in zip(audit['frames'],original['frames']):
                assert f['frame']==old['frame'] and f['mask']==(old['mask_status']=='ok')
                assert f['sham']==(old['sham_status']=='ok') and f['geometry_ok']==old['mechanical_frame_pass']
                assert abs(f['bbox_iou']-old['mask_bbox_iou'])<1e-12
                assert (f['other_gt_intersection_pixels']>0)==old['intersects_other_bee']
                assert f['mask_pixels']==old['qa']['mask_pixels']
                if 'alpha_support_min_samples' in old:assert f['background_support_min']==old['alpha_support_min_samples']
    supplement_shams=None
    if 'background_recheck' in e:
        s=e['background_recheck']['summary'];a=e['background_recheck']['acceptance']
        original=next(m for m in donors.values() if (m['seq'],m['track_id'],m['f'])==(s['seq'],s['track_id'],s['f']))
        assert s['background_dates']==original['background_frames']
        assert s['original_qa_good_frames']==original['qa_good_frames'] and not s['original_engineering_pass']
        assert s['no_new_segmentation'] and s['no_replacement_dates'] and s['original_artifacts_unchanged']
        invalid=[]
        for day in s['background_day_records']:
            rows=day['all_original_gt_rows'];indices=day['original_row_indices']
            bad=[i for i,r in zip(indices,rows) if r[4]<=0 or r[5]<=0]
            assert bad==day['nonpositive_original_row_indices'] and day['trusted_date']==(not bad)
            if bad:invalid.append(day['frame'])
        assert [f['frame'] for f in s['frames']]==[f['frame'] for f in original['frames']]
        for f,old in zip(s['frames'],original['frames']):
            assert f['invalid_background_dates']==invalid
            assert f['true_png_sha256']==f['original_png_sha256']==old['png_sha256']
            assert (f['x'],f['y'],f['crop_shape'])==(old['x'],old['y'],old['crop_shape'])
            ready=f['unsupported_alpha_pixels']==0 and f['alpha_support_min_samples']>=3
            assert (f['sham_status']=='ok')==ready
        audits=e['background_recheck']['evidence']['frames']
        assert len(audits)==17
        for f,audit in zip(s['frames'],audits):
            assert f['frame']==audit['frame']
            assert [f['x'],f['y']]==audit['origin'] and f['crop_shape']==audit['crop_shape']
            for field in ('unsupported_alpha_pixels','alpha_support_min_samples'):assert f[field]==audit[field]
            assert audit['sham_ready']==(f['sham_status']=='ok')
        supplement_shams=sum(f['sham_status']=='ok' for f in s['frames'])
        assert supplement_shams==s['sham_successes']==a['sham_successes'] and len(s['frames'])==a['frames']==17
        assert a['all_17_alpha_and_origin_identical'] and a['all_15_dates_unchanged']
        assert a['whole_invalid_days_provide_zero_support'] and a['background_counts_medians_and_shams_match']
    return dict(donors=len(donors),frames=frames,masks=masks,original_shams=shams,
                geometry_pass=good,rectangle_touch=touch,separate_recheck_shams=supplement_shams)
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('evidence',type=Path);a=p.parse_args()
    raw=a.evidence.read_bytes();raw=gzip.decompress(raw) if a.evidence.suffix=='.gz' else raw
    print(json.dumps(replay(json.loads(raw))))
