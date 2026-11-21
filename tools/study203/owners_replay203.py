"""Reconstruct203 native/single-insertion S/D/U from deduplicated annotated rows."""
from tools.study202.owner_numeric202 import check_row


def classify(rows):
    if not rows or not all(r['qualified'] for r in rows):return 'U'
    if len({r['official_owner'] for r in rows})==1:return 'S'
    return 'D' if len({r['original_gt_id'] for r in rows})>1 else 'U'


def replay(archive):
    counts={};labels={}
    for key in archive.keys('owner_unit_meta'):
        unit=archive.get('owner_unit_meta',key);rows=archive.get('owner_rows',key)
        for member in archive.keys('owner_member_actions'):
            if not member.startswith(key+'|'):continue
            for saved in archive.array('owner_member_actions',member):
                action=archive.resolve(saved,rows)
                for threshold,label in action['labels'].items():
                    host=label['original_host_rows'];donor=label['original_donor_rows']
                    inserted=label.get('inserted_rows',label.get('inserted_host_rows',host)+label.get('inserted_suffix_rows',[]))
                    for row in host+donor+inserted:check_row(row,float(threshold))
                    assert label['class']==classify(host+donor+inserted)
                    assert label['native_class']==classify(host+donor) and label['insert_class']==classify(inserted)
                    token=(unit['domain'],threshold,label['class']);counts[token]=counts.get(token,0)+1
                labels[action['action_id']]={t:v['class'] for t,v in action['labels'].items()}
        # Native host answers may be empty, but never valid on mixed/ambiguous anchors.
        for saved in archive.array('owner_hosts',key):
            host=archive.resolve(saved,rows)
            for threshold,supervision in host['supervision'].items():
                for row in supervision['anchors']:check_row(row,float(threshold))
                assert supervision['valid']==(classify(supervision['anchors'])=='S')
            assert host['event_target_valid']==host['supervision']['0.5']['valid']
        del rows
    return dict(counts={'|'.join(k):v for k,v in counts.items()},action_classes=labels)
