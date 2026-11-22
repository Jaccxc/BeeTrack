"""Post-result arithmetic only: moving the G counterexample 53 -> 61.

25 fps, 30 one-second phases, 30-second bins. Other event counts/directions
are held fixed. This proves no window change; it says nothing about F1.
"""
import json


def proof():
    before=[(53+25*offset-1)//750 for offset in range(30)]
    after=[(61+25*offset-1)//750 for offset in range(30)]
    assert before==after
    return dict(schema='phase212-posthoc-clock-v1',nframes=75,
        before_frame=53,after_frame=61,phases=30,bin_indices_equal=True,
        changed_window_entries=0,window_L1_change=0,
        scope='One event date only; all other events/directions/counts fixed; no F1 inference.')


if __name__=='__main__':
    print(json.dumps(proof(),sort_keys=True))
