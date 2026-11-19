TrackletLink models used by `beetrack/p0.py` (loaded with `p0.link_model(name)`; `beetrack/link_model.json` is a different fit, not used by P0).

- `m_dev_strict.json`: the test link model of P0 (log §125); on val + dev it is reported only (19.56%).
- `m_v0_val.json`, `m_v0_val_s1.json`, `m_v0_val_s2.json` on val and `m_v0_dev*.json` on dev: the three cross-fitted link models of §74 (seeds 0, 1, 2), averaged for the val + dev numbers of Table A (§91, §96).
- `m_noW.json`: the whiteboard-free link model of the leave-whiteboard fold (§64, used by §157's `tools/whiteboard_fold.py`), a byte copy of the scratch `loho/m_noW.json`. Fitted as `python -m beetrack.link --dets y11s_640 --fit train val --skip` the 13 whiteboard sequences (BEE24-01–11, 14, 15), seed 0; a refit on that path is byte-identical.
- `m_noG.json`: the gray-board-free link model of the leave-gray-board fold (§159, used by `tools/gray_fold.py`). Fitted as `python -m beetrack.link --dets y11s_640 --fit train val --skip` the 16 gray-board sequences (BEE24-17–32); a refit on that path is byte-identical.
