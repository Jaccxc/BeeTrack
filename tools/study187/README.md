# §187 two-bee 2→1→2 occlusion crossings (CPU gate)

Base: the sealed §186 C-arm MOT (`D:/beetrack_runs/endpoints186/c_{33,35}_s{0,1,2}.npy`, distilled boxes pl, score ≥ 0.4, m_dev_strict, P0, dev heads, detector seeds 0–2). Outputs go to `D:/beetrack_runs/t187/`.

Run from `final_project` with `PYTHONIOENCODING=utf-8 PYTHONPATH=. CUDA_VISIBLE_DEVICES=-1 uv run --offline --no-sync python`, in this order:

1. `tools/study187/selfcheck187.py` synthetic checks (extraction, bound, joint rewrite)
2. `tools/study187/extract187.py` candidates from detections only; writes `freeze.json` before any GT read (needs an empty output folder)
3. `tools/study187/gtcheck187.py` (optional `--iou 0.3`) Y / U per head; refuses if the freeze hashes do not match
4. `tools/study187/repair187.py` (optional `--gt gt_check_iou0.3.json`) diagnostics 甲 / 乙, full-MOT rescoring, per-window optimistic bound
5. `tools/study187/reuse187.py` post-hoc sensitivity over the candidates that abstained for piece reuse

Results and reading: `docs/experiment_log.md` §187. A fresh rerun into an empty folder reproduced every output file byte for byte.
