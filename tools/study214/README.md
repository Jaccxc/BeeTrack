# Study214: endpoint image recentering

From the BeeTrack repository root:

```powershell
uv run --offline --no-sync python tools/study214/replay214.py tools/study214/evidence214.json.gz --out D:/beetrack_runs/study214.replay.json
uv run --offline --no-sync python tools/study214/supplement214.py tools/study214/evidence214.json.gz D:/beetrack_runs/study214.replay.json --out D:/beetrack_runs/study214.supplement.json
```

The first command recomputes four source-only folds/eight two-axis ridge heads, grouped weights/scaling, all endpoint shifts and Z, all90 state graph matchings/windows, individual W, three-member E3/F1 gates, localization and18 member-clip/12 pooled member A/B bounds. The second reports registered E3 envelopes and explicitly post-result integer deductions (two endpoints outside B; 33's36+64 cancellation change). Neither command reads images, MOT matrices or original GT identities. See [results](results.md).

Evidence SHA-256: `47abd69c57e2a39c1ca1146885480c107e7789c1459524c7bf6f3eafbec5fa78`. Expected core report using the repository's `uv` Python3.11.0: `5f845b20d4737fc2e4ed5276320df1b82006e81036e1b481b710172f38d38b58`. Upstream Python3.13.5 report SHA was `7dacdf662025af9d0568f70ca2274e30037b0f79257dd4bce8f26f2bfc5ca784`: 199 floating MAE/F1 sum leaves differ by at most1.11e−16 due to Python's updated float summation; integer scores/bounds, models and gate truth values are identical. Use the repo command for byte comparison.

Upstream evidence bytes had SHA `c97841cd0b6a31a4ca33dd359aae2fb632863b32c3b297a86a882fdf1bb09203`. Release scanning changed only98 genuine decimal tokens to equivalent exponent spelling. Every decoded float type/value and the full JSON are identical; the released bytes reproduce the same Python3.11 report SHA above.

Independent receipt: `8c8ba40f673b06389fc8440719163f2a9315b42f9e77c82fa838706879693eb8`. Producer/audit seals are inside the evidence; `source_provenance.json` separates original from portable source hashes. `manifest.json` hashes every released file.

## Scope

The anonymous bundle has15026 endpoints/867 eligible,155 receiver source rows, four fold orders/models, and18×5 new legal graph/window states. These90 scores are freshly evaluated physical matrices, not study212 score reuse. E3 averages three counts before absolute error; short clips pool integers. F1 pools each member's short clips then averages the three members' F1. The individual W bounds have18 clip cells and12 pooled domain/member cells, distinct from E3 bounds.

O is a prescribed owner-based correction reference, not an upper bound. A permits any eligible endpoint side, including geometrically unreachable choices. B permits only original/O sides and fixes unknown points, so it does not bound arbitrary learned directions. Coordinatewise intervals can mix incompatible choices across windows. A/B lower bounds carry neither feasible-witness nor F1 guarantees.

Native verification checked5600 source images, every eligible patch/G6/χ24, all18 full-scene original GT assignments, eight independent ridge fits, all shifts and90 complete physical states at both IoUs, with independent matching cardinality/clock loops. Shared low-level primitives are cv2 remap/polygon, SciPy Hungarian and official legal-event semantics. Numeric replay starts from suppliedχ/G/GT signed-distance summaries and legal edges; it does not independently decode pixels or reassociate GT boxes.

External reconstruction needs separately supplied original frames, six old K clip/member matrices, zones/annotations and exact registered parent seals. Included prepare/label/fit/eval/audit sources accept explicit historical input paths; some original output guards retain the D-drive policy. Portable modified source hashes are separate and historical code pins stay enforced. Do not present newly generated artifacts as the old execution seals. No natural crop, GT box/identity or private machine path is distributed.

Image tracking has box refinement work such as [Yan et al., Alpha-Refine, CVPR2021](https://openaccess.thecvf.com/content/CVPR2021/html/Yan_Alpha-Refine_Boosting_Tracking_Performance_by_Precise_Bounding_Box_Estimation_CVPR_2021_paper.html). This experiment tests bee-track endpoint sides and whole-scene30-second counts; it does not benchmark that method directly. All findings remain limited to six old development clips with upstream unlabeled-target CS exposure.
