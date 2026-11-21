# Study202: native orphan continuation

[繁體中文結果與全部臂的兩 IoU 表](results.md)

From the BeeTrack repository root, with the existing locked environment:

```powershell
uv run --offline --no-sync python tools/study202/replay202.py tools/study202/evidence.zip
```

No images, detector caches, external GT files or native MOT are needed for this numeric replay. It uses the existing NumPy/SciPy environment and public study197/198/199 tools and pinned evidence archives. Run from the repository root. `--work-dir <temporary-directory>` optionally places streamed extraction and one-fold disk-backed numeric designs on a chosen disk.

## 公開數值重播

`evidence.zip` is deterministic ZIP_STORED: fixed timestamps, sorted names, and exactly one common manifest plus its1535 listed gzip shards. It contains no inflated private staging files or per-frame MOT. The CLI checks the exact ZIP allowlist before streaming extraction, then checks every compressed and inflated shard SHA256. Directory input remains supported for debugging.

The replay reconstructs32 rational integer matching bounds and feasible masks,8 source-only softmax refits, the deployed matching/M shuffle, all9 applicable arms' E3 and member-pooled F1 at0.5/0.3, investment/attribution gates,396 request/357 unique-state reuse accounting,60 three-exact source receipts, and the final independent-receipt joins. Per-member union-row tables share forward/reverse annotated IoU neighbors and source detector SHA/row index. These reproduce S/D/U,541/607 S-extension denominators and4/39 nonzero-window counts, plus distinct censor cases7/23 and restored-side cases2/23. U means undetermined. Continuation ownership uses the fixed K-plus-all-referenced-orphans union scene; cut classes retain study199 native-K ownership and are not relabelled in that union.

Capacity action windows are sparse integer columns, reconstructed to exactly the original complete30-phase objective. Parent and orphan capacities are shared by cuts/extensions; zero-delta candidates remain in owner/model evidence even though they need not enter a count-optimum search. Lower=upper is the only optimality criterion. Capacity witnesses see labels and are not deployed models or F1 maxima.

The ZIP is23,826,167 bytes. SHA256: `ef5fd41a7ebadf1425fa840cc73b1c412f5b5f76d049c7c58f0f4c66196d6788`.

This study uses six old development clips,20 units and60 member-clips. Router fitting holds out one hive's labels; the detector/student pipeline has previously seen the unlabelled target clips. This is not a fresh unseen-hive confirmation. No seed15–32 outputs, GPU work or test data were read. P0/calibrated ByteTrack/TOPICTrack comparisons retain the historical input/recipe limits in studies193/200.

## 數值圖與完整原生核對的界線

The numeric replay checks dated events,30-phase windows and maximum matching on published annotated threshold graphs. It checks forward/reverse owner-neighbor consistency, not full-frame IoU geometry from absent original boxes. All357 unique configurations were actually materialized and officially rescored in their full new scene by the original scorer;39 calls shared an identical complete state within that job. It never interpolated198 fixed-scene F1 after adding boxes.

Independent auditing checked all396 native matrix/provenance/event/window/graph-cardinality receipts. It independently regenerated full-frame matching at both IoUs on one changed configuration in each of the four domains. Remaining spatial graph generation is sealed producer evidence. TP_ID was independently checked only on those four fresh samples; compact replay does not claim TP_ID regeneration. The independent main receipt is `aab2a77771d1d3fff4211e926bfc6bad612c4eee3b0a35219f6749e666c27153`; the separate60-row-source allowlist supplement is `7bf32b47204e520e2ad745a4f3b69f7bb53117c47ec95d04b079f53bf62aa1a0`.

`source_manifest.json` records original versus published portable code fingerprints (published Git/LF bytes; platform checkout line endings may differ). `execution_sources.zip` preserves the three exact historical execution snapshots plus the final original verifier, including their original bytes/line endings, for provenance; it is not the fresh-run entry. The unified portable verifier is `verify_orphans.py` (original source SHA20c88a3b9090eb6101a2b43d6e99b52ad92398621242229c9cc2a7d34e50a257). Registration/freeze/model/score ordering is established by run receipts, not by claiming an after-the-fact verifier witnessed the run in advance.

## 有外部資料時的原生重建

The following are input-dependent reconstruction entry points, not commands satisfied by the public numeric ZIP. Supply the original sealed195 frozen archive (the42 registered members and both native arms), original197 frozen/score archive, and original199 owners/capacity archive. Their original fingerprints remain checked. Also supply only the corresponding allowed original detector `.npz` caches, pinned strict/noW/noG linker models, polygon metadata and the six prepared original GT files. The detector cache is the existing frame/xyxy/confidence numeric cache consumed by `source_rows.py`, not a CSV of final trajectories. No new inference or pixels are required. See `freeze_orphans.py` and the source receipts for the exact allowlist. Public old evidence gzip files do not replace these external native archives.

Use fresh output roots. Portable copies accept explicit local paths; changing code/output roots changes provenance seals, so use the newly generated frozen/owner seals below. Historical originals and portable fresh-run seals are distinct; no integrity check is disabled. These whole native runs were not repeated as part of this publication; portable freeze/owner synthetic and CLI checks passed. The published original run and independent receipt supply the field evidence.

```powershell
# Set these to separately supplied, hash-pinned native archives and a fresh output root.
$s195 = "<original-195-frozen>"
$s197 = "<original-197-frozen>"
$s197score = "<original-197-score>"
$s199 = "<original-199-root>"
$work = "<fresh-output-root>"
$prereg = "e2d3871227319b6674c3b6569f07d73e3663bdd8"
uv run --offline --no-sync python tools/study202/freeze_orphans.py --repo . --source195 $s195 --source197 $s197 --out "$work/frozen" --prereg $prereg
uv run --offline --no-sync python tools/study202/owners_orphans.py --repo . --frozen "$work/frozen" --source197 $s197 --owners199 "$s199/owners" --out "$work/owners" --prereg $prereg
$frozenSHA = (Get-FileHash "$work/frozen/manifest.json" -Algorithm SHA256).Hash.ToLower()
$ownersSHA = (Get-FileHash "$work/owners/manifest.json" -Algorithm SHA256).Hash.ToLower()
uv run --offline --no-sync python tools/study202/fit_orphans.py --run-field --frozen "$work/frozen" --owners "$work/owners" --out "$work/predictions" --prereg $prereg --frozen-sha $frozenSHA --owners-sha $ownersSHA
uv run --offline --no-sync python tools/study202/capacity_stage.py --root $work --old199 "$s199/capacity_v2" --old197-summary "$s197score/summary.json" --prereg $prereg --fit-finished
uv run --offline --no-sync python tools/study202/score_orphans.py --score --repo . --root $work --out "$work/score" --source197-score $s197score --prereg $prereg
```

The verifier additionally takes all five newly generated stage seals and those same original native archives. Default/synthetic modes do not open field data. A real verification requires explicit `--phase all --field-go --score-worker-finished --repo . --root <fresh-root> --source195 <195-frozen> --source197 <197-frozen> --source199 <199-root> --prereg <registered-SHA>` plus `--frozen-sha/--owners-sha/--predictions-sha/--capacity-sha/--score-sha`. These arguments make a field read explicit and keep historical versus fresh receipt identities separate.
