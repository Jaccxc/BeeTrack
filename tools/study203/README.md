# Study203: observed peer suffixes

From the BeeTrack repository root, with its existing NumPy/SciPy environment:

```powershell
uv run --offline --no-sync python tools/study203/replay203.py tools/study203/evidence.parts.json --work-dir D:/beetrack_runs
```

Keep both `evidence.zip.001` and `evidence.zip.002` beside the descriptor. This single entry verifies each part's complete SHA256, the joined ZIP SHA256, its exact common-manifest allowlist and every compressed/inflated component. Parts are transport segments, not separate experiments. It streams the join and extracts only the listed gzip components into a temporary directory, then removes that temporary extraction. `--work-dir` can be another existing directory with enough free space. Directory/unsplit-ZIP inputs are also accepted. No Git LFS is used.

The existing `tools/study202/evidence.zip` is required, pinned to SHA256 `ef5fd41a7ebadf1425fa840cc73b1c412f5b5f76d049c7c58f0f4c66196d6788`. No original images, detector caches, GT files or complete MOT are needed for this numeric command. All numeric archive components are source-projected, shared or sparsely encoded without changing their decoded values. Source JSON SHA256 values remain separate from public serialization hashes. Complete private inflated scan files are not distributed.

## Replayed evidence

The command passed on the complete finalized numerical archive: 24 tight integer capacity certificates and feasible witnesses, 12 deterministic source-only refits, fixed fourfold decoder choices, 191 distinct complete-scene event graphs at IoU0.5/0.3, all applicable arm/group E3 and member-pooled F1, investment/attribution gates, owner labels from annotated forward/reverse neighbors, complete GT-run return-probe denominators and the registered T/M descriptive control. Refits use bounded disk-backed objectives; compatible-set loss is not globally convex, so convergence is local numerical convergence, not a global loss optimum.

The experiment requested 432 physical scores: 191 distinct configurations were originally freshly scored and 241 requests shared those configurations. A metadata correction then resumed/validated all191 completed states without rerunning official matching. Reusing an unchanged numeric K metric from study202 is transport deduplication after complete source-file hash and value equality; it is not historical physical-score reuse.

`member` pools the original global member across W/G clips for F1; `host_member` identifies unit|slot for native-cache/resource identity. E3 averages predictions before absolute error. F1 is a member/group mean of pooled TP/pred/GT, not an ensemble F1.

This replay verifies annotated threshold graphs and maximum event matching, not their full-frame spatial IoU construction. Original full-evidence hashes and the independent native audit establish that geometry. Full candidate-feature geometry requiring every receiving-frame box likewise belongs to external native reconstruction; numerical refit checks the published source feature records. The GT universe contains every canonical run, including no-return cases; observed dates/sides are tied to prepared-GT file/source-index/row hashes. No pixels are published.

## External native reconstruction

The source entries are `freeze203.py`, `owners203.py`, `fit203.py`, `stage203.py`, `capacity203.py`, `score203.py`, `verify203.py` and their sibling helpers. They require the separately supplied **complete sealed study202 native frozen tree**, its original detector-row lineage, the sealed study202 native score reference, six permitted prepared GT files and polygon metadata. The study202 public numeric ZIP cannot substitute for native per-frame caches. Rebuilding those native inputs is described in study202. No new upstream data is fabricated here. All commands run from the repository root; historical on-site native writers require fresh explicit D: outputs. `source_fingerprints.json` records original versus portable source hashes; `executed_sources.zip` preserves both exact independent executed snapshots. Fresh portable seals can differ because source fingerprints differ; integrity checks remain enabled.

Registration: `69d52924d596d74c56edd25f1bc9bb7c8e66edb4`. The external sequence is freeze -> owners -> fourfold fit -> capacity -> native score -> independent verification. Each step consumes the prior complete seals; use the respective `--help` for required explicit inputs and pass the actual new manifest hashes, not historical hashes to a newly built tree. The following source-only checks read no field data:

```powershell
uv run --offline --no-sync python tools/study203/fit203.py --self-test
uv run --offline --no-sync python tools/study203/capacity203.py
```

The historical full native audit checked all9888 owner actions, 12refits, 24certificates and all432 requests/191distinct native matrices, source rows, event windows and newly reconstructed graph cardinalities. Independent fresh full-frame rematching covered four configurations, one per domain, at both IoUs; TP_ID was independently checked only for those four. It does not claim191 independent spatial rematches.

## Scope

Six reused development clips,20 three-member units. CS had previously seen unlabelled target images; holding out router labels/fits is not a new-hive blind evaluation. No GPU, reserve seeds15–32 or test arm was used. Return probes are GT-selected dated cases, not deployment identity accuracy. See [results.md](results.md) for all arms, groups and limits.

For an externally supplied native202 tree, these are the explicit stage commands (replace angle-bracket paths; they are not public-archive inputs):

```powershell
$prereg = '69d52924d596d74c56edd25f1bc9bb7c8e66edb4'
$native203 = 'D:/your_fresh_native203'
uv run --offline --no-sync python tools/study203/freeze203.py --repo . --source202 <native202-frozen> --out "$native203/frozen" --prereg $prereg
uv run --offline --no-sync python tools/study203/owners203.py --repo . --frozen "$native203/frozen" --out "$native203/owners" --prereg $prereg
$fr = (Get-Content "$native203/frozen/SEALED.sha256").Trim()
$ow = (Get-Content "$native203/owners/SEALED.sha256").Trim()
uv run --offline --no-sync python tools/study203/fit203.py --run-field --frozen "$native203/frozen" --owners "$native203/owners" --out "$native203/predictions" --frozen-sha $fr --owners-sha $ow --prereg $prereg
$pr = (Get-Content "$native203/predictions/SEALED.sha256").Trim()
uv run --offline --no-sync python tools/study203/stage203.py --root $native203 --prereg $prereg --frozen-seal $fr --owners-seal $ow --prediction-seal $pr --fit-finished
uv run --offline --no-sync python tools/study203/score203.py --score --repo . --root $native203 --out "$native203/score" --source202-score <native202-score> --prereg $prereg
uv run --offline --no-sync python tools/study203/verify203.py --phase all --field-go --score-worker-finished --repo . --root $native203 --source202 <native202-frozen> --source202-score <native202-score> --prereg $prereg
```

This native template requires the supplied complete trees, original prepared GT and the repo's existing permitted detector/model assets. It has not been presented as a fresh public-only numeric run. The exact historical executed snapshots verify the historical receipts; portable outputs use their own freshly recorded source seals. The packaged numeric replay command above is the independently rerunnable public-data entry.
