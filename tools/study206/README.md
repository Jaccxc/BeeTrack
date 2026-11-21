# Study206: one confirmed return box

From the BeeTrack repository root, using its existing environment:

```powershell
uv run --offline --no-sync python tools/study206/replay206.py tools/study206/evidence.zip --work-dir D:/beetrack_runs
```

`--work-dir` may be another existing directory with enough temporary space. Keep public study203's two evidence parts/descriptor and study202's evidence.zip in their published locations. The command validates the exact archive allowlist, every gzip/inflated hash, the pinned study203 joined-ZIP SHA and study202 reference, then replays the incremental numbers. No raw images, complete MOT, detector caches or external GT files are read. It uses one process; the archive reuses prior references rather than copying study203's 191 physical configurations (including the 60 K baselines) or its 1,958 annotated runs.

Evidence ZIP SHA256: `72af4e9f90701c99a23f2475dba8ed56df42d5233e8db99f959c3a7e387607ef` (8,089,997 bytes). It has one manifest and207 gzip components. Source full-JSON hashes are distinct from public serialization hashes. Three source-derived full hash values and523 floating-point JSON spellings were checked before exact value-preserving serialization; no private inflated scan files ship.

## What is verified

- Complete4169 original-event query index against all60 original study203 K signed-event rosters, including ineligible/empty searches; anchors, fixed25-frame dates/circle, both-IoU annotated ownership/unknown classes and fixed earliest candidate evidence. Source annotations/earlier-date rejection reasons are tied to independent receipts, not claimed as newly computed full-frame geometry.
- All30 offsets, partial windows, all-event/eligible/confirmed count-gain ceilings, exact rational group/clip weights and the465-unit/16-member-event necessary count on35. These bounds are optimistic ceilings, not attainable solutions.
- Every stable highest-positive greedy selection/rejection and resource/joint-safety certificate, each adopted prefix's complete group windows/TP/FP/FN, final K/R05/R03 windows, mean member W and pooled mean-member F1 at both IoUs. GT-run counts deduplicate across groups directly; they are not summed per-group distinct counts.
- All85 supplied event graphs:25 genuinely new configurations and60 exactly bound historical K states. Maximum matching and signed-event windows are recomputed. Historical full graphs are read from the pinned public203 reference, not copied. Original selected events' legal same-direction edges are checked separately: S identity evidence is not automatically official FP status.
- Complete GT-return denominator rescanned from public203's all-run universe, including no-return runs. All expected dated probes and all-member zero-hit geometry are retained. Published per-date numeric boxes permit the given full-scene IoU/circle coverage arithmetic; completeness of those scene rows and original IoU/Hungarian construction is the external native audit's scope.

E3 averages the three members' counts before absolute error. F1 pools TP/pred/GT across W/G clips for the same physical member, then averages members/groups. It is not ensemble F1. A better E3 can accompany worse mean member W.

## Native reconstruction is a separate input contract

Source entries are freeze206.py, queries.py, owners206.py, score206.py/stage206.py and verify206*.py. They need the separately supplied complete sealed203 native frozen tree, its native score tree and the original verified203 GT-universe JSON, plus the six permitted prepared GT files and polygons. Public numeric ZIPs do not contain complete original per-frame K. No new upstream data or image model is invented. The original/native constants and code fingerprints remain pinned; a newly generated portable receipt is not mislabeled as the original historical run.

Example with those external original inputs (run from repository root; replace paths):

```powershell
$prereg = '0528579a3678f2367a514920ef625f85f9b3fa10'
$root206 = 'D:/your_fresh_206'
uv run --offline --no-sync python tools/study206/freeze206.py --repo . --source203 <sealed203-frozen> --out "$root206/frozen" --prereg $prereg
uv run --offline --no-sync python tools/study206/owners206.py --repo . --frozen "$root206/frozen" --out "$root206/owners" --prereg $prereg --universe203 <original-verified203-universe.json>
$fr = (Get-Content "$root206/frozen/SEALED.sha256").Trim()
$ow = (Get-Content "$root206/owners/SEALED.sha256").Trim()
uv run --offline --no-sync python tools/study206/score206.py --score --repo . --root $root206 --prereg $prereg --frozen-seal $fr --owners-seal $ow --source203-score <sealed203-score> --source203-score-seal dbbc5837f2747024539a395f2c3f239848475fe711dc7cfc38aacc7998928263 --owners-released --field-go
$sc = (Get-Content "$root206/score/SEALED.sha256").Trim()
uv run --offline --no-sync python tools/study206/verify206.py --phase all --field-go --score-worker-finished --repo . --root $root206 --source203 <sealed203-frozen> --source203-score <sealed203-score> --frozen-sha $fr --owners-sha $ow --score-sha $sc
```

Portable copies adjust only repository-root imports/explicit input paths; original/published hashes are in source_fingerprints.json. Exact historical executed snapshots are in executed_sources.zip. Four native entry `--help` checks passed. The native template requires external data and has not been advertised as a public-data-only fresh native rerun. The numeric archive command above is the actually tested public-data entry.

Historical independent audit checked all queries/thresholds, every bound and prefix,85 full matrices/source provenance/windows and supplied graph matchings. Fresh full-frame both-IoU rematching was independently repeated for one changed configuration each on35/33/W; TP_ID was independently recomputed only on these3. G had no change. Do not infer85 independent fresh spatial rematches.

## Scope

Six reused development clips,20 three-member video units,60 original K member tracks; CS saw unlabelled target clips. Held-out routing labels are not a new-hive blind test. This is an oracle-return source diagnostic, not a trained206 model. No GPU, reserve seeds15–32 or test was used. Private engineering synthetic heads are not included. See [results.md](results.md) for both IoUs, unknown taxonomy, coverage, distinct runs and TP/FP/FN.
