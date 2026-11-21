# Endpoint identity / 切口身分、同軌保護與觀測中斷

[Results](results.md). Preregistration `530ee05ab8401e4313a4fc53e50318d2a37fe74e`, `[endpoint-identity-v1]`.

Run from the BeeTrack repository root:

```powershell
uv run --offline --no-sync python tools/study199/replay199.py tools/study199/evidence.json.gz
```

Keep the published197/198 archives under `tools/study197/` and `tools/study198/`. Their complete gzip SHA values are verified first; `--reference197 <file> --reference198 <file>` accepts those same bytes elsewhere.197 is checked without loading its large JSON, and198 supplies the unchanged features/state graphs. No original MOT, images or GT files are needed for this numeric command.

Replay checks1196 cut classifications at.5/.3 from annotated official owners/degrees,40 restricted binary certificates, two joint minimum-S exact integer proofs/coherent masks, four source-only direct classifiers/12 heads, all eight arms' graph counts/F1/E3, classification/ROC/migration tables,24 actual head-witness score JSONs, and censor-run deduplication. It derives the4/0 minimum, the posthoc mean-F1 guard result, distinct-run1/0 and the exact four239 observations from the same saved masks. It does not rerun the integer optimizer or native MOT scorer.

**Recompute boundary:** owner rows preserve official assignment, explicit originalGT inputID/run, date/side, all threshold neighbors of the prediction and annotated degree of each GT neighbor. Numeric replay validates these annotations and the resulting isolation/class logic. It does not regenerate full-frame geometry or establish missing-neighbor completeness from a compressed row. The independent native audit rebuilt full-frame ownership/threshold degrees and sides from the original allowed inputs. Per-row hashes and raw indices link the two scopes. A GT inputID is never inferred by canonicalID modulo100000.

S means the two cut rows uniquely map to one officialGT gap segment; D means accepted rows with distinct original inputIDs; other cases, including one originalID split into separate gap runs, are U. This is annotation-relative, not biological identity. The four35 S cuts share GT239/run0 and dates3411→3412; their parent endpoints plus cut rows did not all qualify as one isolated GT run, so no return is confirmed. Whether every solution must cut this particular239 run remains **undetermined**.

Fourfold identity fitting is unconditional on capacity. Only eligible D/S rows in the other3hives are fitted; U has no training weight. Add/drop classes, not D/S labels, share clip weight. Empty fits abstain; pD is clipped and must be strictly>.5. Model fitting opens domain label shards separately, not the heldout shard or old198 label models. These old scenes have repeatedly informed design, and CS saw unlabeled target images: this is not a new-hive blind test.

Synthetic commands (no field inputs):

```powershell
$env:PYTHONPATH=(Get-Location).Path
uv run --offline --no-sync python tools/study199/replay199.py
uv run --offline --no-sync python tools/study199/owners199.py --repo . --self-test
uv run --offline --no-sync python tools/study199/cap199.py
uv run --offline --no-sync python tools/study199/model199.py --self-test
uv run --offline --no-sync python tools/study199/score199.py
uv run --offline --no-sync python tools/study199/verify199.py
```

Native reconstruction needs the author's original197 fixed-F60 archive and198 frozen features, masks and first-stage graphs, plus the original permitted six clips' GT/seqinfo/polygons. Public compact archives are not those native inputs. Original source pins197 `e7fd48078f8d0a32bb9018f2bac93dc4b5270d374d5bcd0a631fa97dd4d790e4`,198 features `d53e70b757984aa42a7d2e8b6c6ea0f287bedfe564d3a6a1fde685dc2d85765f`, masks `9adafc2ecc6dff3ba85b674653bd3a55ce91183bf0917ce34bfea2e5c14c5843` remain enforced in code. No images/inference, new seeds or test data are requested. Set `$identityRoot` to a fresh output root and supply `$source197`/`$source198` explicitly:

```powershell
function FileSHA($p) { (Get-FileHash $p -Algorithm SHA256).Hash.ToLower() }
$prereg='530ee05ab8401e4313a4fc53e50318d2a37fe74e'
uv run python tools/study199/owners199.py --repo . --source197 "$source197/frozen" --out "$identityRoot/owners" --prereg $prereg
uv run python tools/study199/model199.py --run-field --repo . --root $identityRoot --peer198 $source198 --owners "$identityRoot/owners" --owners-sha (FileSHA "$identityRoot/owners/manifest.json") --prereg $prereg
uv run python tools/study199/cap199.py --owners "$identityRoot/owners" --owners-seal (FileSHA "$identityRoot/owners/manifest.json") --graphs "$source198/stage1/graph_data.json" --graphs-sha (FileSHA "$source198/stage1/graph_data.json") --out "$identityRoot/capacity_v2" --prereg $prereg
uv run python tools/study199/score199.py --score --repo . --out "$identityRoot/score" --source197-frozen "$source197/frozen" --peer198 $source198 --owners "$identityRoot/owners" --capacity "$identityRoot/capacity_v2" --predictions "$identityRoot/predictions" --prereg $prereg
uv run python tools/study199/verify199.py --field-go --repo . --root $identityRoot --source197 $source197 --source198 $source198 --prereg $prereg
```

This is an input-dependent recipe, not an additional field rerun made during publication. Portable copies require explicit paths and remove the score drive restriction. Current stage seals/code hashes are always checked; `verify199.py --require-original-seals` additionally pins the original four199 stages. For an existing historical bundle with original owner ledger/receipt, preserve it with a fresh audit directory and the original producer code:

```powershell
uv run python tools/study199/verify199.py --field-go --repo . --root $historical199 --source197 $source197 --source198 $source198 --prereg $prereg --require-original-seals --reuse-verified-owners --source-code-dir $original199Code --audit-out $freshAudit
```

It verifies the saved owner ledger hash before copying it and writes new reports separately. Fresh outputs must not impersonate the old seals. `verify199_owners_run.py` preserves the initial owner-audit source for provenance; use the documented final CLI. Full native audits can consume substantial memory; compact replay is the normal reader path.

Physical accounting:216 requests were115 fresh unique outputs plus101 within-job reuses. Independent audit checked all216 sealed matrix/hash/rank/event/graph references, without fresh per-arm native rematching; graph replay does not infer strict TP_ID. The same head witnesses prove minimumS4/0 even with hive-mean official F1@.5≥K. That is not F1 maximization, an each-seed/.3 guard, or four independent bees. Supported censor observations0/40/11/9 deduplicate to0/15/4/4 annotated runs, not new independent specimens. Seal chronology is receipt evidence, not a retroactive witness.

Manifest hashes every public file and original stage. JSON spelling changes preserve exactly the parsed values;108 finite decimal pattern hits use equivalent notation. Two copies of one row-SHA were independently recomputed from the pinned native numeric row before equal-value escaping. All other decoded strings/keys are strictly checked. No original MOT, pixels, fullGT trajectories or private machine paths ship.
