# Endpoint peer evidence / 跨成員端點旁證

[Results](results.md). Preregistration `48f6e45bb27c5513d52ecf46f16775cab3d6f534`, `[endpoint-peer-v1]`.

From BeeTrack repository root:

```powershell
uv run --offline --no-sync python tools/study198/replay198.py tools/study198/evidence.json.gz
```

No original images/MOT/GT needed. Keep the already published `tools/study197/evidence.json.gz` beside this study: its full SHA is checked before reading. An explicit `--reference197 <file>` accepts the same archive elsewhere. This avoids duplicating420 historical arms.

Numeric replay recomputes1196 SAME-peer eligibility decisions and N11/B15/I22 from frozen per-date IoUs,20 restricted integer certificates/coherent O masks,420 original graph references, four source-only y/z labels and12 ridge heads, every blind mask/control,600 new graph results/140 groups, all gates and the posthoc3072-configuration exact-add bound. It checks the60 first-stage O physical summaries against the same reused final O. It does not reconstruct IoUs from full trajectories or rerun native matching; those require original data and the independent field checker. Strict TP_ID is not reconstructible from event graphs and is never inferred.

正式資格須同一旁觀成員先辨核心再觀測短段,不把兩個peer各有一半證據拼起來。逐日自由ID覆蓋只作診斷。第一關先用舊目標標註篩研究方向,通過後才做第二關,所以四折不是外部確認。CS也曾見無標註目標片。計數可達不代表此模型學到,O僅是標註可見實體見證。

Synthetic commands (no field inputs):

```powershell
$env:PYTHONPATH=(Get-Location).Path
uv run --offline --no-sync python tools/study198/replay198.py
uv run --offline --no-sync python tools/study198/peers.py
uv run --offline --no-sync python tools/study198/capacity.py
uv run --offline --no-sync python tools/study198/graph198.py
uv run --offline --no-sync python tools/study198/model198.py --self-test
uv run --offline --no-sync python tools/study198/split_inputs198.py --self-test
uv run --offline --no-sync python tools/study198/physical198.py --repo . --self-test
uv run --offline --no-sync python tools/study198/run198.py
uv run --offline --no-sync python tools/study198/verify198.py
```

Native reconstruction separately needs the author's original §197 sealed archive (F60 NPZ `rows`, original masks, scores/physical receipts) and the same permitted six clips' GT/seqinfo/polygons. Required original seals: frozen `e7fd48078f8d0a32bb9018f2bac93dc4b5270d374d5bcd0a631fa97dd4d790e4`, predictions `9a726ff7c097e0f2c595c63e2a50bd4b05390006439bd12c0c539eac4a6f752f`, score `3b21533355dd1ba4ff441c2fd4d6bdc518c22b6b806b456e7c699fdb9a56e7d8`. Detection-only arrays or the public compact197 archive are not those native inputs. No new upstream producer or raw data is supplied here.

The explicit-path sequence below is input-dependent; publication tested numeric/synthetic entry points, not an additional field rerun. Set `$source197` to the original archive root, `$peerRoot` to a fresh output root, `$countRoot` to a different fresh directory. Derive SHA variables from actual files, retaining full integrity checks:

```powershell
function FileSHA($p) { (Get-FileHash $p -Algorithm SHA256).Hash.ToLower() }
$prereg='48f6e45bb27c5513d52ecf46f16775cab3d6f534'
uv run python tools/study198/freeze.py --repo . --source197 "$source197/frozen" --out "$peerRoot/frozen" --prereg $prereg
uv run python tools/study198/capacity.py --input "$peerRoot/frozen/deployment.json" --sha256 (FileSHA "$peerRoot/frozen/deployment.json") --prereg $prereg --phase1 --source-deployment197 "$source197/frozen/deployment.json" --source-deployment-sha (FileSHA "$source197/frozen/deployment.json") --source-summary197 "$source197/score/summary.json" --source-summary-sha (FileSHA "$source197/score/summary.json") --out $countRoot
uv run python tools/study198/phase1.py --repo . --root $peerRoot --source197 $source197 --count "$countRoot/capacity.json" --out "$peerRoot/stage1"
uv run python tools/study198/verify198.py --phase 1 --repo . --root $peerRoot --source197 $source197 --capacity "$peerRoot/stage1/capacity.json" --physical-batch "$peerRoot/stage1/physical_batch.json" --physical-root "$peerRoot/stage1/O_physical" --prereg $prereg --field-go
```

**Stop if the finalized first-stage resource gate fails or remains unknown.** Only a passing count witness with all60 actual physical checks authorizes the second stage:

```powershell
uv run python tools/study198/split_inputs198.py --stage1 "$peerRoot/stage1" --out "$peerRoot/source_shards" --manifest-sha (FileSHA "$peerRoot/stage1/manifest.json") --graph-sha (FileSHA "$peerRoot/stage1/graph_data.json")
uv run python tools/study198/model198.py --run-field --repo . --root $peerRoot --source197 $source197 --shards "$peerRoot/source_shards" --shards-sha (FileSHA "$peerRoot/source_shards/manifest.json") --guard "$peerRoot/stage1/capacity.json" --guard-sha (FileSHA "$peerRoot/stage1/capacity.json") --prereg $prereg
uv run python tools/study198/run198.py --score --repo . --out "$peerRoot/score" --stage1 "$peerRoot/stage1" --peers "$peerRoot/frozen" --source197-frozen "$source197/frozen" --source197-score "$source197/score" --predictions "$peerRoot/predictions" --prereg $prereg
uv run python tools/study198/verify198.py --phase 2 --repo . --root $peerRoot --source197 $source197 --score-root "$peerRoot/score" --prereg $prereg --field-go
```

Training reads only the other3source-domain graph shards per fold. All four models/masks seal before deployment scoring. No GPU, new seed or test split. Public copies require explicit root/source paths and remove only the scorer's drive restriction; fresh source fingerprints/seals need not equal historical ones. The original peer freeze/source197 pins remain enforced.

For the historical bundle, the model producer code hash differs from the portable CLI copy. Supply the original producer directory and a new audit destination, preserving the historical receipts:

```powershell
uv run python tools/study198/verify198.py --phase 2 --repo . --root $historicalPeer --source197 $source197 --score-root "$historicalPeer/score" --prereg $prereg --field-go --source-code-dir $originalCode --audit-out $freshAudit
```

It needs the existing original phase1 receipt under that archive root. This is not the fresh sequence. `verify198_phase1sealed.py` preserves the original first-stage verifier source for provenance; use `verify198.py` for the documented interface. Full verification can consume substantial memory; the first compact command is the normal public path.

Independent acceptance: first-stage1196×2peer/420 legacy graphs/20 proofs/O60; second-stage all y/z/refits/masks,600 graph results,140 groups and I60+24 control sealed native matrices/hashes/events/edges. No fresh native rematching calls in that later audit: it relies on sealed official scores plus197 per-frame owner/rank invariance. Original IoUs/geometry were independently rebuilt from native data. Shared official primitives and decoder definitions are explicit; seal chronology is receipt evidence, not retroactive observation.

Manifest lists original seals and every shipped file. All decoded strings/keys/types are privacy checked;252 decimal spelling changes preserve exactly the parsed JSON. No original MOT/images/individual pixels, full GT trajectories or private machine paths ship. Posthoc bounds are necessary only, not new model thresholds or sufficient recipes.
