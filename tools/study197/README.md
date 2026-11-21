# Endpoint ID cuts / 入口端點切段

[Results](results.md). Preregistration `6c121a384949944dab001318721ffe7599c6e2f4`, `[endpoint-cut-v1]`.

Run from BeeTrack repository root with its existing environment:

```powershell
uv run --offline --no-sync python tools/study197/replay197.py tools/study197/evidence.json.gz
```

No original MOT, images or GT files are needed. Evidence includes1196 local k+3 observation windows/frame/box/confidence/parent lengths, polygons, candidate signed-event deltas, frozen baseline vectors, four source-only fits/masks, seven arms' signed events/legal graphs, integer certificates and the exact saved oracle witness. Replay reconstructs all12 features/H,30phase deltas/source labels, weighted four-fold fits, strict>.10 choices and matched M,420 arm metrics, member-pooled F1, all group E3 and gates, plus explicitly posthoc add-only bounds and singleton-versus-batch gains. It never reruns official scoring or tracker inference.

無原資料的重播含完整局部數值觀測,不是僅相信封存特徵值。原始完整MOT、影像/像素及完整GT軌跡不公開。相同舊設計場景與seed重用,CS曾見無標註目標片;隔離的是router標註/擬合,不是全流程新箱盲驗。

Synthetic checks (no field inputs):

```powershell
$env:PYTHONPATH=(Get-Location).Path
uv run --offline --no-sync python tools/study197/replay197.py
uv run --offline --no-sync python tools/study197/candidates.py
uv run --offline --no-sync python tools/study197/model.py --self-test
uv run --offline --no-sync python tools/study197/oracle.py
uv run --offline --no-sync python tools/study197/local_choice_certificate.py --self-test
uv run --offline --no-sync python tools/study197/physical.py --repo . --self-test
uv run --offline --no-sync python tools/study197/run197.py
uv run --offline --no-sync python tools/study197/verify.py
```

Native reconstruction is separate. Supply the **original §195 native archive**, frozen manifest `4f78f24d7fd13dc2eca9b206501a5326b4e42d476e1a9c4dd1ba98fe4bf61377`, its original score/native_scores.json and full manifests, and the original permitted six clips' GT/seqinfo plus polygons. Only fixed F60 arms are read (head t50, clean W/G t40), not all120 arms. These are already tracked complete NPZ `rows`, not detection arrays. The archive and original data are not included here; differently fingerprinted fresh §195 outputs cannot impersonate that pinned archive.

Set `$source195` to that archive's parent containing frozen/score and `$cutRoot` to a fresh explicit output root:

```powershell
$prereg='6c121a384949944dab001318721ffe7599c6e2f4'
uv run python tools/study197/freeze.py --repo . --source195 "$source195/frozen" --out "$cutRoot/frozen" --prereg $prereg
uv run python tools/study197/model.py --run-field --repo . --root $cutRoot --prereg $prereg
uv run python tools/study197/run197.py --score --repo . --frozen "$cutRoot/frozen" --predictions "$cutRoot/predictions" --source195-score "$source195/score" --out "$cutRoot/score" --prereg $prereg
uv run python tools/study197/verify.py --repo . --root $cutRoot --source195 $source195 --prereg $prereg --field-go
```

Freeze reads no GT; each source fold reads only the other3domains; all four predictions seal before total scoring/oracle. Every current file hash, source195 binding, mask and source boundary remains checked. Public model accepts a freshly verified197 freeze by default; `--require-original-freeze` additionally pins the historical197 freeze. Output-drive restriction is removed from scoring, physical synthetic uses a normal temporary directory, oracle imports its certificate helper as a sibling. These source changes produce their own fingerprints/seals. The recipe is input-dependent, not a claim that publication reran full native field work.

Historical verification instead needs the original197 bundle and original producer source directory; current public source fingerprints differ. Use fresh receipt destinations or copies, never overwrite historical receipts:

```powershell
uv run python tools/study197/verify.py --repo . --root $historicalCut --source195 $source195 --prereg $prereg --field-go --require-original-seals --source-code-dir $originalCode --audit-out $freshAudit
```

This validates the original three exact seals and original producer code hashes. It is not the fresh chain above. The independent verifier can consume substantial memory and reads full native receipts; the compact command is the lightweight public path.

Independent acceptance covers1196 candidates, four refits/masks,420 physical outputs at both IoUs,98 groups,20 tight integer proofs and their same physical O witness. It directly called official match_events84 times (60 K plus24 arm samples); all420 event graphs/F1 were independently rebuilt using exact per-frame raw-index/rank invariance to reuse frame owners. Original detector/linker inference was not rerun. Historical seal order is receipt evidence, not retroactive independent observation.

Count-optimal O also passes the registered mean F1 guard in every hive. Thus it achieves the F1-constrained minimum mean **window error** within this finite family; it does not maximize F1 or prove all tied count optima satisfy F1. It is label-visible, not a deployable model.

Manifest lists original seals and every public file hash. JSON decimal/escape spelling changes only avoid accidental privacy-pattern matches: parsed values are identical. Decoded keys/strings/types are checked; the sole matching M hash is recomputed from its exact canonical metadata. No private machine paths ship.
