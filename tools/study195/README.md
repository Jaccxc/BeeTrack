# Whole-member router / 整片三成員選軌

Results: [results.md](results.md). Preregistration: `b7e85bb9679d771394a578c6962b4b870f15b5b9`, `[whole-member-router-v1]`.

From the BeeTrack repository root, using its existing uv environment:

```powershell
uv run --offline --no-sync python tools/study195/replay195.py tools/study195/evidence.json.gz
```

No images, native MOT or external labels are needed. The archive supplies20 video units,120 complete-arm signed events/official matching graphs/count vectors, four source-only models and all448 physical configurations. Replay reconstructs30phase bins, checks maximum matching TP and F1, refits J/D, reproduces M and tie choices, enumerates all configurations and verifies the exact F1-constrained across-group optimum. It also recomputes the explicitly posthoc truth-only marginal diagnostic. It does not regenerate detections or infer native tracks.

無原片讀者只需上面命令。重播先跨短片累加各實體成員的TP/預測/GT再算F1,不把平均計數當事件軌跡,也不拼接不同配置的最佳計數與最佳F1。這是舊設計片的router留箱擬合;CS曾見未標註目標片,不是全流程新箱盲驗。

Synthetic checks (no field inputs):

```powershell
$env:PYTHONPATH=(Get-Location).Path
uv run --offline --no-sync python tools/study195/freeze.py --repo . --self-check
uv run --offline --no-sync python tools/study195/model.py
uv run --offline --no-sync python tools/study195/run.py --self-test
uv run --offline --no-sync python tools/study195/verify.py
```

Native reproduction requires separately supplied **already tracked complete-arm NPZ caches**, fields `rows` and `s1`, not detection-only arrays. Provide both `<tag>_<split>_<clip>_0.40.npz` and `_0.50.npz` for the exact registered60 memberclips. Corresponding `runs/dets/<tag>/<split>/<clip>.npy` detector fingerprints, prepared permitted GT/seqinfo, zones and pinned clean linker models must also be available. Head uses `m_dev_strict`, W/G use `m_noW`/`m_noG`. Input/model/algorithm fingerprints remain in evidence. This release does not supply original data or create a new upstream cache producer.

With those existing inputs, assign explicit local paths and a fresh output root:

```powershell
$prereg='b7e85bb9679d771394a578c6962b4b870f15b5b9'
uv run python tools/study195/freeze.py --repo . --out "$routerRoot/frozen" --head-cache $headCache --clean-cache $cleanCache --prereg $prereg --section 195
uv run python tools/study195/run.py --repo . --root $routerRoot --prereg $prereg --stage train-predict
uv run python tools/study195/run.py --repo . --root $routerRoot --prereg $prereg --stage score
uv run python tools/study195/verify.py --repo . --root $routerRoot
```

The first step reads no GT. Training opens only other3domain labels per fold; final evaluation opens all4 only after all predictions seal. The last verifier needs full native outputs and original inputs; it is not the compact command. It refuses an existing independent receipt. These are input-dependent reconstruction commands, not a claim of another field rerun from this public copy. Public `run.py` removes only the live machine's D-drive output restriction. Fresh source fingerprints/seals may differ; optional `freeze.py --reference193` is an additional historical deployment check, not required input. No test, new seeds or GPU are involved.

Independent verification rebuilt120 native events/graphs,120 fullprecision MOT roundtrips,448 configurations, all four fits and Fraction F1 guards. It shares official low-level `by_track/frame_match/match_events`; detector/linker inference was not rerun. Run receipts record seal chronology; the later verifier is not an advance witness.

`manifest.json` lists public file hashes and original seals. Decimal spelling changes only avoid accidental numeric privacy-pattern matches; parsed JSON remains exactly equal. Decoded strings and field types are checked separately. No image pixels, per-frame MOT, private machine paths or full GT trajectories ship.
