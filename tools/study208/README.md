# Study208: natural within-track switch hazards

Run from the BeeTrack repository root, with the repository's existing offline environment:

```powershell
uv run --offline --no-sync python tools/study208/replay208.py --evidence tools/study208/evidence.zip --work-dir D:/beetrack_runs --duplicate-census tools/study208/duplicate_census.json
```

This numeric replay needs no images, crops, individual GT identities, native MOT files or tensor weights. It checks all 12 source-negative calibration thresholds, 18 fixed target score arrays, strict threshold decisions, date NMS, the displaced-score control, cut and mutual-nearest relinking, signed events, all 30 window offsets, and both-IoU event-graph maximum matchings for all 54 physical configurations. The six clips contain one member each; white/gray clips are pooled before W and F1. These are not three-member ensemble E3 scores. Target AP pools clips within a hive before computing tied-score AP; its population is represented valid finite-score known adjacent gaps. Unscored known positives remain in the separate pipeline-FN denominator. Unknown identity is not a negative label.

These are six previously used development clips, not new hives. Upstream detector preparation already used unlabeled target footage; held-out supervision here concerns the switch model's labels and fitting. Passing a single-member study would not establish a multi-seed result.

All heads register 809 parameters, but the geometry-only arm has only 129 parameters with a possible loss gradient. This compares inputs/losses, not equal effective model capacity.

The compact K path stores original-run side changes and endpoint geometry needed by the actual selected cuts. Its row/source fingerprints bind the projection to frozen inputs. The event graphs are annotated sufficient statistics: replaying their maximum matching does not recompute original frame IoUs or prove GT identity from pixels. All natural image/crop arrays and original GT IDs are omitted. Fixed probabilities are not a reproduction of CNN forward inference or training.

Public numeric replay is distinct from external-data reconstruction. The latter requires the six permitted original training clips/annotations and the separately retained study206 frozen K bundle, whose seal is checked by `data208.py`; it is not included in this archive. `data208.py`, `label208.py`, `fit208.py`, `physical208.py` and their sibling helpers preserve the reconstruction implementation. The public copies move helper imports into this directory and the existing `tools/study203` directory. Output restrictions to a fresh D: directory remain explicit. Fresh outputs have different source-code seals; they must not be substituted for historical byte-pinned receipts. No fresh full training/native reconstruction is claimed by this release.

Source-level checks, which do not read field data:

```powershell
uv run --offline --no-sync python tools/study208/replay208.py --self-test
```

繁中說明: 這個輕量包可重算固定分數之後的門檻、切點、重接、54份事件與窗數、兩個 IoU 配對圖及投資條件。它保留沒有輸入或裁片不可用的已知正例分母,不把未知當陰性。它不提供自然裁片、原圖或個別 GT 身分,也不宣稱只靠這個包就能重新訓練影像模型。原生重建需要另備允許的六片素材與封存 K 快取。

See `results.md` for the numerical results and `release_manifest.json` for shipped-file hashes, upstream seals and the independent audit scope.
