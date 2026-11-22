# Study211: peer event capacity

From the BeeTrack repository root, replay both separately registered assessments:

```powershell
uv run --offline --no-sync python tools/study211/replay211.py tools/study211/evidence211.json.gz --joint tools/study211/joint211.json.gz --repo . --out D:/beetrack_runs/study211.replay.json
```

Omit `--joint` for the primary 22 configurations alone. The joint family contains 86 configurations including those 22, not 108 distinct configurations. No images, neural inference or original annotations are needed for this numeric replay. See [results](results.md) for both policy tables and limitations.

The replay recomputes K-event deletion labels and peer features from complete anonymous legal graphs, four source-only constrained Q/L/X fits, policy choices, maximum event matching at both IoU thresholds, and 30-phase directional window errors. W/G pool integer numerators and event counts across their two clips. This is one receiver per clip; W is not three-member E3. Shared scoring/window helpers are the published `tools/study209/replay209.py` and `replay_release209.py`. Joint choices reuse the primary coefficients without fitting to joint outcomes.

J counts left events with any legal neighbor; M is capacity-one maximum matching on the same graph. Peer0/peer1 and true-GT vertices occupy separate anonymous namespaces. Cross-state event IDs establish numeric linkage, not biological identity. The packages contain signed-event vertices and legal edges, not boxes or frame-level ownership reconstruction.

## Evidence integrity

| Input | SHA-256 |
| --- | --- |
| evidence211.json.gz | 4bc397f359905a9df770f15af755d4bf7148af762c1f0ed9f694aaedfd69f7a0 |
| joint211.json.gz | 4f61018500e43e3bfd058ed4acb86bb51d0f66abe1827b46905b52752e5e262c |

Primary independent receipt: `665e00b436924f963a985b192cbbae8e0097f3359f9ea073f44d22a8ba409d34`. Joint independent receipt: `7a62d84d0c67bd86e9db0f336879e00594fd1e85fd2af5d245f00318bc9e56b3`; its manifest seal is `f92e21d6fb58ffc7b7bcf66efba45db49d47a265cd6285ce46e8d9b2279e9a9d`. Producer/assessment seals are inside each package. The file manifest records every shipped file; portable versus original source hashes are separately recorded in `source_provenance.json`.

## External native reconstruction

The included producers and audit sources expose historical input contracts. They require separately supplied original six-clip K/peer matrices, study208 inputs and study209/210 sealed catalogs/features/predictions, annotations/zones, and exact registered seals. They are not a fresh-data download pipeline. Raw assets, images, MOT matrices, GT rectangles and individual GT identities are not distributed here.

`freeze211.py` builds peer states from explicit sources; `true211.py` creates source labels and assessment after its gates; `fit211.py` seals four source-only folds. `build_joint_job211.py` creates the joint input contract; `run_joint211.py` separates peer-only preparation from gated assessment. `audit211.py`/`field211.py` and `verify_joint_stage211.py` check native scenes, sharing official event/legal-edge definitions and ordered geometry helpers. `verify_joint211.py` independently implements merge/clock/matching/selection arithmetic. Some native output guards retain the original D-drive policy. Code-receipt pins remain enforced: portable modified sources are not byte-identical to original execution, and fresh outputs must not be passed off as pinned historical seals.

The command above verifies released numeric evidence, not new frame-level geometry, original owner assignments or images. Native verification and numeric replay have distinct scopes. The six old development clips and single fixed receivers provide no new-seed or external-hive test; upstream CS already saw the unlabeled target domain.
