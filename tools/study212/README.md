# Study212: observed-date phases

Run from the BeeTrack repository root:

```powershell
uv run --offline --no-sync python tools/study212/replay212.py tools/study212/evidence212.json.gz --public211 tools/study211/joint211.json.gz --repo . --out D:/beetrack_runs/study212.replay.json
uv run --offline --no-sync python tools/study212/clock212.py
```

The first command recomputes date/inside features, all16 deterministic random salts, four source-only folds (12 G/R/T fits), S using T's coefficients, event probabilities, guarded choices, all18 original K member Briers, investment/attribution checks and the six policy scores. The 86 configurations' signed events/legal graphs/windows are reused from the exact published study211 bundle. The second command proves the separately marked post-result 53→61 clock identity, not a new deployment score or F1 result. See [results](results.md).

| Input | SHA-256 |
| --- | --- |
| evidence212.json.gz | 19031cee0dd4388d4bbf7316afa6ab6da6d9654ff5fb2156e51aa5106b665131 |
| ../study211/joint211.json.gz | 4f61018500e43e3bfd058ed4acb86bb51d0f66abe1827b46905b52752e5e262c |

Expected full report SHA-256: `44f5f6321937c2847e94628f0a7106564b34ba952e85c1de3c0784504056a980`. Independent native receipt: `d32e925c1e84bc17fb9ee95df63a8f1caa9406d0f067c8ae8dc2ecb84ee56a52`. Original producer/verification and separately modified portable source hashes are in the evidence receipts and `source_provenance.json`; `manifest.json` lists every shipped file.

Upstream numeric serialization SHA was `2e1b55e2eea423fcc272561c0444369c34c88339187b2649fa4e27ae5dd9499a`. Two decimal tokens were written with equivalent exponent spelling during release scanning. Their decoded float types/values and the complete decoded JSON are unchanged; the final released bytes reproduce the identical report SHA above.

The compact increment contains 1058 deduplicated ordered native-date/inside sequences, opaque event linkage, source rows/order, model/choice references and anonymous new peer truth graphs. The target label is existence of any legal GT event neighbor, not biological identity, matched TP or deletion necessity. The random controls preserve phase sample counts. Unstable phases can also describe a true short crossing.

## Native inputs and verification limits

`prepare212.py` requires the separately supplied original study211 frozen K/peer matrices and joint job/choices, linked to the old study208 inputs. `truth212.py` and `truth212_late.py` need the original annotations/zones and enforce source-label and late-target gates; `fit212.py`, `finalize212.py` and `evaluate212.py` implement the source-only fitting and assessment contracts. External data and exact historical seals must be supplied explicitly; the package does not download source imagery or provide original matrices. Some native output guards retain the original D-drive policy. Portable source path changes have separate hashes; historical code-receipt checks are not disabled, and fresh output must not be claimed to have the old seal.

Native verification reconstructed all86 physical carriers plus18 original K date/inside inputs, independently refit 12 source models, and recomputed every guard/choice. Full-scene owners and maximum event matching were freshly checked at both IoUs for all18 original K and five known singleton scenes, sharing official low-level geometry. Old86 outcomes reuse their already verified graphs/windows; there were zero new86 full MOT scorers. `audit212.py` and `field212_*.py` preserve this scope.

Numeric replay does not reassign GT ownership from boxes, decode images or repeat frame matching. Forensic owner/net/timing summaries are verified native findings, not geometry recomputed from the public bundle. No raw images, original MOT, GT boxes or individual GT identities are distributed. Fixed old clips/receivers and correlated peer members do not establish external-hive generalization; upstream CS already used the unlabeled target domain.
