# Pair texture source models / 兩蜂局部紋理來源模型

[Results](results.md). Preregistration `c5c55868165aa944f76c9a383597d3a0c519cd63`, `[pair-texture-v1]`.

From the BeeTrack repository root, no original images or GT needed:

```powershell
uv run --offline --no-sync python tools/study196/replay196.py tools/study196/evidence.json.gz
```

This refits both source-only folds from OLD pairs' numeric four-crop12D descriptors, geometry and blind-order hashes; checks all4584 NEW decisions, original/time-shift/four shuffle summaries,382 physical arms' signed-event windows and graph maximum matching, and every mechanism/physical gate. It also recomputes the explicitly posthoc T-minus-C cost decomposition. Q signatures bind aggregate descriptors, not individual pixels: checking actual color-preserving pixel shuffles requires original images and the independent field checker.

新配對首次評分不等於新蜂/新片/新箱。W28/30、G115/161新配對重用已知舊來源蜂,同片/幀也可重用。S是非空片等權,棄權仍在分母;時移沒有身分真值。物理試次只含兩條指定完整真run,接觸後交換ID但框集合不變,不是自然偵測成績。零GT試次照計N與誤報,先加總再算W/F1,零總分母為null。

Synthetic checks (no field data):

```powershell
$env:PYTHONPATH=(Get-Location).Path
uv run --offline --no-sync python tools/study196/replay196.py
uv run --offline --no-sync python tools/study196/features.py --self-test
uv run --offline --no-sync python tools/study196/model.py --self-test
uv run --offline --no-sync python tools/study196/physical.py --repo . --self-test
uv run --offline --no-sync python tools/study196/evaluate.py --self-test
uv run --offline --no-sync python tools/study196/check_texture.py
```

Native reproduction needs original permitted25train JPEGs and officialGT, plus the **original full selection.json** and SELECTION.sha256 sidecar. It does not accept the differently serialized public RGB compact file as that historical input. The existing public RGB producer can regenerate the original selection with those allowed inputs:

```powershell
$rgbPrereg='adab39fbcef0d7cb3e4507d7934598f30a4db388'
uv run python tools/paired_rgb_source/track_choice_source_probe.py --repo . --out $rgbOriginal --prereg $rgbPrereg
(Get-FileHash "$rgbOriginal/selection.json" -Algorithm SHA256).Hash.ToLower()
```

Choose a fresh `$rgbOriginal`; expected selection SHA is `027e127674826303042df7074b3b855bb282b7caaf2e7205da3030cb6a4943a4`. The prior RGB manifest and this archive retain producer/source fingerprints. Selection is sealed before RGB; §196 source pixels are read only after its new roster seals. This is a reproducible input recipe, not a claim of an additional field rerun during publication.

With `$textureRoot` a new explicit output root and that original selection:

```powershell
$prereg='c5c55868165aa944f76c9a383597d3a0c519cd63'
uv run python tools/study196/features.py --repo . --selection "$rgbOriginal/selection.json" --out "$textureRoot/features" --prereg $prereg
uv run python tools/study196/model.py --features "$textureRoot/features/features.json" --out "$textureRoot/predictions" --prereg $prereg
uv run python tools/study196/physical.py --repo . --roster "$textureRoot/features/roster.json" --predictions-dir "$textureRoot/predictions" --out "$textureRoot/physical" --prereg $prereg
uv run python tools/study196/evaluate.py --predictions-dir "$textureRoot/predictions" --physical-dir "$textureRoot/physical" --feature-dir "$textureRoot/features" --out "$textureRoot/score"
uv run python tools/study196/check_texture.py --repo . --root $textureRoot --selection "$rgbOriginal/selection.json" --prereg $prereg --field-go
```

The feature producer uses JPEGs/selection metadata, not GT; physical scoring opens allowed GT only after all predictions seal. Public producer/physical copies remove only the original machine's output-drive restriction. Fresh code fingerprints/seals can differ from historical receipts: checker verifies its actual sibling sources and actual bundle hashes. Do not use a fresh bundle to impersonate the old seal, and do not overwrite old receipts. No GPU, test split or new seeds are required. Full native reconstruction requires original inputs; the first numeric command does not.

Independent verification covered747 source images,336 identity-stratum cells,10 physical summaries,382 arms and4584 decisions; maximum descriptor difference3.44e−15. Original decoding and official low-level event/frame definitions are shared; selection, descriptors, fits, graph matching, windows and aggregation were rebuilt separately. It is a controlled source study, not a new-hive confirmation or TOPICTrack comparison.

Manifest records every public file and original seals. Decimal/JSON-escape spelling only avoids accidental privacy-pattern matches; parsed JSON equals the original. Decoded keys/strings are checked, and the only matching SHA-derived strings are fully recomputed from candidate metadata or Q aggregate serialization. Images, sampled grids, per-pixel values, complete GT tracks and private machine paths do not ship.
