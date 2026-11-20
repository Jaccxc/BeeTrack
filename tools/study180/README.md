# Source-only hard-negative training (study 180)

The two fine-tuning arms share architecture, initialization, positive samples, candidate locations and training budget. H trains against the highest-scoring quarter of the source model's negative candidates. U selects a uniformly random quarter of the same pool. Source-tail frames 1–488 select thresholds. Frames 513–1000 evaluate all three arms, including the unchanged model. No cross-hive evaluation was triggered by the registered gate.

The held clips contained 27 true events on hive 33 and 16 on hive 35. The H-minus-U gains in position-and-time F1 were 0.0187 and 0.0430. Window error increased on hive 33 and decreased on hive 35. Both F1 gains fell below the registered 0.05 threshold.

### Reproduce the results

The published event lists use one-based frames relative to the 488-frame evaluation interval. The same standard-library scorer used for study 179 replays all six arms.

```sh
python tools/study179/replay179.py docs/study180_events.json.gz
```

To repeat training, first reproduce the study 179 caches and freezes, then run the commands below from the repository root. The prior directory contains study 179 outputs. Each freeze loads the published numeric T weights and checks them against the prior model state. It saves the fitted weights, calibration metrics and sealed predictions under the new output directory.

```sh
uv run python tools/study180/run_mining.py freeze --source 33 --repo . --prior results179 --out results180 --prereg d1e649e6734fe28888cf9458496401e25147df72
uv run python tools/study180/run_mining.py freeze --source 35 --repo . --prior results179 --out results180 --prereg d1e649e6734fe28888cf9458496401e25147df72
uv run python tools/study180/run_mining.py score --repo . --prior results179 --out results180 --prereg d1e649e6734fe28888cf9458496401e25147df72
```
