# Boundary video event model (study 179)

One small convolutional network estimates entries and exits at 32 positions along the entrance polygon. Arm T uses nine frames spanning one second at 25 fps. Arm S repeats the current frame through the same network. Both arms use the same sampling geometry and training budget. Predictions contain the source frame, boundary cell and direction. They have no bee identity or box.

The temporal arm reached a position-and-time F1 of 0.1933 on hive 35 and 0.0649 on hive 33. Its 30-second window error exceeded P0 on both hives. The weights are research artifacts for reproducing this result. The registered production candidate remains P0.

### Replay the published result

From the repository root, this command needs only Python's standard library.

```sh
python tools/study179/replay179.py docs/study179_events.json.gz
```

Each fold lists the target ground-truth events, the inferred events from each arm, and the source calibration threshold. The replay reproduces maximum-cardinality matches by direction and the 30-phase window error, including windows cut by a sequence boundary.

### Infer events from images

Supply a directory of contiguous, numerically named JPEGs at 25 fps and a polygon JSON file containing an array of raw-image pixel vertex pairs. Use a writable output directory with space for the temporary strip cache. Weights are named by their **training** hive. For example, the following runs the model trained on hive 33.

```sh
uv run python tools/study179/predict_events.py --weights tools/study179/weights/source33_T.npz --frames frames --polygon entrance.json --output events.json
```

### Repeat training and evaluation

Use the prepared BeeTrack data described by the repository. Pass the same output directory and registration SHA `ce382bc2b1b1a986db990a5d68a5eb2b696cfa39` to each command. Run `prepare` for both clips (33 and 35) and both parts (head and tail), then `freeze` once for each source. Run `score` last. Preparation caches the oriented grayscale patches. Each freeze fits both arms, sets their thresholds on the source tail and seals the opposite head's predictions before the comparison with target labels.

```sh
uv run python tools/study179/run_boundary.py prepare --repo . --out results --prereg ce382bc2b1b1a986db990a5d68a5eb2b696cfa39 --clip 33 --part head
uv run python tools/study179/run_boundary.py freeze --repo . --out results --prereg ce382bc2b1b1a986db990a5d68a5eb2b696cfa39 --source 33
uv run python tools/study179/run_boundary.py score --repo . --out results --prereg ce382bc2b1b1a986db990a5d68a5eb2b696cfa39
```
