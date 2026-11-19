# BeeTrack results, v2 protocol (draft)

Status: draft. This file is the number source for the final report under the v2 protocol. Every number is copied from `docs/experiment_log.md`; the section is in brackets, like (§95). `TODO(§N)` means the number was not found in the log.

- The test column (Table D) stays empty ("—") until the single locked test run (§125). Nothing in this file was measured on test.
- The numbers in `RESULTS.md` (the "/341" ones) belong to the old protocol. They are replaced, not mixed in (§2, §125).
- All tables use the 30-second window counting error unless said otherwise. Lower is better.

## 1. Protocol

### 1.1 Metrics

- Main metric: 30-second window counting error, in %. Cut the video into 30-second windows, compare the predicted in count and out count with the true ones window by window, take the absolute differences, repeat for 30 window start offsets and average, sum in and out separately, divide by the number of GT events (§95, §125).
- Counting rule: an in-event is a track whose first centre is outside the entrance zone and last centre is inside; out is the reverse. An ID with a gap of more than 25 frames counts as two bees (§0, §2).
- Event time is the frame of the last crossing of the track in that direction (§41). The event rule used by P0 is locked as in §92 (§125).
- Secondary metric: event F1 at IoU 0.5 (and 0.3), with matched / false / missed events, and the bias = predicted events minus GT events (§1, §95, §125).
- Test pooling: per video, average the 30 offsets; add the 5 videos; divide by the 405 GT events of the 5 videos (§125, §2).

### 1.2 Splits and GT event counts

| Split | Content | GT events | Source |
|---|---|---:|---|
| val + dev ("seen hives") | val (BEE24-10/15/24/28 and the last 1000 frames of BEE24-33 / 35) plus dev (BEE24-33-head, BEE24-35-head) | 486 | §0 (173 + 313), §91 |
| 35-head | BEE24-35, frames 1..3950 | 129 | §148 |
| 33-head | BEE24-33, frames 1..3950 | 184 | §148 |
| 35-tail | BEE24-35, frames 4001..5000 | 37 | §152 |
| 33-tail | BEE24-33, frames 4001..5000 | 50 | §152 |
| test | BEE24-12, 13, 16, 34, 36 (10,762 frames: 250, 258, 254, 5,000, 5,000) | 405 (1/3, 10/11, 26/37, 84/88, 91/54 in/out per video) | §0, §2, §125 |

- "val + dev" measures the tracking side: the four-hive detectors saw these hives in training (§42).
- "Leave-one-hive-out" (留一蜂箱) is the proxy for an unseen hive. Fold 留 35 trains without hive 35 and is scored on 35-head. Fold 留 33 trains without hive 33 and is scored on 33-head (§64, §95).
- 9 detector seeds per fold, split into 3 groups by seed number: 0-2, 3-5, 6-8 (§95). The group range (max minus min of the 3 group values) is the noise scale (§95).
- Tails are the same videos, 50 frames after the head. They were not used before §152 (§152).

### 1.3 Main pipeline P0

- Detector: YOLO11-s at fixed 640. Test detectors: `y11s_640`, `y11s4_s1`, `y11s4_s2` (four hives seen) (§91, §125).
- First stage: PASS1 as it is (`track.PRESETS['link']` without linking and minimum length; "v0" / "舊" of §74). A tracklet may survive a skip of 2 to 4 frames (33-head has 2016 skips) (§128). §74 picked the "cut at holes" version (新二, v2) as the base for the linker experiments, but the locked P0 numbers come from v0 (§91, the `q_t74_v0_*` files), and cutting at skips was tested again as S1 and not adopted (§129).
- Linker: TrackletLink, threshold 0.15 (§74, §91). For test the linker is `m_dev_strict.json` (§125). For val + dev reproduction the three cross-fitted linkers of §74 are used, because `m_dev_strict.json` saw dev while fitting; on val + dev it gives 19.56% (report only) (§125, clarification 1).
- Drop tracks with fewer than 3 boxes, so tracks of 1 or 2 boxes (`drop_short(rows, 3, 0.0)`; the log says "丟掉短於 3 的"); then the event rule locked in §92 (§91, §125).
- Val + dev numbers per detector are the mean over the 3 cross-fitted linkers (§91).

### 1.4 Arms

| Arm | Definition | Introduced |
|---|---|---|
| P0 | main setting above, fixed 640 | §91, §92, §125 |
| P1 | like P0, but per video pick the scale by flicker. Run 352, 448, 640, 960, 1280, 1600 on the whole video, keep boxes with score >= 0.25, flicker = mean absolute change of box count between neighbouring frames divided by mean box count; pick the smallest flicker; a relative difference under 2% is a tie and the smaller scale wins; videos under 1000 frames use 640 | §91 (rule), §68 (flicker), §92 (side arm) |
| E3 | 3 detectors (or 3 seeds) each run the whole P0 pipeline; per 30 s window the in and out counts are averaged (fractions allowed) | §95 |
| F8 | one detector runs on the 8 frame symmetries (original, left-right, up-down, rotate 180, transpose, anti-transpose, rotate 90 cw and ccw); each version runs P0; window counts averaged over the 8 | §143 (4 versions, F4), §144 (8 versions) |
| E3 x F8 | E3 and F8 together: 3 x 8 = 24 members averaged | §147 |
| E3 x P1 | each of the 3 detectors runs P0 at the scale P1 picks; the 3 window counts are averaged | §148 |
| E3 x P1 x F8 | each detector runs the 8 symmetries at its P1 scale; 24 members averaged | §149 |
| E3 x P1↓ x F8 | like E3 x P1 x F8, but a P1 pick above 640 (960, 1280, 1600) becomes 640. Designed after looking at §148 and §149 (post hoc) | §151 |
| Calibrated ByteTrack (校正 ByteTrack) | default ByteTrack; in and out counts multiplied by k = true total / predicted total, fitted once per detector on val + dev and then frozen. Test k_in / k_out: 2.016 / 1.396, 1.866 / 1.364, 1.969 / 1.192 for the 3 detectors | §93, §125 |
| Default ByteTrack | Ultralytics `bytetrack.yaml` defaults, no calibration | §93, §96, §125 |
| Paper's tracker A (論文的 tracker A) | the paper's tracker with its ReID loading fixed ("二修"), threshold 0.2, min_hits 3; the setting was picked on the proxy, not on test | §96, §99, §102, §125 |

## 2. Table A: val + dev (seen hives, 486 GT events)

Each cell is the 30 s window error in %. For the single-detector arms the three columns are the three detectors (each is a mean over the 3 cross-fitted linkers). E3-type arms mix the 3 detectors into one number, so the three columns are the three cross-fitted linkers instead (marked "linker").

| Arm | y11s_640 | y11s4_s1 | y11s4_s2 | Mean | Range over detectors | Event F1@0.5 | Source |
|---|---:|---:|---:|---:|---:|---:|---|
| P0 | 18.89 | 17.34 | 19.15 | 18.46 | 1.81 (§96) | 0.740 (0.753 / 0.740 / 0.728) | §91, §96, §125 |
| P1 | 18.82 | 16.24 | 18.90 | 17.99 | 2.66 (computed) | 0.741 (0.749 / 0.738 / 0.735) | §91, §125 |
| Default ByteTrack | 41.37 | 38.13 | 41.04 | 40.18 | 3.24 (§96) | 0.545 | §93, §96 |
| Calibrated ByteTrack | 33.78 | 29.09 | 41.60 | 34.82 | 12.51 (§96) | TODO(§93) | §93, §96 |
| Paper's tracker A (0.2, 3) | 36.63 | 34.02 | 35.10 | 35.25 | 2.61 (§102) | 0.592 | §102, §125 |
| F8 | 15.81 | 15.51 | 16.83 | 16.05 | 1.32 (computed) | TODO(§144) | §144 |
| E3 (3 detectors, one number) | - | - | - | 17.07 | - | n/a (averaged counts) | §144, §148 |
| E3 x F8 (linker 1 / 2 / 3) | 15.48 | 15.40 | 15.72 | 15.53 | - | n/a (averaged counts) | §147 |
| E3 x P1 (linker 1 / 2 / 3) | 16.26 | 16.32 | 16.95 | 16.51 | - | n/a (averaged counts) | §148 |
| E3 x P1 x F8 (linker 1 / 2 / 3) | 15.67 | 15.22 | 15.64 | 15.51 | - | n/a (averaged counts) | §149 |
| E3 x P1↓ x F8 (linker 1 / 2 / 3) | 15.77 | 15.51 | 15.94 | 15.74 | - | n/a (averaged counts) | §151 |

Notes.
- Other values the log gives: F4 15.93 / 15.49 / 17.40, mean 16.27 (§144); P1 x F8 mean 15.98 (§149).
- P0 event F1@0.3 is 0.750 and default ByteTrack 0.557 (§96).
- E3 x P1↓ x F8 is 2.72 points below P0 (§151). It changes only one val + dev cell: `y11s4_s1` on BEE24-33-head goes from 960 to 640 (§151).
- The three linkers in the E3-type rows are not the same thing as the three detectors in the other rows; do not compare columns across the two groups, compare the mean (§147, §148, §149, §151).
- P0 per linker: `y11s_640` 18.74 / 19.01 / 18.93, `y11s4_s1` 17.48 / 16.49 / 18.07, `y11s4_s2` 19.25 / 18.16 / 20.04 (§91).
- The "keeps seen hives safe" gate for every new arm was "mean over the 3 linkers not above 18.46%" (§147, §148, §149, §151).
- Reproduce the P0, P0 with `m_dev_strict.json`, default and calibrated ByteTrack rows (and the test k) from the repo alone: `PYTHONPATH=. uv run python tools/repro_table_a.py` (CPU, about 5 min; asserts every number above, outputs byte-identical to the original pipeline).

## 3. Table B: leave-one-hive-out heads (9 seeds, 3 groups)

Group cells are seed 0-2 / 3-5 / 6-8. Mean is the mean of the three groups. Range is max minus min of the three groups.

| Row | 留 35 (35-head, GT 129) groups | Mean | Range | 留 33 (33-head, GT 184) groups | Mean | Range |
|---|---|---:|---:|---|---:|---:|
| P0 single (§151) | 17.76 / 19.93 / 28.85 | 22.18 | 11.09 | 23.27 / 26.50 / 21.94 | 23.90 | 4.55 |
| E3 (§147) | 15.87 / 17.81 / 27.68 | 20.45 | 11.82 | 16.80 / 23.95 / 17.13 | 19.29 | 7.15 |
| E3 x F8 (§151) | 15.07 / 15.29 / 20.80 | 17.05 | 5.73 | 17.42 / 17.62 / 18.35 | 17.80 | 0.92 |
| E3 x P1 (§151) | 8.65 / 11.94 / 12.63 | 11.07 | 3.98 | 20.57 / 22.87 / 22.40 | 21.95 | 2.31 |
| E3 x P1 x F8 (§151) | 9.26 / 11.76 / 12.80 | 11.28 | 3.54 | 16.05 / 22.07 / 17.80 | 18.64 | 6.02 |
| E3 x P1↓ x F8 (§151) | 9.26 / 10.54 / 10.98 | 10.26 | 1.72 | 17.42 / 17.62 / 18.35 | 17.80 | 0.92 |
| Calibrated ByteTrack E3 (§147) | 42.84 / 43.09 / 52.09 | 46.01 | 9.25 | 41.32 / 47.48 / 35.51 | 41.44 | 11.97 |
| Default ByteTrack, single (§95) | 27.27 / 25.58 / 27.84 | 26.90 | 2.26 | 35.92 / 27.46 / 36.01 | 33.13 | 8.55 |
| Default ByteTrack E3 (§95) | 25.99 / 23.45 / 26.56 | 25.33 | 3.11 | 35.16 / 24.82 / 34.14 | 31.37 | 10.34 |

(Mean and range of the two default ByteTrack rows are computed here from the §95 group values; the single-row means equal the 9-seed means in §95.)

Extra, 9-seed single-seed means (range over the 9 seeds) from §95:
- P0 single: 留 35 22.18 (33.80), 留 33 23.90 (13.26).
- Default ByteTrack single: 留 35 26.90 (10.90), 留 33 33.13 (15.18).
- Calibrated ByteTrack single: 留 35 47.38 (45.60), 留 33 44.57 (28.84).
- Event F1 of P0 over 9 seeds: 留 35 0.765, 留 33 0.324 (§95). Default ByteTrack: 0.561 and 0.254 (§115, §118).
- Bias: P0 over-counts on 留 35 for all 9 seeds (+25.0 events on average); on 留 33 the bias is about zero on average (+3.6) with both signs (§95).

### 3.1 Criteria per arm and fold

- Criterion 1 (generalization): per group d = arm minus P0 single; the mean of the three d must be negative and its absolute value larger than max(range of the arm, range of P0 single) (§95).
- Criterion 2 (second baseline): per group arm minus calibrated ByteTrack E3; same reading with the larger range (§95).

| Arm | Fold | Criterion 1 | Criterion 2 | Source |
|---|---|---|---|---|
| E3 | 留 35 | no: d mean −1.73, threshold 11.82 | yes: −25.55 vs 11.82 | §95 |
| E3 | 留 33 | no: −4.61 vs 7.15 | yes: −22.14 vs 11.97 | §95 |
| E3 x F8 | 留 35 | no: −5.13 vs 11.09 | yes: −28.96 vs 9.25 | §147 |
| E3 x F8 | 留 33 | yes: −6.11 vs 4.55 (1.56 points of margin) | yes: −23.64 vs 11.97 | §147 |
| E3 x P1 | 留 35 | yes, barely: −11.11 vs 11.09 (0.02 points) | yes: −34.94 vs 9.25 | §148 |
| E3 x P1 | 留 33 | no: −1.96 vs 4.55 | yes: −19.49 vs 11.97 | §148 |
| E3 x P1 x F8 | 留 35 | no: −10.91 vs 11.09 (short by 0.19) | yes: −34.73 vs 9.25 | §149 |
| E3 x P1 x F8 | 留 33 | no: −5.26 vs 6.02 (short by 0.76) | yes: −22.79 vs 11.97 | §149 |
| E3 x P1↓ x F8 | 留 35 | yes: −11.92 vs 11.09 (0.83 points of margin) | yes: −35.75 vs 9.25 | §151 |
| E3 x P1↓ x F8 | 留 33 | yes: −6.11 vs 4.55 (same as E3 x F8) | yes: −23.64 vs 11.97 | §151 |

Reading notes.
- Before §151 no arm passed criterion 1 on both folds: 留 35 passed only with E3 x P1 (by 0.02 points) and 留 33 only with E3 x F8 (§148, §149, §150).
- E3 x P1↓ x F8 was designed after looking at the per-seed results of §148 and §149 (a P1 pick below 640 helped 7 of 7 seeds, above 640 only 5 of 11) (§150, §151). So its pass on the two heads is post hoc. It is the 8th rule tried on the same two heads (§151). It is not counted as criterion (b) reached (§151).
- On 留 33 all 9 seeds go back to 640 under P1↓, so the arm equals E3 x F8 there; the 留 33 pass comes entirely from F8 (§151).
- On 留 35 only seeds 4 and 8 change (picked 1280 and 960, now 640): window error at those scales 20.56 and 19.96, at 640 11.88 and 14.10 (§151).
- Seen-hive gate for E3 x P1↓ x F8: 15.74% (P0 18.46%) (§151).

## 4. Table C: unseen-hive tails (§152 result)

Tails are the 1000 frames after each head (GT 37 events on 35-tail, 50 on 33-tail). The rules were registered before the run and not changed (§152). Cells are groups 0-2 / 3-5 / 6-8.

| Row | 留 35 (35-tail) groups | Mean | Range | 留 33 (33-tail) groups | Mean | Range |
|---|---|---:|---:|---|---:|---:|
| P0 single | 25.59 / 20.24 / 30.21 | 25.35 | 9.97 | 26.09 / 29.07 / 30.89 | 28.68 | 4.80 |
| E3 | 21.26 / 17.84 / 27.27 | 22.12 | 9.43 | 13.24 / 21.56 / 24.18 | 19.66 | 10.93 |
| E3 x F8 (640) | 13.03 / 12.27 / 19.41 | 14.91 | 7.14 | 17.57 / 18.71 / 21.12 | 19.13 | 3.55 |
| E3 x P1 | 16.28 / 9.49 / 14.41 | 13.39 | 6.79 | 10.44 / 20.84 / 17.02 | 16.10 | 10.40 |
| E3 x P1↓ x F8 | 6.90 / 8.92 / 4.83 | 6.88 | 4.09 | 17.57 / 18.71 / 21.12 | 19.13 | 3.55 |
| Calibrated ByteTrack E3 | 53.28 / 47.23 / 56.26 | 52.26 | 9.03 | 14.58 / 24.37 / 26.80 | 21.92 | 12.22 |

(Source: §152, result table.) E3, E3 x F8 and E3 x P1 are report-only in §152.

Criteria (§152):
- Criterion 1, 留 35: d = −18.69 / −11.32 / −25.38, mean −18.46, threshold 9.97: pass, margin 8.49.
- Criterion 1, 留 33: d = −8.52 / −10.36 / −9.77, mean −9.55, threshold 4.80: pass, margin 4.75.
- Criterion 2, 留 35: d = −46.38 / −38.31 / −51.44, mean −45.38, threshold 9.03: pass.
- Criterion 2, 留 33: d = +2.99 / −5.66 / −5.68, mean −2.78, threshold 12.22: fail, short by 9.43 points.

Registered outcome (§152): "tails confirm only one fold". Only 留 35 passes both criteria, so the main setting stays P0, and E3 x P1↓ x F8 stays report-only.

Notes.
- The criterion 2 failure on 留 33 is not the arm getting worse (19.13% on 33-tail, 17.80% on 33-head). The baseline changed: calibrated ByteTrack E3 is 21.92% on 33-tail against 41.44% on head, same coefficients, and its per-seed values run from 19.31 to 57.87 (§152). The reason is an inference, not measured (§152).
- P1 scale picks on tail against head: same side of 640 in 6/9 seeds (留 35) and 8/9 seeds (留 33). On 留 33 P1↓ puts all 9 back to 640, so E3 x P1↓ x F8 = E3 x F8 there (§152).
- On 33-tail P1 with large scales did not hurt: E3 x P1 16.10% is 3.03 points below E3 x P1↓ x F8. This is the opposite of head (21.95% vs 17.80%). The idea "shrinking is safe, enlarging is not" was not reproduced on 33-tail (§152).
- Weak points (§152): only 37 and 50 events, one 40-second window is almost the whole video; tail and head are the same hive, so this does not test a new hive.
- End-of-log status: criterion (b) was not cleanly reached on the proxy; only test can say more (§152, §153).

## 5. Table D: test (filled once, after the locked run)

Cells are 30 s window error in %, 5 test videos pooled over 405 GT events (§125). One run only; no setting is changed after seeing it (§125).

| Arm | y11s_640 | y11s4_s1 | y11s4_s2 | Mean | Range | Event F1 |
|---|---|---|---|---|---|---|
| P0 | — | — | — | — | — | — |
| P1 | — | — | — | — | — | — |
| Calibrated ByteTrack | — | — | — | — | — | — |
| Default ByteTrack | — | — | — | — | — | — |
| Paper's tracker A (report only) | — | — | — | — | — | — |
| E3 (report only) | one number | one number | one number | — | P0 range is the threshold | — |
| F8 (report only) | — | — | — | — | — | — |
| E3 x F8 (report only) | one number | one number | one number | — | P0 range is the threshold | — |
| E3 x P1 (report only) | one number | one number | one number | — | P0 range is the threshold | — |
| E3 x P1↓ x F8 (report only) | one number | one number | one number | — | P0 range is the threshold | — |

Per video and total, report also: event F1 at IoU 0.5 and 0.3 (matched / false / missed) and bias = predicted events minus GT events (§125). The bias is report-only, not judged (§125). Also report which scale P1 picks for each of 3 detectors x 5 videos; if all picks are 640 or above, E3 x P1↓ x F8 equals E3 x F8 and this is stated (§125, addendum for §151).

### 5.1 Reading rules (set before test, §125)

- P0 against each baseline: take the mean difference over the 3 detectors and compare it with the larger of the two arms' ranges over the 3 detectors.
  - Mean difference negative and its absolute value larger than that range: write "more accurate on test".
  - Mean difference negative but smaller than the range: write "lower, but within the differences between detectors".
  - Otherwise write the numbers as they are.
- P1 against P0 is read the same way, report only; the main setting does not change.
- F8 is read like P1 (threshold = larger range of the two arms over 3 detectors), report only.
- E3-type arms are one number each (no range between detectors). Compare with the P0 mean of the 3 detectors; the difference is compared with P0's range over the 3 detectors, with the same three phrasings.
- Whatever the result, report and slides use this number set; every old /341 number is replaced (§125).

### 5.2 Expectations written before test (§125 and its addenda)

Probabilities are the main controller's own guesses, written before test.

| Arm | Expectation | Probability | Source |
|---|---|---:|---|
| F8 | mean over the 3 detectors lower than P0 | 70% | §125 addendum (§144) |
| F8 | difference between −1 and −3 points | 50% | same |
| F8 | read as "lower, but within the differences between detectors" | 55% | same |
| E3 x F8 | lower than P0 mean | 75% | §125 addendum (§147) |
| E3 x F8 | lower than F8 mean | 60% | same |
| E3 x F8 | lower than E3 | 60% | same |
| E3 x F8 | difference larger than P0's range | 35% | same |
| E3 x P1 | lower than P0 mean | 70% | §125 addendum (§148) |
| E3 x P1 | lower than P1 mean | 60% | same |
| E3 x P1 | lower than E3 | 50% | same |
| E3 x P1 | difference larger than P0's range | 30% | same |
| E3 x P1↓ x F8 | lower than P0 mean | 70% | §125 addendum (§151) |
| E3 x P1↓ x F8 | lower than E3 x F8 | 45% | same |
| E3 x P1↓ x F8 | lower than E3 x P1 | 55% | same |
| E3 x P1↓ x F8 | difference larger than P0's range | 35% | same |

- Tooling (§125, last addendum): `t152/arms125x.py` computes F8, E3 x F8, E3 x P1, E3 x P1↓ x F8 and prints P0 and E3 as reference. The self-check on val + dev matched the logged values to under 1e−9 (§125).
- The paper's tracker A only runs on test as report-only; its wrapper reproduced the 35.25 on val + dev bit for bit (§125, clarification 2 and the note after it).
- Planned GPU: P1 scales about 4 hours, 7 extra symmetries at 640 about 4.7 hours, extra symmetries at 352 / 448 up to about 3.8 hours (§125).

## 6. What did not work

Grouped by the four research issues. "Not adopted" means the registered rule said so. Numbers are window error unless said otherwise; first number 留 35, second 留 33, P0 = 17.76 / 23.27 (§95, §151).

### Issue #1 detector

- Frozen backbone (A1): 17.76 → 12.42 on 留 35, 23.27 → 33.99 on 留 33; 25-52 more over-counted events on 留 33 (§131).
- Hive adversary (A2): 17.76 → 24.79, 23.27 → 30.46 (§132).
- Frozen DINOv2 features (A3): 17.76 → 9.18, 23.27 → 32.32; seen-hive mAP50 down by 0.056 (§133, §136).
- Up-down flip in training: 17.76 → 31.15, 23.27 → 26.82; val mAP rose 6 of 6 times while counting got worse (§145).
- Averaging 5 checkpoints of one training run: 17.76 → 32.97, 23.27 → 20.08 (§120).
- Half-perfect detection: precision perfect 16.24 / 39.55, recall perfect 33.02 / 84.67; a one-sided fix does not carry to counting (§124).
- Distilled pseudo-label detector on a third hive: event F1 −0.031 / −0.035 and +9.2 points count error, so not a fixed pipeline (§84, §80).
- Choosing the scale without labels (P1) is unstable on unseen hives: another 1000-frame clip flips the pick; 5 bad picks in 24 (§92). Shrinking helped 7 of 7 seeds, enlarging 5 of 11 (§150).
- Later checkpoints: the trend with training is not stable, so `best.pt` stays (§101).

### Issue #2 tracking and linking

- Geometric joining after stage one: not adopted; dense hive got 4.3 points worse (§76).
- Removing duplicate or overlapping tracklets by geometry (relative-score drop, merging, shadow tracks): no effect on val + dev (difference 0.00 for shadow tracks) (§87, §89, §138).
- Appearance for linking: with the paper's ReID loaded properly the right match is picked 0.289 of the time, below the small network's 0.507 (§98).
- Paper's tracker: 47.65% with its shipped parameters, 35.25% with the fixed setting A, 26.01% with the best setting picked after the fact, against P0 18.46%; no one of 20 settings wins on all detectors (§102).
- Perfect linking on real detections is worse than P0: 17.76 → 21.58 and 23.27 → 25.97, because P0's links also absorb noise fragments (§123).
- Cutting tracklets at skipped frames before linking: 17.76 → 15.90 (range 18.40), 23.27 → 25.10, val + dev +1.17 (§129).
- Better linking after changing the detector: even with refitting the linker, 留 33 never gets below P0's own swept 21.87% (§137).

### Issue #3 counting logic

- Averaging lines (E3, cross-family average, F4, F8): criterion 1 not passed on both folds in any single arm (§95, §142, §143, §144, §150).
- Events for fast tracks starting near the entrance: more than 2 points worse on 4 settings (§90).
- Cancelling an in-out pair: worse on every seen-hive setting; 留 35 −2.85 but inside the range (§121).
- Minimum track length chosen from the target hive's short-track ratio: 留 35 9.18 (not triggered), 留 33 23.93 vs P0 23.27 (§140).
- A box-based counting rule from the literature: worse for every tracker, TrackletLink 10-20 points worse in three places (§134).
- Cross-detector support to drop extra events: 留 35 E3 20.45 → 11.49 but 留 33 19.29 → 39.46, so only one fold (§108).
- Raising the tracking score threshold on new hives: not adopted (§70).

### Issue #4 error decomposition (what the errors are made of)

- On the dense hive detection fixes do not move the window error (§71); fakes and misses cancel in total (§95).
- 留 35 error is about half bias: 10.41 of 22.18 (9-seed mean) and all 9 seeds over-count. 留 33 is mostly window noise (20.90 of 23.90) (§95).
- Averaging removes only random window noise; a bias needs a signal from the target hive, and the only one that helped is the P1 flicker, which has a direction (§150).
- One detector seed can be far off: 留 35 seed 6 gives 46.51% with 87 false events, mostly tracks on empty background (§95).

## 7. Error decomposition and upper bounds

- Perfect boxes (GT boxes) through P0: 35-head 7.86%, 33-head 29.17% (§122, §123). With perfect linking on top: 2.33% and 23.77% (§123).
- On 33-head the 23.77% is not the protocol limit: every box replaced by the true bee gives 1.63%, but pure tracklets plus perfect linking give 23.95%. Cause: 1249 of 15017 neighbouring tracklet pairs of the same bee overlap in time, because stage one skips frames (2016 skips), and an "end to start" linker cannot join them (§128).
- Perfect linking on real detections is worse than P0 (21.58 / 25.97 against 17.76 / 23.27) (§123).
- Perfect precision or perfect recall alone are worse than P0 on 留 33 (39.55, 84.67) and recall perfect is worse on 留 35 (33.02) (§124). Perfect precision on 留 35 gives 16.24 (§124).
- Remove only the extra boxes on real bees (D2) and add perfect linking: 4.24% on 留 35, the only arm under the 5.05% bar; D2 with the normal linker gives 16.24%, perfect linking on raw boxes 21.58% (§135).
- A3 detector with a refitted linker (optimistic, after the sweep): lowest 留 35 is 6.87%; on 留 33 nothing beats P0's own swept 21.87% (§137).
- With all links correct, the base detector plus scale pick still has 18.6% on 33-head (three seeds): missed bees cannot be counted (§72, §81).
- The thresholds used for these bounds: 留 35 needs a mean below 5.05% and 留 33 below 20.95% (§122, §123).
- Window error of a flat "true total spread over the windows" reference: 留 35 23.29%, 留 33 22.04%, val + dev 18.28% (§94).

## 8. Table E: robustness on val + dev (§153, report only)

Same 3 detectors at 640, not retrained; P0 uses the 3 cross-fitted linkers as in Table A; calibrated ByteTrack keeps the clean cross-fitted factors (not refitted per condition). Cell: 30 s window error in %, mean over the 3 detectors [range], then the change from clean (§153). The clean row reproduces Table A exactly (§153).

| Condition | P0 | Calibrated ByteTrack | Default ByteTrack | P0 event F1@0.5 |
|---|---|---|---|---:|
| clean | 18.46 [1.81] | 34.82 [12.51] | 40.18 [3.24] | 0.740 |
| dark (x 0.4) | 18.02 [1.16], −0.44 | 34.39 [6.73], −0.43 | 41.32 [4.83], +1.13 | 0.738 |
| bright (x 1.6 + 20) | 19.23 [3.68], +0.76 | 37.14 [14.27], +2.32 | 40.81 [1.93], +0.63 | 0.710 |
| blur (1 x 9) | 19.88 [2.79], +1.42 | 37.22 [9.47], +2.40 | 39.74 [2.98], −0.44 | 0.694 |
| noise (sigma 15) | 22.61 [0.89], +4.15 | 40.48 [13.05], +5.66 | 46.63 [3.55], +6.45 | 0.681 |
| frame drop k = 2 (12.5 fps) | 22.81 [3.21], +4.35 | 35.96 [3.30], +1.14 | 51.56 [2.18], +11.38 | |
| frame drop k = 3 (8.3 fps) | 33.69 [2.57], +15.22 | 43.85 [2.15], +9.02 | 60.06 [3.55], +19.88 | |
| frame drop k = 5 (5 fps) | 47.47 [1.35], +29.01 | 54.22 [0.57], +19.40 | 63.84 [2.51], +23.65 | |

Notes.
- Bias (predicted − GT events, 486 events): P0 clean −30.0; frame drop k = 2, 3, 5: −87.4, −153.3, −223.9; corruptions −13.9 to −25.3. Default ByteTrack clean −177.7 (§153).
- Default ByteTrack event F1: clean 0.545, corruptions 0.495 to 0.540 (§153).
- P0's lead over calibrated ByteTrack: 16.4 points clean, 13.2 / 10.2 / 6.8 at k = 2 / 3 / 5 (§153).
- In frame drop, P0 drops tracks with fewer than 3 kept boxes; the linker was fitted at 25 fps and is not refitted (§153).
- Cause of the frame-drop loss (tracks breaking into short pieces) is an inference, not measured (§153).

Clean, per video, sorted by GT bees per frame (mean of the 3 detectors; per-video output of the §153 run):

| Video | Split | Frames | Bees / frame | GT events | P0 | Calibrated ByteTrack | Default ByteTrack |
|---|---|---:|---:|---:|---:|---:|---:|
| BEE24-35-tail | val | 1000 | 9.83 | 37 | 12.61 | 20.59 | 29.73 |
| BEE24-35-head | dev | 3950 | 9.89 | 129 | 8.83 | 31.45 | 31.35 |
| BEE24-10 | val | 250 | 9.93 | 13 | 21.48 | 45.48 | 62.22 |
| BEE24-15 | val | 240 | 12.89 | 56 | 32.08 | 54.77 | 62.54 |
| BEE24-24 | val | 75 | 13.28 | 5 | 56.30 | 94.31 | 60.44 |
| BEE24-28 | val | 75 | 13.51 | 12 | 30.80 | 53.68 | 15.19 |
| BEE24-33-tail | val | 1000 | 37.31 | 50 | 21.10 | 27.83 | 39.47 |
| BEE24-33-head | dev | 3950 | 40.24 | 184 | 19.48 | 32.27 | 41.39 |

- Density has only two groups here: the two hive-33 segments and everything else. It cannot be separated from the hive effect (§153).
- The four short videos (75 to 250 frames, 5 to 56 events) are shorter than one 30 s window, so each is one noisy number (§153).
- Test versions of these conditions are registered as report-only in the §125 addendum (2026-11-19) and fill no row until the locked test run.

## 9. Table F: window length (§154, report only)

Window error % for windows of 10, 30, 60 and 120 s and for the whole video as one window. L s windows are scored at L start offsets one second apart, so 30 s is the main metric of every other table. The window grids nest, so no arm's error can grow with the window length. The main metric stays 30 s; this table makes no choice.

val + dev (486 GT events), mean over the 3 detectors [range]:

| Arm | 10 s | 30 s | 60 s | 120 s | Whole video | Bias |
|---|---:|---:|---:|---:|---:|---:|
| P0 | 23.93 [2.99] | 18.46 [1.81] | 15.55 [1.85] | 13.55 [1.57] | 11.57 [1.51] | −30.0 |
| P0 with `m_dev_strict.json` | 24.95 [2.59] | 19.56 [0.29] | 16.09 [1.26] | 13.32 [1.48] | 10.97 [3.09] | −16.7 |
| Calibrated ByteTrack | 47.95 [11.72] | 34.82 [12.51] | 31.55 [13.48] | 30.39 [14.61] | 29.48 [15.57] | +27.0 |
| Default ByteTrack | 44.29 [4.57] | 40.18 [3.24] | 39.72 [3.41] | 39.37 [3.56] | 39.03 [3.70] | −177.7 |

Leave-one-hive-out heads, single detector seeds, mean over 9 seeds [range]:

| Fold | Arm | 10 s | 30 s | 60 s | 120 s | Whole video | Bias |
|---|---|---:|---:|---:|---:|---:|---:|
| 留 35 (35-head, GT 129) | P0 | 27.73 [32.40] | 22.18 [33.80] | 20.40 [36.30] | 19.82 [37.13] | 19.38 [37.98] | +25.0 |
| | Default ByteTrack | 33.80 [11.32] | 26.90 [10.90] | 24.79 [12.58] | 23.79 [13.62] | 22.22 [13.95] | −17.1 |
| | Calibrated ByteTrack | 58.20 [40.22] | 47.38 [45.60] | 45.79 [46.06] | 44.73 [46.15] | 43.91 [46.25] | +56.6 |
| 留 33 (33-head, GT 184) | P0 | 39.03 [12.07] | 23.90 [13.26] | 17.81 [12.37] | 14.35 [14.47] | 11.71 [17.39] | +3.6 |
| | Default ByteTrack | 46.98 [13.04] | 33.13 [15.18] | 29.77 [19.24] | 28.24 [26.10] | 27.42 [29.89] | −50.4 |
| | Calibrated ByteTrack | 64.80 [23.50] | 44.57 [28.84] | 39.72 [34.15] | 37.81 [34.85] | 35.87 [34.52] | +47.3 |

Notes.
- The 30 s column equals Table A and the §95 single-seed values (asserted). Reproduce: `PYTHONPATH=. uv run python tools/window_lengths.py [--heads]` (CPU, about 5 and 20 min).
- P0 is the lowest arm at every length, on val + dev and on both heads. Calibrated ByteTrack is the worst arm at every length on both heads (§154).
- On val + dev, calibrated ByteTrack is worse than default ByteTrack at 10 s and better from 30 s on: scaling the counts fixes the totals and also scales the noise of each short window (§154).
- At long windows the remaining error is mostly the total-count bias. On 留 35 every seed over-counts, and P0's whole-video error (19.38) equals the mean over-count of 25 events / 129 (§154).
- On 留 35 the whole-video gap between P0 and default ByteTrack (2.84) is far smaller than P0's seed range (37.98), so the two cannot be told apart there at long windows (§154).
- Val videos are 3 to 40 s long and the two dev videos 158 s, so the 60 s, 120 s and whole-video columns differ mainly through the dev videos.
- Test: the same lengths will be computed from the locked §125 test outputs, report only.
