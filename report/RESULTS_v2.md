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
| whiteboard (留白板) | the 13 whiteboard clips: BEE24-01..09, 11, 14 (train) and BEE24-10, 15 (val), 1,600 frames, each under 15 s | 137 (77 in / 60 out) | §157 |
| test | BEE24-12, 13, 16, 34, 36 (10,762 frames: 250, 258, 254, 5,000, 5,000) | 405 (1/3, 10/11, 26/37, 84/88, 91/54 in/out per video) | §0, §2, §125 |

- "val + dev" measures the tracking side: the four-hive detectors saw these hives in training (§42).
- "Leave-one-hive-out" (留一蜂箱) is the proxy for an unseen hive. Fold 留 35 trains without hive 35 and is scored on 35-head. Fold 留 33 trains without hive 33 and is scored on 33-head (§64, §95).
- 9 detector seeds per fold, split into 3 groups by seed number: 0-2, 3-5, 6-8 (§95). The group range (max minus min of the 3 group values) is the noise scale (§95).
- Tails are the same videos, 50 frames after the head. They were not used before §152 (§152).
- Fold 留白板 trains without any whiteboard clip (detectors `loho_noW_s0..8`, linker `m_noW.json`, which was fitted without whiteboard GT). The 11 train clips had no entrance polygon; they reuse the BEE24-10 / 15 polygons, shifted by the camera offset found by phase correlation (§157).

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
| F (distilled-student committee) | 3 members. Each member's detector is fine-tuned on its own pseudo-labels (teacher scale picked by the rule CS), runs at 640, keeps boxes >= 0.50, then runs the unchanged P0 tracking and events. The 3 members' in and out counts are averaged per 30 s window, as in E3. Details below. Report only | §177, §181, §183, §201, §125 addendum (2026-11-23) |
| E3 with the 0.50 box cut (E3₅₀) | control for F: the same 3 detectors at 640, boxes >= 0.50, the same member pipeline as F, no distillation; counts averaged as in E3. Report only | §125 addendum (2026-11-23) |

How F is built (§177, §181, §201, §125 addendum 2026-11-23):
- Members: 3 detectors. On test they are the 3 test detectors of P0. On the proxy they are leave-one-hive-out detectors (§125 addendum, §177).
- Teacher scale, rule CS (§181). It uses no labels and has no parameters. The 3 members run at the 6 scales of P1 on unlabeled frames (every 5th frame, boxes with score >= 0.25). For each scale, take the 3 pairs of members, measure how well their boxes agree (F1 of the two-way coverage at IoU >= 0.5) and average the pairs. Pick the scale with the highest mean; an exact tie goes to the smaller scale. All 3 members use that scale (§181).
- Why CS: it mainly avoids teachers that break down at large scales. Under the P1 rule two 35-head teachers picked 1280 and 960 and gave 37.5 and 21.1 boxes per frame, where the truth has 9.8 (§177, §181).
- Pseudo-labels: 600 random frames; the teacher is the member's own detector at the CS scale, and its boxes with score >= 0.25 become the labels (§177, §125 addendum).
- Fine-tune: from the same detector, 15 epochs, with the pseudo-label frames mixed into the detector's own training list; the last checkpoint is used; inference at 640 (§177, §125 addendum).
- Member pipeline: keep boxes >= 0.50, then the unchanged P0 tracking: first stage, TrackletLink `m_dev_strict.json`, drop tracks with fewer than 3 boxes, P0 event rule (§177, §201, §125 addendum).
- Count: average the 3 members' in and out counts per 30 s window, as in E3 (§125 addendum).
- Frames: on the proxy the unlabeled frames come from the tail of the same video (§177, §181). On test they come from the scored video itself (all frames if it has fewer than 600), sampling seed 0 (§125 addendum).

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

### 3.2 Table B2: distilled-student committee F (§188, §201, §204)

F is the main result of the model direction. It is report-only on test and does not replace P0 (§125 addendum 2026-11-23, §201).

These runs use new leave-one-hive-out detector seeds, not the seeds of Table B. Design groups 2-5 are seeds 3-14 (§188); groups 6-7 are seeds 15-20 (§201); groups 8-11 are seeds 21-32 (§204). Every cell is an E3-type number: the members' window counts are averaged. For groups 8-11 the cell is the mean over the 6 committees of 6 members (two groups each), with the jackknife standard error in brackets (§204). "TrackletLink" is the P0 pipeline (640, all boxes) on the same base detectors, counts averaged the same way (§95, §204). The bars are 5.05 on 35-head and 20.95 on 33-head (§122).

| Row | 35-head groups 2-5 | 35-head groups 6-7 | 35-head groups 8-11 | 33-head groups 2-5 | 33-head groups 6-7 | 33-head groups 8-11 | Source |
|---|---:|---:|---:|---:|---:|---:|---|
| F (CS students, boxes >= 0.50) | 4.34 | 6.37 | 5.51 (0.30) | 19.07 | 20.40 | 20.05 (1.11) | §188, §201, §204 |
| Same students, boxes >= 0.40 | 6.83 | 7.92 | 6.91 (1.66) | 19.36 | 20.82 | 19.41 (0.74) | §188, §201, §204 |
| 640 direct inference, boxes >= 0.40, no distillation ("control ii" in §204) | 7.92 | 11.67 | 6.90 (0.88) | 25.88 | 28.50 | 33.43 (2.74) | §188, §201, §204 |
| TrackletLink (P0 pipeline, base detectors) | - | 27.30 | 27.07 (7.80) | - | 21.11 | 20.62 (2.12) | §201, §204 |
| Calibrated ByteTrack | - | 55.06 | 49.44 (6.00) | - | 41.71 | 34.64 (1.85) | §201, §204 |
| P0 single | - | 28.01 | - | - | 23.45 | - | §201 |

Per group, F is 4.72 / 3.84 / 4.58 / 4.20 on 35-head and 19.18 / 21.61 / 16.49 / 19.00 on 33-head for groups 2-5 (§125 addendum, §188), and 5.24 / 7.51 and 19.29 / 21.52 for groups 6 / 7 (§201). The 6 committees of groups 8-11 give 4.88 / 5.67 / 5.38 / 5.56 / 5.67 / 5.91 on 35-head; only groups 8+9 is below 5.05 (§204).

Paired differences on groups 8-11; beyond noise means an absolute value above 2.0 (§204):

| Difference | 35-head | 33-head |
|---|---:|---:|
| F − control ii | −1.39 (0.96) | −13.37 (1.88) |
| F − TrackletLink | −21.56 (7.77) | −0.57 (1.32) |
| F − calibrated ByteTrack | −43.93 (5.84) | −14.59 (2.79) |

On groups 6-7 the differences were: F − 640 direct at 0.40 −5.30 / −8.09; F − TrackletLink −20.92 / −0.70 (on 33-head the two groups had opposite signs, +0.75 / −2.15); F − calibrated ByteTrack −48.69 / −21.30 (§201).

Reading notes.
- 35-head: F is clearly lower than TrackletLink and calibrated ByteTrack, beyond seed noise, on both new seed sets (§201, §204).
- 35-head: F is still above the 5.05 bar: 6.37 on groups 6-7 and 5.51 on groups 8-11. The registered sentence is "above the bar" (§201, §204).
- 35-head: against the same committee without distillation (control ii), F is −1.39, "within the seed differences". So the distillation gain on 35-head did not reproduce in the 6-member committee on new seeds (§204). In the 3-member groups 6-7 it was −5.30 (§201).
- 33-head: F 20.05 is below the bar but not below the noise line 19.25. The TrackletLink committee (20.62) is also below the bar, and F − TrackletLink is −0.57 (SE 1.32). The registered sentence is "the 33-head bar cannot separate F and TrackletLink" (§204). Groups 6-7 gave the same picture: 20.40 against 21.11 (§201).
- 33-head: F is 13.37 below control ii, beyond noise, so on 33-head distillation helps against direct inference (§204).
- The new seeds are worse than the design groups. Two-group committees of F gave 3.99 / 18.77 on the design groups and 5.51 / 20.05 on groups 8-11, so 1.52 / 1.28 higher (§204).
- The 0.50 box cut was chosen while looking at the labels of these two videos (§183, §188, §204). At 0.50 the design split was the best of 15,400 ways to split the seeds (4.34; the mean over all splits is 5.03), so the design numbers are optimistic (§188). The new-seed check only asks whether the result comes back with other detector seeds; it does not test unseen hives (§204).
- Do-no-harm of the 0.50 cut (§188): on the whiteboard hive all three detector types got 2.25 to 4.25 points worse, and on the seen hives (486 events) the error went from 17.97 to 19.23. The gain at 0.50 belongs to the 35-head CS students; it is not a general rule (§188).
- Criterion (b): §204 never writes "criterion (b) passed" (§204). On 35-head every known method family is above 5.05 in the 6-member committee (F 5.51, control ii 6.90, TrackletLink 27.07, calibrated ByteTrack 49.44). On 33-head the bar does not separate F from TrackletLink (§204). The known families fell short by these numbers. This does not show that the bar cannot be reached.
- Error left in F (E12, all 12 members of groups 8-11): 35-head 5.01, two thirds of it over-counts, mostly exits; 33-head 19.70, nearly nine tenths under-counts (§204).
- Seeds 21-32 count as seen from §204 on. No new seeds are trained for the committee (§204).

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

### 4.1 Table C2: leave-whiteboard fold (§157 result)

Registered before the run and not changed (§157). Measurement only: whatever the result, §125 keeps P0 as the main setting (§156, §157). Every clip is shorter than 30 s, so the 30 s window error is close to the per-clip count error (§157). Clips under 1000 frames use 640 under P1, so E3 x P1↓ x F8 = E3 x F8 on this fold (§91, §157).

| Row | Groups 0-2 / 3-5 / 6-8 | Mean | Range | 9-seed mean [range] | Bias | Event F1 |
|---|---|---:|---:|---|---:|---:|
| P0 single | 34.53 / 35.57 / 31.79 | 33.97 | 3.78 | 33.97 [10.07] | −20.8 | 0.681 |
| E3 (report) | 31.91 / 34.37 / 29.68 | 31.99 | 4.69 | - | −20.8 | 0.681 |
| E3 x F8 (= E3 x P1↓ x F8) | 37.26 / 34.40 / 34.80 | 35.49 | 2.86 | - | −25.6 | 0.643 |
| Default ByteTrack single | 58.98 / 54.21 / 55.41 | 56.20 | 4.77 | 56.20 [10.71] | −70.4 | 0.405 |
| Calibrated ByteTrack single | 51.47 / 45.76 / 47.81 | 48.35 | 5.71 | 48.35 [15.30] | −40.5 | 0.405 |
| Calibrated ByteTrack E3 | 50.04 / 42.95 / 43.66 | 45.55 | 7.09 | - | −40.5 | - |

(Source: §157, result tables. Bias = k x predicted minus GT events, mean over seeds or members; event F1 of E3-type rows is the mean over members.)

Criteria (§157):
- P0 vs calibrated ByteTrack, 9 seeds: P0 lower in 9 / 9 seeds, mean d −14.38, threshold 15.30 (range of calibrated ByteTrack): fail, short by 0.92 points.
- P0 vs default ByteTrack, 9 seeds (report): mean d −22.23, threshold 10.71: pass, margin 11.53. On 留 35 and 留 33 this difference did not pass (§95).
- Criterion 1, E3 x F8 vs P0 single: d = +2.73 / −1.18 / +3.01, mean +1.52, threshold 3.78: fail (the arm is higher).
- Criterion 2, E3 x F8 vs calibrated ByteTrack E3: d = −12.78 / −8.55 / −8.86, mean −10.06, threshold 7.09: pass, margin 2.97.

Notes.
- P0 vs calibrated ByteTrack over the three folds: lower on all three, but the §95 reading passes on two (留 35, 留 33) and misses by 0.92 points on 留白板 (§95, §157). The report must say "two of three folds", not "better than calibrated ByteTrack on every unseen hive" (§157).
- Pre-registered checks of E3 x P1↓ x F8: 35-tail passes both criteria, 33-tail fails criterion 2, 留白板 fails criterion 1 (§152, §157). Criterion (b) is still not reached on the proxies (§157).
- The whole P0 under-count comes from one clip: on BEE24-15 (56 GT events in 240 frames, the most crowded clip) P0 predicts 35.2 events on average. This clip is about 45% of P0's error; the bias of the other 12 clips sums to about zero (§157).
- F8 lowers counts here (bias −25.5 vs −20.8 for single P0), which hurts on an under-counting fold. Whether F8's gain on the two heads is partly the same count reduction is an inference, not measured (§157).
- Weak points (§157): 13 short clips, one clip dominates; the two whiteboard val clips were used before as do-no-harm checks (§84, §89, §90).
- Why entries are lost (§158, diagnostic only): P0 (s0..s2) counts 58-60 in vs 77 GT, 57-60 out vs 60 GT. Entering bees cross in flight (median 1.19 body lengths / frame, 75% above 0.5), leaving bees walk (0.38). The strict first stage cuts fast bees into one-frame tracklets, so entries depend on TrackletLink: P0 finds 47% of fast and 75% of slow entries (53% / 83% of exits). GT boxes through P0 find fewer entries (29), because `m_noW.json` was fitted on real-detection tracklets, so they are not an upper bound here. No new fix registered: §90 and §129 already tried the label-free ones.

### 4.2 Table C3: leave-gray-board fold (§159 result)

Registered before the run and not changed (§159). Measurement only: whatever the result, §125 keeps P0 as the main setting (§159). The 16 gray-board clips (BEE24-17 to -32, val -24 and -28 included) have 75 frames each, so the 30 s window error is the per-clip count error. The 14 train clips use BEE24-28's polygon moved by phase correlation (§159). Clips under 1000 frames use 640 under P1, so E3 x P1↓ x F8 = E3 x F8 on this fold (§91, §159).

| Row | Groups 0-2 / 3-5 / 6-8 | Mean | Range | 9-seed mean [range] | Bias | Event F1 |
|---|---|---:|---:|---|---:|---:|
| P0 single | 35.86 / 38.35 / 35.60 | 36.60 | 2.75 | 36.60 [7.81] | −14.0 | 0.596 |
| E3 (report) | 33.96 / 32.25 / 29.94 | 32.05 | 4.02 | - | −14.0 | 0.596 |
| E3 x F8 (= E3 x P1↓ x F8) | 30.98 / 30.90 / 31.06 | 30.98 | 0.16 | - | −21.7 | 0.547 |
| Default ByteTrack single | 34.05 / 47.46 / 38.99 | 40.17 | 13.41 | 40.17 [25.96] | −47.0 | 0.398 |
| Calibrated ByteTrack single | 42.16 / 43.92 / 44.04 | 43.37 | 1.87 | 43.37 [16.91] | +10.6 | 0.398 |
| Calibrated ByteTrack E3 | 34.80 / 37.07 / 35.42 | 35.77 | 2.27 | - | +10.6 | - |

(Source: §159, result tables. Bias = k x predicted minus GT events, mean over seeds or members; event F1 of E3-type rows is the mean over members.)

Criteria (§159):
- P0 vs calibrated ByteTrack, 9 seeds: P0 lower in 8 / 9 seeds, mean d −6.77, threshold 16.91 (range of calibrated ByteTrack): fail, short by 10.14 points.
- P0 vs default ByteTrack, 9 seeds (report): mean d −3.56, threshold 25.96: fail.
- Criterion 1, E3 x F8 vs P0 single: d = −4.88 / −7.45 / −4.54, mean −5.63, threshold 2.75: pass, margin 2.88.
- Criterion 2, E3 x F8 vs calibrated ByteTrack E3: d = −3.83 / −6.18 / −4.36, mean −4.79, threshold 2.27: pass, margin 2.52.

Notes.
- P0 vs calibrated ByteTrack over the four folds: lower on the mean on all four, the §95 reading passes on two (留 35, 留 33), misses by 0.92 points on 留白板 and by 10.14 on 留灰板 (§157, §159). The report says "two of four folds".
- Pre-registered checks of E3 x P1↓ x F8: 35-tail and 留灰板 pass both criteria, 33-tail fails criterion 2, 留白板 fails criterion 1: two of four (§152, §157, §159). No arm passes on every fold, so criterion (b) is not reached on the proxies (§159).
- P0 under-counts exits here (75.6 vs 91 GT), entries are about right (62.4 vs 61); on the whiteboard it was the entries (§158, §159). Side measurement after the result (`tools/whiteboard_speed.py gray`): exits are the faster direction here too (median 0.52 vs 0.41 body lengths / frame); P0 finds 35% of fast and 77% of slow exits, 47% / 68% of entries (§159).
- No clip dominates: the largest P0 clip error is about 9% of the total (§159, per-clip check from the run's cache).
- F8 lowers the counts (bias −21.7 vs −14.0) but its error is lower on 12 of 16 clips, so the §157 guess that part of F8's gain is "fewer counts" does not hold on this fold (§159).
- Weak points (§159): 3 s clips; BEE24-24 and -28 are in val, where P0's settings were chosen; calibrated ByteTrack's factors were fitted on other hives and over-count here (+10.6).

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
| F (report only) | one number | one number | one number | — | P0 range is the threshold | — |
| E3 with the 0.50 box cut (report only) | one number | one number | one number | — | P0 range is the threshold | — |

Per video and total, report also: event F1 at IoU 0.5 and 0.3 (matched / false / missed) and bias = predicted events minus GT events (§125). The bias is report-only, not judged (§125). Also report which scale P1 picks for each of 3 detectors x 5 videos; if all picks are 640 or above, E3 x P1↓ x F8 equals E3 x F8 and this is stated (§125, addendum for §151).

F and E3₅₀ are also reported separately for the short videos BEE24-12, 13, 16 and the long videos BEE24-34, 36 (the addendum first called them seen and unseen; corrected before the lock: 12 has a frame size no training video has, 13 and 16 share the 950 x 600 size of training hives, so they may or may not be seen hives): F on 12 / 13 / 16 —, F on 34 / 36 —, E3₅₀ on 12 / 13 / 16 —, E3₅₀ on 34 / 36 —. Also print F − E3₅₀ (—) and F − E3 (—) (§125 addendum, 2026-11-23).

### 5.1 Reading rules (set before test, §125)

- P0 against each baseline: take the mean difference over the 3 detectors and compare it with the larger of the two arms' ranges over the 3 detectors.
  - Mean difference negative and its absolute value larger than that range: write "more accurate on test".
  - Mean difference negative but smaller than the range: write "lower, but within the differences between detectors".
  - Otherwise write the numbers as they are.
- P1 against P0 is read the same way, report only; the main setting does not change.
- F8 is read like P1 (threshold = larger range of the two arms over 3 detectors), report only.
- E3-type arms are one number each (no range between detectors). Compare with the P0 mean of the 3 detectors; the difference is compared with P0's range over the 3 detectors, with the same three phrasings.
- F and E3₅₀ (§125 addendum, 2026-11-23) are read like the other E3-type arms, report only; the main setting does not change.
  - F − E3₅₀ is the difference from distillation alone. It is read with the same three phrasings; the threshold is P0's range on the same videos.
  - F − E3 is printed as a number only, with no sentence. On val + dev the 0.50 box cut alone moves E3 from 18.40 to 19.79, which is why E3₅₀ was added.
  - The short videos 12, 13, 16 are reported separately; 13 and 16 may come from a training hive, so possible harm on seen hives would show there.
- How F runs on test (§125 addendum, method details):
  - The CS scale and the pseudo-labels come from the scored video itself, without labels. This is transductive and may favour F.
  - 12, 13 and 16 have about 250 frames each; all of them are used as pseudo-labels.
  - The fine-tune mixes the pseudo-label frames into the base detector's own training list (train 3,740 images plus the pseudo-label frames, val 839 images). The fine-tune seed is the base detector's own training seed (0 / 1 / 2).
  - There is one committee (seeds 0-2), so there is no spread between groups.
  - A unit that fails is rerun once. If it fails again, F is recorded as missing; it is not replaced by 2 members.
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
| F | lower than the P0 mean of the 3 detectors | 60% | §125 addendum (2026-11-23) |
| F | lower than E3 | 55% | same |
| F | difference larger than P0's range | 30% | same |
| F | on 34 and 36 only, lower than E3 | 60% | same |
| F | lower than E3₅₀ | 60% | same, method details |
| F | on 34 and 36 only, lower than E3₅₀ | 65% | same, method details |
| F | on 12, 13 and 16 only, lower than E3₅₀ | 45% | same, method details |

- Tooling for F (§125 addendum, method details): `score_ftest.py`. Rerun on val + dev, it gives P0 (`m_dev_strict.json`) 19.53 / 19.71 / 19.42 and E3 18.40, and F on groups 2-5 4.34 (35-head) and 19.07 (33-head); it matches `ana.py` to under 1e-9.
- Schedule (§125 addendum, 2026-11-23): the §125 lock moved to 11/25 23:59; the test runs once on 11/26-27.
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
- Committee line for F (§201, §204): the 0.50 cut on the same students was confirmed on groups 6-7 (35-head −1.55, 5 of 6 members lower), but F stayed above the 35-head bar (6.37) (§201). On seeds 21-32 the 6-member committees gave 5.51 / 20.05, and on 35-head F was only −1.39 against the same committee without distillation, within noise. The committee line was closed; no new seeds are trained for it (§204).
- TC student, pseudo-labels picked by the teacher's own tracks (§215): stopped at the CPU gate, not trained. Entrance-band recall changed by −0.043 / +0.000 (needed +0.03) and precision by −0.040 / −0.054 (allowed −0.02). The first stage is a fragmenter, so a "trusted track" meant a slow bee. Even fixing the box scores of F's members with GT (oracle) made E3 worse, by +1.93 / +1.24 (§215).

### Issue #2 tracking and linking

- Geometric joining after stage one: not adopted; dense hive got 4.3 points worse (§76).
- Removing duplicate or overlapping tracklets by geometry (relative-score drop, merging, shadow tracks): no effect on val + dev (difference 0.00 for shadow tracks) (§87, §89, §138).
- Appearance for linking: with the paper's ReID loaded properly the right match is picked 0.289 of the time, below the small network's 0.507 (§98).
- Paper's tracker: 47.65% with its shipped parameters, 35.25% with the fixed setting A, 26.01% with the best setting picked after the fact, against P0 18.46%; no one of 20 settings wins on all detectors (§102).
- Perfect linking on real detections is worse than P0: 17.76 → 21.58 and 23.27 → 25.97, because P0's links also absorb noise fragments (§123).
- Cutting tracklets at skipped frames before linking: 17.76 → 15.90 (range 18.40), 23.27 → 25.10, val + dev +1.17 (§129).
- Better linking after changing the detector: even with refitting the linker, 留 33 never gets below P0's own swept 21.87% (§137).
- Identity patches on F's fixed tracks (§202, §203, §206, §208-§212, §214): none of the nine passed its own registered investment line. Most helped one head and hurt the other, so patching F's fixed tracks stopped (§207, §213). Numbers are F's E3 on the design groups (unpatched 4.34 / 19.07) unless marked "single member":
  - §202, reattach dropped detections to the same bee (model J): 33-head 1.21 lower, 35-head 1.71 higher. Only 4 of 541 (35) and 39 of 607 (33) same-bee short tracks change any window (§202, §207).
  - §203, borrow another member's later track part: best 35-head E3 4.07, at most 0.26 better, which rules out a 1.4-point gain for this family (§203).
  - §206, withdraw a false event with one same-bee return box: 35-head 0 of 1,568 queries confirmed, E3 unchanged; 33-head 19.07 → 18.90 (§206).
  - §208, a model for bee switches inside a track (single member): 35-head 9.10 → 8.89 and F1 0.868 → 0.851, so not the 1-point gain and the F1 guard failed (§208).
  - §209, cross-part motion (single member): 35-head 0.31 worse (§209).
  - §210, alternating offset (single member): the source folds had no training cases, so the correction was 0; 35-head only 0.47 better (§210).
  - §211, event capacity of the peer members (single member): 35-head 9.10 → 6.82, 33-head unchanged at 20.98; whiteboard and gray board worse, 39.52 / 30.59 → 40.97 / 36.08 (§211).
  - §212, phase of the frame dates (single member): 35-head unchanged, 33-head 0.43 better, gray board 30.59 → 36.08 (§212).
  - §214, re-centre track end boxes from the image (seeds 3-5 only, unpatched E3 4.72 / 19.18): 35-head about 0.35 better, 33-head about 0.60 worse (§214, §207).
  - §208-§212 use one fixed member per clip on six old clips, so their window error is not the 3-member E3 (§211, §212, §213).

### Issue #3 counting logic

- Averaging lines (E3, cross-family average, F4, F8): criterion 1 not passed on both folds in any single arm (§95, §142, §143, §144, §150).
- Events for fast tracks starting near the entrance: more than 2 points worse on 4 settings (§90).
- Cancelling an in-out pair: worse on every seen-hive setting; 留 35 −2.85 but inside the range (§121).
- Minimum track length chosen from the target hive's short-track ratio: 留 35 9.18 (not triggered), 留 33 23.93 vs P0 23.27 (§140).
- A box-based counting rule from the literature: worse for every tracker, TrackletLink 10-20 points worse in three places (§134).
- Cross-detector support to drop extra events: 留 35 E3 20.45 → 11.49 but 留 33 19.29 → 39.46, so only one fold (§108).
- Raising the tracking score threshold on new hives: not adopted (§70).
- Local crossing counter without track identity, CrossNet (§207): stopped before a registered run. Even with GT boxes and GT identities, cancelling back-and-forth crossings within 8 frames leaves 27.91 / 90.22; it needs a 75-frame (3 s) window to reach 5.27 / 19.06, about F's 4.34 / 19.07. One trained CrossNet-G (about 100k parameters, GT boxes of ±4 frames, leave-one-scene) gave 30.19 / 42.90 (§207).
- Field calibration, a person counts the first 60 s of the new hive and the later counts are scaled by that minute's true / predicted ratio: 留 35 25.05 → 19.42 (7 of 9 seeds) but inside the seed range 37.88; 留 33 20.72 → 24.15; seen hives (dev) 8.71 points worse. Default ByteTrack with the same field factor: 29.03 against 46.09 for calibrated ByteTrack (mean of the two folds), still above P0 without calibration (22.89) (§155).

### Issue #4 error decomposition (what the errors are made of)

- On the dense hive detection fixes do not move the window error (§71); fakes and misses cancel in total (§95).
- 留 35 error is about half bias: 10.41 of 22.18 (9-seed mean) and all 9 seeds over-count. 留 33 is mostly window noise (20.90 of 23.90) (§95).
- Averaging removes only random window noise; a bias needs a signal from the target hive, and the only one that helped is the P1 flicker, which has a direction (§150).
- One detector seed can be far off: 留 35 seed 6 gives 46.51% with 87 false events, mostly tracks on empty background (§95).
- Oracle split of F's errors (§205, after the fact, seeds 3-20, no prediction written before): on 33-head each member misses 79.6 events; 53.6 had a box at the crossing (lost by tracking or linking) and 25.9 had none (detector); over-counts are 64.2, mainly returns 20.2 and bee switches 22.0. On 35-head over-counts are 11.7 (returns 5.2, duplicates 4.1) and misses 8.9 (7.8 of them had a box; 52% split into two tracks); fake-box tracks are only 0.2. So on 33-head the main error is identity, on 35-head duplicates and returns (§205, §207).
- Over-counts and misses of F cancel inside windows: deleting all over-counts makes it worse, 4.34 → 6.79 and 19.07 → 41.76 (§205). A one-sided fix moves the two heads in opposite directions, because 35-head over-counts and 33-head under-counts (§207).

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
