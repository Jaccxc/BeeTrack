# Individual contribution statements (draft — each member should edit to match what they actually did)

> Names and IDs are {{NAMEi}} / {{IDi}} placeholders filled from the local proposal/team.json; they are not committed.

## {{NAME1}} ({{ID1}}): data and evaluation
- Downloaded and organised BEE24; designed the sequence-level train / val / test split (val = last 1000 frames of BEE24-33 and 35 with a 50-frame gap, to avoid leakage).
- Wrote the YOLO label conversion and the evaluation ground truth (fixing duplicate IDs within a frame in the GT).
- Built the TrackEval pipeline (HOTA, MOTA, IDF1, IDSW) and the subsampled GT for the frame-drop study (k = 2, 3, 5).
- Evaluated and tabulated the robustness experiments.

## {{NAME2}} ({{ID2}}): detection and Experiment 2
- Trained the YOLO11-s detectors at 640 and 1280 input size, with early stopping and an equal effective batch size.
- Cached detections so every tracker uses identical detections; implemented the oracle (GT-box) detector.
- Experiment 2: detection mAP, tracking metrics, counting error and speed at 640 vs 1280.
- Implemented the image corruptions (dark, bright, blur, noise) on the detection side.

## {{NAME3}} ({{ID3}}): tracking, improvement and Experiment 1
- Designed BeeTracker on top of ByteTrack: adaptive buffered IoU, velocity damping of lost tracks, tuned Kalman velocity noise.
- Tuned and ablated on val only; tried and documented the ideas that did not help (second-stage BIoU, interpolation, short-track filter, the effect of tracklet stitching on zone counting).
- Experiment 1: ByteTrack vs ByteTrack-tuned vs BeeTracker on test, plus the oracle analysis.

## {{NAME4}} ({{ID4}}): counting, visualisation, demo and report
- Designed the counting protocols (banded mid-height line and entrance zone) and analysed why the line rule fails.
- Trajectory, heatmap, in/out count and track-length figures, and the failure-case figures.
- Demo program (live zone counting, FPS overlay) and the demo video.
- Integrated the final reports (zh and en) and slides.
