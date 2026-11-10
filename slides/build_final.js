// Build the final-presentation decks (English + Chinese), same theme and helpers as proposal/build_deck.js.
// run (from the project root): set NODE_PATH to a node_modules that has pptxgenjs, then
//   node slides/build_final.js <path to pptx skill apply_theme.js>
// numbers: report/RESULTS.md (copied by hand); figures: runs/figs, runs/demo, proposal/img; dataset stats: proposal/stats.json
const pptxgen = require("pptxgenjs");
const fs = require("fs");
const path = require("path");
const { applyTheme } = require(process.argv[2]);

const ROOT = path.join(__dirname, "..");
const S = JSON.parse(fs.readFileSync(path.join(ROOT, "proposal", "stats.json"), "utf8"));
const IMG = (p) => path.join(ROOT, p);
const fmt = (n) => n.toLocaleString("en-US");
// team.json (not in git: personal data) = [["name", "student id"], ...]; placeholders if it is missing
const TEAM_FILE = path.join(ROOT, "proposal", "team.json");
const TEAM = fs.existsSync(TEAM_FILE) ? JSON.parse(fs.readFileSync(TEAM_FILE, "utf8")) : [1, 2, 3, 4].map((i) => [`Member ${i}`, "000000000"]);
const [P1, P2, P3, P4] = TEAM.map(([n]) => n);
const cjk = (t) => (/[㐀-鿿]/.test(t) ? { fontFace: "Microsoft JhengHei" } : {});

// ---------- numbers (report/RESULTS.md) ----------
const MAIN = [ // tracker, HOTA, IDF1, MOTA, IDSW, zone /341, banded /316, plain /753
  ["ByteTrack", "40.2", "48.1", "15.0", "2186", "122", "263", "479"],
  ["ByteTrack-tuned", "45.2", "56.4", "43.2", "1382", "139", "264", "509"],
  ["Ours", "45.2", "55.7", "46.7", "1314", "119", "262", "492"],
  ["Ours + stitching", "42.2", "52.0", "46.9", "952", "164", "192", "359"],
  ["Oracle dets + ours", "79.2", "83.9", "96.2", "904", "100", "186", "312"],
  ["Oracle dets + ours + stitching", "74.8", "76.5", "96.4", "715", "163", "110", "212"],
];
const ABL = [ // setting key, HOTA, AssA, IDF1, MOTA, IDSW, banded /95, zone /173
  ["ours", "55.7", "51.6", "72.5", "78.3", "407", "78", "83"],
  ["biou", "53.5", "47.8", "68.3", "77.8", "518", "77", "89"],
  ["margin", "53.9", "48.2", "69.5", "78.2", "489", "74", "87"],
  ["damp", "55.5", "51.2", "71.9", "78.2", "413", "80", "86"],
  ["q", "55.5", "51.1", "72.0", "78.6", "392", "78", "88"],
  ["newtrack", "55.4", "51.2", "71.1", "77.1", "615", "78", "74"],
  ["buffer", "55.7", "51.6", "72.5", "78.3", "401", "79", "81"],
  ["bt_tuned", "54.8", "50.0", "71.0", "78.5", "427", "72", "82"],
  ["bt", "53.7", "48.5", "68.6", "76.6", "632", "76", "76"],
];
const RES = { labels: ["HOTA", "IDF1", "MOTA"], ours640: [45.2, 55.7, 46.7], ours1280: [48.0, 60.0, 60.3], bt1280: [47.8, 60.8, 59.2] };
const DROP = { k: ["1", "2", "3", "5"], bt: [40.2, 39.4, 38.0, 35.1], tuned: [45.2, 43.9, 42.1, 39.9], ours: [45.2, 43.5, 44.0, 42.0] }; // k=1 = main test table
const CORR = { tuned: [139, 145, 174, 156, 166], ours: [119, 118, 141, 119, 127] }; // none, dark, bright, blur, noise; zone err /341
const DENS = { // sorted by GT bees/frame
  seq: ["BEE24-12", "BEE24-36", "BEE24-16", "BEE24-34", "BEE24-13"], bpf: ["7.9", "10.5", "14.8", "20.7", "25.0"],
  bt: [71.0, 42.5, 53.9, 36.9, 67.8], tuned: [71.1, 49.1, 57.8, 41.2, 67.2], ours: [71.8, 46.4, 57.1, 42.5, 66.3], ours1280: [69.9, 44.2, 62.2, 47.6, 66.6],
};

// ---------- content ----------
const T = {
  en: {
    file: "BeeTrack_Final_EN.pptx", lang: "en-US", font: "Calibri",
    footer: "DLIA Final Project · BeeTrack",
    s1: {
      title: "BeeTrack",
      tag: "Detecting, Tracking and Counting Honeybees at the Hive Entrance",
      sub: "Final Presentation · Project 1: Real-Time Multi-Object Detection and Tracking",
      meta: "Deep Learning for Image Analysis (M5261) · 24 Dec 2026",
      notes: "Good morning. We are the BeeTrack team. In the proposal we promised a system that detects honeybees at a hive entrance, follows each bee with a stable ID, and counts how many go in and out. Today we show what we built and what we measured: the pipeline, our tracker BeeTracker, the counting protocol, the main test results, two controlled experiments, three robustness studies, the failure cases, and a live demo. All numbers are on the BEE24 test set, and every parameter was tuned on validation only.",
    },
    s2: {
      title: "Problem and Motivation",
      cap: "BEE24 sequence {seq}, frame {frame} (raw video frame)",
      items: [
        ["Why it matters", "Entrance traffic, bees leaving and returning, is a direct sign of colony activity and health. Counting it by hand does not scale."],
        ["Why it is hard", "Dozens of small, identical bees per frame; fast, sudden turns; bees crowd and occlude each other inside a dark entrance."],
        ["What we deliver", "Boxes, stable IDs, trajectories and in/out counts per hive entrance, faster than real time on one laptop GPU."],
      ],
      notes: "Why bees? Beekeepers watch the entrance because the number of bees leaving and returning tells you how active and healthy the colony is, and nobody can count that by eye for hours. It is also a hard tracking problem. In this raw frame you can see that the bees are small and look the same, they turn suddenly, and they pile up in the entrance where they hide each other. Our system takes the entrance video and outputs boxes, IDs, trajectories and in-and-out counts, in real time.",
    },
    s3: {
      title: "BEE24 Dataset and Split",
      stats: [[fmt(S.videos), "videos"], [fmt(S.boxes), "annotated boxes"], [fmt(S.tracks), "bee identities"], [S.avgPerFrame.toFixed(1), "bees per frame (mean)"]],
      head: ["Split", "Sequences", "Used for"],
      rows: [
        ["Train", "27 official training sequences (long videos without their last 1,050 frames)", "Detector training"],
        ["Val", "BEE24-10, 15, 24, 28 + last 1,000 frames of 33 and 35 (50-frame gap)", "All tuning, ablation"],
        ["Test", "Official 5: BEE24-12, 13, 16, 34, 36", "Run once per setting"],
      ],
      cap: "Ground-truth boxes and IDs from the dataset",
      notes: "We use BEE24 from the TOPICTrack paper: {videos} videos, about {boxesK} thousand boxes, every box with a track ID. The official split has {trainSeq} training and {testSeq} test sequences. We built validation from whole sequences, never single frames: four short training videos, plus the last thousand frames of the two long videos, with a fifty-frame gap so neighbouring frames cannot leak. Every threshold in this talk was chosen on validation. The test set was run once per setting.",
    },
    s4: {
      title: "Pipeline",
      cols: [
        ["Video", ["Hive entrance", "25 fps"]],
        ["Detector", ["YOLO11-s, fine-tuned", "640 or 1280 input"]],
        ["Cached dets", ["conf 0.05, NMS 0.7", "Shared by all trackers"]],
        ["Tracker", ["ByteTrack baseline", "BeeTracker (ours)"]],
        ["Counting", ["Entrance-zone in/out", "Trajectories, heatmap"]],
      ],
      improve: "our improvement",
      eval: "Speed (640): detector ~17 ms/frame incl. JPEG decode · tracker 1.63 ms/frame · live demo 66 FPS, real time at 25 fps",
      notes: "This is the pipeline. A fine-tuned YOLO11 small detector runs once per video at a low confidence threshold of 0.05, and we cache its boxes. Every tracker then reads exactly the same cached detections, so any difference between trackers comes from association alone, not from the detector. The tracker is either the ByteTrack baseline or our BeeTracker, highlighted. The last stage turns tracks into in-and-out counts with an entrance zone, plus trajectories and heatmaps. Speed: the detector takes about 17 milliseconds per frame including decoding, the tracker under 2, and the live demo runs at 66 FPS.",
    },
    s5: {
      title: "BeeTracker: Motion-Adaptive Association",
      dia: { prev: "t−1", pred: "Kalman prediction", buf: "buffered box", det: "detection", note: "Plain IoU = 0 → track lost. Buffered IoU > 0 → matched." },
      formula: "r = 0.1 + 0.1 · frames lost + 0.5 · min(speed / h, 1)",
      steps: [
        ["Buffered IoU, adaptive margin", "Match on enlarged boxes. The margin r grows with the bee's speed and with the frames the track has been lost."],
        ["Velocity damping", "A lost track's velocity is multiplied by 0.8 per frame, so a hidden bee does not drift away from where it disappeared."],
        ["Kalman velocity noise", "Process noise 1/80 instead of ByteTrack's 1/160: bees accelerate much harder than pedestrians."],
        ["Track birth and memory", "New-track threshold 0.5 (also given to ByteTrack-tuned, for fairness) and a 60-frame lost buffer."],
      ],
      notes: "Here is the core of our improvement. ByteTrack matches a predicted box to a detection by IoU. A bee is small and fast, so the prediction often lands beside the real bee: the IoU is zero and the track is lost. BeeTracker enlarges both boxes before computing IoU. The enlargement, the margin r, starts at 0.1 and grows with the bee's speed in box heights per frame and with how long the track has been lost. Second, while a track is lost we damp its velocity by 0.8 per frame so it stays near where it vanished. Third, we double the Kalman velocity noise. Finally we raise the new-track threshold to 0.5; we gave the same change to a tuned ByteTrack so the comparison is fair.",
    },
    s6: {
      title: "Counting Protocol",
      cap: "BEE24-36: our trajectories and the entrance zone (dashed)",
      rows: [
        ["Plain line", "753", "Counts every wiggle across a mid-height line: hovering bees make GT \"crossings\" that are not traffic."],
        ["Banded line (±0.05·H)", "316", "Hysteresis band fixed a priori. Better, but a broken track still loses its crossing."],
        ["Entrance zone (main)", "341", "One polygon per camera, drawn on the first frame without labels. Start outside → end inside = in; reverse = out."],
      ],
      gtLabel: "GT crossings (test)",
      foot: "Error = |pred in − GT in| + |pred out − GT out|, same rule for GT and predictions. Zone counting cut our relative counting error on test from 83% to 35%.",
      notes: "BEE24 has no count labels, so we derive counts from trajectories with the same rule for ground truth and predictions. Our first idea, a plain line across the image, failed: the ground truth alone produces 753 crossings on test, because bees wiggle back and forth on the line. A hysteresis band helps but still depends on one long unbroken track. Our main protocol is an entrance zone: one polygon per camera, drawn from the first frame only, without looking at labels or predictions. A track that starts outside and ends inside is one bee in, and the reverse is one bee out. This cut our relative counting error from 83 to 35 percent.",
    },
    s7: {
      title: "Main Results on the Test Set",
      head: ["Tracker (640 detector)", "HOTA", "IDF1", "MOTA", "IDSW", "Zone err /341", "Banded /316", "Plain /753"],
      tiles: [["45.2", "HOTA, +5.0 vs ByteTrack"], ["46.7", "MOTA, best non-oracle"], ["119", "zone error, best of all trackers"], ["−40%", "ID switches vs ByteTrack"]],
      foot: "Oracle (GT boxes) + ours still has 29% zone error → the remaining counting error is fragmentation in the crowded entrance, not detection.",
      notes: "These are the main test numbers, all from TrackEval on the same cached detections. Against default ByteTrack, our tracker gains 5 HOTA points, triples MOTA and cuts ID switches by 40 percent. Against the tuned ByteTrack, HOTA is tied at 45.2, but we have higher MOTA, fewer switches and the best zone count, 119 errors out of 341 crossings. Stitching broken tracklets helps the line counts but hurts zone counting and HOTA, so it is not our main result. The oracle rows feed ground-truth boxes to the tracker: even then the zone error is 29 percent, so what remains is track fragmentation, not detection.",
    },
    s8: {
      title: "Ablation on Validation",
      names: { ours: "Ours (full)", biou: "− buffered IoU (plain IoU)", margin: "− adaptive margin (fixed 0.1)", damp: "− velocity damping", q: "Default Kalman Q", newtrack: "Default new-track thresh 0.25", buffer: "track_buffer 30", bt_tuned: "ByteTrack-tuned", bt: "ByteTrack" },
      head: ["Setting", "HOTA", "AssA", "IDF1", "MOTA", "IDSW", "Band /95", "Zone /173"],
      take: [
        ["Buffered IoU matters most", "Without it: −2.2 HOTA, −3.8 AssA, +111 ID switches."],
        ["Adaptive margin", "A fixed 0.1 margin loses 1.8 HOTA and adds 82 switches."],
        ["Smaller effects", "Damping and Kalman Q add a few tenths each; the 60-frame buffer only helps zone count."],
      ],
      notes: "This ablation is on validation, removing one component at a time from the full tracker. The buffered IoU is the most important part: with plain IoU we lose 2.2 HOTA and get 111 more ID switches. Making the margin fixed instead of adaptive costs 1.8 HOTA. Velocity damping and the Kalman noise each add a few tenths. To be honest about the trade-offs: the default new-track threshold gives a better zone count on validation, 74, but almost doubles ID switches, and we chose the threshold for tracking quality before looking at test. We also tried and rejected buffered IoU in the low-score stage, gap interpolation and a short-track filter.",
    },
    s9: {
      title: "Experiment 2: Detector Input 640 vs 1280",
      chartTitle: "Test set, tracking metrics",
      series: ["Ours 640", "Ours 1280", "ByteTrack-tuned 1280"],
      tiles: [["+13.6", "MOTA (46.7 → 60.3)"], ["+2.8", "HOTA (45.2 → 48.0)"], ["119 → 120", "zone error: unchanged"], ["+9%", "time (17.0 → 18.5 ms/frame)"]],
      foot: "Val mAP50 0.946 → 0.955, mAP50-95 0.513 → 0.536. Same YOLO11-s, effective batch 64 at both sizes.",
      notes: "Experiment two keeps the tracker fixed and changes only the detector input size. At 1280 the detector finds more of the small bees: validation mAP50-95 rises from 0.513 to 0.536, and on test our tracker gains 2.8 HOTA and 13.6 MOTA. At 1280 ByteTrack-tuned and ours are close in HOTA, but ours keeps a better zone count. Interestingly the zone count itself does not improve with resolution, 119 versus 120, which again points to fragmentation rather than missed detections. The cost is small: only 9 percent slower, because the pipeline is bound by JPEG decoding.",
    },
    s10: {
      title: "Robustness: Frame Drop and Corruptions",
      t1: "Frame drop (keep every k-th frame), test HOTA", t2: "Image corruptions, test zone error /341",
      series: ["ByteTrack", "ByteTrack-tuned", "Ours"], k: "k", corr: ["none", "dark", "bright", "blur", "noise"],
      n1: "At k = 5: +2.1 HOTA and −37% ID switches vs ByteTrack-tuned. The margin grows with speed and frames lost.",
      n2: "Ours keeps the best zone count under every corruption. Noise hurts most; darkening even helps (fewer false positives).",
      notes: "Two robustness studies on test. On the left we simulate fast motion and low frame rates by keeping only every k-th frame, with the ground truth subsampled the same way. All trackers degrade, but ours degrades most gracefully: at k equal to 5 it is 2.1 HOTA above tuned ByteTrack with 37 percent fewer ID switches, because the adaptive margin grows when bees move further between frames. On the right we corrupt the images before detection: darkening, brightening, motion blur and Gaussian noise. Noise is the hardest. Darkening slightly helps, because fewer unlabeled bees in the dark get detected. Under every corruption our tracker has the lowest zone counting error.",
    },
    s11: {
      title: "Robustness: Crowd Density",
      chartTitle: "Test HOTA per sequence (GT bees per frame)",
      series: ["ByteTrack", "ByteTrack-tuned", "Ours", "Ours 1280"],
      items: [
        ["Turnover, not density", "BEE24-13 is the densest (25.0 bees/frame) but its 54 bees mostly stay: HOTA 66.3."],
        ["The hardest sequence", "BEE24-34 has 1,143 GT tracks. Ours 42.5 HOTA vs 41.2 tuned; 1280 input lifts it to 47.6."],
        ["Easy scene", "BEE24-12: 19 tracks, ours 71.8 HOTA, 4 ID switches, zone error 2/4."],
      ],
      notes: "Our third robustness condition is crowd density, split per test sequence and sorted by the number of bees per frame. The surprise is that density alone does not predict difficulty. The densest sequence, BEE24-13, still scores 66 HOTA because its bees mostly sit still. What hurts is turnover: bees entering, leaving and hiding in the entrance. BEE24-34 has over a thousand ground-truth tracks and is the hardest for every tracker; there ours beats tuned ByteTrack, and the 1280 detector adds another 5 points.",
    },
    s12: {
      title: "Failure Analysis",
      c1: "BEE24-34, frame 391: unmatched predictions",
      b1: "Many \"false positives\" are real bees: GT boxes are much looser than ours (IoU < 0.5), and bees on flowers or the ground at the border are unlabeled. BEE24-34 alone holds ~51k of our FP.",
      c2: "BEE24-36, frames 6 → 13: ID switch in the entrance cluster",
      b2: "Touching bees occlude each other; the track restarts with a new ID. The zone rule then sees two half-tracks and misses the crossing.",
      foot: "Counting errors are almost all undercounts: bees vanish into the dark entrance (GT follows them in), and fast bees break tracks.",
      notes: "Two kinds of failure. On the left, false positives in BEE24-34. When we look at them, many are real bees: the ground-truth boxes are much looser than our tight boxes, so the IoU falls under 0.5, and bees on flowers or the ground at the image border were never labeled. That one sequence holds about 51 thousand of our false positives. On the right, the real problem for counting: inside the entrance cluster, touching bees hide each other, the track breaks and restarts with a new ID. The zone rule then sees two half-tracks, neither of which crosses, so the bee is not counted. That is why almost all our counting errors are undercounts.",
    },
    s13: {
      title: "Demo: BEE24-36",
      cap: "Live demo still: zone, IDs, running in/out count and FPS",
      cap2: "Crossings per 20 s, GT (solid) vs ours (dashed)",
      tiles: [["40 / 22", "ours in / out (GT 49 / 32)"], ["66 FPS", "detector + tracker, 640"]],
      video: "Video played live: runs/demo/BEE24-36_ours.mp4",
      notes: "Now the demo. This still is from the BEE24-36 test video. The white polygon is the entrance zone, every box carries its ID, and the header shows the running in and out count and the frame rate. The header shows the current frame rate; on average the detector plus tracker runs at 66 FPS on the laptop GPU, more than twice real time. On this video we count 40 in and 22 out against 49 and 32 in the ground truth, the same numbers as our offline analytics. The chart on the right shows crossings over time; we follow the main peaks but undercount. Let me play the video.",
    },
    s14: {
      title: "Conclusions and Limitations",
      blocks: [
        ["What worked", "Motion-adaptive buffered IoU: +5.0 HOTA, −40% ID switches over ByteTrack, best zone count, most graceful under frame drop and corruptions, real time."],
        ["What remains", "Fragmentation: 390 tracks vs 163 GT on BEE24-36, median 0.64 s vs 2.24 s. Even GT boxes leave 29% zone error."],
        ["Negative results", "Tracklet stitching helps line counts but hurts zone count and HOTA. BIoU in the low-score stage, gap interpolation and short-track filtering did not help on val."],
        ["Future work", "Appearance re-identification (ReID) to reconnect a bee after occlusion in the entrance."],
      ],
      cap: "Track lengths on BEE24-36: ours are much shorter than GT",
      notes: "To conclude. Our motion-adaptive association gives a clear gain over default ByteTrack and the best zone count, and it degrades most gracefully under frame drop and corruptions, all in real time. The main limitation is fragmentation: on the demo video we produce 390 tracks for 163 real bees, and our median track lasts 0.64 seconds against 2.24 in the ground truth. Even with perfect boxes the zone error stays at 29 percent. We also report what did not work: tracklet stitching, low-score buffered matching, interpolation and short-track filtering. The natural next step is appearance re-identification, so a bee that disappears in the entrance gets its old ID back.",
    },
    s15: {
      title: "Team Contributions",
      members: [
        ["Data and evaluation", "Dataset preparation, val split, counting protocol and zones, TrackEval pipeline"],
        ["Detection and Exp 2", "YOLO11-s fine-tuning, 640 vs 1280, detection metrics"],
        ["Tracking and Exp 1", "ByteTrack baseline, BeeTracker improvement, ablation, robustness runs"],
        ["Counting and report", "Counting and visualization, demo application and video, report integration"],
      ],
      foot: "Every member can explain the full pipeline and code. AI assistance was used and is disclosed in the report.",
      thanks: "Thank you — questions?",
      notes: `Finally, who did what. ${P1} owned the data and evaluation: the split, the counting protocol and the TrackEval pipeline. ${P2} owned the detector and experiment two. ${P3} owned the tracker, our improvement and experiment one. ${P4} owned the counting, visualization, the demo and the report. We reviewed each other's code every week, so each of us can explain the whole system. We used AI assistance and disclose it in the report. Thank you, we are happy to take questions.`,
    },
  },
  zh: {
    file: "BeeTrack_Final_ZH.pptx", lang: "zh-TW", font: "Microsoft JhengHei",
    footer: "DLIA 期末專題 · BeeTrack",
    s1: {
      title: "BeeTrack",
      tag: "蜂巢出入口的蜜蜂偵測、追蹤與計數",
      sub: "期末發表 · Project 1：即時多目標偵測與追蹤",
      meta: "圖像分析的深度學習（M5261）· 2026 年 12 月 24 日",
      notes: "大家好，我們是 BeeTrack 這組。提案時我們說要做一個系統：偵測蜂巢出入口的蜜蜂、給每隻蜜蜂固定的 ID，並計算進出的數量。今天報告我們做出來的東西和量到的結果：整體流程、我們的追蹤器 BeeTracker、計數規則、測試集主要結果、兩組對照實驗、三種穩健性分析、失敗案例，最後是現場 demo。所有數字都來自 BEE24 測試集，所有參數都只在驗證集上調。",
    },
    s2: {
      title: "問題與動機",
      cap: "BEE24 序列 {seq} 第 {frame} 格（原始影片畫面）",
      items: [
        ["為什麼重要", "出入口流量（出巢與回巢的蜜蜂數）直接反映蜂群的活動力與健康，人工計數無法長時間進行。"],
        ["難在哪裡", "每格數十隻又小又長得一樣的蜜蜂，飛得快、突然轉向，還會在昏暗的巢口擠在一起互相遮擋。"],
        ["我們的成果", "每個巢口輸出偵測框、固定 ID、軌跡與進出計數，在一張筆電 GPU 上比即時還快。"],
      ],
      notes: "為什麼是蜜蜂？養蜂人會觀察巢口，因為出巢和回巢的數量能看出蜂群的活動力和健康，但沒有人能用眼睛連續數好幾個小時。這也是很難的追蹤問題。從這張原始畫面可以看到，蜜蜂很小、長得一樣、會突然轉向，而且在巢口擠成一團互相遮住。我們的系統輸入巢口影片，輸出偵測框、ID、軌跡和進出數量，而且是即時的。",
    },
    s3: {
      title: "BEE24 資料集與切分",
      stats: [[fmt(S.videos), "支影片"], [fmt(S.boxes), "個標註框"], [fmt(S.tracks), "個蜜蜂 ID"], [S.avgPerFrame.toFixed(1), "每格平均蜜蜂數"]],
      head: ["切分", "序列", "用途"],
      rows: [
        ["訓練", "27 段官方訓練序列；長影片去掉最後 1,050 格", "訓練偵測器"],
        ["驗證", "BEE24-10、15、24、28，加上 33 與 35 的最後 1,000 格（間隔 50 格）", "所有調參、消融實驗"],
        ["測試", "官方 5 段：BEE24-12、13、16、34、36", "每個設定只跑一次"],
      ],
      cap: "資料集內的標註框與 ID",
      notes: "我們使用 TOPICTrack 論文的 BEE24 資料集：{videos} 支影片、約 {boxesK} 千個框，每個框都有追蹤 ID。官方切成訓練 {trainSeq} 段、測試 {testSeq} 段。驗證集我們都是整段切出來，不是抽單張：四段較短的訓練影片，加上兩段長影片的最後一千格，中間隔五十格，避免相鄰影格洩漏。今天提到的每一個門檻都在驗證集上決定，測試集每個設定只跑一次。",
    },
    s4: {
      title: "系統流程",
      cols: [
        ["影片", ["蜂巢出入口", "25 fps"]],
        ["偵測器", ["YOLO11-s 微調", "輸入 640 或 1280"]],
        ["偵測快取", ["conf 0.05、NMS 0.7", "所有追蹤器共用"]],
        ["追蹤器", ["ByteTrack baseline", "BeeTracker（我們的）"]],
        ["計數", ["巢口區域進出計數", "軌跡、熱度圖"]],
      ],
      improve: "我們的改進",
      eval: "速度（640）：偵測器約 17 ms/格（含 JPEG 解碼）· 追蹤器 1.63 ms/格 · 現場 demo 66 FPS，25 fps 下即時有餘",
      notes: "這是整個流程。微調過的 YOLO11 small 偵測器對每支影片只跑一次，信心門檻設很低，0.05，結果存成快取。之後每個追蹤器讀的都是完全相同的偵測結果，所以追蹤器之間的差異只來自關聯方法，不是偵測器。追蹤器可以是 ByteTrack baseline 或我們的 BeeTracker，也就是標成橘色的那格。最後一步把軌跡轉成巢口區域的進出計數，並畫出軌跡和熱度圖。速度方面，偵測器含解碼每格約 17 毫秒，追蹤器不到 2 毫秒，現場 demo 有 66 FPS。",
    },
    s5: {
      title: "BeeTracker：運動自適應關聯",
      dia: { prev: "t−1", pred: "Kalman 預測", buf: "放大後的框", det: "偵測框", note: "一般 IoU = 0 → 跟丟；buffered IoU > 0 → 配對成功" },
      formula: "r = 0.1 + 0.1 · 跟丟格數 + 0.5 · min(速度 / h, 1)",
      steps: [
        ["Buffered IoU、自適應邊界", "用放大的框計算 IoU；放大量 r 隨蜜蜂速度與跟丟的格數增加。"],
        ["速度衰減", "跟丟的軌跡每格速度乘以 0.8，被遮住的蜜蜂不會漂離消失的位置。"],
        ["Kalman 速度雜訊", "process noise 用 1/80，而不是 ByteTrack 的 1/160：蜜蜂的加速度比行人大得多。"],
        ["軌跡建立與保留", "新軌跡門檻 0.5（ByteTrack-tuned 也用同樣設定以求公平），跟丟後保留 60 格。"],
      ],
      notes: "這是我們改進的核心。ByteTrack 用 IoU 把預測框和偵測框配對。蜜蜂又小又快，預測框常常落在真正的蜜蜂旁邊，IoU 等於 0，軌跡就跟丟了。BeeTracker 在計算 IoU 之前先把兩個框放大，放大量 r 從 0.1 起跳，會隨著蜜蜂的速度（以框高為單位）和跟丟的格數增加。第二，軌跡跟丟時每格把速度乘以 0.8，讓它留在消失的地方附近。第三，把 Kalman 的速度雜訊加倍。最後把新軌跡門檻提高到 0.5；我們也把同樣的設定給了 ByteTrack-tuned，比較才公平。",
    },
    s6: {
      title: "計數規則",
      cap: "BEE24-36：我們的軌跡與巢口區域（虛線）",
      rows: [
        ["單純計數線", "753", "每次越過中線都算：在線上徘徊的蜜蜂讓真值本身就產生大量假的「穿越」。"],
        ["容差帶計數線（±0.05·H）", "316", "事先固定的遲滯帶，比較好，但軌跡一斷還是會漏算。"],
        ["巢口區域（主要）", "341", "每台攝影機一個多邊形，只看第一格畫、不看標註。從外面開始、裡面結束＝進；反之＝出。"],
      ],
      gtLabel: "真值穿越數（測試）",
      foot: "誤差 = |預測進 − 真值進| + |預測出 − 真值出|，真值與預測用同一規則。巢口區域把我們在測試集的相對計數誤差從 83% 降到 35%。",
      notes: "BEE24 沒有計數標註，所以我們從軌跡推算數量，真值和預測用同一套規則。一開始想用一條橫線，結果失敗了：光是真值在測試集上就有 753 次穿越，因為蜜蜂會在線上來回晃。加一個遲滯帶有改善，但還是要靠一條沒斷的長軌跡。我們的主要規則是巢口區域：每台攝影機一個多邊形，只看第一格畫，完全不看標註和預測。軌跡從區域外開始、在區域內結束就算進一隻，反過來算出一隻。這讓相對計數誤差從 83% 降到 35%。",
    },
    s7: {
      title: "測試集主要結果",
      head: ["追蹤器（640 偵測器）", "HOTA", "IDF1", "MOTA", "IDSW", "區域誤差 /341", "容差帶中線 /316", "單純線 /753"],
      tiles: [["45.2", "HOTA，比 ByteTrack +5.0"], ["46.7", "MOTA，非 oracle 最高"], ["119", "區域計數誤差，所有追蹤器最低"], ["−40%", "ID 換錯次數（對比 ByteTrack）"]],
      foot: "Oracle（真值框）＋我們的追蹤器仍有 29% 區域誤差 → 剩下的計數誤差主要是擁擠巢口的軌跡斷裂，而不是偵測。",
      notes: "這是測試集的主要數字，全部用 TrackEval、同一份偵測快取算出。和預設 ByteTrack 相比，我們的追蹤器 HOTA 高 5 分、MOTA 是三倍、ID 換錯少了 40%。和調過的 ByteTrack 相比，HOTA 都是 45.2，但我們的 MOTA 較高、換錯較少，區域計數也最好，341 次穿越只錯 119。把斷掉的片段接起來（stitching）對計數線有幫助，卻會傷害區域計數和 HOTA，所以不當主要結果。Oracle 那兩列是直接給追蹤器真值框：即使如此區域誤差還有 29%，所以剩下的問題是軌跡斷裂，不是偵測。",
    },
    s8: {
      title: "驗證集消融實驗",
      names: { ours: "我們的（完整）", biou: "− buffered IoU（一般 IoU）", margin: "− 自適應邊界（固定 0.1）", damp: "− 速度衰減", q: "預設 Kalman Q", newtrack: "預設新軌跡門檻 0.25", buffer: "track_buffer 30", bt_tuned: "ByteTrack-tuned", bt: "ByteTrack" },
      head: ["設定", "HOTA", "AssA", "IDF1", "MOTA", "IDSW", "容差帶 /95", "區域 /173"],
      take: [
        ["Buffered IoU 最關鍵", "拿掉後：HOTA −2.2、AssA −3.8、ID 換錯 +111。"],
        ["自適應邊界", "改成固定 0.1：HOTA −1.8、換錯 +82。"],
        ["影響較小的部分", "速度衰減與 Kalman Q 各加零點幾分；保留 60 格只對區域計數有幫助。"],
      ],
      notes: "這是驗證集上的消融實驗，每次從完整追蹤器拿掉一個元件。最重要的是 buffered IoU：改回一般 IoU，HOTA 掉 2.2，ID 換錯多 111 次。把邊界改成固定、不自適應，HOTA 掉 1.8。速度衰減和 Kalman 雜訊各貢獻零點幾分。誠實說明取捨：預設的新軌跡門檻在驗證集的區域計數比較好，74，但 ID 換錯幾乎加倍；我們是在看測試集之前、以追蹤品質為準選的門檻。我們也試過並放棄：在低分階段用 buffered IoU、補洞插值、過濾短軌跡。",
    },
    s9: {
      title: "實驗 2：偵測器輸入 640 對比 1280",
      chartTitle: "測試集追蹤指標",
      series: ["我們的 640", "我們的 1280", "ByteTrack-tuned 1280"],
      tiles: [["+13.6", "MOTA（46.7 → 60.3）"], ["+2.8", "HOTA（45.2 → 48.0）"], ["119 → 120", "區域誤差：幾乎不變"], ["+9%", "時間（17.0 → 18.5 ms/格）"]],
      foot: "驗證 mAP50 0.946 → 0.955，mAP50-95 0.513 → 0.536。同一個 YOLO11-s，兩種大小的等效 batch 都是 64。",
      notes: "實驗二固定追蹤器，只改偵測器的輸入大小。1280 能找到更多小蜜蜂：驗證集 mAP50-95 從 0.513 升到 0.536，測試集上我們的追蹤器 HOTA 加 2.8、MOTA 加 13.6。在 1280 下 ByteTrack-tuned 和我們的 HOTA 很接近，但我們的區域計數仍比較好。有趣的是區域計數本身沒有因為解析度變好，119 對 120，這再次說明問題在軌跡斷裂，而不是漏偵測。代價很小，只慢了 9%，因為瓶頸在 JPEG 解碼。",
    },
    s10: {
      title: "穩健性：掉幀與影像劣化",
      t1: "掉幀（每 k 格取 1 格），測試集 HOTA", t2: "影像劣化，測試集區域誤差 /341",
      series: ["ByteTrack", "ByteTrack-tuned", "我們的"], k: "k", corr: ["原圖", "變暗", "變亮", "模糊", "雜訊"],
      n1: "k = 5 時比 ByteTrack-tuned HOTA +2.1、ID 換錯 −37%：邊界會隨速度和跟丟格數放大。",
      n2: "每種劣化下我們的區域計數都最好。雜訊影響最大；變暗反而有幫助（誤偵測變少）。",
      notes: "測試集上的兩種穩健性分析。左邊用每 k 格取一格來模擬快速移動和低幀率，真值也用同樣方式抽樣。所有追蹤器都會變差，但我們退化得最平緩：k 等於 5 時，HOTA 比調過的 ByteTrack 高 2.1，ID 換錯少 37%，因為蜜蜂在兩格之間移動越遠，自適應邊界就放得越大。右邊是在偵測前先把影像劣化：變暗、變亮、動態模糊和高斯雜訊。雜訊最難。變暗反而略有幫助，因為暗處沒標註的蜜蜂比較少被偵測到。每一種劣化下，我們的區域計數誤差都最低。",
    },
    s11: {
      title: "穩健性：擁擠程度",
      chartTitle: "各測試序列 HOTA（每格真值蜜蜂數）",
      series: ["ByteTrack", "ByteTrack-tuned", "我們的", "我們的 1280"],
      items: [
        ["關鍵是進出，不是密度", "BEE24-13 最擁擠（每格 25.0 隻），但 54 隻蜜蜂大多停著不動：HOTA 66.3。"],
        ["最難的序列", "BEE24-34 有 1,143 條真值軌跡。我們 42.5 HOTA，tuned 41.2；1280 輸入提升到 47.6。"],
        ["簡單的場景", "BEE24-12：19 條軌跡，我們 71.8 HOTA、4 次換錯、區域誤差 2/4。"],
      ],
      notes: "第三種穩健性條件是擁擠程度，按測試序列分開，依每格蜜蜂數排序。意外的是，密度本身並不能預測難度。最擁擠的 BEE24-13 仍有 66 的 HOTA，因為蜜蜂大多停著不動。真正困難的是進出的流動：蜜蜂進來、離開、躲進巢口。BEE24-34 有超過一千條真值軌跡，對每個追蹤器都是最難的；在那裡我們贏過調過的 ByteTrack，用 1280 偵測器再多 5 分。",
    },
    s12: {
      title: "失敗案例分析",
      c1: "BEE24-34 第 391 格：沒配對到的預測框",
      b1: "很多「誤偵測」其實是真的蜜蜂：真值框比我們的框鬆很多（IoU < 0.5），畫面邊緣花上、地上的蜜蜂也沒標。光 BEE24-34 就佔了我們約 51k 個 FP。",
      c2: "BEE24-36 第 6 → 13 格：巢口蜂群中的 ID 換錯",
      b2: "互相碰觸的蜜蜂彼此遮擋，軌跡以新 ID 重新開始；區域規則只看到兩段半截軌跡，漏算這次穿越。",
      foot: "計數誤差幾乎都是少算：蜜蜂消失在昏暗的巢口裡（真值仍跟著進去），飛得快的蜜蜂讓軌跡斷掉。",
      notes: "兩種失敗。左邊是 BEE24-34 的誤偵測。仔細看會發現很多其實是真的蜜蜂：真值框比我們緊貼的框鬆很多，IoU 低於 0.5，而畫面邊緣在花上或地上的蜜蜂根本沒有標。光這一段就佔了我們大約五萬一千個誤偵測。右邊才是計數真正的問題：在巢口的蜂群裡，互相碰觸的蜜蜂把彼此遮住，軌跡斷掉後用新的 ID 重來。區域規則只看到兩段半截的軌跡，兩段都沒有完整穿越，於是這隻蜜蜂沒被算到。這就是為什麼我們的計數誤差幾乎都是少算。",
    },
    s13: {
      title: "Demo：BEE24-36",
      cap: "現場 demo 截圖：巢口區域、ID、即時進出計數與 FPS",
      cap2: "每 20 秒穿越數，真值（實線）對比我們的（虛線）",
      tiles: [["40 / 22", "我們的進／出（真值 49 / 32）"], ["66 FPS", "偵測器＋追蹤器，640"]],
      video: "現場播放影片：runs/demo/BEE24-36_ours.mp4",
      notes: "接下來是 demo。這張截圖來自測試影片 BEE24-36。白色多邊形是巢口區域，每個框都有 ID，上方顯示即時的進出計數和幀率。上方顯示的是當下的幀率，平均起來偵測器加追蹤器在筆電 GPU 上跑 66 FPS，是即時的兩倍以上。這支影片我們算出進 40、出 22，真值是 49 和 32，和離線分析的數字完全一樣。右邊的圖是穿越數隨時間的變化，主要的高峰都有跟到，但整體少算。現在播放影片。",
    },
    s14: {
      title: "結論與限制",
      blocks: [
        ["有效的部分", "運動自適應 buffered IoU：比 ByteTrack HOTA +5.0、ID 換錯 −40%，區域計數最好，掉幀與劣化下退化最平緩，而且即時。"],
        ["尚未解決", "軌跡斷裂：BEE24-36 上 390 條軌跡對比真值 163 條，中位長度 0.64 秒對比 2.24 秒；即使給真值框，區域誤差仍有 29%。"],
        ["負面結果", "Tracklet stitching 對計數線有幫助，但傷害區域計數與 HOTA；低分階段 BIoU、補洞插值、短軌跡過濾在驗證集都沒有幫助。"],
        ["未來工作", "外觀重識別（ReID），讓在巢口被遮住的蜜蜂重新接回原本的 ID。"],
      ],
      cap: "BEE24-36 軌跡長度：我們的軌跡比真值短很多",
      notes: "總結。我們的運動自適應關聯比預設 ByteTrack 有明顯提升、區域計數最好，在掉幀和影像劣化下退化得最平緩，而且是即時的。主要限制是軌跡斷裂：demo 影片裡 163 隻真實的蜜蜂我們產生了 390 條軌跡，軌跡長度中位數 0.64 秒，真值是 2.24 秒。就算給完美的框，區域誤差仍有 29%。我們也報告沒用的嘗試：tracklet stitching、低分階段的 buffered 配對、插值和短軌跡過濾。下一步很自然是外觀重識別，讓在巢口消失的蜜蜂拿回原本的 ID。",
    },
    s15: {
      title: "組員貢獻",
      members: [
        ["資料與評估", "資料集準備、驗證集切分、計數規則與巢口區域、TrackEval 評估流程"],
        ["偵測與實驗 2", "YOLO11-s 微調、640 對比 1280、偵測指標"],
        ["追蹤與實驗 1", "ByteTrack baseline、BeeTracker 改進、消融實驗、穩健性實驗"],
        ["計數與報告", "計數與視覺化、demo 程式與影片、報告整合"],
      ],
      foot: "每位組員都能解釋完整流程與程式碼；我們有使用 AI 輔助，並已在報告中揭露。",
      thanks: "謝謝，歡迎提問",
      notes: `最後是分工。${P1}負責資料和評估：資料切分、計數規則和 TrackEval 評估流程。${P2}負責偵測器和實驗二。${P3}負責追蹤器、我們的改進和實驗一。${P4}負責計數、視覺化、demo 和報告。我們每週互相看彼此的程式碼，所以每個人都能解釋整個系統。我們有使用 AI 輔助，並在報告中揭露。謝謝大家，歡迎提問。`,
    },
  },
};

const fill = (s) => s.replace(/\{(\w+)\}/g, (_, k) => (k === "boxesK" ? Math.round(S.boxes / 1000) : k in S ? (typeof S[k] === "number" ? fmt(S[k]) : S[k]) : `{${k}}`));

// ---------- theme (same as the proposal) ----------
const COL = { ink: "1F1F1F", muted: "5C5C5C", card: "F4F4F2", line: "D9D9D6", honey: "B7791F", honeySoft: "FBF1DE" };
const SER = { bt: "3A3A3A", tuned: "8C8C8C", ours: COL.honey, ours1280: "D9A441" }; // chart colors, hex only

async function build(L) {
  const THEME = {
    name: "BeeTrack Minimal", headFontFace: L.font, bodyFontFace: L.font,
    colors: { dk1: COL.ink, lt1: "FFFFFF", dk2: COL.muted, lt2: COL.card, accent1: COL.honey, accent2: "3A3A3A", accent3: "8C8C8C", accent4: "D9A441", accent5: "6B6B6B", accent6: "C9C9C4", hlink: "8A5A12", folHlink: "6B6B6B" },
  };
  const pres = new pptxgen();
  pres.layout = "LAYOUT_WIDE"; // 13.333 x 7.5 in
  pres.theme = { headFontFace: L.font, bodyFontFace: L.font };
  pres.title = L.s1.title;
  const C = pres.SchemeColor;
  const W = 13.333, M = 0.6;

  pres.defineSlideMaster({
    title: "Title", background: { color: "FFFFFF" },
    objects: [
      { placeholder: { options: { name: "title", type: "title", x: M, y: 1.6, w: 5.9, h: 1.3, fontSize: 60, bold: true, color: C.text1, valign: "bottom", align: "left", margin: 0 }, text: "" } },
      { placeholder: { options: { name: "body", type: "body", x: M, y: 3.1, w: 5.9, h: 3.6, fontSize: 15, color: C.text2, valign: "top", margin: 0 }, text: "" } },
    ],
  });
  pres.defineSlideMaster({
    title: "Content", background: { color: "FFFFFF" },
    objects: [
      { placeholder: { options: { name: "title", type: "title", x: M, y: 0.4, w: W - 2 * M, h: 0.9, fontSize: 34, bold: true, color: C.text1, valign: "middle", align: "left", margin: 0 }, text: "" } },
      { text: { text: L.footer, options: { x: M, y: 7.0, w: 8, h: 0.3, fontSize: 10, color: C.text2, margin: 0 } } },
    ],
    slideNumber: { x: W - M - 0.6, y: 7.0, w: 0.6, h: 0.3, fontSize: 10, color: COL.muted, align: "right" },
  });

  const TB = (o) => ({ isTextBox: true, lang: L.lang, margin: 0, color: C.text1, fontSize: 16, valign: "top", ...o });
  const content = (title) => { const s = pres.addSlide({ masterName: "Content" }); s.addText(title, { placeholder: "title" }); return s; };
  const caption = (s, text, x, y, w) => s.addText(text, TB({ x, y, w, h: 0.3, fontSize: 11, color: C.text2 }));
  const pic = (s, file, x, y, w, h, alt) => s.addImage({ path: IMG(file), x, y, w, h, altText: alt });
  const num = (s, n, x, y) => s.addText(String(n).padStart(2, "0"), TB({ x, y, w: 0.6, h: 0.42, fontSize: 20, bold: true, color: C.accent1, valign: "middle" }));
  const box = (s, x, y, w, h, color) => s.addShape(pres.shapes.RECTANGLE, { x, y, w, h, fill: { color }, line: { color } });
  // hot = index of the row to highlight (our tracker)
  // numeric tables center every column but the first; text tables (num = false) stay left-aligned
  const table = (s, head, rows, x, y, w, colW, fs = 14, hot = -1, rowH = 0.42, num = true) => {
    const al = (j) => (j && num ? "center" : "left");
    const hdr = head.map((h, j) => ({ text: h, options: { bold: true, color: C.text1, fill: { color: C.background2 }, align: al(j) } }));
    const body = rows.map((r, i) => r.map((c, j) => ({ text: c, options: { align: al(j), ...(i === hot ? { bold: true, fill: { color: COL.honeySoft } } : {}) } })));
    s.addTable([hdr, ...body], {
      x, y, w, colW, fontSize: fs, color: C.text1, lang: L.lang, valign: "middle",
      border: { type: "solid", pt: 0.75, color: COL.line }, margin: [0.04, 0.1, 0.04, 0.1], rowH,
    });
  };
  const tiles = (s, items, x0, y, w, h, gap = 0.3) => items.forEach(([v, k], i) => {
    const x = x0 + i * (w + gap);
    box(s, x, y, w, h, C.background2);
    s.addText(v, TB({ x: x + 0.2, y: y + 0.12, w: w - 0.4, h: 0.65, fontSize: v.length > 7 ? 24 : 30, bold: true, color: C.accent1, valign: "middle" }));
    s.addText(k, TB({ x: x + 0.2, y: y + 0.8, w: w - 0.4, h: h - 0.9, fontSize: 13, color: C.text2 }));
  });
  // native chart, quiet frame, same label look on every chart
  const chart = (s, type, data, o) => s.addChart(type, data, {
    titleFontFace: "+mn-lt", titleFontSize: 14, titleColor: COL.ink, showTitle: !!o.title,
    catAxisLabelFontFace: "+mn-lt", valAxisLabelFontFace: "+mn-lt", legendFontFace: "+mn-lt", dataLabelFontFace: "+mn-lt",
    catAxisLabelFontSize: 12, valAxisLabelFontSize: 11, legendFontSize: 12, dataLabelFontSize: 10,
    catAxisLabelColor: COL.muted, valAxisLabelColor: COL.muted, dataLabelColor: COL.ink, legendColor: COL.ink,
    valGridLine: { color: COL.line, size: 0.75 }, catGridLine: { style: "none" },
    showLegend: true, legendPos: "b", ...o,
  });

  // 1 title
  let s = pres.addSlide({ masterName: "Title" });
  s.addText(L.s1.title, { placeholder: "title" });
  s.addText([
    { text: L.s1.tag, options: { breakLine: true, fontSize: 22, bold: true, color: C.text1, paraSpaceAfter: 28 } },
    { text: L.s1.sub, options: { breakLine: true, paraSpaceAfter: 6 } },
    { text: L.s1.meta, options: { breakLine: true, paraSpaceAfter: 18 } },
    ...TEAM.flatMap(([n, id], i) => [
      { text: n, options: { color: C.accent1, bold: true, ...cjk(n) } },
      { text: "  " + id, options: { color: C.text2, breakLine: i < TEAM.length - 1, paraSpaceAfter: 4 } },
    ]),
  ], { placeholder: "body" });
  pic(s, "proposal/img/hive_title.jpg", 7.0, 0, W - 7.0, 7.5, "Hive entrance from the BEE24 dataset");
  s.addNotes(L.s1.notes);

  // 2 problem: photo left, three numbered points right
  s = content(L.s2.title);
  pic(s, "proposal/img/hive_raw.jpg", M, 1.55, 6.0, 6.0 * S.imgRatio, "Raw BEE24 frame of a hive entrance with many bees");
  caption(s, fill(L.s2.cap).replace("{seq}", S.rawSeq).replace("{frame}", S.rawFrame), M, 1.6 + 6.0 * S.imgRatio, 6.0);
  L.s2.items.forEach(([h, b], i) => {
    const y = 1.6 + i * 1.75;
    num(s, i + 1, 7.0, y);
    s.addText(h, TB({ x: 7.6, y: y + 0.02, w: 5.1, h: 0.4, fontSize: 18, bold: true }));
    s.addText(b, TB({ x: 7.6, y: y + 0.45, w: 5.1, h: 1.2, fontSize: 15 }));
  });
  s.addNotes(L.s2.notes);

  // 3 dataset: stat callouts, split table left, GT image right
  s = content(L.s3.title);
  L.s3.stats.forEach(([v, k], i) => {
    const x = M + i * 3.07;
    box(s, x, 1.45, 2.85, 1.3, C.background2);
    s.addText(v, TB({ x: x + 0.2, y: 1.55, w: 2.45, h: 0.7, fontSize: 32, bold: true, color: C.accent1 }));
    s.addText(k, TB({ x: x + 0.2, y: 2.25, w: 2.45, h: 0.4, fontSize: 13, color: C.text2 }));
  });
  table(s, L.s3.head, L.s3.rows, M, 3.1, 7.2, [1.0, 4.2, 2.0], 14, -1, 0.42, false);
  const gw = 4.75; pic(s, "proposal/img/gt_boxes.jpg", W - M - gw, 3.1, gw, gw * S.imgRatio, "Ground-truth boxes and IDs drawn on a BEE24 frame");
  caption(s, L.s3.cap, W - M - gw, 3.15 + gw * S.imgRatio, gw);
  s.addNotes(fill(L.s3.notes));

  // 4 pipeline: five stage headers with arrows, sub-steps under each, tracker highlighted, speed bar
  s = content(L.s4.title);
  const bw = 2.05, gap = 0.47, hy = 2.0, hh = 0.75, sy = 3.0, sh = 0.6, sp = 0.75, ey = 5.2;
  L.s4.cols.forEach(([h, subs], i) => {
    const x = M + i * (bw + gap), hot = i === 3;
    box(s, x, hy, bw, hh, hot ? COL.honey : COL.ink);
    s.addText(h, TB({ x, y: hy, w: bw, h: hh, fontSize: 17, bold: true, color: C.background1, align: "center", valign: "middle" }));
    subs.forEach((t, j) => {
      box(s, x, sy + j * sp, bw, sh, hot ? COL.honeySoft : COL.card);
      s.addText(t, TB({ x: x + 0.1, y: sy + j * sp, w: bw - 0.2, h: sh, fontSize: 13, align: "center", valign: "middle" }));
    });
    if (i < 4) s.addShape(pres.shapes.LINE, { x: x + bw + 0.06, y: hy + hh / 2, w: gap - 0.12, h: 0, line: { color: COL.muted, width: 1.5, endArrowType: "triangle" } });
    if (hot) s.addText(L.s4.improve, TB({ x, y: hy - 0.45, w: bw, h: 0.35, fontSize: 13, bold: true, color: C.accent1, align: "center" }));
  });
  s.addShape(pres.shapes.RECTANGLE, { x: M, y: ey, w: W - 2 * M, h: 0.85, fill: { color: "FFFFFF" }, line: { color: COL.line, width: 1 } });
  s.addText(L.s4.eval, TB({ x: M + 0.3, y: ey, w: W - 2 * M - 0.6, h: 0.85, fontSize: 15, valign: "middle" }));
  s.addNotes(L.s4.notes);

  // 5 BeeTracker: buffered-IoU diagram left, four numbered changes right
  s = content(L.s5.title);
  const D = L.s5.dia, px = M, py = 1.55, pw = 5.0, ph = 4.95;
  box(s, px, py, pw, ph, COL.card);
  const bx = (x, y, w, h, line, dash, fillC) => s.addShape(pres.shapes.RECTANGLE, { x, y, w, h, fill: fillC ? { color: fillC } : { type: "none" }, line: { color: line, width: 2, dashType: dash || "solid" } });
  const lab = (t, x, y, w, color, align = "left") => s.addText(t, TB({ x, y, w, h: 0.3, fontSize: 13, bold: true, color, align }));
  bx(1.05, 3.75, 0.7, 0.55, COL.muted);                         // previous position
  lab(D.prev, 1.05, 4.35, 0.7, C.text2, "center");
  s.addShape(pres.shapes.LINE, { x: 1.8, y: 3.45, w: 0.75, h: 0.4, flipV: true, line: { color: COL.muted, width: 1.5, endArrowType: "triangle" } });
  bx(2.05, 2.55, 1.9, 1.6, COL.honey, "dash");                   // enlarged (buffered) prediction
  lab(D.buf, 2.05, 2.2, 2.4, C.accent1);
  bx(2.55, 2.95, 0.9, 0.7, COL.ink);                             // Kalman prediction
  lab(D.pred, 1.6, 4.25, 2.0, C.text1, "center");
  bx(3.5, 3.7, 0.9, 0.7, COL.honey, null, COL.honeySoft);        // detection: misses the prediction, overlaps the buffer
  lab(D.det, 3.35, 4.45, 1.2, C.accent1, "center");
  s.addText(L.s5.formula, TB({ x: px + 0.25, y: 5.05, w: pw - 0.5, h: 0.4, fontSize: 14, bold: true }));
  s.addText(D.note, TB({ x: px + 0.25, y: 5.55, w: pw - 0.5, h: 0.7, fontSize: 14, color: C.text2 }));
  L.s5.steps.forEach(([h, b], i) => {
    const y = 1.55 + i * 1.25;
    num(s, i + 1, 6.1, y);
    s.addText(h, TB({ x: 6.7, y: y + 0.02, w: 6.0, h: 0.4, fontSize: 17, bold: true }));
    s.addText(b, TB({ x: 6.7, y: y + 0.45, w: 6.0, h: 0.75, fontSize: 14 }));
  });
  s.addNotes(L.s5.notes);

  // 6 counting protocol: trajectories with zone left, three protocols right (main one highlighted)
  s = content(L.s6.title);
  const tw = 5.9, th = tw * 563 / 954;
  pic(s, "runs/figs/BEE24-36_ours_trajectories.png", M, 1.55, tw, th, "Trajectories of our tracks on BEE24-36 with the entrance zone drawn as a dashed polygon");
  caption(s, L.s6.cap, M, 1.6 + th, tw);
  const rx = 6.85, rw = W - M - rx;
  L.s6.rows.forEach(([h, n, b], i) => {
    const y = 1.55 + i * 1.38, main = i === 2;
    box(s, rx, y, rw, 1.25, main ? COL.honeySoft : COL.card);
    s.addText(h, TB({ x: rx + 0.25, y: y + 0.12, w: rw - 1.6, h: 0.35, fontSize: 16, bold: true, color: main ? C.accent1 : C.text1 }));
    s.addText([{ text: n, options: { fontSize: 22, bold: true, color: main ? C.accent1 : C.text1, breakLine: true } }, { text: L.s6.gtLabel, options: { fontSize: 10, color: C.text2 } }],
      TB({ x: rx + rw - 1.45, y: y + 0.12, w: 1.25, h: 0.8, align: "right" }));
    s.addText(b, TB({ x: rx + 0.25, y: y + 0.5, w: rw - 1.75, h: 0.7, fontSize: 13, color: C.text2 }));
  });
  s.addText(L.s6.foot, TB({ x: M, y: 5.85, w: W - 2 * M, h: 0.75, fontSize: 15, bold: true }));
  s.addNotes(L.s6.notes);

  // 7 main results: table, highlighted row, four stat tiles, oracle note
  s = content(L.s7.title);
  table(s, L.s7.head, MAIN, M, 1.45, W - 2 * M, [3.43, 1.1, 1.1, 1.1, 1.1, 1.4, 1.4, 1.5], 14, 2, 0.4);
  tiles(s, L.s7.tiles, M, 4.7, (W - 2 * M - 0.9) / 4, 1.3);
  s.addText(L.s7.foot, TB({ x: M, y: 6.15, w: W - 2 * M, h: 0.7, fontSize: 14, color: C.text2 }));
  s.addNotes(L.s7.notes);

  // 8 ablation: table left, three takeaways right
  s = content(L.s8.title);
  table(s, L.s8.head, ABL.map(([k, ...v]) => [L.s8.names[k], ...v]), M, 1.45, 8.3, [3.05, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75], 13, 0, 0.44);
  L.s8.take.forEach(([h, b], i) => {
    const y = 1.45 + i * 1.6;
    box(s, 9.2, y, W - M - 9.2, 1.4, i === 0 ? COL.honeySoft : COL.card);
    s.addText(h, TB({ x: 9.4, y: y + 0.15, w: W - M - 9.6, h: 0.35, fontSize: 15, bold: true, color: i === 0 ? C.accent1 : C.text1 }));
    s.addText(b, TB({ x: 9.4, y: y + 0.55, w: W - M - 9.6, h: 0.8, fontSize: 13, color: C.text2 }));
  });
  s.addNotes(L.s8.notes);

  // 9 Exp2: clustered bar chart left, four tiles right (2 x 2), detector note
  s = content(L.s9.title);
  chart(s, pres.charts.BAR, L.s9.series.map((name, i) => ({ name, labels: RES.labels, values: [RES.ours640, RES.ours1280, RES.bt1280][i] })), {
    x: M, y: 1.45, w: 7.0, h: 4.7, barDir: "col", barGapWidthPct: 60, title: L.s9.chartTitle,
    chartColors: [SER.ours, SER.ours1280, SER.tuned], showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.0", valAxisMinVal: 0, valAxisMaxVal: 70,
  });
  const t9w = (W - M - 8.0 - 0.3) / 2;
  tiles(s, L.s9.tiles.slice(0, 2), 8.0, 1.6, t9w, 1.35);
  tiles(s, L.s9.tiles.slice(2), 8.0, 3.25, t9w, 1.35);
  s.addText(L.s9.foot, TB({ x: 8.0, y: 4.95, w: W - M - 8.0, h: 1.1, fontSize: 13, color: C.text2 }));
  s.addNotes(L.s9.notes);

  // 10 robustness: frame-drop line chart left, corruption bar chart right
  s = content(L.s10.title);
  const cw2 = (W - 2 * M - 0.4) / 2;
  chart(s, pres.charts.LINE, L.s10.series.map((name, i) => ({ name, labels: DROP.k.map((k) => `${L.s10.k} = ${k}`), values: [DROP.bt, DROP.tuned, DROP.ours][i] })), {
    x: M, y: 1.45, w: cw2, h: 4.2, title: L.s10.t1, chartColors: [SER.bt, SER.tuned, SER.ours],
    lineSize: 2.5, lineDataSymbol: "circle", lineDataSymbolSize: 7, valAxisMinVal: 30, valAxisMaxVal: 50,
    showValue: false,
  });
  chart(s, pres.charts.BAR, L.s10.series.slice(1).map((name, i) => ({ name, labels: L.s10.corr, values: [CORR.tuned, CORR.ours][i] })), {
    x: M + cw2 + 0.4, y: 1.45, w: cw2, h: 4.2, barDir: "col", barGapWidthPct: 60, title: L.s10.t2, chartColors: [SER.tuned, SER.ours],
    showValue: true, dataLabelPosition: "outEnd", valAxisMinVal: 0, valAxisMaxVal: 200,
  });
  s.addText(L.s10.n1, TB({ x: M, y: 5.85, w: cw2, h: 0.8, fontSize: 14, color: C.text2 }));
  s.addText(L.s10.n2, TB({ x: M + cw2 + 0.4, y: 5.85, w: cw2, h: 0.8, fontSize: 14, color: C.text2 }));
  s.addNotes(L.s10.notes);

  // 11 density: per-sequence bar chart left, three findings right
  s = content(L.s11.title);
  chart(s, pres.charts.BAR, L.s11.series.map((name, i) => ({ name, labels: DENS.seq.map((q, j) => `${q} (${DENS.bpf[j]})`), values: [DENS.bt, DENS.tuned, DENS.ours, DENS.ours1280][i] })), {
    x: M, y: 1.45, w: 7.6, h: 5.2, barDir: "col", barGapWidthPct: 50, title: L.s11.chartTitle,
    chartColors: [SER.bt, SER.tuned, SER.ours, SER.ours1280], valAxisMinVal: 0, valAxisMaxVal: 80, catAxisLabelFontSize: 11,
  });
  L.s11.items.forEach(([h, b], i) => {
    const y = 1.6 + i * 1.65;
    num(s, i + 1, 8.6, y);
    s.addText(h, TB({ x: 9.2, y: y + 0.02, w: W - M - 9.2, h: 0.4, fontSize: 16, bold: true }));
    s.addText(b, TB({ x: 9.2, y: y + 0.45, w: W - M - 9.2, h: 1.05, fontSize: 14, color: C.text2 }));
  });
  s.addNotes(L.s11.notes);

  // 12 failure analysis: two figures with captions and explanations
  s = content(L.s12.title);
  const fw = 6.3, fh = fw * 470 / 950, iw2 = W - 2 * M - fw - 0.4, ih = iw2 * 240 / 646, ix = M + fw + 0.4;
  pic(s, "runs/figs/failure_fp_BEE24-34.png", M, 1.45, fw, fh, "BEE24-34 frame 391 with unmatched predictions and loose ground-truth boxes");
  s.addText(L.s12.c1, TB({ x: M, y: 1.55 + fh, w: fw, h: 0.35, fontSize: 14, bold: true }));
  s.addText(L.s12.b1, TB({ x: M, y: 1.95 + fh, w: fw, h: 1.0, fontSize: 14, color: C.text2 }));
  pic(s, "runs/figs/failure_idsw_BEE24-36.png", ix, 1.45, iw2, ih, "BEE24-36 frames 6 and 13: an ID switch inside the entrance cluster");
  s.addText(L.s12.c2, TB({ x: ix, y: 1.55 + ih, w: iw2, h: 0.35, fontSize: 14, bold: true }));
  s.addText(L.s12.b2, TB({ x: ix, y: 1.95 + ih, w: iw2, h: 1.0, fontSize: 14, color: C.text2 }));
  box(s, M, 6.0, W - 2 * M, 0.7, COL.honeySoft);
  s.addText(L.s12.foot, TB({ x: M + 0.25, y: 6.0, w: W - 2 * M - 0.5, h: 0.7, fontSize: 14, bold: true, valign: "middle" }));
  s.addNotes(L.s12.notes);

  // 13 demo: still left, count chart + tiles right, video note
  s = content(L.s13.title);
  const dw = 6.6, dh = dw * 548 / 950;
  pic(s, "runs/demo/BEE24-36_ours.jpg", M, 1.45, dw, dh, "Live demo frame on BEE24-36 with entrance zone, track IDs, in/out count and FPS");
  caption(s, L.s13.cap, M, 1.5 + dh, dw);
  const cx = M + dw + 0.4, cwd = W - M - cx, chh = cwd * 441 / 1029;
  pic(s, "runs/figs/BEE24-36_ours_counts.png", cx, 1.45, cwd, chh, "In and out crossings per 20 seconds on BEE24-36, ground truth versus ours");
  caption(s, L.s13.cap2, cx, 1.5 + chh, cwd);
  tiles(s, L.s13.tiles, cx, 4.2, (cwd - 0.3) / 2, 1.35);
  box(s, M, 5.95, W - 2 * M, 0.7, COL.honeySoft);
  s.addText(L.s13.video, TB({ x: M + 0.25, y: 5.95, w: W - 2 * M - 0.5, h: 0.7, fontSize: 16, bold: true, color: C.accent1, valign: "middle" }));
  s.addNotes(L.s13.notes);

  // 14 conclusions: four blocks left, track-length histogram right
  s = content(L.s14.title);
  L.s14.blocks.forEach(([h, b], i) => {
    const y = 1.45 + i * 1.32;
    box(s, M, y, 7.4, 1.17, i === 1 ? COL.honeySoft : COL.card);
    s.addText(h, TB({ x: M + 0.25, y: y + 0.1, w: 6.9, h: 0.35, fontSize: 16, bold: true, color: i === 1 ? C.accent1 : C.text1 }));
    s.addText(b, TB({ x: M + 0.25, y: y + 0.45, w: 6.9, h: 0.68, fontSize: 13, color: C.text2 }));
  });
  const lw = W - M - 8.4, lh = lw * 445 / 681;
  pic(s, "runs/figs/BEE24-36_ours_lengths.png", 8.4, 1.6, lw, lh, "Histogram of track lengths on BEE24-36, ground truth versus ours");
  caption(s, L.s14.cap, 8.4, 1.65 + lh, lw);
  s.addNotes(L.s14.notes);

  // 15 contributions: four cards in member order, thanks
  s = content(L.s15.title);
  const mw = (W - 2 * M - 0.9) / 4;
  L.s15.members.forEach(([role, task], i) => {
    const x = M + i * (mw + 0.3), [who, id] = TEAM[i];
    box(s, x, 1.6, mw, 3.3, C.background2);
    num(s, i + 1, x + 0.3, 1.8);
    s.addText(who, TB({ x: x + 0.3, y: 2.3, w: mw - 0.45, h: 0.45, fontSize: 18, bold: true, ...cjk(who) }));
    s.addText(id, TB({ x: x + 0.3, y: 2.75, w: mw - 0.6, h: 0.3, fontSize: 13, color: C.text2 }));
    s.addText(role, TB({ x: x + 0.3, y: 3.15, w: mw - 0.6, h: 0.4, fontSize: 16, bold: true, color: C.accent1 }));
    s.addText(task, TB({ x: x + 0.3, y: 3.6, w: mw - 0.6, h: 1.2, fontSize: 14, color: C.text2 }));
  });
  s.addText(L.s15.foot, TB({ x: M, y: 5.15, w: W - 2 * M, h: 0.5, fontSize: 14, color: C.text2 }));
  s.addText(L.s15.thanks, TB({ x: M, y: 5.8, w: W - 2 * M, h: 0.8, fontSize: 28, bold: true, color: C.accent1 }));
  s.addNotes(L.s15.notes);

  const out = path.join(__dirname, L.file);
  await pres.writeFile({ fileName: out });
  await applyTheme(out, THEME);
  console.log("wrote", out);
}

(async () => {
  for (const k of ["en", "zh"]) await build(T[k]);
})();
