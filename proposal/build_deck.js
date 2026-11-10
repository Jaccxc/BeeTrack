// Build the proposal decks (English + Chinese), one .pptx each, speaker notes in the deck's language.
// run (from this folder): set NODE_PATH to a node_modules that has pptxgenjs, then
//   node build_deck.js <path to pptx skill apply_theme.js>
// inputs: img/*.jpg and stats.json written by make_figs.py
const pptxgen = require("pptxgenjs");
const fs = require("fs");
const path = require("path");
const { applyTheme } = require(process.argv[2]);

const S = JSON.parse(fs.readFileSync(path.join(__dirname, "stats.json"), "utf8"));
const IMG = (n) => path.join(__dirname, "img", n);
const fmt = (n) => n.toLocaleString("en-US");
// names stay in their original script in both decks; order = role order on the team slide
// team.json (not in git: personal data) = [["name", "student id"], ...]; placeholders if it is missing
const TEAM_FILE = path.join(__dirname, "team.json");
const TEAM = fs.existsSync(TEAM_FILE) ? JSON.parse(fs.readFileSync(TEAM_FILE, "utf8")) : [1, 2, 3, 4].map((i) => [`Member ${i}`, "000000000"]);
const [P1, P2, P3, P4] = TEAM.map(([n]) => n);
const cjk = (t) => (/[㐀-鿿]/.test(t) ? { fontFace: "Microsoft JhengHei" } : {});

// ---------- content ----------
const T = {
  en: {
    file: "DLIA_Proposal_BeeTrack_EN.pptx", lang: "en-US", font: "Calibri",
    footer: "DLIA Final Project Proposal · BeeTrack",
    s1: {
      title: "BeeTrack",
      tag: "Detecting, Tracking and Counting Honeybees at the Hive Entrance",
      sub: "Final Project Proposal · Project 1: Real-Time Multi-Object Detection and Tracking",
      meta: "Deep Learning for Image Analysis (M5261) · 12 Nov 2026",
      notes: "Good morning. Our team chose Project 1, real-time multi-object detection and tracking. Instead of pedestrians, we will track honeybees at the entrance of a beehive. The system is called BeeTrack. It detects every bee in a video, gives each bee a stable ID, draws its trajectory, and counts how many bees go in and out of the hive. In the next ten minutes we will cover the problem, the dataset, our baseline, the improvement we plan, how we will evaluate it, and who does what.",
    },
    s2: {
      title: "Motivation",
      cap: "BEE24 sequence {seq}, frame {frame} (raw video frame)",
      items: [
        ["Why it matters", "Entrance traffic, the number of bees leaving and returning, is a direct sign of colony activity and health. Counting it by hand is impractical."],
        ["Why it is hard", "Dozens of small bees per frame look almost identical, fly fast and change direction suddenly, and overlap each other. Trackers built for pedestrians swap IDs."],
        ["Input → output", "Input: hive-entrance video. Output: boxes, track IDs, trajectories and directions, in/out counts at a line, and processing FPS."],
      ],
      notes: "Why bees? Beekeepers and researchers watch the hive entrance because the number of bees leaving and returning tells you how active and healthy the colony is. Doing this by eye is impossible at scale. It is also a hard tracking problem. As you can see in this raw frame, the bees are small, they all look the same, they move fast and change direction suddenly, and they crowd together. Standard trackers assume objects move smoothly, like pedestrians, so they lose or swap IDs. Our input is a hive-entrance video; our output is boxes, IDs, trajectories, in-and-out counts, and the processing speed.",
    },
    s3: {
      title: "Project 1: Multi-Object Detection and Tracking",
      tiles: [
        ["Detection", "YOLO, fine-tuned on BEE24"],
        ["Tracking IDs", "ByteTrack-style tracker"],
        ["Identity over time", "Motion-adaptive association"],
        ["Occlusion", "Lost-track buffer, re-activation"],
        ["Trajectories", "Per-ID trails and heading"],
        ["Counting", "In/out line at the entrance"],
        ["Evaluation", "mAP · HOTA · IDF1 · MOTA · FPS"],
      ],
      ours: ["Our case", "Honeybees at the hive entrance, BEE24 dataset"],
      foot: "Section 7A minimum scope covered: ≥ 2 instances, ID-based metrics, counting, two tracking settings compared",
      notes: "We chose Project 1, real-time multi-object detection and tracking. Each tile is one required capability of Project 1, with the component of our system that delivers it. A YOLO detector handles detection, and a ByteTrack-style tracker assigns IDs. Keeping identities through fast motion and overlaps is where our improvement goes. Lost tracks are kept in a buffer and re-activated. We draw trajectories and directions, count bees crossing a line at the entrance, and evaluate both detection and tracking. Our case is honeybees at a hive entrance, using the BEE24 dataset, and the minimum scope in Section 7A is covered as well.",
    },
    s4: {
      title: "BEE24 Dataset",
      stats: [
        [fmt(S.videos), "videos"],
        [fmt(S.boxes), "annotated boxes"],
        [fmt(S.tracks), "bee identities"],
        [S.avgPerFrame.toFixed(1), "bees per frame (mean)"],
      ],
      items: [
        "Source: Cao et al., IEEE TIP 2025 (TOPICTrack); public download, 2.2 GB, no registration",
        "Annotations: box + track ID per bee per frame (MOT format)",
        "Official split: {trainSeq} training / {testSeq} test sequences, {frames} frames, {res}",
        "Validation: whole sequences held out from training; the test set is never used for tuning",
        "License: none stated on the dataset page; we use it for coursework only and cite the paper",
      ],
      cap: "Ground-truth boxes and IDs from the dataset (not model output)",
      notes: "Our dataset is BEE24, published with the TOPICTrack paper in IEEE Transactions on Image Processing in 2025. It has {videos} videos and about {boxesK} thousand annotated boxes. Every box carries a track ID, which is exactly what tracking metrics need. The authors already split it into {trainSeq} training and {testSeq} test sequences. We will hold out whole training sequences as validation, never single frames, so there is no leakage between neighbouring frames. The dataset page does not state a license, so we use it only for this course and cite the paper.",
    },
    s5: {
      title: "Dataset Annotations",
      head: ["What we measure", "Metric", "Annotation used"],
      rows: [
        ["Detection", "Precision, recall, mAP@0.5, mAP@0.5:0.95", "Boxes"],
        ["Tracking", "HOTA, IDF1, MOTA, ID switches", "Boxes + track IDs"],
        ["In/out counting", "Counting error, crossing precision/recall", "Derived from ground-truth trajectories"],
        ["Speed", "FPS, latency per frame", "None (measured)"],
      ],
      proto: "Counting protocol: a bee counts as one crossing when its ground-truth trajectory crosses the entrance line; direction gives in or out. The same rule is applied to predicted tracks.",
      cap: "Ground-truth trajectories with the counting line",
      notes: "The guidelines say annotations must support the metrics, so here is the mapping. Boxes give detection metrics. Track IDs give HOTA, IDF1, MOTA and ID switches. The dataset has no count labels, so we write a documented protocol: a ground-truth trajectory that crosses the entrance line counts as one crossing, and its direction tells in or out. We apply exactly the same rule to our predicted tracks and compare. Speed needs no labels; we measure it. Everything else that has no ground truth will be shown only as qualitative evidence and labelled that way.",
    },
    s6: {
      title: "System Architecture",
      cols: [
        ["Input", ["Hive-entrance video", "Resize 640 / 1280"]],
        ["Detection", ["YOLO11-s, fine-tuned", "Boxes + scores"]],
        ["Tracking", ["Kalman prediction", "Two-stage association", "Lost-track buffer"]],
        ["Analytics", ["Trajectory + heading", "Line-crossing counter"]],
        ["Output", ["Annotated video", "CSV · FPS log"]],
      ],
      improve: "our improvement",
      eval: "Evaluation (same held-out test set): mAP · HOTA · IDF1 · MOTA · ID switches · count error · FPS",
      notes: "This is the pipeline. A hive-entrance video is resized to 640 or 1280 pixels, which is our second experiment, and goes into a YOLO11 small detector fine-tuned on BEE24 that outputs boxes with confidence scores. The tracker then works in three steps: a Kalman filter predicts each bee's next position, a two-stage association matches high- and low-confidence detections to existing tracks, and a buffer keeps lost tracks so they can be re-activated. The highlighted tracking stage is where our improvement lives. The analytics stage turns tracks into trajectories with headings and counts line crossings. Outputs are an annotated video, a CSV of tracks and counts, and an FPS log, all scored on the same held-out test set.",
    },
    s7: {
      title: "Baseline System",
      cards: [
        ["Detector", "YOLO11-s, COCO-pretrained, fine-tuned on the BEE24 training sequences. COCO has no bee class, so fine-tuning is required."],
        ["Tracker", "ByteTrack with default settings: constant-velocity Kalman filter, IoU association, two-stage matching of high and low scores."],
        ["Setup", "RTX 4060 Laptop 8 GB · Python 3.11 · PyTorch · Ultralytics · TrackEval for metrics · fixed seeds, all settings logged."],
      ],
      ref: "Published reference on the BEE24 test set (TOPICTrack, authors' repository)",
      refs: [["46.6", "HOTA"], ["59.7", "IDF1"], ["66.7", "MOTA"]],
      notes: "Our baseline is a standard, reproducible pipeline. The detector is YOLO11 small, pretrained on COCO and fine-tuned on the BEE24 training sequences. A COCO model alone cannot be the baseline, because COCO has no bee class. The tracker is ByteTrack with its default settings. We train on one RTX 4060 laptop GPU, use TrackEval for the official metrics, fix the random seeds and log every setting. For context, the dataset authors report HOTA 46.6 and IDF1 59.7 on the test set with their own tracker, so this is a genuinely difficult benchmark.",
    },
    s8: {
      title: "Motion-Adaptive Association",
      prob: ["Observed problem", "Bees are small and fast. The constant-velocity prediction lands beside the next detection, IoU drops to zero, and the track is lost or swapped."],
      steps: [
        ["Buffered matching", "Match on slightly enlarged boxes and on size-normalised centre distance, so small fast bees still overlap their prediction."],
        ["Adaptive motion noise", "Raise the Kalman process noise for tracks that are turning or accelerating, so the search region follows erratic flight."],
        ["Occlusion recovery", "Keep lost tracks longer and re-activate them near their last position after overlaps."],
      ],
      goal: "Success criterion: higher IDF1 and HOTA and fewer ID switches than the baseline, on the same test set, with the same detector.",
      notes: "Our improvement targets the tracker, because that is where bees break standard methods. ByteTrack predicts that each object keeps moving at constant velocity and matches by box overlap. A bee is small and turns suddenly, so its predicted box often does not overlap its real box at all, and the track is lost or swapped with a neighbour. We plan three changes: match on enlarged boxes and size-normalised distance, increase the motion uncertainty for bees that are turning, and keep lost tracks alive longer. We count it as a success only if IDF1 and HOTA go up and ID switches go down on the same test set with the same detector.",
    },
    s9: {
      title: "Experiments and Robustness",
      exps: [
        ["Experiment 1 · Association", "Baseline ByteTrack vs. motion-adaptive association. Fixed: detector weights, input size, test set."],
        ["Experiment 2 · Detector input size", "640 vs. 1280 pixels. Fixed: tracker settings, training budget, test set."],
      ],
      robust: ["Robustness conditions (≥ 3)", "Crowd density (bees per frame)", "Fast motion (frame dropping)", "Lighting and shadow (scene groups)", "Motion blur (synthetic)"],
      fair: "Fairness: one principal factor changed at a time; tuning on validation sequences only; each experiment reports objective, settings, results table, interpretation and failure cases.",
      notes: "We will run at least two controlled comparisons, changing one factor at a time. Experiment one compares the baseline association with ours while keeping the detector identical. Experiment two compares two detector input sizes while keeping the tracker identical, because small bees may need higher resolution. For robustness we split the test results by crowd density, simulate fast motion by dropping frames, compare bright and shadowed scenes, and add synthetic motion blur. All tuning is done on validation sequences, and every experiment ends with failure cases, not just numbers.",
    },
    s10: {
      title: "Timeline and Risks",
      plan: [
        ["11/13–11/19", "Data pipeline, detector fine-tuning"],
        ["11/20–11/26", "ByteTrack baseline, full evaluation"],
        ["11/27–12/03", "Motion-adaptive association"],
        ["12/04–12/10", "Experiments 1–2, robustness"],
        ["12/11–12/17", "Demo application, report"],
        ["12/18–12/24", "Slides, demo video, rehearsal"],
      ],
      risks: [
        ["Small objects missed", "Higher input size, Experiment 2"],
        ["8 GB GPU memory", "Small model, batch tuning"],
        ["License not stated", "Course use only, cite the paper"],
        ["Exam week 12/21", "Report done by 12/17"],
      ],
      riskHead: "Risks and mitigation",
      notes: "Here is the plan from now until the final presentation on December 24. The first two weeks build a complete baseline with real numbers. Weeks three and four are the improvement and the controlled experiments. Week five is the demo application and the report, and the last week is slides, the demo video and rehearsal. Final exams start December 21, so we aim to finish the report by December 17. The main risks are missing very small bees, the 8 GB GPU, and the unstated license; each has a mitigation on the right.",
    },
    s11: {
      title: "Team Roles",
      members: [
        ["Member 1", "Data and evaluation", "Dataset preparation, validation split, counting protocol, TrackEval pipeline"],
        ["Member 2", "Detection model", "YOLO fine-tuning, Experiment 2, detection metrics"],
        ["Member 3", "Tracking and improvement", "ByteTrack baseline, motion-adaptive association, Experiment 1"],
        ["Member 4", "Application and report", "Counting and visualization, demo video, report integration"],
      ],
      foot: "Every member will be able to explain the full pipeline and code. AI assistance is used and will be disclosed in the report.",
      notes: `We split the work into four parts. ${P1} owns the data and the evaluation pipeline, including the counting protocol. ${P2} owns the detection model and experiment two. ${P3} owns the tracker and our improvement, which is experiment one. ${P4} owns the counting, visualization, demo application and report integration. We review each other's code weekly so that every member can explain the whole system. We use AI assistance and will disclose it, as the guidelines require.`,
    },
    s12: {
      title: "Expected Outcomes",
      items: [
        "Annotated video: boxes, IDs, trajectories, in/out counts, FPS",
        "Baseline vs. improved tracker on the same BEE24 test set",
        "Two controlled experiments and three robustness conditions",
        "Code, weights, report and demo video",
      ],
      cap: "Illustration drawn from ground-truth tracks; the final demo will show model predictions",
      thanks: "Thank you — questions?",
      notes: "To sum up, at the final presentation we will show a working system that turns a hive-entrance video into boxes, IDs, trajectories and in-and-out counts, together with the processing speed. We will compare our tracker with ByteTrack on the same test set, report two controlled experiments and three robustness conditions, and submit the code, weights, report and demo video. This picture is drawn from the ground-truth tracks to show the target output. Thank you, we are happy to take questions.",
    },
  },
  zh: {
    file: "DLIA_Proposal_BeeTrack_ZH.pptx", lang: "zh-TW", font: "Microsoft JhengHei",
    footer: "DLIA 期末專題提案 · BeeTrack",
    s1: {
      title: "BeeTrack",
      tag: "蜂巢出入口的蜜蜂偵測、追蹤與計數",
      sub: "期末專題提案 · Project 1：即時多目標偵測與追蹤",
      meta: "圖像分析的深度學習（M5261）· 2026 年 11 月 12 日",
      notes: "大家好。我們這組選的是 Project 1，即時多目標偵測與追蹤。我們追蹤的不是行人，而是蜂巢出入口的蜜蜂，系統叫做 BeeTrack。它會偵測影片裡每一隻蜜蜂、給每隻一個固定的 ID、畫出軌跡，並計算進出蜂巢的數量。接下來十分鐘會依序介紹問題、資料集、baseline、我們打算做的改進、評估方式，以及分工。",
    },
    s2: {
      title: "研究動機",
      cap: "BEE24 序列 {seq} 第 {frame} 格（原始影片畫面）",
      items: [
        ["為什麼重要", "出入口流量（出巢與回巢的蜜蜂數）直接反映蜂群的活動力與健康，人工計數不切實際。"],
        ["難在哪裡", "每格畫面有數十隻小蜜蜂、長得幾乎一樣、飛得快又突然轉向，還會互相重疊；為行人設計的追蹤器很容易換錯 ID。"],
        ["輸入 → 輸出", "輸入：蜂巢口影片。輸出：偵測框、追蹤 ID、軌跡與方向、穿越計數線的進出數量、處理 FPS。"],
      ],
      notes: "為什麼是蜜蜂？養蜂人和研究者會觀察蜂巢出入口，因為出巢和回巢的數量可以看出蜂群的活動力和健康狀況，但靠人眼數根本做不到。這同時也是很難的追蹤問題。大家看這張原始畫面：蜜蜂很小、長得都一樣、飛得快又會突然轉向，還常常擠在一起。一般的追蹤器假設物體像行人一樣平順移動，所以很容易跟丟或換錯 ID。我們的輸入是蜂巢口影片，輸出是偵測框、ID、軌跡、進出數量和處理速度。",
    },
    s3: {
      title: "Project 1：即時多目標偵測與追蹤",
      tiles: [
        ["偵測", "以 BEE24 微調的 YOLO"],
        ["追蹤 ID", "ByteTrack 式追蹤器"],
        ["跨影格維持 ID", "運動自適應關聯"],
        ["遮擋處理", "跟丟軌跡暫存與重新啟用"],
        ["軌跡視覺化", "每個 ID 的軌跡與方向"],
        ["計數", "蜂巢口進出計數線"],
        ["評估", "mAP · HOTA · IDF1 · MOTA · FPS"],
      ],
      ours: ["我們的題材", "蜂巢出入口的蜜蜂，BEE24 資料集"],
      foot: "符合第 7A 節最低門檻：兩個以上物件、以 ID 為基礎的指標、計數分析、比較兩種追蹤設定",
      notes: "我們選的是 Project 1，即時多目標偵測與追蹤。每一格是 Project 1 要求的一項能力，下面是我們系統裡負責它的元件。偵測用 YOLO，ID 由 ByteTrack 式的追蹤器指派。在快速移動和重疊時維持 ID，就是我們改進的重點。跟丟的軌跡會暫存並重新啟用。我們會畫出軌跡和方向、在蜂巢口設一條計數線，並同時評估偵測和追蹤。我們的題材是蜂巢出入口的蜜蜂，使用 BEE24 資料集，第 7A 節的最低門檻也都涵蓋。",
    },
    s4: {
      title: "BEE24 資料集",
      stats: [
        [fmt(S.videos), "支影片"],
        [fmt(S.boxes), "個標註框"],
        [fmt(S.tracks), "個蜜蜂 ID"],
        [S.avgPerFrame.toFixed(1), "每格平均蜜蜂數"],
      ],
      items: [
        "來源：Cao 等人，IEEE TIP 2025（TOPICTrack）；公開下載 2.2 GB，不需註冊",
        "標註：每格每隻蜜蜂都有框＋追蹤 ID（MOT 格式）",
        "官方切分：訓練 {trainSeq} 段／測試 {testSeq} 段，共 {frames} 格，{res}",
        "驗證集：從訓練集整段保留，測試集完全不拿來調參",
        "授權：資料集頁面沒有寫明；僅用於課程並引用論文",
      ],
      cap: "資料集內的標註框與 ID（不是模型輸出）",
      notes: "我們用的資料集是 BEE24，隨 TOPICTrack 論文發表在 2025 年的 IEEE Transactions on Image Processing。它有 {videos} 支影片、約 {boxesK} 千個標註框，而且每個框都有追蹤 ID，正好是追蹤指標需要的。作者已經切好訓練 {trainSeq} 段、測試 {testSeq} 段。驗證集我們會從訓練集整段保留，不是隨機抽單張，避免相鄰影格洩漏。資料集頁面沒有寫授權，所以我們只用於本課程並引用論文。",
    },
    s5: {
      title: "資料集標註",
      head: ["量測對象", "指標", "使用的標註"],
      rows: [
        ["偵測", "Precision、Recall、mAP@0.5、mAP@0.5:0.95", "偵測框"],
        ["追蹤", "HOTA、IDF1、MOTA、ID 換錯次數", "偵測框＋追蹤 ID"],
        ["進出計數", "計數誤差、穿越事件的 precision／recall", "由真值軌跡推得"],
        ["速度", "FPS、每格延遲", "不需標註（實測）"],
      ],
      proto: "計數規則：一條真值軌跡穿越蜂巢口計數線就算一次穿越，依方向判斷進或出；預測軌跡套用完全相同的規則。",
      cap: "真值軌跡與計數線",
      notes: "規範要求標註必須撐得起指標，所以這裡列出對應。偵測框可以算偵測指標，追蹤 ID 可以算 HOTA、IDF1、MOTA 和 ID 換錯次數。資料集沒有計數標註，所以我們寫一條明確的規則：真值軌跡穿越蜂巢口計數線就算一次，方向決定是進還是出。預測的軌跡套用完全一樣的規則再比較。速度不需要標註，直接量。其他沒有真值的部分只會當成質性證據，並且清楚標示。",
    },
    s6: {
      title: "系統架構",
      cols: [
        ["輸入", ["蜂巢口影片", "縮放 640／1280"]],
        ["偵測", ["YOLO11-s 微調", "框＋信心分數"]],
        ["追蹤", ["Kalman 預測", "兩階段關聯", "跟丟軌跡暫存"]],
        ["分析", ["軌跡＋方向", "穿越線計數"]],
        ["輸出", ["標註影片", "CSV · FPS 紀錄"]],
      ],
      improve: "我們的改進",
      eval: "評估（同一份保留測試集）：mAP · HOTA · IDF1 · MOTA · ID 換錯 · 計數誤差 · FPS",
      notes: "這是整個流程。蜂巢口影片先縮放成 640 或 1280 像素，這也是我們的實驗二，再進入以 BEE24 微調的 YOLO11 small 偵測器，輸出偵測框和信心分數。追蹤分三步：Kalman filter 預測每隻蜜蜂下一格的位置、兩階段關聯把高分和低分的偵測配對到既有軌跡、跟丟的軌跡先暫存，之後可以重新啟用。標成橘色的追蹤階段就是我們改進的地方。分析階段把軌跡轉成路徑和方向，並計算穿越計數線的次數。輸出是標註影片、軌跡與計數的 CSV，以及 FPS 紀錄，全部在同一份保留測試集上評分。",
    },
    s7: {
      title: "Baseline 系統",
      cards: [
        ["偵測器", "YOLO11-s，COCO 預訓練，再用 BEE24 訓練序列微調。COCO 沒有蜜蜂類別，所以一定要微調。"],
        ["追蹤器", "ByteTrack 預設設定：等速 Kalman filter、IoU 關聯、高低分數兩階段配對。"],
        ["環境", "RTX 4060 筆電 8 GB · Python 3.11 · PyTorch · Ultralytics · TrackEval 算指標 · 固定 seed、記錄全部設定。"],
      ],
      ref: "已發表的參考值（BEE24 測試集，TOPICTrack，取自作者 repo）",
      refs: [["46.6", "HOTA"], ["59.7", "IDF1"], ["66.7", "MOTA"]],
      notes: "我們的 baseline 是標準、可重現的流程。偵測器是 YOLO11 small，先用 COCO 預訓練，再用 BEE24 的訓練序列微調；單用 COCO 模型不能當 baseline，因為 COCO 根本沒有蜜蜂這個類別。追蹤器是預設設定的 ByteTrack。我們用一張 RTX 4060 筆電顯卡訓練、用 TrackEval 算官方指標、固定隨機種子並記錄所有設定。作為參考，資料集作者用他們自己的追蹤器在測試集上只有 HOTA 46.6、IDF1 59.7，可見這是很有挑戰性的資料集。",
    },
    s8: {
      title: "運動自適應關聯",
      prob: ["觀察到的問題", "蜜蜂又小又快，等速預測的框落在實際偵測旁邊，IoU 變成 0，軌跡就跟丟或換錯。"],
      steps: [
        ["緩衝配對", "用稍微放大的框，以及依大小正規化的中心距離來配對，讓又小又快的蜜蜂仍能和預測位置重疊。"],
        ["自適應運動雜訊", "對正在轉彎或加速的軌跡提高 Kalman 的 process noise，讓搜尋範圍跟得上不規則的飛行。"],
        ["遮擋恢復", "跟丟的軌跡保留更久，重疊結束後在最後位置附近重新啟用。"],
      ],
      goal: "成功標準：同一份測試集、同一個偵測器下，IDF1 與 HOTA 高於 baseline，ID 換錯次數更少。",
      notes: "我們的改進針對追蹤器，因為蜜蜂正是在這裡讓標準方法失效。ByteTrack 假設物體以等速移動，並用框的重疊來配對。蜜蜂很小又會突然轉向，預測的框常常和實際的框完全沒有重疊，於是軌跡跟丟或和旁邊的蜜蜂對調。我們打算做三件事：用放大的框和依大小正規化的距離來配對、對正在轉彎的蜜蜂提高運動的不確定性、讓跟丟的軌跡多保留一段時間。只有在同一份測試集、同一個偵測器下，IDF1 和 HOTA 提高、ID 換錯減少，才算成功。",
    },
    s9: {
      title: "對照實驗與穩健性",
      exps: [
        ["實驗 1 · 關聯方法", "Baseline ByteTrack 對比運動自適應關聯。固定：偵測器權重、輸入大小、測試集。"],
        ["實驗 2 · 偵測器輸入大小", "640 對比 1280 像素。固定：追蹤器設定、訓練預算、測試集。"],
      ],
      robust: ["穩健性條件（至少 3 種）", "擁擠程度（每格蜜蜂數）", "快速移動（跳格模擬）", "光線與陰影（場景分組）", "動態模糊（人工合成）"],
      fair: "公平性：一次只改一個主要因素；只在驗證序列上調參；每個實驗都寫出目的、控制條件、結果表、解讀與失敗案例。",
      notes: "我們至少做兩組對照實驗，一次只改一個因素。實驗一比較 baseline 的關聯方法和我們的方法，偵測器完全不變。實驗二比較兩種偵測器輸入大小，追蹤器完全不變，因為小蜜蜂可能需要更高的解析度。穩健性方面，我們會依擁擠程度分組看結果、用跳格模擬快速移動、比較明亮和有陰影的場景，並加上人工合成的動態模糊。所有調參都只用驗證序列，每個實驗最後都會附上失敗案例，而不是只有數字。",
    },
    s10: {
      title: "時程與風險",
      plan: [
        ["11/13–11/19", "資料流程、偵測器微調"],
        ["11/20–11/26", "ByteTrack baseline、完整評估"],
        ["11/27–12/03", "運動自適應關聯"],
        ["12/04–12/10", "實驗 1–2、穩健性分析"],
        ["12/11–12/17", "Demo 程式、報告"],
        ["12/18–12/24", "簡報、Demo 影片、排練"],
      ],
      risks: [
        ["小目標漏偵測", "提高輸入大小（實驗 2）"],
        ["GPU 只有 8 GB", "小模型、調整 batch"],
        ["授權未寫明", "僅課程使用、引用論文"],
        ["12/21 起期末考", "12/17 前完成報告"],
      ],
      riskHead: "風險與對策",
      notes: "這是從現在到 12 月 24 日期末發表的計畫。前兩週先做出有實際數字的完整 baseline，第三、四週做改進和對照實驗，第五週做 demo 程式和報告，最後一週準備簡報、demo 影片和排練。期末考從 12 月 21 日開始，所以報告目標在 12 月 17 日前完成。主要風險是很小的蜜蜂偵測不到、GPU 只有 8 GB，以及授權沒寫明，右邊列出各自的對策。",
    },
    s11: {
      title: "組員分工",
      members: [
        ["組員 1", "資料與評估", "資料集準備、驗證集切分、計數規則、TrackEval 評估流程"],
        ["組員 2", "偵測模型", "YOLO 微調、實驗 2、偵測指標"],
        ["組員 3", "追蹤與改進", "ByteTrack baseline、運動自適應關聯、實驗 1"],
        ["組員 4", "應用與報告", "計數與視覺化、Demo 影片、報告整合"],
      ],
      foot: "每位組員都能解釋完整流程與程式碼；我們有使用 AI 輔助，會在報告中揭露。",
      notes: `我們把工作分成四塊。${P1}負責資料和評估流程，包括計數規則。${P2} 負責偵測模型和實驗二。${P3}負責追蹤器和我們的改進，也就是實驗一。${P4}負責計數、視覺化、demo 程式和報告整合。我們每週互相看彼此的程式碼，確保每個人都能解釋整個系統。我們有使用 AI 輔助，會依規範在報告中揭露。`,
    },
    s12: {
      title: "預期成果",
      items: [
        "標註影片：偵測框、ID、軌跡、進出數量、FPS",
        "同一份 BEE24 測試集上比較 baseline 與改進的追蹤器",
        "兩組對照實驗與三種穩健性條件",
        "程式碼、模型權重、報告與 Demo 影片",
      ],
      cap: "示意圖以真值軌跡繪製；期末 Demo 會顯示模型的預測結果",
      thanks: "謝謝，歡迎提問",
      notes: "總結一下，期末發表時我們會展示一個能運作的系統：把蜂巢口影片變成偵測框、ID、軌跡和進出數量，並顯示處理速度。我們會在同一份測試集上比較我們的追蹤器和 ByteTrack，報告兩組對照實驗和三種穩健性條件，並繳交程式碼、權重、報告和 demo 影片。這張圖是用真值軌跡畫的，用來示意目標輸出。謝謝大家，歡迎提問。",
    },
  },
};

const fill = (s) => s.replace(/\{(\w+)\}/g, (_, k) => (k === "boxesK" ? Math.round(S.boxes / 1000) : k in S ? (typeof S[k] === "number" ? fmt(S[k]) : S[k]) : `{${k}}`));

// ---------- theme ----------
const COL = { ink: "1F1F1F", muted: "5C5C5C", card: "F4F4F2", line: "D9D9D6", honey: "B7791F", honeySoft: "FBF1DE" };

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
  const table = (s, head, rows, x, y, w, colW, fs = 14) => {
    const hdr = head.map((h) => ({ text: h, options: { bold: true, color: C.text1, fill: { color: C.background2 } } }));
    s.addTable([hdr, ...rows.map((r) => r.map((c) => ({ text: c })))], {
      x, y, w, colW, fontSize: fs, color: C.text1, lang: L.lang, valign: "middle",
      border: { type: "solid", pt: 0.75, color: COL.line }, margin: [0.06, 0.12, 0.06, 0.12], rowH: 0.42,
    });
  };

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
  pic(s, "hive_title.jpg", 7.0, 0, W - 7.0, 7.5, "Hive entrance from the BEE24 dataset");
  s.addNotes(L.s1.notes);

  // 2 motivation: photo left, three numbered points right
  s = content(L.s2.title);
  pic(s, "hive_raw.jpg", M, 1.55, 6.0, 6.0 * S.imgRatio, "Raw BEE24 frame of a hive entrance with many bees");
  caption(s, fill(L.s2.cap).replace("{seq}", S.rawSeq).replace("{frame}", S.rawFrame), M, 1.6 + 6.0 * S.imgRatio, 6.0);
  L.s2.items.forEach(([h, b], i) => {
    const y = 1.6 + i * 1.75;
    num(s, i + 1, 7.0, y);
    s.addText(h, TB({ x: 7.6, y: y + 0.02, w: 5.1, h: 0.4, fontSize: 18, bold: true }));
    s.addText(b, TB({ x: 7.6, y: y + 0.45, w: 5.1, h: 1.2, fontSize: 15 }));
  });
  s.addNotes(L.s2.notes);

  // 3 Project 1 capabilities: 4 x 2 tiles, the 8th is our case
  s = content(L.s3.title);
  const qw = (W - 2 * M - 0.9) / 4, qh = 1.75;
  [...L.s3.tiles, L.s3.ours].forEach(([h, b], i) => {
    const x = M + (i % 4) * (qw + 0.3), y = 1.65 + Math.floor(i / 4) * (qh + 0.3), ours = i === 7;
    s.addShape(pres.shapes.RECTANGLE, { x, y, w: qw, h: qh, fill: { color: ours ? COL.honeySoft : COL.card }, line: { color: ours ? COL.honeySoft : COL.card } });
    if (!ours) num(s, i + 1, x + 0.3, y + 0.2);
    s.addText(h, TB({ x: x + 0.3, y: y + 0.65, w: qw - 0.6, h: 0.42, fontSize: 18, bold: true, color: ours ? C.accent1 : C.text1 }));
    s.addText(b, TB({ x: x + 0.3, y: y + 1.07, w: qw - 0.6, h: 0.6, fontSize: 14, color: C.text2 }));
  });
  s.addText(L.s3.foot, TB({ x: M, y: 5.95, w: W - 2 * M, h: 0.4, fontSize: 13, color: C.text2 }));
  s.addNotes(L.s3.notes);

  // 4 dataset: stat callouts, bullets left, GT image right
  s = content(L.s4.title);
  L.s4.stats.forEach(([v, k], i) => {
    const x = M + i * 3.07;
    s.addShape(pres.shapes.RECTANGLE, { x, y: 1.45, w: 2.85, h: 1.3, fill: { color: C.background2 }, line: { color: C.background2 } });
    s.addText(v, TB({ x: x + 0.2, y: 1.55, w: 2.45, h: 0.7, fontSize: 32, bold: true, color: C.accent1 }));
    s.addText(k, TB({ x: x + 0.2, y: 2.25, w: 2.45, h: 0.4, fontSize: 13, color: C.text2 }));
  });
  s.addText(L.s4.items.map((t, i) => ({ text: fill(t), options: { bullet: true, breakLine: i < L.s4.items.length - 1 } })),
    TB({ x: M, y: 3.1, w: 6.6, h: 3.6, fontSize: 15, paraSpaceAfter: 8 }));
  const gw = 5.2; pic(s, "gt_boxes.jpg", W - M - gw, 3.1, gw, gw * S.imgRatio, "Ground-truth boxes and IDs drawn on a BEE24 frame");
  caption(s, L.s4.cap, W - M - gw, 3.15 + gw * S.imgRatio, gw);
  s.addNotes(fill(L.s4.notes));

  // 5 annotation suitability
  s = content(L.s5.title);
  table(s, L.s5.head, L.s5.rows, M, 1.5, 7.4, [1.7, 3.3, 2.4], 13);
  s.addShape(pres.shapes.RECTANGLE, { x: M, y: 4.0, w: 7.4, h: 1.2, fill: { color: C.background2 }, line: { color: C.background2 } });
  s.addText(L.s5.proto, TB({ x: M + 0.25, y: 4.15, w: 6.9, h: 1.2, fontSize: 14 }));
  const tw = 4.5; pic(s, "gt_traj.jpg", W - M - tw, 1.5, tw, tw * S.imgRatio, "Ground-truth trajectories and the counting line on a BEE24 frame");
  caption(s, L.s5.cap, W - M - tw, 1.55 + tw * S.imgRatio, tw);
  s.addNotes(L.s5.notes);

  // 6 architecture: five stage headers with arrows, sub-step boxes under each, tracking highlighted, evaluation bar
  s = content(L.s6.title);
  const bw = 2.05, gap = 0.47, hy = 2.0, hh = 0.75, sy = 3.0, sh = 0.6, sp = 0.75, ey = 5.65;
  L.s6.cols.forEach(([h, subs], i) => {
    const x = M + i * (bw + gap), hot = i === 2;
    s.addShape(pres.shapes.RECTANGLE, { x, y: hy, w: bw, h: hh, fill: { color: hot ? COL.honey : COL.ink }, line: { color: hot ? COL.honey : COL.ink } });
    s.addText(h, TB({ x, y: hy, w: bw, h: hh, fontSize: 17, bold: true, color: C.background1, align: "center", valign: "middle" }));
    subs.forEach((t, j) => {
      s.addShape(pres.shapes.RECTANGLE, { x, y: sy + j * sp, w: bw, h: sh, fill: { color: hot ? COL.honeySoft : COL.card }, line: { color: hot ? COL.honeySoft : COL.card } });
      s.addText(t, TB({ x: x + 0.1, y: sy + j * sp, w: bw - 0.2, h: sh, fontSize: 13, align: "center", valign: "middle" }));
    });
    if (i < 4) s.addShape(pres.shapes.LINE, { x: x + bw + 0.06, y: hy + hh / 2, w: gap - 0.12, h: 0, line: { color: COL.muted, width: 1.5, endArrowType: "triangle" } });
    if (hot) s.addText(L.s6.improve, TB({ x, y: hy - 0.45, w: bw, h: 0.35, fontSize: 13, bold: true, color: C.accent1, align: "center" }));
  });
  const ox = M + 4 * (bw + gap) + bw / 2, oy = sy + sp + sh;
  s.addShape(pres.shapes.LINE, { x: ox, y: oy + 0.08, w: 0, h: ey - oy - 0.16, line: { color: COL.muted, width: 1.5, endArrowType: "triangle" } });
  s.addShape(pres.shapes.RECTANGLE, { x: M, y: ey, w: W - 2 * M, h: 0.75, fill: { color: "FFFFFF" }, line: { color: COL.line, width: 1 } });
  s.addText(L.s6.eval, TB({ x: M + 0.3, y: ey, w: W - 2 * M - 0.6, h: 0.75, fontSize: 15, valign: "middle" }));
  s.addNotes(L.s6.notes);

  // 7 baseline: three cards + reference line
  s = content(L.s7.title);
  const cw = (W - 2 * M - 0.6) / 3;
  L.s7.cards.forEach(([h, b], i) => {
    const x = M + i * (cw + 0.3);
    s.addShape(pres.shapes.RECTANGLE, { x, y: 1.6, w: cw, h: 2.9, fill: { color: C.background2 }, line: { color: C.background2 } });
    num(s, i + 1, x + 0.3, 1.85);
    s.addText(h, TB({ x: x + 0.3, y: 2.4, w: cw - 0.6, h: 0.45, fontSize: 19, bold: true }));
    s.addText(b, TB({ x: x + 0.3, y: 2.95, w: cw - 0.6, h: 1.45, fontSize: 15 }));
  });
  s.addText(L.s7.ref, TB({ x: M, y: 4.95, w: W - 2 * M, h: 0.4, fontSize: 14, color: C.text2 }));
  L.s7.refs.forEach(([v, k], i) => {
    const x = M + i * (cw + 0.3);
    s.addText([{ text: v, options: { fontSize: 36, bold: true, color: C.text1 } }, { text: "  " + k, options: { fontSize: 16, color: C.text2 } }],
      TB({ x, y: 5.45, w: cw, h: 0.8, valign: "middle" }));
  });
  s.addNotes(L.s7.notes);

  // 8 improvement: problem box left, three steps right, goal bottom
  s = content(L.s8.title);
  s.addShape(pres.shapes.RECTANGLE, { x: M, y: 1.6, w: 4.3, h: 3.95, fill: { color: COL.honeySoft }, line: { color: COL.honeySoft } });
  s.addText(L.s8.prob[0], TB({ x: M + 0.3, y: 1.9, w: 3.7, h: 0.45, fontSize: 20, bold: true, color: C.accent1 }));
  s.addText(L.s8.prob[1], TB({ x: M + 0.3, y: 2.55, w: 3.7, h: 2.8, fontSize: 17 }));
  L.s8.steps.forEach(([h, b], i) => {
    const y = 1.6 + i * 1.4;
    num(s, i + 1, 5.3, y);
    s.addText(h, TB({ x: 5.9, y: y + 0.02, w: 6.8, h: 0.4, fontSize: 17, bold: true }));
    s.addText(b, TB({ x: 5.9, y: y + 0.45, w: 6.8, h: 0.85, fontSize: 14 }));
  });
  s.addText(L.s8.goal, TB({ x: M, y: 5.95, w: W - 2 * M, h: 0.7, fontSize: 15, bold: true }));
  s.addNotes(L.s8.notes);

  // 9 experiments: two experiment cards left, robustness list right, fairness bottom
  s = content(L.s9.title);
  L.s9.exps.forEach(([h, b], i) => {
    const y = 1.6 + i * 1.75;
    s.addShape(pres.shapes.RECTANGLE, { x: M, y, w: 7.3, h: 1.5, fill: { color: C.background2 }, line: { color: C.background2 } });
    s.addText(h, TB({ x: M + 0.3, y: y + 0.2, w: 6.7, h: 0.4, fontSize: 18, bold: true }));
    s.addText(b, TB({ x: M + 0.3, y: y + 0.65, w: 6.7, h: 0.75, fontSize: 14 }));
  });
  const rx = 8.3;
  s.addText(L.s9.robust[0], TB({ x: rx, y: 1.6, w: W - M - rx, h: 0.45, fontSize: 18, bold: true }));
  s.addText(L.s9.robust.slice(1).map((t, i, a) => ({ text: t, options: { bullet: true, breakLine: i < a.length - 1 } })),
    TB({ x: rx, y: 2.2, w: W - M - rx, h: 2.6, fontSize: 15, paraSpaceAfter: 10 }));
  s.addText(L.s9.fair, TB({ x: M, y: 5.3, w: W - 2 * M, h: 0.9, fontSize: 14, color: C.text2 }));
  s.addNotes(L.s9.notes);

  // 10 timeline (vertical list with dots) + risk table
  s = content(L.s10.title);
  L.s10.plan.forEach(([d, t], i) => {
    const y = 1.55 + i * 0.82;
    s.addShape(pres.shapes.OVAL, { x: M, y: y + 0.08, w: 0.24, h: 0.24, fill: { color: i === 5 ? C.accent1 : COL.line }, line: { color: i === 5 ? C.accent1 : COL.line } });
    if (i < 5) s.addShape(pres.shapes.LINE, { x: M + 0.12, y: y + 0.36, w: 0, h: 0.54, line: { color: COL.line, width: 1.5 } });
    s.addText(d, TB({ x: M + 0.45, y, w: 1.7, h: 0.4, fontSize: 15, bold: true }));
    s.addText(t, TB({ x: M + 2.2, y, w: 4.2, h: 0.4, fontSize: 15 }));
  });
  s.addText(L.s10.riskHead, TB({ x: 7.4, y: 1.5, w: 5.3, h: 0.45, fontSize: 18, bold: true }));
  s.addTable(L.s10.risks.map(([r, m]) => [{ text: r, options: { bold: true } }, { text: m }]), {
    x: 7.4, y: 2.1, w: W - M - 7.4, colW: [2.3, W - M - 7.4 - 2.3], fontSize: 14, color: C.text1, lang: L.lang, valign: "middle",
    border: { type: "solid", pt: 0.75, color: COL.line }, margin: [0.06, 0.12, 0.06, 0.12], rowH: 0.62,
  });
  s.addNotes(L.s10.notes);

  // 11 team: four cards
  s = content(L.s11.title);
  const mw = (W - 2 * M - 0.9) / 4;
  L.s11.members.forEach(([, role, task], i) => {
    const x = M + i * (mw + 0.3), [who, id] = TEAM[i];
    s.addShape(pres.shapes.RECTANGLE, { x, y: 1.6, w: mw, h: 3.3, fill: { color: C.background2 }, line: { color: C.background2 } });
    s.addText(who, TB({ x: x + 0.3, y: 1.9, w: mw - 0.45, h: 0.45, fontSize: 18, bold: true, ...cjk(who) }));
    s.addText(id, TB({ x: x + 0.3, y: 2.35, w: mw - 0.6, h: 0.3, fontSize: 13, color: C.text2 }));
    s.addText(role, TB({ x: x + 0.3, y: 2.8, w: mw - 0.6, h: 0.75, fontSize: 16, bold: true, color: C.accent1 }));
    s.addText(task, TB({ x: x + 0.3, y: 3.6, w: mw - 0.6, h: 1.2, fontSize: 14, color: C.text2 }));
  });
  s.addText(L.s11.foot, TB({ x: M, y: 5.3, w: W - 2 * M, h: 0.7, fontSize: 14, color: C.text2 }));
  s.addNotes(L.s11.notes);

  // 12 outcome + thanks: list left, illustration right
  s = content(L.s12.title);
  s.addText(L.s12.items.map((t, i, a) => ({ text: t, options: { bullet: true, breakLine: i < a.length - 1 } })),
    TB({ x: M, y: 1.6, w: 6.2, h: 3.2, fontSize: 17, paraSpaceAfter: 12 }));
  s.addText(L.s12.thanks, TB({ x: M, y: 5.3, w: 6.2, h: 0.8, fontSize: 28, bold: true, color: C.accent1 }));
  const iw = 5.6; pic(s, "demo_mock.jpg", W - M - iw, 1.6, iw, iw * S.imgRatio, "Illustration of the target output drawn from ground-truth tracks");
  caption(s, L.s12.cap, W - M - iw, 1.65 + iw * S.imgRatio, iw);
  s.addNotes(L.s12.notes);

  const out = path.join(__dirname, L.file);
  await pres.writeFile({ fileName: out });
  await applyTheme(out, THEME);
  console.log("wrote", out);
}

(async () => {
  for (const k of ["en", "zh"]) await build(T[k]);
})();
