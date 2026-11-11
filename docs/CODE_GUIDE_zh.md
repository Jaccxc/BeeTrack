# BeeTrack 程式導讀

老師可能請任何一位組員解釋任何一段程式。這份導讀說明每個檔案在做什麼、為什麼這樣設計、怎麼執行，最後附上可能被問的問題與簡短回答。建議對照原始碼讀一遍，每個模組都不到 160 行。

## 1. 整體流程

```
data/BEE24（MOT 格式 gt.txt + jpg 影格）
   │ prep.py ──► runs/data/{train,val,test}.txt、bee24.yaml、split.json        （偵測器影像清單）
   │         └─► runs/gt/{val,test,test_step2,3,5}/<seq>/gt/gt.txt + seqinfo.ini   （TrackEval 真值）
   ▼
train.py ──► runs/detect/y11s_<imgsz>/weights/best.pt
   ▼
detect.py ──► runs/dets/<tag>/<split>/<seq>.npy（frame, x1, y1, x2, y2, score）＋ <seq>.json 計時
   ▼
track.py ──► runs/trackers/<split>/<name>/data/<seq>.txt（MOT 格式）＋ timing.json
   ▼
evaluate.py ──► HOTA／IDF1／MOTA／IDSW／計數誤差，runs/eval/<split>/<name>.json
sweep.py    ──► 在 val 上一次跑多組追蹤設定，runs/eval/val/sweep_<grid>_<dets>.json
analytics.py ─► runs/figs/<seq>_<tracker>_{trajectories,heatmap,counts,lengths}.png
demo.py     ──► runs/demo/<name>_<tracker>.mp4 + .csv（線上：逐格偵測＋追蹤）
```

所有指令都在 repo 根目錄以 `uv run python -m beetrack.<模組> ...` 執行。環境只要先 `uv sync`（依 `uv.lock` 建立），再用 `uv run python tools/fetch_bee24.py` 下載資料。

## 2. 各模組說明

### `common.py`：路徑、資料切分、MOT 讀寫
- `ROOT`、`DATA`、`RUNS`：路徑都從套件所在位置推出，clone 到哪裡都能跑。
- `split_specs(split)` 是**資料切分唯一的定義**，回傳 `(評估名稱, 來源序列, 起始格, 結束格)`：
  - `test`：官方 5 支測試影片，整段。
  - `val`：`VAL_WHOLE` = BEE24-10/15/24/28，加上每支長訓練影片（≥ `LONG` = 1000 格，即 BEE24-33、-35）的最後 `VAL_TAIL` = 1000 格，命名為 `<seq>-tail`。
  - `train`：其餘全部；長影片在驗證尾段前 `GAP` = 50 格就停止。
- `gt_slice` 把真值切到指定影格範圍並從 1 重新編號（驗證尾段在 TrackEval 中從第 1 格開始）。
- `write_mot` 輸出 MOTChallenge 格式 `frame,id,x,y,w,h,score,-1,-1,-1`。

**為什麼這樣切**：測試集絕對不能碰（規範 7B）。提案原本打算用整段序列當驗證集，但短的訓練影片很稀疏，而測試的 34、36 是 5,000 格的擁擠長影片；在稀疏資料上調參會選到錯的設定。加入長影片尾段就有長而擁擠的片段；50 格間隔避免幾乎相同的相鄰影格跨界洩漏。

### `prep.py`：標籤、影像清單、評估真值
- `yolo_labels`：每張影格旁邊放一個 `.txt`（`0 cx cy w h`，正規化），框先裁切到影像內，小於 1 像素的丟掉；沒有蜜蜂的影格給空白檔，當作負樣本。
- 影像清單：短影片每格都用；長影片訓練每 5 格取 1（`STRIDE_LONG`），驗證／測試每 10 格取 1（`STRIDE_VAL_LONG`），因為 25 fps 的相鄰影格幾乎一樣。結果是 3,740／840／1,762 張。
- `write_eval_gt`：產生 TrackEval 資料夾。**重複 ID 修正**：BEE24-14/15/33/35 有 993 列在同一格把同一個 ID 給了兩隻蜜蜂，TrackEval 不接受，所以多出來的那一筆保留框、改給新 ID。測試序列沒有這種情形。`step=k` 時另外寫出跳格真值（`test_step2/3/5`），保留第 1、1+k、… 格並重新編號，規則和 `track.run_sequence` 一樣。
- 最後的迴圈會 **assert 沒有任何一格同時屬於兩個切分**。
- 執行：`uv run python -m beetrack.prep`

### `train.py`：偵測器微調
- YOLO11-s 的 COCO 權重 → 單一類別「bee」。固定預算 50 epoch、patience 15、seed 0、`deterministic=True`；640 時 batch 16、1280 時 batch 4（8 GB GPU）；mosaic，最後 10 個 epoch 關閉（`close_mosaic=10`）；只做水平翻轉（蜂巢口的上下方向有意義）。
- **兩種輸入大小用同樣的預算**，實驗二才公平。
- 執行：`uv run python -m beetrack.train --imgsz 640`（以及 `--imgsz 1280`）。
- 目前結果：640 px 在第 22 個 epoch 提前停止（0.32 小時），驗證 mAP50 0.946、mAP50-95 0.513。

### `detect.py`：偵測快取、影像劣化、oracle
- 對 val 或 test 的每一格**只跑一次**偵測器，把 `frame, x1, y1, x2, y2, score` 存成 `.npy`。之後每種追蹤器都讀同一個檔案，所以實驗一只改變追蹤器（沒有 FP16 或每次執行的偵測差異）。
- `MIN_SCORE = 0.05`：保留低分框，因為 ByteTrack 第二階段會用分數 0.1 到 0.25 之間的框。
- `CORRUPT`：在偵測前套用的穩健性條件：`dark`（×0.4）、`bright`（×1.6 + 20）、`blur`（9 像素水平模糊核）、`noise`（高斯 σ = 15，seed 0）。一次一種、只用於測試集、不重新訓練。
- `--weights gt`：**oracle** 偵測器，直接把真值框當偵測（分數 1），用來單獨看關聯錯誤：預設 ByteTrack 在驗證 oracle 上只有 HOTA 71.3／IDF1 76.0／IDSW 431。
- 計時：每次呼叫前後 `torch.cuda.synchronize()`；前 10 格當暖機不計；時間包含影像解碼。
- 執行：`uv run python -m beetrack.detect --weights runs/detect/y11s_640/weights/best.pt --imgsz 640 --split test [--corrupt dark]`；oracle：`--weights gt --split val`。

### `track.py`：baseline 與本文追蹤器
- `BYTETRACK`：Ultralytics 預設 `bytetrack.yaml` 的值（baseline）。
- `OURS`：同上，再加 `track_buffer=60, buffer=0.3, grow=0.1, speed=0.5, damp=0.9, q_vel=1/80`（預設值，最終值由驗證集掃描決定）。
- `Dets`：小小的轉接類別，讓 NumPy 陣列看起來像 `BYTETracker.update` 需要的 Ultralytics `Results.boxes`。
- `BeeTracker(BYTETracker)` 覆寫三個方法：
  - `get_dists` → **緩衝 IoU**。軌跡的緩衝 `r = buffer + grow·min(跟丟格數, 10) + speed·min(|v|/h, 1)`，偵測框的緩衝是 `buffer`。兩個框四邊各放大 r × 寬／高後算 IoU，再像 ByteTrack 一樣與偵測分數融合（`fuse_score`）。用在第一階段（高分框）與未確認軌跡；第二階段（低分框）維持一般 IoU。
  - `multi_predict` → **速度衰減**：跟丟軌跡在預測前速度乘上 `damp`；同時設定共用 Kalman filter 的速度雜訊。
  - `get_kalmanfilter` → **process noise** `q_vel`（ByteTrack 為 1/160）。
- `run_sequence(dets, n, tracker, step)`：把第 1、1+step、… 格餵給追蹤器，輸出影格重新編號，只計追蹤器本身的時間。
- 執行：`uv run python -m beetrack.track --dets y11s_640 --split test --tracker ours`（或 `bytetrack`；用 `--set buffer=0.2 track_buffer=90 --name x` 覆寫參數；`--step 3` 做跳格，輸出到 `test_step3`）。

**為什麼用緩衝 IoU（要能講出的理由）**：蜜蜂框約 46 像素。蜜蜂一格移動超過自己的寬度，預測框和偵測框就不重疊，IoU = 0，任何門檻都配不上，軌跡跟丟後換新 ID。把框放大就能恢復重疊。讓放大幅度**自適應**，是為了讓蜂巢口擁擠處的慢速蜜蜂維持小緩衝（避免和鄰居配錯），只有快速或跟丟中的軌跡才放大，因為它們的位置不確定性真的比較大。

### `link.py`：TrackletLink（兩階段追蹤，為進出計數設計）
- 用法：`track.py --tracker link`。第一階段是 `track.py` 的 `PASS1`：只接幾乎完全重疊的框（緩衝 IoU ≥ 0.7、不融合分數、跟丟 2 格就結束），每個框都輸出，之前先做第二次 NMS（IoU 0.5，去掉同一隻蜂上的重複框）。結果是很碎、但身分很純的 tracklet。
- 第二階段 `link(rows, thr)`：`candidates` 列出「tracklet j 在 i 結束後 25 格內開始、而且夠近」的候選連結；`features` 算 25 個成對幾何特徵（間隔、距離、兩端速度外插的落差、框形變化、共線支持等，長度以身長為單位）；`model` 讀 `link_model.json`（2 × 32 的小型 MLP，推論只用 NumPy）給每條候選一個機率；由高到低貪婪連結，每個尾、每個頭只用一次，低於 `thr`（0.15）不接。最後丟掉少於 3 個框的軌跡。
- 擬合：`labelled` 用真值決定每條候選的標籤（同一隻蜂的下一段 = 1），`fit` 用 torch 訓練後存成 JSON。執行：`uv run python -m beetrack.link --dets y11s_640 --fit train val`（需要 train、val 的偵測快取與 `prep.py` 寫出的真值）。
- 檔案載入時會跑 `_check()`：兩隻合成蜂（一隻跳兩個身長、一隻三點一線飛過）驗證連結邏輯與共線特徵。

**為什麼分兩階段（要能講出的理由）**：逐格決策的追蹤器遇到蜜蜂跳動或起飛只能二選一：放掉牠（漏計），或把 ID 留給鄰居（誤計）。先只接確定的，再看完整段序列決定哪個尾接哪個頭，可用的資訊比較多。所有實驗（含負面結果）在 `docs/experiment_log.md`。

### `evaluate.py`：指標
- `trackeval_metrics`：TrackEval `MotChallenge2DBox`，計算 HOTA、CLEAR、Identity。重要設定：`DO_PREPROC=False`（MOT17 的前處理會移除干擾類別；我們的真值只有蜜蜂，全部有效）、`SKIP_SPLIT_FOL=True`、自己的 seqmap。類別名稱「pedestrian」只是 TrackEval 對類別 1 的稱呼。檔案開頭那一小段補回 `np.float` 等別名，因為 TrackEval 還在用、NumPy ≥ 1.24 已移除。
- `crossings(rows, line_y)`：**計數規則**。每條軌跡依影格排序，取（中心 y − 線）的正負號，0 去掉；每次變號算一次穿越；往上（+ → −，y 變小）算進，往下算出。真值與預測用的是同一個函式。
- `count_metrics`：線放在畫面一半高度（由 `seqinfo.ini` 的 `imHeight/2`）；每段序列誤差 `|預測進 − 真值進| + |預測出 − 真值出|`。
- 執行：`uv run python -m beetrack.evaluate --split test --trackers bytetrack ours`（也可 `--split val`、`--split test_step2`）。

### `sweep.py`：調參與消融（只用驗證集）
- `GRIDS`：`ablation`（每列把本文的一個元件改回 ByteTrack 的值）、`buffer`（24 組）、`motion`（27 組）、`thresh`（36 組）。
- 在快取的驗證偵測上跑每組設定、評估、依 HOTA 排序印出表格、存 JSON，最後刪掉暫存的追蹤輸出。
- 執行：`uv run python -m beetrack.sweep --dets y11s_640 --grid ablation`（`--base buffer=0.2` 會套用到每一列 ours）。

### `analytics.py`：圖表
- 針對一段序列：前 60 秒的軌跡、佔據熱度圖、每 20 秒的進出次數（真值對預測，用同一條穿越規則）、軌跡長度直方圖（斷裂＝很多短軌跡）、速度中位數。
- 執行：`uv run python -m beetrack.analytics --split test --tracker ours --seq BEE24-36`

### `demo.py`：可運作的應用程式
- 線上路徑：讀影片或影格資料夾，逐格偵測（`conf=0.05`、FP16）與追蹤，畫出框、ID、一秒尾跡、計數線，以及顯示進／出、蜜蜂數、FPS（偵測＋追蹤的 30 格滑動平均）的狀態列，輸出 `.mp4` 與 `.csv`。
- 即時計數器使用和 `evaluate.crossings` 相同的變號規則，對每個 ID 逐格累計。
- 執行：`uv run python -m beetrack.demo --source data/BEE24/test/BEE24-36/img1 --tracker ours [--line 0.5 --max-frames 500]`

## 3. 完整重現順序
```
uv sync
uv run python tools/fetch_bee24.py
uv run python -m beetrack.prep
uv run python -m beetrack.train --imgsz 640
uv run python -m beetrack.detect --weights runs/detect/y11s_640/weights/best.pt --imgsz 640 --split val
uv run python -m beetrack.sweep --dets y11s_640 --grid buffer      # 以及 motion、thresh、ablation
uv run python -m beetrack.detect --weights runs/detect/y11s_640/weights/best.pt --imgsz 640 --split test
uv run python -m beetrack.track --dets y11s_640 --split test --tracker bytetrack
uv run python -m beetrack.track --dets y11s_640 --split test --tracker ours
uv run python -m beetrack.evaluate --split test --trackers bytetrack ours
uv run python -m beetrack.analytics --tracker ours --seq BEE24-36
uv run python -m beetrack.demo --source data/BEE24/test/BEE24-36/img1
```
穩健性：先 `detect.py --corrupt <條件>`，再 `track.py --dets y11s_640_<條件>`；跳格：`track.py --step k`，再 `evaluate.py --split test_step<k>`。
報告 PDF：`powershell -ExecutionPolicy Bypass -File report\build.ps1`。

## 4. 老師可能會問的問題

**問：一句話說明你們的改進？**
運動自適應關聯：ByteTrack 的 IoU 改在放大後的框上計算，放大幅度隨軌跡速度與跟丟格數增加；另外跟丟軌跡的速度會衰減、Kalman 速度雜訊調高、跟丟軌跡保留更久。

**問：怎麼知道問題在追蹤器而不是偵測器？**
Oracle 實驗：把真值框當偵測，預設 ByteTrack 在驗證集仍只有 HOTA 71.3、IDF1 76.0、431 次 ID 換錯，這些錯誤全部來自關聯。

**問：比較公平嗎？**
公平。兩種追蹤器讀同一份偵測快取（`runs/dets/...npy`），門檻、測試集、評估程式都一樣；所有調參只用驗證集，測試集每個最終設定只評估一次。

**問：有沒有在測試集上調參？**
沒有。`sweep.py` 預設 `--split val`，參數在跑測試集之前就固定。

**問：為什麼不用外觀特徵（DeepSORT、BoT-SORT 的 ReID）？**
蜜蜂幾乎長得一樣、框只有約 46 像素，ReID 特徵很難區分；我們觀察到的失敗是幾何上的（框不重疊），用運動面的修正直接處理，而且仍能即時。

**問：為什麼不直接用一個固定的大緩衝？**
蜂巢口很擠，大緩衝會讓相鄰的慢速蜜蜂重疊而互換 ID。自適應緩衝只在不確定性大的地方（快速或跟丟的軌跡）放大。消融表的「− 自適應」那一列就是在測這件事。

**問：為什麼第二階段（低分框）還是用一般 IoU？**
低分框比較常是誤偵測，只有真的重疊才接受，避免軌跡黏到雜訊上。

**問：HOTA、IDF1、MOTA 各量什麼？**
MOTA = 1 −（FN + FP + IDSW）/ 真值數，主要受偵測影響。IDF1 是 ID 正確配對的 F1，衡量 ID 維持正確多久。HOTA = √(DetA·AssA)，在 IoU 門檻 0.05–0.95 上取平均，同時平衡偵測與關聯。

**問：TrackEval 為什麼設 `DO_PREPROC=False`？**
MOT17 的前處理會移除和干擾類別（例如靜止的人）配對的框；我們的真值只有一個類別且每個框都有效，沒有東西需要移除。

**問：沒有計數標註，計數怎麼定義？**
在畫面一半高度放一條線，軌跡的框中心換邊就算一次穿越（往上＝進、往下＝出），真值與預測用完全相同的規則；每段序列的誤差是 |進的差| + |出的差|。測試集真值共進 398、出 355，合計 753 次。

**問：驗證集為什麼不像提案寫的只用整段序列？**
短的訓練影片很稀疏，測試的 34、36 又長又擠。我們加入兩支長訓練影片的最後 1000 格，中間隔 50 格，讓調參看得到長而擁擠的片段，又不會洩漏。

**問：重複 ID 是什麼問題？**
BEE24-14/15/33/35 有 993 列在同一格把同一 ID 給兩隻蜜蜂，TrackEval 不接受。評估真值中多出的那一筆改給新 ID；測試集沒有這個問題，所以測試結果不受影響。

**問：為什麼要微調，不直接用 COCO 的 YOLO？**
COCO 沒有蜜蜂類別。

**問：實驗二為什麼比 640 和 1280？batch 不同算不算干擾因素？**
蜜蜂很小，高解析度可能幫助偵測，但計算量是四倍。1280 時 batch 必須降到 4 才放得進 8 GB，這是所研究因素不可避免的結果，我們在報告中寫明。

**問：FPS 怎麼量？**
CUDA 同步後量實際時間；偵測器每格時間（含影像解碼，排除前 10 格暖機）加上追蹤器每格時間；demo 顯示偵測＋追蹤的 30 格滑動 FPS。

**問：AI 做了什麼？**
Claude（Anthropic）協助撰寫程式、實驗與文字，已在報告中揭露。每位組員都檢查過程式並能解釋；所有數字都來自我們自己執行的結果。
