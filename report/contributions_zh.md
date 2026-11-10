# 個人貢獻說明（草稿，請各組員依實際情況修改）

> 姓名與學號以 {{NAMEi}} / {{IDi}} 佔位，由本機 proposal/team.json 填入，不進 git。

## {{NAME1}}（{{ID1}}）：資料與評估
- 下載並整理 BEE24，設計以序列為單位的 train / val / test 切分（val 取 BEE24-33、35 的最後 1000 幀並留 50 幀間隔，避免資料洩漏）。
- 撰寫 YOLO 標註轉換與評估用 GT（處理 GT 中同幀重複 ID 的問題）。
- 建立 TrackEval 評估流程（HOTA、MOTA、IDF1、IDSW），以及掉幀（k = 2、3、5）用的子取樣 GT。
- 負責 robustness 實驗的評估與表格整理。

## {{NAME2}}（{{ID2}}）：偵測與 Experiment 2
- 訓練 YOLO11-s 偵測器（640 與 1280 兩種輸入解析度），設定 early stopping 與等效 batch。
- 偵測結果快取，讓所有追蹤器用同一份偵測，確保比較公平；實作 oracle（GT 框）偵測器。
- Experiment 2：比較 640 與 1280 的偵測 mAP、追蹤指標、計數誤差與速度。
- 影像劣化實驗（偏暗、過曝、模糊、雜訊）的偵測端實作。

## {{NAME3}}（{{ID3}}）：追蹤、改進方法與 Experiment 1
- 以 ByteTrack 為基準，設計 BeeTracker：自適應 buffered IoU、遺失軌跡速度阻尼、Kalman 速度雜訊調整。
- 只在 val 上調參與做消融實驗；嘗試並記錄無效的方法（第二階段 BIoU、內插、短軌跡過濾、tracklet stitching 對區域計數的影響）。
- Experiment 1：ByteTrack、ByteTrack-tuned 與 BeeTracker 在 test 上的比較，以及 oracle 分析。

## {{NAME4}}（{{ID4}}）：計數、視覺化、demo 與報告
- 設計計數規則：中線（含容差帶）與巢口區域計數，並分析中線規則失敗的原因。
- 軌跡、熱度圖、進出計數、軌跡長度等分析圖，以及失敗案例圖。
- demo 程式（即時區域計數、FPS 顯示）與 demo 影片。
- 統整中英文期末報告與簡報。
