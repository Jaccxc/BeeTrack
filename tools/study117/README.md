# §117 用兩幀前的特徵辨認當前弱框

一個蜂框的位置可能已經抓對,分類信心卻太低。§117 保留當前彩色影像、640 輸入及當幀的原生框回歸,僅讓高解析度 P3 分類層上部分弱格借用兩幀前的訊息。研究比較移動後對齊的特徵、同座標特徵與直接調整分類分數,並檢查改變是否傳到最終的進出計數。

兩個來源片段都是完整 1,000 幀。留 33 折用 no33 偵測器評 35-tail,留 35 折用 no35 評 33-tail。每折使用三個 seed(0–2),每個 seed 對六臂各跑完整 P0,合計 36 份追蹤結果。同一片的三個 seed 是同場景重播。[預登記](https://github.com/Jaccxc/BeeTrack/commit/064d5bf1786d0a9733ec7c99496e46679548fb3f)

35-tail 上,三個 seed 的平均 30 秒窗誤差 W 在原生 U 與位移對齊 A 均為 8.288%,官方事件 F1@0.5 均為 0.9255。33-tail 上,W 從 21.333% 升至 21.689%,F1@0.5 從 0.6642 升至 0.6701。兩片的 A 與零位移 Z 平均 W 均相等,全片效用及額外複雜度兩項門檻皆未通過。依預登記止於來源片段。

兩來源共有 33,386 個 A 合格受援格。下表各格依序列 seed 0、1、2。IoU 0.5 下,六次比較的 U 與 A 都看全同一批穿越,全片新增偵測 TP 也都少於新增 FP。以原框或裁切框判定供體歸屬,均為 0/3 個 seed 通過各自來源的機制門檻。

| 來源 | A 合格格數 | 機制母體 | U 與 A 均看全兩端的 GT 段數 | 全片逐幀 ΔTP/ΔFP |
|---|---:|---:|---|---|
| 33-tail | 28,130 | 34 段 | 23、25、22 | 38/260、45/168、39/162 |
| 35-tail | 5,256 | 21 段 | 14、13、14 | 4/12、1/6、7/15 |

## 用公開包重播

從已安裝專案鎖定環境的 repo 根目錄執行。逐片、逐 seed 核對正式 30 秒窗誤差 W、官方事件 F1 與機制對照。[§117 重播資料包](../../docs/study117_replay.json.gz)

重播包共 6,549,732 bytes,SHA-256 為 `e8e582051f44337aa5c7f25a093857ec3eb34a84ad55bd3963e1f0ad04f76b72`。可重算 36 份本次 P0、六份歷史 P0、12 份實體追蹤器對照與六份套用分向倍率的 ByteTrack 計數。

```powershell
$env:PYTHONPATH = (Get-Location).Path
$env:CUDA_VISIBLE_DEVICES = '-1'
uv run --offline --no-sync python tools/study117/release117.py replay docs/study117_replay.json.gz --table
```

加上 `--report public_report.json` 可輸出每臂的入出數、兩個 IoU 下的 TP/FP/FN/F1、逐 seed 的供體見證與來源門檻。輸出 SHA-256 為 `1706c9672bd9b623738580297053ffcf6bf0980dace1f5d9a491925c4c7ec759`。公開包保存有方向的事件幀、可配對的事件邊、供體機制所需的配對證據、階段成本及原始封存指紋。從相機像素重生 P3 特徵、偵測及十欄 MOT 仍需原始影片和權重,逐框重建 GT 幾何另需原始標註。

要查留 33 折 seed 0 的每一個供體,可匯出整段 1,000 幀的紀錄。當中有原框及裁切框各自前兩名 GT 的 IoU,以及逐幀全場 Hungarian 配對證據,保留原始數值精度。

```powershell
uv run --offline --no-sync python tools/study117/release117.py evidence docs/study117_replay.json.gz 33 0 donor_33_0.json
```

## 六臂如何分辨時間與位置的作用

| 臂 | P3 分類方式 |
|---|---|
| U | 直接用原單幀分類輸出。 |
| A | 從兩幀前搜尋外觀相符、互相選中且歷史信心高的格。將其特徵與當前弱格特徵各取一半,再跑 P3 分類塔。 |
| Z | 只在兩幀的同一座標檢查弱格、強格及外觀相似度,直接平均該處特徵。 |
| Q | 使用與 A 完全相同的受援格和供體格,平均兩個原生分類 logit,省下重新跑分類塔的成本。 |
| Zm | 使用 A 選出的同一批受援格,歷史特徵改從相同座標取得,用來隔離位置對齊的作用。 |
| D | 保留 A 的受援格與同幀全部供體特徵,依格子排序循環平移一位後再配對。只有一個供體時,D 與 A 相同。 |

A 的當前弱格原信心在 [0.05,0.25),歷史強格 ≥0.25。先在 5×5 鄰域比對已去通道平均、正規化的特徵。兩格必須互為唯一最近者,雙方第一名 cosine ≥0.8,且各比第二名高 ≥0.05。忽略 letterbox 填充與近零向量。各處理臂只在指定格上取更新及原生 logit 兩者較大值,其他分級及全部原生當幀回歸框保留。NMS 仍可能改變其他框是否留存。

歷史一律存原生特徵及原生分數。換片、換權重、幾何變動或幀不連續時清空歷史。每片首兩幀維持 U。六臂共用每個 seed 的當幀 backbone/neck,合計 6,000 次 640 CPU FP32 原生前向。固定每臂的 NMS 偵測流、格子與供體證據、36 份完整 MOT 後才配對 GT。原始 P3 張量、影像及逐幀原始 GT 留在資料端。

## 三項分開判讀

- **全片計數。** W 按 30 個整秒起點逐窗逐方向算絕對計數差,除以 30 倍全片 GT 事件數再乘 100。每來源平均 W(U)−W(A) 至少 2 個百分點,三個 seed 至少兩個 A 嚴格低於 U,且平均官方 F1@0.5(A) ≥ F1@0.5(U),才通過效用門檻。各臂另報 IoU 0.3 的事件 TP/FP/FN/F1。
- **額外分類成本。** A 的平均 W 必須同時嚴格低於 Z 和 Q,且平均 F1@0.5 對兩者都不下降。W 嚴格改善採差 >1e−9 個百分點,F1 不下降容許 1e−9。只有兩個來源都同時通過全片效用與額外成本兩項門檻,才另登記未見蜂箱驗證。Zm 與 D 是機制對照。
- **同蜂、異格的供體。** 機制分析扣掉 §116 已抽過的 GT 段。因此 33-tail 的分母為 50−16=34 段,35-tail 為 37−16=21 段。每段只看最後一次同方向穿越,兩端都須在全幀一對一配對中配到該 GT,且中心側別正確。在 IoU 0.5,每個 seed 須同時達成 A 的雙端可見數高於 U、Z、Q,至少一段由 U 不完全可見轉成 A 完全可見並有同蜂異格見證,以及全片逐幀偵測相對 U 的 ΔTP > max(0,ΔFP)。每個來源至少 2/3 seed 同時滿足以上三條才通過此門檻。

同蜂異格見證對應到 U 原失敗的那一端。A 的最終框須通過 detector NMS、分數 ≥0.25,且在全幀配對中配到該 GT。該框原列須正是 A 的受援格,其 t−2 供體位於不同格。主要歸屬使用未裁至感光畫面邊界的原生解碼框。兩格在各自時刻對同一切段 GT 均須 IoU ≥0.5,對其他任何 GT 都 <0.3。以裁切框重新判定歸屬另列為敏感度檢查。兩套歸屬皆用裁切後的 NMS 存活框進行全幀 Hungarian 指派,判定穿越兩端是否可見。全片逐幀 ΔTP/ΔFP 涵蓋所有 1,000 幀,不要求入口側別,與官方完整軌跡的事件 TP/FP 分別計算。

主要參照 U 來自這次同精度 CPU 推論。預設 ByteTrack 另用此 U 偵測快取。校正 ByteTrack 只產生 W,而此來源 tail 參與過倍率估計。§108 原生 P0 與 §99 已在來源選參數的 TOPICTrack 使用歷史偵測,用於說明舊基準、精度及成本差異。

## 從原影像重現六臂

在 repo 根目錄備妥 `data/BEE24/train/BEE24-{33,35}/img1` 的第 4001–5000 幀、`runs/detect/loho_no{33,35}_s{0,1,2}/weights/best.pt` 六份權重、P0 原版 `m_dev_strict.json` 和入口多邊形。歷史對照另需 §114 的 `source_inputs.json` 及其指向的舊偵測快取,以及 §99 六份已選 TOPICTrack MOT。`$strict`、`$prior` 分別設成連結器及 §114 清單的實際路徑。歷史清單核對各路徑的內容指紋。[§114 來源清單的重建方法](../study114/README.md)

第一段只取偵測與追蹤證據。每折依序完成三個 seed,保存當次模型指紋及每張原始影像的 SHA-256。兩折一起封存,再由 36 份快取各自跑完整 P0。預設 ByteTrack 使用相同 U 快取另跑六份。

```powershell
$pkg = 'tools/study117'
$run = Join-Path (Get-Location) 'runs/study117'
$reg = '064d5bf1786d0a9733ec7c99496e46679548fb3f'
$acq = "$pkg/acquisition"
$track = "$pkg/tracking/run117.py"
$ref = "$pkg/baselines117.py"
$env:PYTHONPATH = (Get-Location).Path
$env:CUDA_VISIBLE_DEVICES = '-1'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
function sha117($path) { (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant() }
foreach ($fold in 33,35) {
    uv run --offline --no-sync python "$acq/run117.py" run --repo . --output $run --prereg $reg --fold $fold --threads 2
}
uv run --offline --no-sync python "$acq/run117.py" seal --output $run --prereg $reg
$det = sha117 "$run/detector_done.json"
uv run --offline --no-sync python $track track $run --prereg $reg --detector-index-sha $det --weights $strict --finalize
$mot = sha117 "$run/tracking/index.json"
uv run --offline --no-sync python $ref --root $run history --prior $prior
$hist = sha117 "$run/baselines/history.json"
uv run --offline --no-sync python $ref --root $run track --history-sha $hist --detector-sha $det --prereg $reg
$bt = sha117 "$run/baselines/index.json"
```

確認兩個來源、三個 seed、六臂的 MOT 全部封存後,才讀 `runs/gt/val/BEE24-{33,35}-tail/gt/gt.txt`。以下從十欄 MOT 算 W 與官方 F1,由逐幀證據辨認供體,再計算其他追蹤器成績。打包使用三份成績的 SHA-256,並重播全部判準。重新推論時應比對原圖、權重、偵測列及成績。含執行時間的收據也會記錄該次量測。

```powershell
uv run --offline --no-sync python $track score $run --prereg $reg --tracking-index-sha $mot --history docs/study108_source.json.gz
uv run --offline --no-sync python "$pkg/donors.py" --run $run --repo . --store-dir $acq --detector-sha $det --tracking-sha $mot --prereg $reg
uv run --offline --no-sync python $ref --root $run score --baseline-sha $bt --primary-sha $mot --prereg $reg
$score = sha117 "$run/scores/index.json"
$mechanism = sha117 "$run/mechanism/index.json"
$baseline = sha117 "$run/baselines/replay.json.gz"
uv run --offline --no-sync python "$pkg/release117.py" pack $run "$run/study117_replay.json.gz" --history docs/study108_source.json.gz --score-sha $score --mechanism-sha $mechanism --baseline-sha $baseline
```

## 單獨執行 A

`deploy117.py detect` 接受影片或一個圖片資料夾,只用一份權重執行 A。以下以 35 號蜂箱的尾段及 no33 seed 0 為例,`--first` 與 `--last` 選取起訖原始幀,`--weights-sha256` 可另行核對權重。輸出為可直接供 P0 使用的六欄 float64 陣列,依序是片內幀號、原圖 x1/y1/x2/y2 和信心。若用 `--names` 選取不連續的影格,片內幀號保留間隔,缺幀後清空歷史。

```powershell
$env:PYTHONPATH = (Get-Location).Path
uv run --offline --no-sync python tools/study117/acquisition/deploy117.py detect data/BEE24/train/BEE24-35/img1 A.npy --weights runs/detect/loho_no33_s0/weights/best.pt --first 4001 --last 5000 --stats A_stats.json
uv run --offline --no-sync python tools/study117/acquisition/deploy117.py self-check
```
