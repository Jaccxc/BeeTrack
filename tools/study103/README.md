# §103 重跑方式

本實驗使用固定 640 的 P0、BEE24-33-head 和 BEE24-35-head，各 3,950 幀。程式從同一組進出事件重算 30 秒窗誤差與時間平移對照。[預登記](https://github.com/Jaccxc/BeeTrack/commit/e62e337760ea91526a23f5590804b9b991676734) · [結果與逐 seed 表](../../docs/experiment_log.md) · [事件資料](../../docs/study103_results.json)

## 用隨附資料核對

從 repo 根目錄執行。加 `--table` 可列出 60 組原始誤差、非零位移的中位數，以及位移誤差不高於原始值的比例。

```sh
uv run python tools/study103/replay.py --table
```

這一步只使用 NumPy 與隨附 JSON。每個有號整數代表一個事件，絕對值為從 1 開始的幀號，正數進、負數出。重播逐一計算 157 個非零位移，核對各分位數、排名和原始值，並枚舉每個 seed 所有可行的互斥交換。校正 ByteTrack 和預設 ByteTrack 共用時間資料，兩個方向的倍率另存為分數。F1 由已配對的 TP、FP、FN 核算。

## 從追蹤框重新產生結果

重新找出交換候選及做 IoU 事件配對，需要 §95、§99 的既有 MOT 輸出。在專案的 `runs` 下準備 dev GT、區域設定、`q_lk_loho_no{33,35}_s{0..8}`、`q_bt_loho_no{33,35}_s{0..8}` 及 §99 選定的六份 `p99_*` 輸出。校正係數還要用來源蜂箱的 ByteTrack 軌跡或快取的偵測框。每個方法載入的檔名可在 `swap_score.py`、`comparison_inputs.py` 核對。

以下 PowerShell 指令從 repo 根目錄執行。將輸出目錄設為可寫的位置。單獨重跑某一格時可直接指定蜂箱及 seed。

```powershell
$env:PYTHONPATH = (Get-Location).Path
$env:CUDA_VISIBLE_DEVICES = '-1'
$studyDir = 'D:/beetrack_runs/study103'
foreach ($hiveId in 33,35) {
    foreach ($seedId in 0..8) {
        uv run python tools/study103/run_study.py $hiveId $seedId "$studyDir/p0/${hiveId}_${seedId}.json"
        if ($LASTEXITCODE -ne 0) { throw 'P0 run failed' }
    }
}
uv run python tools/study103/comparison_inputs.py --output "$studyDir/comparisons.json"
uv run python tools/study103/pack_results.py $studyDir "$studyDir/rebuilt.json"
uv run python tools/study103/replay.py --input "$studyDir/rebuilt.json"
```

`comparison_inputs.py` 可加 `--calibration-cache <目錄>` 重用 §95 的來源軌跡。沒有快取時，它從已有的來源偵測框在記憶體重算 ByteTrack。兩個原始資料入口會核對 §95、§99 的已發表基準。重播資料中的彙整值保留五位小數，事件幀號與校正係數保留精確值。
