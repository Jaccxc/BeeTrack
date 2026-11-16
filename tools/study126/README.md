# 固定 P0 下選框會怎樣改變計數

在 35-head,固定追蹤器收到全部真框時的 30 秒窗誤差為 7.86%。只保留真實穿越蜂抵達前後各 12 幀的原框,誤差降為 6.20%,事件 F1@0.5 從 0.938 升到 0.968。兩種輸入的追蹤器與評分 GT 完全相同。因此全部真框給出的誤差可以再低,實測的 6.20% 仍高於這一片的 5.05% 門檻。在實際偵測框上採同樣選擇時,35-head 與 33-head 的平均窗誤差都上升。選框需要整段真實身分,只用以研究誤差來源。[預登記](https://github.com/Jaccxc/BeeTrack/commit/c9ecd32a5632cd9420859c2718ffafad8dd38797) · [研究紀錄](../../docs/experiment_log.md)

| 輸入 | 35-head 原生 → 只留穿越附近框 | 33-head 原生 → 只留穿越附近框 |
|---|---:|---:|
| 真框,窗誤差 | 7.86% → 6.20% | 29.17% → 23.88% |
| 真實偵測,三種子平均窗誤差 | 17.76% → 25.86% | 23.27% → 75.36% |
| 真實偵測,三種子平均事件 F1@0.5 | 0.806 → 0.820 | 0.337 → 0.373 |

G 是全部真框,U 是全部實際偵測,M 是先在整幀一對一配對後與真框有 IoU 至少 0.5 的實際偵測。E 使用正式穿越蜂整段的框,T 在 E 內再限於抵達的前後 12 幀。C 使用沒有正式進出事件的蜂的全部框。K 每幀取與 T 相同的框數,依固定列號雜湊選取。每條件都以整片原始標註評分。兩段各 3950 幀,三次偵測器訓練重複使用同一段影片。[圖](../../docs/study126_selection.png)

## 從封存的事件重算

在本 repo 根目錄執行。第一條指令只用 Python 標準函式庫,會重新計算 40 格的 30 個窗起點、兩種 IoU 的事件最大配對和跨種子門檻。繪圖使用專案環境中的 matplotlib。[重播資料](../../docs/study126_replay.json.gz)

```powershell
uv run --offline --no-sync python tools/study126/replay126.py docs/study126_replay.json.gz
uv run --offline --no-sync python tools/study126/chart126.py docs/study126_replay.json.gz runs/study126_selection.png
```

檔案保留每筆進出時刻、合法配對邊、計數分子與產生資料的指紋。從頭驗證選框及逐幀的配對,還需要兩段開發影片的 GT、原生偵測快取、相機多邊形和固定連結器權重。

## 從既有偵測框重建追蹤

將 `$strictLinker` 設為本機保存的 m_dev_strict.json 路徑,並使用新輸出目錄。產生器會檢查真框來源、準備好的 GT 和列次序,實際偵測會先做與 §124 相同的全幀指派。P0 只接收座標、幀號與信心分數。

```powershell
$env:PYTHONPATH = (Get-Location).Path
$env:CUDA_VISIBLE_DEVICES = '-1'
$prereg = 'c9ecd32a5632cd9420859c2718ffafad8dd38797'
$outputDir = 'D:/beetrack_runs/study126_raw'
uv run --offline --no-sync python tools/study126/generate126.py --weights $strictLinker --output $outputDir
uv run --offline --no-sync python tools/study126/score126.py "$outputDir/manifest.json" --out D:/beetrack_runs/study126_scores --prereg $prereg
uv run --offline --no-sync python tools/study126/package126.py D:/beetrack_runs/study126_scores D:/beetrack_runs/study126_replay.json.gz
```
