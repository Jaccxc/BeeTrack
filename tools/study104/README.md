# §104　倒播追蹤與互補事件

在同一批固定 640 的偵測框上，F 是完整正播，R 是完整倒播，RF 用倒播產生的片段還原到原幀號，再交給正播連結器。三臂都回到原時間軸才算事件。[預登記](https://github.com/Jaccxc/BeeTrack/commit/c65d5e9396f11d61c525d56ee1726b9e5c59d21d)

## 用事件資料重算全表

從 repo 根目錄執行下列 PowerShell 指令。封存資料包含三臂的事件時間、穿越後的框，以及官方兩個 IoU 門檻下可以配對的邊。程式會重算配對、J、聯集與窗誤差，並核對每一格。從影像重新找出候選邊，則需要下節的軌跡檔。

```powershell
$env:PYTHONPATH = (Get-Location).Path
uv run python tools/study104/package_replay.py replay docs/study104_results.json --table
```

## 從原本的偵測框重跑一格

從專案根目錄執行。以下範例使用 PowerShell，需既有的偵測快取、§95 P0 追蹤檔、準備完成的 dev GT 及 §32 的嚴格連結權重 `m_dev_strict.json`。請將權重路徑換成自己的檔案。程式會核對權重指紋以及正播重跑是否逐列重現 §95，才繼續倒播。

```powershell
$env:PYTHONPATH = (Get-Location).Path
$env:CUDA_VISIBLE_DEVICES = '-1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
$studyDir = 'D:/beetrack_runs/study104'
$strictWeights = 'D:/path/to/m_dev_strict.json'
uv run python tools/study104/track_reverse.py 33 0 --weights $strictWeights --output-dir "$studyDir/tracks"
uv run python tools/study104/score_case.py 33 0 "$studyDir/tracks" "$studyDir/cases/33_0.json"
```

完整資料使用蜂箱 33、35 各自 seed 0 到 8。事件採軌跡首尾的方向，時間用最後一次同向跨界。官方事件配對容許任一次同向跨界在 5 幀內，框 IoU 為 0.5，另報 0.3。跨臂去重一律用代表事件時間相差至多 5 幀、同方向且穿越後的框 IoU 至少 0.5。J 是倒播獨有真事件減去倒播獨有假事件。聯集以全部正播事件為起點，只加上用時間、方向和框配不到正播的倒播事件，全程使用預測本身。
