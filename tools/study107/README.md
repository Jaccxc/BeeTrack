# §107 用入口影像分配 30 秒窗的進出計數

P0 先估計一整段有幾隻蜂進、幾隻出。這個模組保留兩個總數,利用影像中沿入口邊界向內與向外的運動,估計每個時間窗各有幾隻。輸出是窗內的期望數,可為小數。整段影片均參與計算。[預登記與判讀條件](https://github.com/Jaccxc/BeeTrack/commit/f48db619aaff4e312ae3ec2f6a8bf491668cd6b2)

## 直接重算

在 repo 根目錄執行以下指令。只需要 Python、NumPy 和隨附的壓縮 JSON。個別蜜蜂的事件時間已由既有軌跡算出,入口影像也已轉為逐幀的兩個方向通量。

```sh
uv run python tools/study107/replay.py replay docs/study107_events.json.gz
```

每折第一張表給出另一個蜂箱的來源 head、tail 各 3 個 seed 在三種權重下的誤差。先在每段平均 seed,再讓兩段等權決定 α。重播會重新選出兩個 α,比對在目標計分前封存的候選分數、選擇及 SHA-256,然後才計算目標表。這兩個蜂箱交換來源及留出角色,head 作為來源時有部分影格參與該折偵測器訓練。九個 seed 共享同一支目標影片。

目標表逐 seed 給出原始 P0、正確時間的通量、同 α 的均攤、交換進出方向、平移通量的中位誤差、正確時間嚴格勝過多少個位移、是否同時勝過均攤和至少 142 個位移、自由分配時間時的理論誤差下界。所有誤差單位為百分比,改善量用百分點。主表使用來源選的 α,另一張表使用預先固定的 0.25。每折的兩項判讀只依主表。第三張表列同 seed 的 P0、ByteTrack、校正 ByteTrack 及有資料的來源選參數 TOPICTrack。GT 分方向總數攤平是另外一條已知答案的參考線。逐幀方向分布差異是描述性數字,範圍 0–1,0 代表正規化後兩個方向形狀一樣。

以下指令另存完整 JSON,包含主臂及固定臂的 1–157 秒相位控制順序、每個 seed 的備援使用情況、進出總數、所有判讀與基準參考值。

```sh
uv run python tools/study107/replay.py replay docs/study107_events.json.gz --json study107_full.json
```

## 證據與重建

公開包每支影片只存一份通量和 GT 事件。每折每個 seed 的 P0 事件另外保存,即使在另一折用到同一支影片,各自的偵測器仍有自己的預測。每列通量對應片內一幀(第一幀為零),兩欄依序是向內、向外的像素面積。GT 和 P0 事件以 `[片內幀號, 方向]` 儲存,幀號從 1 起算,進為 +1、出為 −1。解開後可核對四份通量的小端 float64 SHA-256、來源封存及完整驅動程式輸出的摘要。每份影像清單以原始 JPG 的 SHA-256 定義。實際重製通量及事件需要原始影像、偵測快取、嚴格連結器權重、軌跡和既有校正標註。重播腳本核對這些輸入所產生的研究數字。

通量欄 `flux` 以二進位保存浮點精度,並記錄 `dtype="<f8"`、`shape=[幀數,2]` 及 `base64`。其餘的來源、事件、參考數字和輸入摘要均是一般 JSON。解碼後可直接核對其中一支影片。

```python
import base64, gzip, json, numpy as np
with gzip.open("docs/study107_events.json.gz", "rt", encoding="utf-8") as stream:
    archive = json.load(stream)
m = archive["clips"]["BEE24-33-head"]["flux"]
flux = np.frombuffer(base64.b64decode(m["base64"]), dtype=m["dtype"]).reshape(m["shape"])
```

從原始資料執行時,先為四個片段各執行一次 `flux.py`,再以 `inputs.py source` 重建來源事件。用 `run.py freeze` 封存兩折的權重與輸入後,才進行 `inputs.py target` 和 `run.py evaluate`。以下 PowerShell 例子需先將 `$strict` 設為既有 `m_dev_strict.json` 的位置。輸出資料夾使用 D 槽。

```powershell
$env:PYTHONPATH = (Get-Location).Path
$env:CUDA_VISIBLE_DEVICES = '-1'
$dir = 'D:/beetrack_runs/study107_reproduction'
foreach ($clip in 'BEE24-33-head','BEE24-33-tail','BEE24-35-head','BEE24-35-tail') {
    uv run python tools/study107/flux.py $clip "$dir/flux"
}
uv run python tools/study107/inputs.py source "$dir/source.json" --weights $strict
uv run python tools/study107/run.py freeze "$dir/source.json" "$dir/flux" "$dir/frozen.json" --prereg f48db619aaff4e312ae3ec2f6a8bf491668cd6b2
$sha = (Get-FileHash "$dir/frozen.json" -Algorithm SHA256).Hash.ToLowerInvariant()
uv run python tools/study107/inputs.py target "$dir/target.json"
uv run python tools/study107/run.py evaluate "$dir/frozen.json" "$dir/target.json" "$dir/flux" "$dir/results" --freeze-sha $sha
uv run python tools/study107/replay.py pack "$dir/results/flow_replay.json" "$dir/results/flow_results.json" "$dir/study107_events.json.gz"
```

打包會先核對完整的來源選擇、18 格主臂與固定臂結果、每個位移的誤差、基準和兩個判讀。若重建使用的 OpenCV、平台或 CPU 不同,影像估出的浮點通量可能因實作差異而略變。研究使用的 OpenCV 版本、像素尺度及原始輸入清單摘要保存在各片段的 metadata。
