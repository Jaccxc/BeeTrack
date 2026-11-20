# §185 弱框選擇器

來源 W/G 的四支既有 val 短片,每折 seed 0–2 混合訓練一個八特徵邏輯斯模型。推論只用原生偵測框,最多延遲三幀。完整結果見 [實驗紀錄 §185](../../docs/experiment_log.md)。預登記為 `d075dc094eb4349f9d71d27f56b9189797cd31b2`。

H 保留分數 ≥ 0.40;L 保留 ≥ 0.25;M 保留強框及模型機率 ≥ 0.5 的弱框;C 每幀保留和 M 同數量的弱框,按原生信心高到低取。所有臂保留原列順序、座標和分數,沿用固定 P0。歷史 P0、校正 ByteTrack 與來源挑 TOPICTrack 是另外的背景臂,不等於 H/L。

## 不讀影像的證據重播

從 BeeTrack repo 根目錄執行。使用既有 uv 環境,不用下載模型或重新推論:

```powershell
$env:PYTHONPATH = (Get-Location).Path
uv run python tools/study185/check185.py --replay tools/study185/evidence.json.gz
```

壓縮證據含 24 格實體 MOT 的事件時間、30 起點分向整數窗、IoU 0.5/0.3 官方事件可配對邊及原始 GT/MOT 的 SHA-256。重播由事件重建整數窗、從事件圖獨立求最大匹配,核對 W 與 TP/FP/FN/F1。六組 M/C 逐幀框預算亦保存;它們在完整獨立驗證時與原框保留清單逐幀核對相等。

## 使用保存的模型

`model33.json`、`model35.json` 是數值模型;前者對應留 33 偵測器,後者留 35。`selector.arms` 接受原始 Nx6 `[frame,x1,y1,x2,y2,score]`,回傳 H/L/M/C 原序框列。下面例子只讀已完成的舊快取:

```powershell
@'
import json
import numpy as np
from pathlib import Path
from tools.study185.selector import arms
model = json.loads(Path('tools/study185/model33.json').read_text())
D = np.load('runs/dets/loho_no33_s0/dev/BEE24-33-head.npy', allow_pickle=False)
selected = arms(D, model)
assert all(v.shape[1] == 6 for v in selected.values())
'@ | uv run python -
```

## 從來源重跑完整流程

需要既有固定 640 偵測快取、修正 GT、入口多邊形、`beetrack/models/m_dev_strict.json`。模型指紋固定為 `fdd788e0dba0e6fa1c2d06feec3127bf84d5e7b2d214cc0f793c452270403dcf`。使用全新的 D 槽輸出目錄,不覆寫封存結果:

```powershell
$env:PYTHONPATH = (Get-Location).Path
$repo185 = (Get-Location).Path
$out185 = 'D:/beetrack_runs/bridge185_reproduction'
$pre185 = 'd075dc094eb4349f9d71d27f56b9189797cd31b2'
foreach ($hive185 in 33,35) {
  uv run python tools/study185/run185.py --repo $repo185 --out $out185 --prereg $pre185 train --hive $hive185
}
foreach ($hive185 in 33,35) {
  foreach ($seed185 in 0,1,2) {
    uv run python tools/study185/run185.py --repo $repo185 --out $out185 --prereg $pre185 freeze --hive $hive185 --seed $seed185
  }
}
uv run python tools/study185/run185.py --repo $repo185 --out $out185 --prereg $pre185 score
New-Item -ItemType Directory -Path "$out185/independent"
uv run python tools/study185/check185.py --repo $repo185 --out $out185 --evidence "$out185/independent/evidence.json"
```

全部六格 H/L/M/C 實體軌跡與保留清單封存後才開目標 GT。來源擬合及推論程式的指紋寫入每次新產物;公開程式位置改變不影響計算,新的執行紀錄會有自己的程式指紋。模型與特徵的合成自檢可執行 `uv run python tools/study185/selector.py`;標籤和獨立算分的合成自檢分別執行 `labels.py` 和 `check185.py`。

本版兩折的配對等量效果與固定門檻守門均未過,停止本版本。兩支 head 都是已知開發場景,三個 seed 不代表三個獨立蜂箱。不更改鎖定 P0,不加入封存測試臂。
