# 穿越前後的框屬於哪一隻蜂

33-head 有一項控制實驗,只把全程沒有正式淨進出的蜂的真框交給 P0。追蹤器仍報出 67 次穿越。按 IoU 0.5 逐筆查證,其中 51 次確實是同一隻蜂短暫改側,15 次是前後不同隻蜂,1 次在同幀有多個可配對的框。0.3 下仍有 46 次可以確認是同一隻無淨進出的蜂。35-head 的這項控制沒有預測事件。[資料與預登記](https://github.com/Jaccxc/BeeTrack/commit/092054e6a94df66bd1c9b931f4aeb8c7d3d4048d)

在實際偵測上,33-head 三個 seed 分別產生 117、100、123 個與真實事件完全沒有合法配對邊的預測。能確認是同一隻蜂暫時改側的有 14、11、14 個。主要限制是穿越前後至少一端無法以正式 IoU 門檻指派到真蜂。GT 蜂段均採專題修復後的身分。研究只追溯穿越時相鄰兩個觀測,同一片上的三個種子共用真蜂。[實驗紀錄](../../docs/experiment_log.md) · [逐事件證據](../../docs/study127_attribution.json.gz) · [圖](../../docs/study127_origins.png)

## 由封存證據重播

在 repo 根目錄執行。資料包保存每個穿越事件兩端的對應 ID、兩側的幀和空間位置分類、雙向合格配對數、真蜂的淨方向及與完整官方事件圖相連的 GT ID。它涵蓋原生 U 的三個 seed、全部真框 G 和只留無淨事件蜂的真框 C,各在兩片重算 IoU 0.5 與 0.3。

```powershell
uv run --offline --no-sync python tools/study127/replay_counts127.py docs/study127_attribution.json.gz
uv run --offline --no-sync python tools/study127/plot127.py docs/study127_attribution.json.gz --output runs/study127_origins.png
```

## 由原始軌跡查核

準備 §126 的既有偵測、相機與 GT,先用該節的產生器準備原始列索引和十欄 MOT。下例的 `$previous` 指向該次輸出,`$target` 是本次查核目錄。若兩個蜂箱來自不同次 §126 執行,在 `--source126` 後列出兩份 manifest。轉接程式會核對 C 的入選原始列及正式 GT 蜂段,再核對十份原生軌跡檔的指紋。[§126 的輸入及指令](../study126/README.md)

```powershell
$previous = 'D:/beetrack_runs/study126_raw'
$target = 'D:/beetrack_runs/study127_reproduction'
New-Item -ItemType Directory -Force -Path $target | Out-Null
uv run --offline --no-sync python tools/study127/prepare127.py --repo . --source126 "$previous/manifest.json" --gt33 runs/gt/dev/BEE24-33-head/gt/gt.txt --gt35 runs/gt/dev/BEE24-35-head/gt/gt.txt --polygon beetrack/zones.json --output "$target/manifest.json"
uv run --offline --no-sync python tools/study127/audit127.py --manifest "$target/manifest.json" --confirmed-prereg 092054e6a94df66bd1c9b931f4aeb8c7d3d4048d --output "$target/expected.json"
```
