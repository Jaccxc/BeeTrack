# §110 兩個偵測器共同報出的事件能補回多少窗計數誤差

§108 讓主偵測器 A 只保留另一個偵測器也支持的進出事件。§110 問的是,在這些保留事件原樣固定的條件下,能否靠另外兩個偵測器 B、C 共同報出的新候選,達到預先指定的 30 秒窗計數目標。候選是在開啟本研究的 GT 計算之前凍結的。後續用已知的 GT 時刻求出這組候選能做到的最低誤差。[預登記](https://github.com/Jaccxc/BeeTrack/commit/fc9133e1f1ffe5dd345f0ad5664cfb3049986b78)

留 33 即使知道答案,三組平均的最低窗誤差仍有 29.75%,超過預登記上限 17.29%。單憑進出方向與整片事件數已可證明誤差至少為 26.09%。留 35 存在可達 9.29% 的子集,但直接加入全部候選使誤差由 11.49% 增至 14.63%,超過 13.49% 上限。兩折的聯合目標因留 33 而排除。18 組均證明最優,其中 14 組使用有理數對偶證書,其餘 4 組的逐格下界已有可達到的子集。

## 重播成績與最佳值證明

從已安裝此專案鎖定環境的 repo 根目錄執行。公開包內含 18 組凍結候選、原生事件流的有號幀、既有 GT 事件的有號幀、參照成績、最佳子集及其下界證明。只需 CPU。重播逐組重算誤差、核對存檔子集的誤差與下界,最後輸出兩個留出蜂箱各自的預登記門檻判讀。[§110 重播資料包](../../docs/study110_replay.json.gz)

```powershell
$env:PYTHONPATH = (Get-Location).Path + ';' + (Resolve-Path 'tools').Path
$env:CUDA_VISIBLE_DEVICES = '-1'
uv run --offline --no-sync python tools/study110/run110.py replay docs/study110_replay.json.gz
uv run --offline --no-sync python tools/study110/verify_witness.py docs/study110_replay.json.gz
```

## 從 §108 的公開預測重建候選

原實驗只讀 §108 已凍結的 `source_masks.json` 與 `target_masks.json`。下列 `from-public` 從公開資料包取出 `masks` 欄位,依 §108 寫入規格還原並驗證原始 SHA-256。接著 `freeze` 再次核對兩份清單指紋,只從這兩個純預測 JSON 重建幾何候選。`synthetic` 可另行檢查競爭同一事件、一對一指派、負方向及界值。[§108 來源資料包](../../docs/study108_source.json.gz) · [§108 留出資料包](../../docs/study108_target.json.gz)

```powershell
$out = 'D:/beetrack_runs/study110_reproduction'
$prereg = 'fc9133e1f1ffe5dd345f0ad5664cfb3049986b78'
uv run --offline --no-sync python tools/study110/proposals110.py synthetic
uv run --offline --no-sync python tools/study110/proposals110.py from-public docs "$out/inputs"
uv run --offline --no-sync python tools/study110/proposals110.py freeze "$out/inputs" "$out/proposals.json" --prereg $prereg
```

來源折在另一個長蜂箱處理 3,950 幀 head 與 1,000 幀 tail,留出折只處理自身 3,950 幀 head。兩折均取三個獨立訓練 seed 組 `(0,1,2)`、`(3,4,5)`、`(6,7,8)`,依序指定 A、B、C,因此共有 12 組來源與 6 組留出資料。每筆原生事件列是 `[軌跡 ID, 有號片內幀, x1, y1, x2, y2]`,正負分別代表進與出,幀從 1 起算,框座標單位是像素。

配對沿用 §108 的同方向、幀差 ≤ 5、IoU ≥ 0.5 規則,以百分之一像素的整數比較。B、C 分別移除與任一原生 A 事件有合法邊的事件,也涵蓋曾與另一個事件競爭同一個 A 的情況。其餘 B、C 依 ID 排序後在完整矩陣求最大數量的一對一指派。每個新增候選取 B 的方向、原時間及框。`proposals.json` 的 `cases[].kept` 列出 §108 保留的 A 事件,`cases[].donors` 逐一保存兩個來源的原始事件列。要讀取包括被 §108 濾去者在內的完整 A、B、C 表,用該列的 `role` 找到對應的純預測檔,再以 `predictions['<fold>_<seed>_<part>']['predictions']` 取得。每組還帶有三份 MOT 與預測輸入的指紋。

## 重新封裝成績

手上若有 §108 已產生的 `source_results.json` 與 `target_results.json`,可用以下指令從重建的候選重新封裝 §110。將 `$hist` 設為存放兩份 §108 成績的資料夾。`score` 會將歷史成績和 §108 資料包所記的指紋逐一比對。前面的公開重播路徑已內含所需的參照資料。

```powershell
$hist = 'D:/beetrack_runs/study108'
$geometrySha = 'a1b7d2735369dce571fcc9d376b3a902f5d3089079f03364cb8c020d4fc0595c'
uv run --offline --no-sync python tools/study110/run110.py score "$out/proposals.json" "$out/score" --geometry-sha $geometrySha --archive-dir docs --historical-dir $hist
uv run --offline --no-sync python tools/study110/oracle110.py self-check
```

## 窗誤差的範圍

對每個方向,若保留 K 筆、有 D 筆可加、真值為 T,則整片可達的最小絕對差是 `max(K−T, T−K−D, 0)`。將兩方向相加可得窗誤差下界。進一步以 750 幀為一窗,在 0 到 725 幀之間每隔 25 幀移一次起點,納入各起點的頭尾部分窗,共 30 組分割。每個「起點、窗、方向」各自套用上式,得到逐格下界。實際計數必須對同一個候選在全部 30 個起點作同一個取捨。

固定每個候選的方向及時間後,將候選依方向、時間排序,每一格涵蓋連續的一段候選。這個區間矩陣具全單模性,用線性規劃可求得與 0/1 子集選擇相同的最小誤差。程式重算選中子集的整數誤差作上界,再以逐格下界或可精確檢查的有理數對偶證書建立下界。兩界相等才記為已證最優。分子是全部格的絕對誤差和,分母固定為 `30 × 全片 GT 數`,乘 100 得百分比。不同最優子集可以得到同一個分子。

每個蜂箱的正式結果將其三組留出主 seed 的誤差等權平均。留 33 須比同三個 P0 的逐格算術平均 E3 至少低 2 個百分點。留 35 須在 §108 保留事件的誤差加 2 點以內,且嚴格優於 E3。這個診斷允許 GT 決定子集,只界定固定事件時間和候選集合的計數能力。它無法辨識候選中的真蜂,也沒有合併 MOT 後的官方事件 F1。評估範圍限於蜂箱 33、35。[§108 協定與結果](../study108/README.md)

| SHA-256 核對對象 | 指紋 |
|---|---|
| §108 來源純預測 JSON | `36fe1cc5470b9155a61bb8bed2f0f06242cccbbbda1b90842607976ce08e831e` |
| §108 留出純預測 JSON | `011b06f9fb98325d0407beac86819fda5ea8058b6627934e34c06c403ae63375` |
| §110 幾何清單 `proposals.json` | `a1b7d2735369dce571fcc9d376b3a902f5d3089079f03364cb8c020d4fc0595c` |
| §110 成績 `results110.json` | `390d6df908bd68997a172130792609f29c36fab91ba80ccd30a0b1130959edf6` |
| §110 公開資料包 `study110_replay.json.gz` 壓縮位元組 | `a44cbca9e2483efb31b0f4a111ac7026058ab0477a992a239ec3b11a2486d6ae` |

獨立驗證程式只用 NumPy 直接把每個事件分入 30 個起點的窗,核對證書與候選,並在兩個方向分別窮舉全部子集。18 個設定合計檢查 537061 個分方向子集。兩個方向的視窗誤差可相加,各方向達到最低值的候選也能合為同一組解。
