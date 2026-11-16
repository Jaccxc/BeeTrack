# §115 將三份 P0 軌跡的平均計數匯出為整數事件

已保存三個 P0 偵測與追蹤結果時，這個工具能交付兩份標準十欄 MOT、原始 seed／ID 對照、三元組的實際候選與支持證據，以及計數保證。兩份輸出的每筆已報事件都有單一原生完整軌跡、實框、方向和實際跨越時刻。它們在正式 30 秒窗的逐格計數相同。工具適用 25 fps、有固定入口多邊形的片段；上游成本仍是三次偵測加追蹤。[整數代表實作](integer_events.py) · [完整 MOT 匯出](e3_mot.py)


在兩個留出蜂箱,固定代表與幾何代表都可追查到原軌跡。相對三路平均,正式 30 秒窗誤差在蜂箱 33 從 19.29% 變為 19.47%,在蜂箱 35 從 20.45% 變為 20.52%。同樣的事件數下,幾何代表選到正確蜂的額外效果在兩個蜂箱都未達預先登記的全部條件。[正式紀錄](../../docs/experiment_log.md) · [研究圖](../../docs/study115_integer.png)
## 使用自己的三份追蹤或重播研究

從 BeeTrack repo 根目錄執行下列 PowerShell 指令。第一次使用可先執行 `uv sync --frozen` 安裝鎖定環境。`beetrack`、`tools/study103`、`tools/study104` 和 `tools/study108` 是同一 repo 內的共用模組。[專案環境說明](../../README.md)

```powershell
$env:PYTHONPATH = (Get-Location).Path
$env:CUDA_VISIBLE_DEVICES = '-1'
$tool = 'tools/study115'
```

公共研究重播包保存 18 組的事件描述、選擇、GT 時刻與合法配對邊。可用下列指令從已發布的包重建整數計數、99 組對照、官方事件最大配對與同秒候選的可達上限。若要從真實框檢查逐幀所有權，請使用下方的原始 MOT 流程。[§115 重播資料包](../../docs/study115_replay.json.gz)

```powershell
uv run --offline --no-sync python "$tool/release115.py" replay docs/study115_replay.json.gz --table
```

## 直接匯出

沿用上述環境，把三個 `--mot` 路徑依固定模型順序填入。輸入接受原 P0 寫檔精度，幾何座標兩位、分數三位小數。`--polygon` 可用單一多邊形或以影片名稱為鍵的 JSON。此入口不需要 GT、舊實驗資料包或預登記。

```powershell
uv run --offline --no-sync python "$tool/export_e3.py" --mot seed0.txt seed1.txt seed2.txt --polygon zones.json --clip my-clip --frames 3950 --output e3-export
```

- `canonical.txt` 固定取每個方向時間排序後的第 2、5、8…個原生事件。
- `geometry.txt` 在同一三元組、同一個 25 幀秒內，選受其他不同 seed 支持數最多的成員。兩臂分別保留其完整原生事件軌。所有來源的非事件軌整段保留，未報告的事件軌依實框所在側分段。保留的原框、信心與幀內欄序在兩臂完全一致。
- `freeze.json` 存每個可報告原生事件、完整父軌 ID 區間、映射到來源 seed/ID 的兩份清單、99 份同秒配額清單、每個時間窗的 E3 三倍計數與整數計數、兩份 MOT 的精確 SHA-256。每幀不同父軌間的競爭不因換代表而改變。相同來源的重跑應產出相同位元組，若目的地既有內容不同則報錯。

`counts_proof.delta_l1_thirds` 給出由這支片自己的預測產生的整體 W 偏差證書。對任何正數 GT 事件總數 G，整數版和 E3 的正式窗誤差最多相差 `100 × delta_l1_thirds / (3 × 30 × G)` 個百分點。任意一個正式分向窗的計數最多差 2/3 隻，整片的各方向最多差 1/3 隻。幾何版在任意整幀起點的一般區間最多差 4/3；採正式的 30 個整秒起點時，幾何版和固定版的每一格都相同。這些是相對三個 P0 的算術平均，真蜂數仍需標註評估。

## 研究模式與凍結次序

每折各用 `012`、`345`、`678` 三個互異訓練 seed 組。來源為 no33 評 35 與 no35 評 33，均做 head 及 tail，合計 12 組。留出為 no33 評 33-head 與 no35 評 35-head，合計 6 組。head 長 3,950 幀、tail 長 1,000 幀。來源與另一折留出共用相同實體蜂箱，不應當成四個獨立場景。檔名盤點可用 `inputs.py paths`，這一步只檢查所需 57 個檔案存在。重建研究時，先準備 §108 的原生追蹤結果。兩份純預測清單可由公開的 §108 資料包逐位元組還原。來源 tail 還需要按 §108 流程重建的 `source/tail_mot/{fold}_{seed}_tail.txt`；來源及留出 head 讀 `runs/trackers/dev/` 下既有的完整檔案。[§108 原始資料重建](../study108/README.md)

```powershell
$out='runs/study115'
$lineage='runs/study108'
uv run --offline --no-sync python tools/study110/proposals110.py from-public docs $lineage
uv run --offline --no-sync python "$tool/inputs.py" paths --lineage-dir $lineage --runs-dir runs --zones-path beetrack/zones.json
$prereg='33ae06e5646fe7eb03d7b4f02e7845c2fd3cefe6'
foreach ($role in 'source','target') {
    uv run --offline --no-sync python "$tool/freeze_e3.py" freeze $role $out --prereg $prereg --lineage-dir $lineage --runs-dir runs --zones beetrack/zones.json
    if ($LASTEXITCODE -ne 0) { throw "$role freeze failed" }
    uv run --offline --no-sync python "$tool/freeze_e3.py" inventory $role $out --prereg $prereg
    if ($LASTEXITCODE -ne 0) { throw "$role inventory failed" }
}
```

分組目錄為 `角色/折_首seed_head或tail/`。載入器先核對整份 §108 純預測清單的 SHA-256，再核對全部原 MOT、入口多邊形及重新由原軌跡抽出的事件與幀。來源 tail 使用 §108 已凍結的 CPU 重建 MOT。迴圈先凍結來源 12 組取得 `source_index.json` 的指紋，再以相同規則凍結留出 6 組，取得 `target_index.json` 指紋。此實作的來源角色是檢查輸入精度、配額與輸出可重現性；兩折留出都須呈報。既有資料研究曾使用這些 GT，故應稱固定規則的留出驗證。

每個可報的三元組固定選 1 筆。每次隨機對照從包含第 2 名的同秒成員中均勻選 1 筆，不改報告數及任何正式窗。每個案例的 99 次使用獨立 `SeedSequence([314159, role, fold, first_seed, part, r])` 傳給 PCG64。來源、留出、一般匯出角色代碼依序為 0、1、2，head、tail、custom 部位代碼依序為 0、1、2，r 為 0–98。先處理進再處理出，再按三元組時間、組內固定名次取樣，保留只有 1 個合法選項的組。包含部位代碼使來源 head 與 tail 的隨機序列可分開重播。全部配額與每次選到的原生 ID 都存檔。

```powershell
$freezeSha='<該組 freeze.json 的 SHA-256>'
uv run --offline --no-sync python "$tool/freeze_e3.py" replay "$out/source/33_0_head/freeze.json" --sha $freezeSha
```

重播可僅由事件描述重建幾何支持、原始分組、99 份清單及精確計數證書。要從框逐一重建兩份實際提交 MOT，需要三份原始軌跡；各檔已記錄來源及指紋。

## 正式計分與可交付結果

兩份索引定版後，每個角色分別指定已公布的指紋。正式計分才開啟對應 §108 標註與原生 GT，重新校驗 GT MOT，並核對舊有三份個別 P0 的成績。下例先算來源，之後用 `target` 與自己的索引指紋重跑。

```powershell
$role='source'
$indexSha='<已發布的 source_index.json SHA-256>'
uv run --offline --no-sync python "$tool/score_e3.py" score $role $out --prereg $prereg --index-sha $indexSha --lineage-dir $lineage --runs-dir runs --zones beetrack/zones.json --archives docs
```

獨立核對程式可從三份原 MOT、固定 GT 和凍結清單重算，再比對主評分的數字與兩種 IoU 的最佳選擇證明。兩個角色各使用對應的索引指紋。

```powershell
uv run --offline --no-sync python "$tool/verify_e3.py" verify $role $out --index-sha $indexSha --lineage-dir $lineage --runs-dir runs --zones beetrack/zones.json --archives docs
uv run --offline --no-sync python "$tool/verify_e3.py" compare $role $out --index-sha $indexSha
```

每組交付 `W(整數)`、`W(E3)`、差值與其資料依存上界，以及雙方向的數量／偏差。官方 F1@0.5 為主，0.3 為敏感度。重讀兩份完整 MOT、重新計算每幀框歸屬、全部同向跨越及整片事件最大匹配。原生 P0 分別報單獨提交的真實分數及只報同一 seed、但保留三路原框競爭時的分數。E3 只提供分數窗基準。票數代表對固定代表和 99 個同秒對照回答軌跡身分的選擇問題。再保留相同事件與三路非事件軌、移掉未報告事件軌的所有框，測試單是拿走原幀競爭者可改多少官方分數。

真正需要判斷的取捨可以直接報量，而不必以來源平均 F1 提升 0.02 等任意門檻決定要不要量留出。每個蜂箱分列三組的數字與完整差值，不把同片三組當獨立樣本。可直接回答以下問題。

1. 已有三份偵測、應用必須交整數與可追溯事件時，把 E3 換成實體輸出的窗誤差代價是多少？同一組的兩種代表及其 99 對照具有完全相同的正式窗計數，實際代價和上界均可核對。
2. 在同一路線、同一數量、同一秒、同一完整框池下，幾何支持是否選到了更多可匹配事件？與固定代表及 99 次任選同報，另列能換代表的三元組數，避免把小分母或全為唯一可選的組誤讀成選擇有效。
3. 即使當前幾何選擇失敗，固定輸出格式還有多少可達的辨識空間？凍結後可將每個三元組所有合法同秒成員的 GT 合法邊取聯集，再求「三元組對 GT」最大一對一匹配。每個配到的三元組選一個提供該邊的實際成員，其餘用固定代表，得到可達到該 TP 的完整選擇。反之任何合法選擇的事件配對都對應這張圖的一個匹配，故上下界相等。此最佳值給「固定框池、同秒配額、此一 IoU」的容量，僅在標註後作診斷，不能當部署選擇器。可將幾何代表與最佳值的差距、原生框競爭移除的差距分別呈現。

不同代表即使無法勝過單模型 P0，兩份標準 MOT、逐方向／逐窗誤差證書與具體可回查的事件仍可交付。若單模型已足以滿足應用，三份推論的額外成本仍須與單模型列在一起；研究結果不等於必須改用三模型。

## 合成核對

```powershell
uv run --offline --no-sync python "$tool/integer_events.py" self-check
uv run --offline --no-sync python "$tool/e3_mot.py" self-check
uv run --offline --no-sync python "$tool/inputs.py" self-check
uv run --offline --no-sync python "$tool/self_check.py"
uv run --offline --no-sync python "$tool/evaluate.py" self-check
uv run --offline --no-sync python "$tool/verify_e3.py" self-check
```

這些檢查只使用合成來源，範圍涵蓋空事件、同幀競爭、完整 MOT 寫讀、99 份對照的 198 份官方閾值比對、E3 的小數計數、全部 12+6 組索引及篡改檢查。

## 研究圖、歷史比較與指紋

以下指令從公開資料重建研究圖及基準表。圖中的三組各包含三個不同 seed,同一蜂箱的三組共用同一支影片。P0、ByteTrack 與 TOPICTrack 歷史表逐一核對已發表的 §103、§108 與 §111。

```powershell
uv run --offline --no-sync python tools/study115/plot115.py --output docs/study115_integer.png
uv run --offline --no-sync python tools/study115/historical115.py --out runs/study115/historical
```

預測索引 SHA-256 為來源 `e5af5c611487d534f65e22425a489f0cb51f2c51f7583e560810289ac5365636`、留出 `3ea1f7e2e55f0031277fa2cd3a03e97d742cecefe3ff197812d7291423c8f921`。正式分數索引為來源 `dd0fac185dd6749cfa5566b8a591edc45480218e6a912b72af3e418a8e7105f8`、留出 `772f4e444063c782ca3d052e4b54fd04590d229c35400f5df52e9e07bb486089`。公開包為 `73ae1f2f6cfb70921c9d8f98a19919abcf922edb38a5f5ffafed78d1010fa3a6`。
