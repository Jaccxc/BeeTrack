# §108 另外兩個偵測器能辨認主偵測器多報的進出事件嗎

將相同架構、不同訓練 seed 的三個偵測器各自接上同一套 P0 追蹤。主偵測器提出進出事件,另外兩個偵測器各自做一對一配對。只要其中一個支持,即原樣保留主偵測器的時間、方向及位置。這項方法需要三份偵測和追蹤的成本,主偵測器本來就漏掉的事件仍然漏算。[預登記](https://github.com/Jaccxc/BeeTrack/commit/5a58655ae718c16f66bb7d38625fdd6a17f55544)

## 從公開資料包重播

在 repo 根目錄執行下列 PowerShell 指令。重播使用四個來源片段(兩個長蜂箱的 head、tail)所對應的預測、GT 事件及官方配對合法邊,無須原始影片。程式重新產生所有研究中的隨機對照,仍須使用封存資料中記錄的 NumPy、SciPy 版本才能精確重現原配對平手及隨機抽樣次序。

```powershell
$env:PYTHONPATH = (Get-Location).Path
uv run python tools/study108/run108.py replay docs/study108_source.json.gz
uv run python tools/study108/verify108.py scores docs/study108_source.json.gz --table
uv run python tools/study108/run108.py replay docs/study108_target.json.gz
uv run python tools/study108/verify108.py scores docs/study108_target.json.gz --table
```

來源 head 的門檻要求三組中至少兩組通過。留 33 用蜂箱 35 作來源,三組各有 0/3、0/3、1/3 個輪換符合條件;留 35 用蜂箱 33 作來源,三組各有 3/3 個輪換符合。依「任一個來源過關」的分支,兩個留出 head 共 18 個設定都進入實測。第一份資料包收錄 36 個來源設定,第二份收錄這 18 個留出設定。正式目標判讀只有三個指定主偵測器 0、3、6,其餘輪換用來檢查對選擇哪個主偵測器有多敏感。同一影片的九個 seed 共用標註。

第一個指令逐案重做幾何配對、99 次循環位移和 99 次配額抽樣,重新計算每個結果的 SHA-256。第二個指令另用整數矩形交集、獨立產生的相位表及完整 ID 排序矩陣重算支持清單,再以獨立的增廣路徑最大配對及累積逐幀計數,核對每個隨機子集的保留 ID、分向 TP/FP/FN、兩種 IoU 的事件 F1、窗誤差、理論下界和各折門檻。

### 怎麼解讀數字

- 支持配對用每一條原生事件最後一次同方向跨界的框。兩個事件方向相同、時間差 ≤5 幀、IoU ≥0.5 時才可能配對。最大配對數相同的解可能對應不同的事件 ID。固定依拆分後的事件 ID 升序排列兩端,採 SciPy 在完整矩陣上的平手結果。錯時對照在片內循環移動兩個支持者,每次重新配對。
- 官方 TP 使用原生逐幀框競爭所決定的合法邊,保留每一次同方向跨界作候選。以原生事件列為全集,保留子集後重新最大配對。正式 TP 差值按最大配對數量決定,同一個 TP 在平手下所配到的 GT 身分可能改變。
- 同數量隨機對照在正確方向內抽出和實際保留名額相同的事件,因此保留總數和進出偏差一致。E3 是同三個偵測器各分向窗計數的算術平均,允許小數且沒有個別蜂的 F1。窗誤差納入完整片段的 30 個起點,單位為百分比。平均窗誤差增益用百分點,F1 增益用 0–1 刻度。來源判讀使用 3950 幀 head;1000 幀 tail 另列。
- 壓縮 JSON 保存各折的輸入摘要、預測事件位置、凍結的保留清單、每個隨機對照及官方候選邊。這些資料可重算研究數字。要重新檢查每幀 IoU 及原軌跡競爭情況,需要原始 MOT、偵測快取、GT 與嚴格連結器權重。

## 從既有偵測與原始 MOT 重建

先將 `$strict` 設為既有 `m_dev_strict.json` 的位置。以下流程先凍結兩折的預測、實際保留清單及隨機名額,再產生該階段的標註分數。來源與留出共用同兩個長蜂箱但交換角色。短 tail 的 P0 可由同一偵測器的原框和嚴格連結模型重建。每個階段結束後保留 SHA-256,後面的步驟以此驗證輸入。

```powershell
$env:PYTHONPATH = (Get-Location).Path
$env:CUDA_VISIBLE_DEVICES = '-1'
$dir = 'D:/beetrack_runs/study108_reproduction'
$prereg = '5a58655ae718c16f66bb7d38625fdd6a17f55544'
uv run python tools/study108/inputs108.py predictions source $dir --weights $strict
uv run python tools/study108/run108.py masks source $dir --prereg $prereg
$masks = (Get-FileHash "$dir/source_masks.json" -Algorithm SHA256).Hash.ToLowerInvariant()
uv run python tools/study108/verify108.py masks "$dir/source_masks.json"
uv run python tools/study108/inputs108.py labels source $dir
uv run python tools/study108/run108.py score source $dir --prereg $prereg --masks-sha $masks
uv run python tools/study108/verify108.py scores "$dir/source_replay.json.gz" --report "$dir/source_results.json" --already-checked-masks
$sourceHash = (Get-FileHash "$dir/source_results.json" -Algorithm SHA256).Hash.ToLowerInvariant()
if ((Get-Content "$dir/source_results.json" -Raw | ConvertFrom-Json).advance) {
    uv run python tools/study108/inputs108.py predictions target $dir
    uv run python tools/study108/run108.py masks target $dir --prereg $prereg --source-sha $sourceHash
    $targetMasks = (Get-FileHash "$dir/target_masks.json" -Algorithm SHA256).Hash.ToLowerInvariant()
    uv run python tools/study108/verify108.py masks "$dir/target_masks.json"
    uv run python tools/study108/inputs108.py labels target $dir
    uv run python tools/study108/run108.py score target $dir --prereg $prereg --source-sha $sourceHash --masks-sha $targetMasks
    uv run python tools/study108/verify108.py scores "$dir/target_replay.json.gz" --report "$dir/target_results.json" --already-checked-masks
}
```

## 對一段新影片輸出有跨 seed 支持的事件

先在同一支影片、同一張入口多邊形和相同影格座標下,以三個不同的固定 640 偵測器 seed 產生各自的 P0 MOT 軌跡。選定其中一個作主種子(anchor),另外兩個只提供支持。MOT 欄位前六欄是 1 起算的幀號、軌跡 ID、左上角 x、y、寬、高。入口多邊形用原影格的像素座標,JSON 可直接是頂點陣列 `[[x,y],...]`,也可用 `--clip` 指定既有多邊形檔內的名稱。

在 repo 根目錄執行以下 PowerShell 指令,將路徑和影片幀數換成自己的輸入。程式接受任意正整數幀數。研究影片為 25 fps,支持配對的時間容許值固定為 5 幀。

```powershell
$env:PYTHONPATH = (Get-Location).Path
uv run python tools/study108/count_supported.py --frames 3950 --polygon beetrack/zones.json --clip BEE24-35-head --anchor anchor.txt --buddy1 seed1.txt --buddy2 seed2.txt --output supported_events.json
```

`anchor_counts` 與 `kept_counts` 各列整段的進、出總數。`events` 按幀號列出每一個保留事件的主軌跡 ID、方向、幀號、主軌跡的代表框 `box_xyxy`,以及兩個支持者各自配到的軌跡 ID。支持者的 ID 分別屬於各自的 MOT 檔。事件時間採主軌跡最後一次同方向跨界。主軌跡以外的事件不會新增。執行只讀軌跡和入口,不需 GT、校正分數或訓練權重。

此輸出可用來核查逐個進出事件。兩個留出蜂箱中,這個篩選只在其中一個通過計數改善門檻。跨場景拿它替代原本 30 秒窗計數的條件仍待驗證。
