# §116 同一次 640 偵測,放大入口能否看全穿越前後的同一隻蜂

局部放大可能保住入口附近的小蜂,同時壓縮畫面其他位置。§116 固定同一偵測權重,比較將像素分給入口或分給其他位置時,一個真實穿越能否在改側前後都配到正確的蜂。使用 §108 來源 35-tail(留 33 折)與 33-tail(留 35 折),每片 1,000 幀、seed 0–2。兩支來源影片與 GT 先前都已研究過。[預登記](https://github.com/Jaccxc/BeeTrack/commit/c8fabf6c55c9737086b02050fabb3360bb5f1db2)

兩片都取得足夠的入口取樣差,但都未達每片至少 2/3 個 seed 通過的門檻。在 33-tail,各 seed 的原圖與入口放大雙端成功數依序為 11→9、10→10、9→10(各 16 筆),只有 seed 2 通過。在 35-tail 則為 8→7、8→8、8→6,沒有 seed 通過。依預登記停止這個固定 1.5 倍配置。

## 用公開包重算

從已安裝專案鎖定環境的 repo 根目錄執行,以 CPU 核對兩個來源、三個 seed 各自的四臂資料。[§116 重播資料包](../../docs/study116_replay.json.gz)

```powershell
$env:PYTHONPATH = (Get-Location).Path
$env:CUDA_VISIBLE_DEVICES = '-1'
uv run --offline --no-sync python tools/study116/release116.py replay docs/study116_replay.json.gz --table
```

公開包共 1,718,951 位元組,SHA-256 為 `959284838dc5ee13a9f02ff19549ae38f38b4666440392bf5f7f6cef24c26c05`。需要逐格明細時,在同一指令加上 `--report report116.json`。包內保留兩種 NMS 的全部預測框、原始順序及原始精度的分數,也保留每幀的 GT 切段 ID、入口側別、是否位在 L 中心區。每幀所有 GT 與分數 ≥0.25 預測之間的每個非零 IoU 都有紀錄,包括低於 0.3 的值。重播把其餘位置補成零,還原完整矩陣後重新做指派。從這些資料可以核對 48 份預測清單、768 次前向的逐幀收據、雙端與周邊成績及研發門檻。原始 JPEG 和 GT 框座標另存於實驗來源中,從像素重跑模型或重算 IoU 需要原始檔。

固定樣本來自每段 GT 最後一次與首尾方向相同的改側。保留相鄰兩個實際觀測幀 u、t 與蜂隻 ID,幀號從 1 開始。按進、出方向及預定 SHA-256 順序各抽 8 筆。33-tail 從 50 個真事件抽 16 個,35-tail 從 37 個抽 16 個,各 8 進、8 出、32 個不同影格。同片三個 seed 共用整份名單,因此總樣本為 32 次獨特穿越。樣本清單在預測前凍結,SHA-256 為 `3f144a920a99ef438bc2552284f77bc8cab579c891e3db956280402ded62566e`。

## 四臂改了什麼

| 臂 | 影像送入同一偵測器的方式 | 可比較的問題 |
|---|---|---|
| U | 原圖、640 | 常規整圖偵測的參照。 |
| L | 原圖先局部放大入口,再送 640 | 用同樣形狀的輸入張量,把更多像素分給真入口。 |
| C | 把與 L 同尺寸的放大矩形移到距離它最遠的合法角落,再送 640 | 同樣的放大倍率與面積換一個位置,看效果是否特別出現在入口。 |
| H | 原圖、960 | 用較大輸入張量,找出 U 遺漏但高尺度看得到的穿越。 |

L 用入口多邊形的外接矩形,四邊向外各加原圖短邊的 5%。從原尺寸 BGR 影像按像素面積重採樣,才做 letterbox。每軸中央等比例放大,周圍壓縮並保留完整視野。兩軸共用最接近 1.5 的可行倍率,外段斜率限制在 [0.5,2]。因 C 的水平或垂直放大帶仍可能涵蓋入口,程式沿實際入口各邊及法向偏移 ±2.5% 短邊的三條取樣帶,計算以弧長加權的兩軸斜率乘積 J。以未四捨五入的平均 J(L)−J(C) 評估實際取樣差。

每個 seed、影格、臂只做一次 CPU FP32 前向,並記錄實際張量形狀。U、L、C 的張量需相同,H 則較大。主分析先將候選框的四條邊逆回相機座標,再做 NMS(信心下限 0.05、IoU 0.7、最多 300 框)。附加對照在同一次前向的變形座標中先做套件原生 NMS,才逆回原圖。這個對照可分辨壓框順序對結果的影響。

所有臂用同一版本、CPU FP32 及同一單工作程序,最多佔兩個邏輯核心,OpenCV 與 PyTorch interop 各一執行緒。收據保存程式及輸入檔指紋、CPU 實際設定、解碼、重採樣、letterbox、前向及兩種 NMS 的耗時。`decode_shared` 是同一 seed 與幀的四臂共用解碼時間,加總時間只能計一次。`total` 包含兩套 NMS 的診斷成本。公開包可逐位核對原收據。重新執行時應比對輸入、模型和套件版本、預測及成績,耗時會隨執行環境變動。

## 如何讀雙端成功與門檻

每個所選幀以全幀 GT 和分數 ≥0.25 的全部預測框做一次一對一指派,先使總 IoU 最大,才用門檻判定。主門檻 IoU 0.5,另報 0.3。對同一穿越,兩端都須配到該 GT ID,且兩個預測框的中心分別落在入口正確側,才算一次雙端成功 V=1。這量到同蜂兩端的偵測可見性。連結器是否給了同一預測 ID 仍需完整追蹤評估。

- **逐蜂比較。** 同一 GT 段的 U、L、C、H 組成四位狀態,按 seed 計成功數、各狀態頻數及 U 改成 L 的新增與損失。在 H 成功而 U 失敗的子組中,另計 L、C 各救回幾筆。前端和抵達端、進和出及各 IoU 分開列。跨 seed 的聯集、交集以同片 GT ID 去重。
- **周邊代價。** 在不重複的所選影格內,分 L 中心矩形內外計全部 GT 的配到數與總數,及未配到 GT 的預測框數。GT 與預測框各按自己的原圖中心分區。同一畫面無論有幾隻入選蜂,周邊統計只計一次。
- **是否值得跑完整片段。** 每個 seed 的 L 雙端成功數須嚴格高於 U 和 C,且 L 須救回至少一筆 H 成功而 U 失敗的事件。每來源要有 16 筆樣本、入口位置劑量差 J(L)−J(C) ≥0.10、至少 2/3 個 seed 同時達標,才另登記該來源完整 1,000 幀 U/L/C/H 與 P0 的實驗。兩片都不過即停止此固定配置。本次計分範圍只含抽樣影格,全片計數窗誤差、官方事件 F1 及留出蜂箱表現留待後續檢驗。

持有原實驗目錄時,可用已封存的偵測及計分 SHA-256 重新打包。

```powershell
$runDir = 'D:/beetrack_runs/study116'
$detectorSha = Read-Host 'Detector SHA-256'
$scoreSha = '07d9c0e6b1c553aee7b3fc4af728c0b6947da9be582733a91166e2514a526c73'
uv run --offline --no-sync python tools/study116/release116.py pack $runDir "$runDir/study116_replay.json.gz" --detector-sha $detectorSha --score-sha $scoreSha
```

## 研究圖與原始輸入重建

```powershell
uv run --offline --no-sync python tools/study116/plot116.py --output docs/study116_sampling.png
```

[查看兩片逐種子的研究圖](../../docs/study116_sampling.png)。以下從原始資料重新執行 CPU 前向,需有兩片來源訓練影片的原始影格、六份歷史偵測權重、prepared tail GT、入口多邊形及 §108 逐種子來源紀錄。研究只讀 33-tail、35-tail。指定一個新的輸出目錄,每個條件共用相同幀,一次前向產生兩種 NMS 的結果。攝影機上的輸入變形與逆座標壓框實作分別見 [入口取樣](entrance_scale.py) 與 [偵測介面](warp_predict.py)。

```powershell
$tool = 'tools/study116'
$out = 'runs/study116_reproduction'
$prereg = 'c8fabf6c55c9737086b02050fabb3360bb5f1db2'
uv run --offline --no-sync python "$tool/sampling.py" freeze-rosters "$out/roster" --lineage-dir runs/study108
uv run --offline --no-sync python "$tool/dose116.py" --out "$out/geometry.json"
uv run --offline --no-sync python "$tool/run_pilot.py" --requests "$out/roster" --output $out --prereg $prereg --threads 2
$pred = (Get-FileHash "$out/detector_done.json").Hash.ToLowerInvariant()
$roster = (Get-FileHash "$out/roster/index.json").Hash.ToLowerInvariant()
$geometry = (Get-FileHash "$out/geometry.json").Hash.ToLowerInvariant()
uv run --offline --no-sync python "$tool/run_scores.py" score $out --prediction-index-sha $pred --roster-index-sha $roster --geometry-sha $geometry
```

原始推論收據含實際耗時、執行環境與原始碼檔案指紋。跨電腦或重跑時應核對同一批幀的框、配對與分數。上方的公開重播包可完整驗證本次封存的指紋。
