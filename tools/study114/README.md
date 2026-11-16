# §114 蜂穿過入口之後,能分辨同蜂折返與另一隻蜂反向穿越嗎

兩隻不同的蜂一進一出,和同一隻蜂進去又出來,在畫面上都會產生一對反向穿越。兩者在首尾側別計數時的意義不同。§114 問的是,已確認一次真實穿越、又可同時看到同蜂折返和異蜂反向穿越時,單一偵測器的運動與框形特徵能否辨認前者。[預登記](https://github.com/Jaccxc/BeeTrack/commit/fc1ce10)

| 偵測器留出折 | 實際研究的來源蜂箱 | 擬合 | 固定模型後評估 |
|---|---|---|---|
| 留 33 | 35 | 35-head,3,950 幀 | 35-tail,1,000 幀 |
| 留 35 | 33 | 33-head,3,950 幀 | 33-tail,1,000 幀 |

每折各用 seed 0、1、2,共 12 格。每個模型都在自己這個 seed 的來源 head 擬合,在同一來源隔開 50 幀的 tail 評估。三個 seed 重複觀察同一批蜂,只能檢查偵測器變動對排名的影響。

兩折各有 0/3 個 seed 通過,原因不同。留 33 折的 35-tail 各 seed 只有 2、1、2 隻蜂同時有可確認的真折返與異蜂競爭,低於至少 10 隻的門檻。直接用該段 GT 的實框也只找到 7 隻有這種競爭的蜂。留 35 折的 33-tail 有 11、13、12 隻,四特徵得分依序為 0.7500、0.7308、0.7083,均達 0.70,但比最近距離只多 0.0909、0.0577、0.0903,均未達預登記的 0.10。

## 用公開包重算

從已安裝專案鎖定環境的 repo 根目錄執行,保留 `docs/` 裡 §§105、108、113 的歷史檔供輸入指紋核對。每格的原生片段 ID、進出方向、最後同向穿越抵達幀與框都要能對回 §108。包內有凍結候選、全部 GT 改側、候選端點同幀 IoU 達 0.3 的 GT 重疊清單及六組 head 擬合紀錄。重播由此重建乾淨標籤、擬合與排名,逐格核對指紋。`--table` 印出 12 格的模型與對照成績。原始偵測及完整 GT 可由下方流程重新驗證。[§114 重播包](../../docs/study114_replay.json.gz) · [§108 來源資料](../../docs/study108_source.json.gz)

```powershell
$env:PYTHONPATH = (Get-Location).Path
$env:CUDA_VISIBLE_DEVICES = '-1'
$source108 = 'docs/study108_source.json.gz'
uv run --offline --no-sync python tools/study114/release114.py --root . replay docs/study114_replay.json.gz --source108 $source108 --table
```

公開包共 23,309,509 位元組,SHA-256 為 `1b2c242b88d9a4c1b0fc9826e975048559d517fffc2c11eae540e70f925aee10`。封存的整數與浮點表分別採 int32 與 float64,每表記欄名、列數和原始位元組 SHA-256。重播針對還原的逐格 JSON 與最終結果核對指紋。四段影片各有一份 GT 改側紀錄,同段三個 seed 共用。加入 `--independent` 可在這 12 格上另外核對分組、排名及判讀。獨立驗證「相應幀是否還有其他重疊 GT」及「原偵測是否少列候選」則需完整 GT 與偵測快取。

## 如何讀排名與覆蓋數

1. **候選穿越。** 取同一 seed 的全片 P0 軌跡中每一次框中心改側,包括最後回到原側的蜂。再從分數 ≥0.25 的 NMS 倖存框補入 1–3 幀內的近鄰異側框對,每個出發框、每個幀差至多留兩個。相同幾何只記一次,保留原框列號與原生軌跡 ID。框的 xywh 用百分之一像素的整數儲存。這個步驟全部做完才開 GT 標籤。
2. **能確認身分的選項。** 起點的兩個實框須獨佔對應同一 GT 片段,各對該蜂 IoU ≥0.5、對其他蜂均 <0.3,側別相合且區間內恰有一次同方向真穿越。從起點往後 1–50 幀,在四個平均身長內找反向候選。正例限同蜂的下一次真穿越,負例限身分已確認的另一隻蜂。同一真穿越在同一池只算一次。較晚的同蜂穿越、未知身分、片尾截斷及池外的真返回均不當負例。同時有正例和已知異蜂的起點才可比排名。另列只有正例、只有異蜂及兩者皆無的起點。
3. **同一組內比較。** 四特徵分別是按身長縮放的距離、抵達幀差除以 50、兩中間框面積比的絕對對數及等速外推殘差。每個 head 的混合起點等權,同組中的各個異蜂等權,用帶 L2 懲罰且無截距的成對 logistic 模型擬合。tail 將真返回和該方法得分最高的異蜂比較,勝得 1、負得 0、分差絕對值 ≤1e−9 得 0.5。先在同一隻原蜂內平均,再對各蜂等權平均。
4. **參照與門檻。** 同組列最近距離、最短時間、最小等速殘差及 P0 是否共用原生片段 ID,另報同幀差異蜂和組內均勻猜測的命中機會。每個 seed 需在 head 與 tail 各有至少 10 隻不同原蜂提供混合起點,tail 的四特徵得分 ≥0.70 且比最近距離高 ≥0.10。每折至少 2/3 個 seed 達標,兩折均過才另登記從未確認身分的起點開始的端到端驗證。

GT 診斷直接從標註的連續實框找相同時空範圍的真返回及異蜂競爭。每段影片只算一次,供該段的三個 seed 共用。GT 與偵測框的形狀和時間可不同,兩者涵蓋數沒有固定大小關係。正式成績衡量已確認身分的候選池內排序,尚需另案測量完整追蹤的窗誤差、事件 F1,以及對留出蜂箱的效果。

## 從來源偵測、MOT 與 GT 重建

以下命令使用同一個專案根目錄與 `$source108`。另需 §108 來源 tail 的完整 P0 軌跡,由 `$old/source/tail_mot/` 讀入,以及 `runs/` 下相應的偵測快取、head 原生 MOT、prepared dev 與 val GT 和專案的入口多邊形。程式用 §§105、108、113 的既有來源檔指紋核對輸入,其中來源 head 的偵測快取指紋首次記於 §114。head 的 GT 已在 prepare 階段修正交接 ID,tail 沿用原標註,兩者再依官方規則切開超過 25 幀的間隔。

```powershell
$out = 'D:/beetrack_runs/study114_reproduction'
$old = 'D:/beetrack_runs/study108'
$prereg = 'fc1ce10a56479b70bc43e4d486788b610b3078f4'
$common = @('--root', '.', '--run', $out, '--study108-root', $old, '--source108', $source108)
uv run --offline --no-sync python tools/study114/run114.py @common preflight
uv run --offline --no-sync python tools/study114/run114.py @common freeze --prereg $prereg --workers 2
```

`preflight` 的 `source_inputs.json` 指紋為 `39b2a95c1c85acebf11194967a495e536ddfb893b96f28c473d35f17a2672ecd`。`source_inventory.json` 同時封存六個 head 與六個 tail 的純預測候選。確認整份指紋後才以 head 的標註建立六組擬合紀錄。若某 seed 的 head 未達最低蜂數,該格保留蜂數與空係數。核對 `models.json` 的全體指紋之後才開 tail 的標註,用原先封存的候選與模型寫出 `results.json`。

```powershell
$freezeSha = '691a2b14d8ce2b5f31acf8248824a92cba87c895333f33ebbe8cdb7481809521'
uv run --offline --no-sync python tools/study114/run114.py @common fit --freeze-sha $freezeSha
$modelsSha = '4f6f58929eb1643a2c84b21d219733779e8db9ddc44041957f5968b36b53e989'
uv run --offline --no-sync python tools/study114/run114.py @common score --freeze-sha $freezeSha --models-sha $modelsSha
uv run --offline --no-sync python tools/study114/release114.py --root . pack $out "$out/study114_replay.json.gz" --freeze-sha $freezeSha --models-sha $modelsSha --source108 $source108
```

持有原始快取及完整 MOT 時,可用下例逐列核對一個 head 的候選。另一套程式以原始 prepared GT 重建所有 head/tail 的標籤、候選池與排名。合成檢查涵蓋重疊框、切軌、下一次改側、同分及每蜂等權的計算。

```powershell
uv run --offline --no-sync python tools/study114/verify_candidates114.py self-check
uv run --offline --no-sync python tools/study114/verify114.py self-check
uv run --offline --no-sync python tools/study114/verify_candidates114.py verify "$out/source/33_0_head.json.gz" runs/dets/loho_no33_s0/dev/BEE24-35-head.npy runs/trackers/dev/q_lk_loho_no33_s0/data/BEE24-35-head.txt beetrack/zones.json
$resultsSha = '53929dd633fee86c21dd96d0e1b35aba17e0b38c9cc540e487d499fa7de5c134'
foreach ($part in 'head','tail') {
    uv run --offline --no-sync python tools/study114/verify114.py raw --root . --run $out --part $part --freeze-sha $freezeSha --models-sha $modelsSha --results-sha $resultsSha --labels
}
```

## 重製研究圖

從 repo 根目錄執行。需可顯示繁體中文的字型。

```powershell
uv run --offline --no-sync python tools/study114/plot114.py --output docs/study114_return.png
```

[查看研究圖](../../docs/study114_return.png)