# 七幀事件定位頭能補回多少進出計數

§119 在兩支各 1,000 幀的來源影片測量一個已知答案的補報器。它保留歷史 P0 的所有軌跡,只允許加入能用同一隻蜂的兩個真框見證、原生事件圖沒有任何合法配對邊的穿越。每個種子都選擇讓 30 秒窗誤差 W 最低的補報子集,再把新增的實框放回原片所有軌跡,重算官方事件配對。這個已知答案的上界可在訓練前判斷這種補報有多少空間。[完整預登記](https://github.com/Jaccxc/BeeTrack/commit/f713b556be94b9eb669ae2565c1532a79cf1b854)

| 來源影片 | 正式 GT 事件 | 七幀可用同蜂兩端表示 | 三個 seed 的最大 W 改善(百分點) | 最佳 W 見證下的雙端同蜂新增數 | 是否值得進入來源訓練 |
| --- | ---: | ---: | --- | --- | --- |
| 35-tail,no33 | 37 | 37 | 2.70 / 8.47 / 10.63 | 1 / 4 / 4 | 本片通過 |
| 33-tail,no35 | 50 | 50 | 2.00 / 0.00 / 0.00 | 2 / 0 / 0 | 本片未過 |

在 33-tail 的 seed 1 與 2,原生事件配對分別漏掉 12 與 13 隻完全無邊的蜂。把其中任何一隻的正確事件單獨補入,都會增加分向窗誤差。兩片的正式 GT 事件全都能在七幀中取得相鄰兩個真框;對 ±3 幀局部改側使用短片首尾標籤,在 35-tail 與 33-tail 分別仍多報 15 與 61 次。三個 seed 共用同一批蜂。依預登記的雙來源條件,本版補報器停在這項來源容量分析。[實驗紀錄](../../docs/experiment_log.md)

## 用公開的重播包重算

在已安裝鎖定依賴的 repo 根目錄執行。資料包約 99 KiB,含 6 組原生事件、全部真實改側、每隻正式蜂在候選時域的觀測、24 組選擇與原始證據摘要。重播逐一列舉所有符合規則的實框對,獨立求最低 W,再用封存的完整合法邊與兩端所有者重算官方事件匹配及各來源判讀。[重播資料包](../../docs/study119_replay.json.gz)

```powershell
$env:PYTHONPATH = (Get-Location).Path
$env:CUDA_VISIBLE_DEVICES = '-1'
$tool = 'tools/study119'
uv run --offline --no-sync python "$tool/replay_capacity.py" replay docs/study119_replay.json.gz --sha256 b8baf33897182108a3e977f8d589f91b5dbee1d17a8792d5c8a1b00392916df4 --history docs/study108_source.json.gz --table --report runs/study119/replayed119.json
uv run --offline --no-sync python "$tool/plot119.py" --report runs/study119/replayed119.json --out runs/study119/study119_event_capacity.png
```

每個來源均比較四種補報方式。正式時刻取一段軌跡最後一次淨方向改側的抵達幀;其寬鬆對照允許抵達在前後五幀內,每隻蜂仍最多選一次。兩者分別在原生完全無邊的蜂及官方固定最大配對漏掉的全部蜂上計算。寬鬆對照可能有多個可行抵達時刻。每個 W 的最佳子集整片共用,並可按真蜂 ID 追查。不同子集可能同時達到最低 W,而在全幀一對一的身分競爭下產生不同的 F1;報告的是已保存見證的真實配對。

## 用原始 GT 與實體軌跡重新檢查

若有專案的 `runs/dets`、`runs/gt` 與 §108 的來源輸出,將下方的 `$prior` 設為內含 `source/tail_mot` 的目錄,`$out` 設為本次輸出目錄。原始分數檔保存每個新增 ID 的完整兩個框。實體驗證另外讀取原始 GT、6 份原生與 24 份擴增十欄 MOT,重建完整逐幀 IoU 競爭、每一個新增框的歸屬與每個時間窗。僅持有公開包時,驗證從已封存的候選邊和所有者開始。[研究程式](run_capacity.py) / [逐幀獨立核驗](verify_physical.py)

```powershell
$prior = 'outputs/study108'
$out = 'outputs/study119'
$registration = 'f713b556be94b9eb669ae2565c1532a79cf1b854'
uv run --offline --no-sync python "$tool/run_capacity.py" --prior $prior --output $out --prereg $registration
$index = (Get-FileHash "$out/index.json" -Algorithm SHA256).Hash.ToLower()
uv run --offline --no-sync python "$tool/verify_physical.py" $out --prior $prior --index-sha $index --output "$out/verified.json"
uv run --offline --no-sync python "$tool/replay_capacity.py" pack $out "$out/study119_replay.json.gz" --index-sha $index --history docs/study108_source.json.gz
```

原實驗成績索引 SHA-256 為 `17488e8a8628effa2d351083aa39c864540965b9a22a5a66ebef2ed0028a727f`。獨立實體驗證收據為 `6aaff4e4070a33fe0b650a3971a8464cb87d0d83a25c83760893c3ed515198da`。固定時刻使用整數窗界;有同蜂多個候選時刻時,用有互斥條件的整數規劃,再以見證逐窗核對目標值。可用以下指令檢查圖片邊界、逾 25 幀重新切段、同時多蜂的配對平手、同一隻蜂多個可行時刻、同 W 不同 F1,以及十欄輸出往返。

```powershell
uv run --offline --no-sync python "$tool/independent.py"
uv run --offline --no-sync python "$tool/check_files.py"
uv run --offline --no-sync python "$tool/replay_capacity.py" self-check
```
