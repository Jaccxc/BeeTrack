# §190 單框端點修復的必要條件

預登記 aeaf95b76e99baef940dddbdbdb91d33611bf3dd。僅用已公開設計組 CS + C 0.40 的 33-head / 35-head,各 3950 幀,seed 3–8。這是已知開發場景的診斷,原軌固定,每個合格端點最多補一個觀測。不得把 GT 見證當成部署成績。

從 BeeTrack repo 根目錄,使用既有 uv 環境。模組放在 tools/study190,相鄰匯入不依賴私人目錄。candidates190.py 定義純預測候選;freeze190.py 封存單格;run_freezes190.py 統一凍結 12 格,全部完成才讓算分讀 GT。labels190.py 核對同蜂標註、漏檢及無衝突的實框見證;bounds190.py 枚舉同段兩端的聯合選項,計算每窗分向包絡;score190.py 重算物理事件、官方 F1、窗誤差及 E3。independent190.py 獨立枚舉合成軌跡,檢查包絡。

完整重跑需要既有設計組快取、GT、入口多邊形和 m_dev_strict.json,使用新的 D: 輸出目錄,不覆寫任何封存:

```powershell
$env:PYTHONPATH = (Get-Location).Path
uv run python tools/study190/run_freezes190.py --repo . --out D:/beetrack_runs/emerge190_reproduction
uv run python tools/study190/score190.py --repo . --out D:/beetrack_runs/emerge190_reproduction --prereg aeaf95b76e99baef940dddbdbdb91d33611bf3dd
uv run python tools/study190/independent190.py --repo . --out D:/beetrack_runs/emerge190_reproduction --verify-sealed
```

合成檢查不讀現場資料:

```powershell
$env:PYTHONPATH = (Get-Location).Path
uv run python -c "import runpy; from beetrack.evaluate import by_track; runpy.run_path('tools/study190/candidates190.py')['self_test'](by_track)"
uv run python tools/study190/labels190.py
uv run python tools/study190/bounds190.py
uv run python tools/study190/score190.py
uv run python tools/study190/independent190.py
```

純 Python 標準函式庫重播不需要影片或偵測快取。先解壓 evidence.json.gz,再執行 `python tools/study190/replay190.py --evidence <解壓的 evidence.json>`。重播核對整數事件與窗、官方最大事件配對、包絡與下界、兩組 E3 及事前守門。

先保存原始浮點軌跡、正式缺口切段對應、純預測候選和來源指紋。12 格各自 receipt 與總 manifest 封存齊全後才讀目標 GT。官方事件 F1 必須從完整實框重新匹配;33-head 另報 IoU 0.3,不取代官方 0.5。E3 每窗先平均三個成員的進出數再取誤差,分 345 / 678 報告,沒有 E3 身分 F1。

第一個包絡放寬身分、位置及影像限制;第二個包絡限於可靠同蜂、未被偵測配到的標註實框。這些是 W 的樂觀下界,不是可達軌跡。GT 缺列或重用 ID 記未知,不宣稱像素中沒有蜂。實框見證才重新產生 MOT,維持原列前綴、原排序及正式段 ID,衝突提案全部棄權。

結果在 results.md。12 格均通過獨立的候選、物理事件、官方配對、窗誤差、兩組 E3、包絡與見證核對。manifest.json 鎖定所有其他套件檔案,外部封存的套件 SHA-256 清單也涵蓋 manifest 本身。只發布 emergence 主線,不包含其他私人工具或未開封資料。

附加的 geometry190.py / geometry.json.gz 是候選凍結後、主控讀算分結果前另記的純預測幾何診斷,不修改兩個預登記界或採用門檻。只有搜尋圓盤整個落在原端點同側的端點才被排除,其餘仍放寬所有 GT 限制;1 像素是保守幾何 guard,不是校準的不確定度。先解壓 sidecar,以 `uv run python tools/study190/geometry190.py --replay <geometry.json>` 重播。完整產生指令為 `uv run python tools/study190/geometry190.py --root D:/beetrack_runs/emerge190_reproduction --out D:/beetrack_runs/emerge190_geometry_reproduction`。圓盤碰到邊界不代表能找到有用觀測;端點數跨 seed 重複,不當成不同蜂隻數。

看到幾何逐窗界之後才加做 coherent190.py / coherent.json.gz。它枚舉每段固定選一個事件、全部 30 起點共用同一組選擇的精確單路最優,不算 E3。解壓後執行 `uv run python tools/study190/coherent190.py <geometry.json> --replay <coherent.json>`;獨立實作 `python tools/study190/exhaust190.py <geometry.json>` 會從同一小份幾何證據重算各格最優。這是理想修復族的事後數學診斷,不保證可部署或有實框/F1 改善。

公開 geometry JSON 僅將兩個十進位數改用等值科學記號,避免數字恰巧包含隱私掃描模式。解析後整份 JSON 嚴格相等,數值與全部窮舉分子不變。manifest 同時保留 original_geometry_sha256 與 public_geometry_sha256;coherent 的來源與程式指紋按公開版本重新生成並完整重播,沒有關閉完整性檢查。
