# §193 配額與身分排序

預登記 f161c64a6310c663d6cb079e83abee335d90e324。只用已知六段影片、四個留出蜂箱、42 名成員與 60 份成員影片。長片 33 / 35 的 CS seed 3–14 用 m_dev_strict;W / G 的舊留一偵測器 seed 0–8 分別用乾淨 m_noW / m_noG。先共同凍結原生預測輸入,每個外折只讀其餘三箱標註,四折全部部署預測封存後才開目標算分。

四臂 B 全留、I 逐事件一次成本、Q 聯合配額+固定 ID、R 相同配額+來源身分排序。Q/R 必須逐成員、秒、方向使用相同 K,由實際 MOT 重算官方事件。O.5 只是在固定配額及合法事件圖上的 F1 上限,不作部署選擇;33 的同臂另列 IoU 0.3。E3 每窗先平均三成員計數再取誤差,沒有 E3 身分 F1。

公開數值證據只保留有號事件、留出預測的保留 ID、GT 計數、兩 IoU 合法配對圖、每組 W/E3/F1、配額、最大流/最小割和來源指紋。原始浮點60份輸入與60×5份實體 MOT 留在私有現場,不重複打包。公開重播需驗證事件與整數窗、Q/R 同秒 K、匹配圖最大配對及配額 F1 上限;原檔重建、模型擬合與物理軌跡實現入口仍保留相鄰程式碼。

重播必須同時保留來源支持的限制:長片完整30秒窗只有另一支長片提供同長來源,W/G 的短片不能當成同長獨立影片。逐折列同長來源數、來源權重、有效交換 pairs、可改選的格與配額分母;無可分辨配對或無配額刪除時,明列退化情況。來源標註隔離只適用本節新頭,底層 CS 曾使用目標無標註影片,不能宣稱整套系統完全隔離。

歷史參照見 references193.md。同影片/種子組可對齊的原640 TrackletLink與校正ByteTrack是背景,跨偵測配方的總差不當成新模型功勞。W/G 舊 m_dev_strict 表有洩漏,本節必須以乾淨linker同軌B為主對照,不挪用13/16段的其他集合數字。

結果在 results.md。主評分對全60格實現B/I/Q/R/O並算官方匹配,獨立核對全60格的事件圖與分數。獨立抽測範圍為四折第一片的成本、每折一格物理重現×五臂,未把共用實現器及次要W/G片的成本稱成完全獨立全量重算。

實作勘誤:fold35 第一次在產生任何模型或預測產物前,被舊 score117 的非負框驗證擋住。BEE24-24 的 996 筆官方 GT 中有 10 列負寬或負高。§193 專用讀取改為保留官方原樣完整列與順序,沒有刪除、改值或重新標註。核對官方 TP、窗誤差及五支長短片首成員後重試;正式指標、模型與 GT 封存規則仍照預登記。這是讀取相容性修正,不是修改評分資料。

compatibility193.json 保留這次停止/修正、五片核對和48個來源成員檢查的無路徑數值摘要;完整來源診斷原證據的SHA-256另列manifest。

純標準函式庫重播不讀影像、GT原檔或完整MOT:

```powershell
python tools/study193/replay193.py tools/study193/score193.json.gz
```

從repo根目錄以既有環境重建全部原生輸入、來源擬合及正式算分。需要§189同一批允許的CS/pl與留一偵測快取、修正GT、入口多邊形和三份雜湊鎖定的linker;不需要私有t189軌跡快取或機器絕對路徑。使用全新D:資料夾:

```powershell
$env:PYTHONPATH = (Get-Location).Path
$run193 = 'D:/beetrack_runs/budget193_reproduction'
$pre193 = 'f161c64a6310c663d6cb079e83abee335d90e324'
uv run python tools/study193/freeze193.py --repo . --out $run193 --prereg $pre193
uv run python tools/study193/field_train193.py --run-field --repo . --root $run193 --prereg $pre193
uv run python tools/study193/field_score193.py --repo . --frozen "$run193/frozen" --predictions "$run193/deploy" --out "$run193/score" --prereg $pre193
uv run python tools/study193/verify193.py --root $run193 --verify-sealed
```

公開重建的來源收據與程式指紋會不同,仍完整驗證60格scope、原偵測/軌跡/段身分雜湊與乾淨linker,再封四折預測。公開scorer以已公布的35/33基準核對B;若有舊§189數值檔可另帶 `--baseline189` 做額外背景核對,不是必需的私有依賴。部署模型的來源/頭規格不變。

合成檢查入口包含 `freeze193.py --repo . --self-check`、`source193.py --self-test`、`field_train193.py`、`field_score193.py --self-check`、`verify193.py` 及 `e2e_selfcheck.py`,均從repo根目錄使用上述PYTHONPATH,不跑真資料。

公開freeze副本已單獨原生重建cs35_s4:48869列軌跡、s1、135個事件的身分/日期/方向/框/十特徵、秒計數和30起點窗均與封存逐值相等。這是入口相容性核對,沒有新增偵測、讀GT或跑新設定。

訓練及完整原檔驗證均要求明確 `--root`。verify193.py 的舊現場附加核對只在提供 `--historical-cache <本地舊cache>` 或 `--baseline189 <本地舊數值檔>` 時執行;公開重建不需要這兩份私有歷史輸入,但仍驗證新封存的全部原生輸入、linker與程式雜湊。

公開score證據只將六個GT收據中的機器絕對路徑改為repo相對路徑,其餘數值/圖/事件/指紋不變。原始score seal及公開檔雜湊分開保留。

models193.json.gz只去掉訓練中可重建的大型診斷列表,保留可部署的配額近鄰模型、身分權重與來源收據;support193.json.gz保留來源有效數、各片交換pair計數及診斷指紋。數字若偶然含隱私掃描模式,僅改等值科學記號,解析後整份JSON嚴格相等。manifest.json列所有內容與上游seal的SHA-256,外部封存清單另涵蓋manifest本身。
