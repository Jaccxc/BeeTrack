# 原始訓練片雙蜂 RGB 來源檢查

來源預登記 adab39fbcef0d7cb3e4507d7934598f30a4db388,標記 [paired-rgb-v1],不佔 §194。只用已允許的25支W/G原始train片,每片最多4對。先按原ID/run及固定最近時刻封樣,再讀前後兩幀影像;時移對照只換影像日期,不換框座標、配對、取樣時刻或名單。整片同ID重幀/官方run別名歧義時停列。

本包只聚焦來源訊號、預登記、原/時移與獨立覆驗。準備碼白名單是 track_choice_source_probe.py、track_choice_features.py、verify_track_choice_source.py;不帶模型擬合、候選軌混合、實體MOT選擇、素材注入或人工影片。features檔是固定描述/門檻函式,不是訓練模型。

公開numeric包含固定名單與SHA-256、選取四框/取樣幀、原與時移的cost/contrast/change/vote、各片/箱/跨界子集的正負棄權分母、片等權結果與兩條主閘門。獨立數值保留180個可量測臂的4×6 RGB均值/母體標準差,可交叉重算兩種接法cost;另6臂不可用,180不等於180次身分投票。原圖、裁圖、個別像素、完整GT軌跡、私人路徑與姓名不隨包公開。

門檻重播須區分crop可讀(valid)與真正能給正負判斷(informative):來源每箱至少8對/3片/25%可判正負、S≥0.20、同名單原S高於時移S。入口交錯解釋還要跨界子集至少8對且S>0。棄權仍留入選分母,可配日期的同名單另報。25%不足必須明列,不能只靠只查8對/3片的coverage旗標宣稱覆蓋充足。

結果見results.md。W原8正1負28棄,G原11正1負44棄,兩箱投票覆蓋及S均未達主閘門。來源及時移數字不能當部署成績或時移影像的身分準確率。

只用公開numeric的標準庫重播:

```powershell
python tools/paired_rgb_source/replay_rgb.py tools/paired_rgb_source/evidence.json.gz
```

它核對25片的排除/合格母群數、固定候選hash排序/每片至多4對、93對的原/時移門檻、180臂描述代價、所有片/箱/跨界分層、日期同名單與雙邊可讀交集、片等權及註冊閘門。這不會重新解碼未公開的像素。完整GT資格與像素描述由獨立驗收碼在原資料上另核對;OpenCV JPEG解碼共用,並非兩套完全不同的解碼器。

合成檢查從repo根目錄執行,不讀真資料:

```powershell
$env:PYTHONPATH = (Get-Location).Path
uv run python tools/paired_rgb_source/track_choice_source_probe.py --self-test
uv run python tools/paired_rgb_source/track_choice_features.py
uv run python tools/paired_rgb_source/verify_track_choice_source.py
```

來源程式的合成暫存與正式輸出使用D:,需已有 `D:/beetrack_runs`。重新建一份來源train包需要原允許25片的影像/GT與repo內入口多邊形,不用任何模型權重或GPU。以下是fresh包及其自身完整獨立驗收,不是驗歷史seal:

```powershell
$freshRGB = 'D:/beetrack_runs/paired_rgb_source_reproduction'
$preregRGB = 'adab39fbcef0d7cb3e4507d7934598f30a4db388'
uv run python tools/paired_rgb_source/track_choice_source_probe.py --repo . --out $freshRGB --prereg $preregRGB
uv run python tools/paired_rgb_source/verify_track_choice_source.py --verify-sealed --repo . --out $freshRGB --prereg $preregRGB
```

fresh輸出必須不存在,驗收另在其independent子目錄落新收據。會驗當前seal/全部檔/來源碼/GT/像素雜湊及完整選樣、描述與聚合,不要求新包冒充歷史收據。沒有原圖的讀者只用第一條numeric命令。

這兩條公共路徑命令已在新的輸出目錄實跑並獨立覆驗:selection、measurements、summary及producer manifest與原封存逐位元一致,獨立numeric_pairs的SHA-256也仍為f5f6e6b16a19b474fb66a9526aaebb8d8b2279d151fe1b13425dc745355aec45。這是第二次重建同來源,不是新樣本。新公開verifier的程式指紋與新驗收收據另存,不冒充歷史驗收器。

本機確實持有歷史原完整封存及原來源圖時,可額外要求其producer seal,並把新驗收放在別處以免覆寫原independent:

```powershell
uv run python tools/paired_rgb_source/verify_track_choice_source.py --verify-sealed --repo . --out $historicalRGB --audit-out $freshHistoricalAudit --prereg $preregRGB --require-producer-seal e7b2a56c93eba882342a12f6b7c95a8cbb85896745f3edebf0406d5d65d1a956
```

兩個路徑變數由持有原封存者明確指定,不是上一段fresh包。預設不會猜本機歷史路徑,不放寬當前包的任何檔案完整性檢查。公開verifier副本只新增通用fresh/選填歷史pin/獨立輸出目錄,原驗收器指紋仍保留在證據。

manifest.json列每個公開檔及上游producer/independent收據。偶合掃描模式的decimal只改等值科學記號,一個候選選取SHA只改等值JSON Unicode escape,整份解析JSON嚴格相等。該完整摘要已從其gt_sha鎖定的允許train GT,用獨立選取函式重算;解碼後所有其他字串仍嚴格掃描,數值則核欄位型別並重播。原圖、裁圖及個別像素均未公開。
