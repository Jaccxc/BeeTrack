# 來源連續遮罩素材預製

這是登記在「下一步」的來源素材準備,不佔 §194 或其他實驗節號。預登記 e90ae7df2d8d07eb0924f030c6f46834be8a5a4f,四項原登記保留。它只問來源素材能否交付,不算任何偵測、追蹤、事件 F1 或窗誤差,也不決定模型採用。

只用已允許的原始 train W/G。按原始蜂 ID 的完整官方軌跡取17幀穿越片段,分箱與方向四格,以預登記ASCII SHA-256排序取每格兩隻、來自不同影片。先封存資格分母、排除理由和選取清單,再逐幀用現有CPU MobileSAM原框提示;不足或失敗不遞補。

真mask、同輪廓背景空貼片及全透明像素控制均只留本地。背景取樣在同片17幀外,按完整時間軸等距至多15幀,遮掉全部標註蜂框後取像素中位數,每個alpha像素至少需3個未遮樣本。機械門檻是17幀全部成功,至少15幀同時達mask框IoU≥0.5且不碰其他蜂框;同格兩隻都過才夠該方向的雙蜂素材。目視混蜂/邊界/時序閃爍註記另列,不替代機械門檻。

公開包只保留selection、逐蜂/逐幀機械數值表、原始證據指紋及可驗證程式。沒有原生像素、遮罩、接觸圖板或可按ID回放的原片。重做像素核對需要按原資料與權重的使用條件在本機取得原檔,本包不重新散布它們。

prepare_assets.py 是來源選取/素材入口,donor.py 提供必要分割和背景處理,verify_assets.py 獨立重算選取與驗收。其他下游注入、偵測、配對或模型工具不在這次包裡。正式像素輸出要求明確的新D:路徑,單CPU,不覆寫既有封存;若公開副本保留此要求,命令會清楚列出。程式從repo根目錄使用相鄰碼,不依賴私人機器目錄。

結果在 results.md。原八隻與17張保守背景補算均已獨立逐像素驗收,主控目視註記另存。四格均未達機械素材門檻,不補挑或手修,沒有任何下游模型成績。

一般讀者只需公開numeric與Python標準函式庫,不讀任何原圖即可重播兩次的選取與數值表:

```powershell
python tools/source_donor_assets/replay_assets.py tools/source_donor_assets/evidence.json.gz
```

它核對51個資格記錄的固定雜湊排序、不同影片選取、136幀框IoU/碰矩形/支援門檻、逐蜂和四格結論,並交叉核對兩份獨立驗收的逐幀數值。補算另核原17張指紋/原點、15日不變、非正標註日全圖零支援規則與0/17結果。這是數值/收據重播,不是從未公開的像素重新判斷分割是否指到同一隻蜂。

需要原資料與現有權重的入口另列。它們要求BeeTrack repo根目錄、原允許的train影格與標註、既有 `mobile_sam.pt`、既有環境。沒有下載或安裝步驟。正式輸出必須是全新的D:路徑,單CPU,不覆寫;合成小檔暫存也使用 `D:/beetrack_runs`,執行前需已有此目錄。

合成檢查 (不讀真資料):

```powershell
$env:PYTHONPATH = (Get-Location).Path
uv run python tools/source_donor_assets/prepare_assets.py --repo . --self-test
uv run python tools/source_donor_assets/donor.py --self-test
uv run python tools/source_donor_assets/verify_assets.py
uv run python tools/source_donor_assets/recheck_background.py --self-test
uv run python tools/source_donor_assets/verify_assets_supplement.py
```

重新製作來源素材與獨立驗收的入口 (將weights改成已有檔案):

```powershell
$freshAssets = 'D:/beetrack_runs/source_assets_reproduction'
$weightsAssets = 'D:/models/mobile_sam.pt'
$preAssets = 'e90ae7df2d8d07eb0924f030c6f46834be8a5a4f'
uv run python tools/source_donor_assets/prepare_assets.py --repo . --out $freshAssets --weights $weightsAssets --prereg $preAssets
uv run python tools/source_donor_assets/verify_assets.py --field --repo . --assets $freshAssets --out 'D:/beetrack_runs/source_assets_reproduction_verify' --prereg $preAssets
```

fresh製作會記錄當時origin/程式指紋,其來源seal不冒充原封存。上面這一組不接補算器;尚未聲稱fresh像素/收據可替代歷史封存。

`recheck_background.py` 只接受本次已封存的原manifest `96561379…015119` 與selection `9c5d63d5…06c5e8`,保留固定G18原遮罩/15日。若本機確有該原封存像素,才以明確路徑執行:

```powershell
uv run python tools/source_donor_assets/recheck_background.py --repo . --original $originalSealedAssets --out $newRecheckRoot --prereg 08bdcb78f8ce482f9c028d6c8e5122c2c4d60384
uv run python tools/source_donor_assets/verify_assets_supplement.py --field --repo . --original $originalSealedAssets --supplement $newRecheckRoot --out $newRecheckVerifyRoot --prereg 08bdcb78f8ce482f9c028d6c8e5122c2c4d60384
```

這三個變數必須由持有原封存的使用者明確指定,不是上一段fresh結果。公開包沒有該原像素;一般讀者用第一條numeric命令即可驗兩次數字,不需這兩條現場命令。沒有放寬原seal的完整性檢查。

manifest.json列所有公開檔與原/補算/獨立/目視證據的雜湊;外部封存清單另涵蓋manifest本身。個別十進位數若恰巧撞隱私掃描模式,僅換等值科學記號,解析後JSON嚴格相等。沒有像素、遮罩、背景陣列或私人機器路徑。
