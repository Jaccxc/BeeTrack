# §191 固定成員的 CS / VT 混組檢查

預登記 8ceeeae5f749e348f4a9b760b65639068bbfe436。只用既有 33-head / 35-head、設計組 seed 345 / 678 的成對 CS 與 VT 學生。每隊每個 seed 恰選一個版本,全 CS、全 VT 加六個混組全部列出,不擇優。六混組與兩純組對每名學生的出場比例相同;平均單路 W 與平均成員官方 F1 必須相等,E3 的差才只剩組隊效果。

公開證據只保留 24 名成員的有號事件及身分、GT 事件、官方事件合法配對圖與來源指紋,不附 24 份完整 MOT。重播需從事件重建 30 起點的分向整數窗,列出八隊 E3、純組/混組均值、全 CS 對照、逐窗恆等式與不讀格內 GT 的改善上限,並由合法圖最大配對重算成員 F1@0.5,密集 33 另報 0.3。E3 沒有單一物理身分 F1。

完整 freeze / score / verify 程式仍保留,使用原官方資料及現有偵測快取可重建所有單路。輸出用全新的 D: 目錄,24 條預測先共同封存才讀 GT;不讀 seed 15–20 或 test。所有公開程式從 BeeTrack repo 根目錄使用,不依賴私人絕對路徑。原生浮點 MOT 為主,文字序列化另行保留供核對。

這個檢查只回答現有兩套訓練配方的組隊效果,不能識別共同標籤本身的因果。兩個 head 和各 seed 組都已參與設計,不當成四個獨立蜂箱。結果在 results.md,兩組方向不一致,固定六學生對全 CS 的能力界不到投資門檻,本輪不配置新訓練。

不讀影片的重播:先解壓 evidence.json.gz,以 `python tools/study191/replay191.py <evidence.json>` 執行。只有 Python 標準函式庫即可從有號事件、合法配對圖和整數計數核對全部 24 成員、四組各八隊 E3、成員曝光、精確 F1 平均、逐窗公式及改善能力界。frozen_member_events 另保留 GT 前凍結的每個事件日期、方向、段 ID 與原軌 ID,與官方算分的事件圖逐一核對。

從 repo 根目錄執行合成檢查:

```powershell
$env:PYTHONPATH = (Get-Location).Path
uv run python tools/study191/score191.py --repo . --self-test
uv run python tools/study191/mix_check.py
uv run python tools/study191/verify191.py
```

manifest.json 包含套件檔案與原始 freeze、score、independent 證據的 SHA-256;外部封存的套件清單也覆蓋 manifest 本身。私有原始 MOT、影片、本機路徑與其他研究分支不隨包發布。

原檔重建用既有 §181 / §184 的 CS/pl 與 VT 640 偵測快取及固定 m_dev_strict.json。若快取尚未產生,需原允許 head 影格與已訓練的對應偵測器權重,以標準 beetrack.detect 格式產生。rebuild191.py 不讀 GT,不需私有 §190 MOT,不訓練或產生新偵測。用全新資料夾完整重建:

```powershell
$out191 = 'D:/beetrack_runs/mix191_reproduction'
foreach ($h191 in 33,35) {
  foreach ($s191 in 3..8) {
    foreach ($r191 in 'CS','VT') {
      uv run python tools/study191/rebuild191.py --repo . --out $out191 --hive $h191 --seed $s191 --recipe $r191
    }
  }
}
uv run python tools/study191/rebuild191.py --repo . --out $out191 --seal-manifest
uv run python tools/study191/score191.py --repo . --out $out191 --prereg 8ceeeae5f749e348f4a9b760b65639068bbfe436
uv run python tools/study191/verify191.py --repo . --out $out191 --verify-sealed
```

重建程式保留正式缺口切段、有號事件、原軌 ID 與原始浮點順序。CS 的一個單格重建已核對 MOT 陣列、文字、事件陣列、事件身分與計數五份產物逐位元相同;全24格完整原檔的獨立驗證亦通過。重建副本的來源/程式收據會有自己的雜湊,不冒充原封存收據。

公開 score191.py 預設用本 repo 的 `tools/study190/evidence.json.gz` 核對 CS 的整數窗、W 與官方 F1,並鎖定該公開證據的 SHA-256。若已有原始 §190 封存,可另帶 `--source190 <原封存目錄>` 加做逐位元核對;完整原始 §190 MOT 不是公開重建的必要依賴。舊現場 freeze 與公開 rebuild 的追蹤及事件結果相同,但後者可用新 D: 路徑,不依賴私人絕對路徑。
