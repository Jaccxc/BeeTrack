# §109 白板場景的跨偵測器事件支持

把 §108 的規則原樣用於留出的白板場景,由主偵測器報出進出事件,另兩個偵測器各自判斷有沒有同時、同地、同方向的預測。主臂固定 seed 0。兩支 val 短片 BEE24-10、BEE24-15 分別有 250、240 幀,各有 13、56 個真事件。三個固定 640 偵測器都是 `loho_noW` 的獨立訓練 seed,沿用 §84 排除白板的連結器。[預登記](https://github.com/Jaccxc/BeeTrack/commit/037ce592d12d80433414efd4219e96de3d2d7620)

官方 IoU 0.5 下,支持規則把主臂的 TP/FP/FN 從 40/7/29 改為 31/5/38。跨片合計的 F1 從 0.690 降至 0.590,保留下來的 31 個 TP 和同數量隨機保留的中位數相同。逐片分向絕對計數誤差從 34.78% 升到 47.83%,同三個 P0 的算術平均 E3 為 30.92%。這兩段短片屬於同一個拍攝場景,長度各約 10 秒,僅能評估事件與整片分向計數。

## 用公開檔案重播

從 repo 根目錄執行。緊湊資料包保存計數所需的有號事件幀、保留 ID、99 次同方向配額抽樣及合法的事件配對邊。這些指令直接從公開資料重算,不需另取原始影片、MOT 或逐框標註。[事件重播包](../../docs/study109_white.json.gz) · [同偵測器的預設 ByteTrack 對照](../../docs/study109_bytetrack.json)

```powershell
$env:PYTHONPATH = (Get-Location).Path + ';' + (Resolve-Path 'tools').Path
$env:CUDA_VISIBLE_DEVICES = '-1'
$out = 'D:/beetrack_runs/study109_reproduction'
uv run --offline --no-sync python tools/study109/score109.py replay docs/study109_white.json.gz --out "$out/white_results.json"
uv run --offline --no-sync python tools/study109/bytetrack109.py replay docs/study109_bytetrack.json
```

配對在每支片內做,才合計 TP、FP、FN。每次隨機對照先在同片、同方向保留與實際規則相同的事件數,再把兩片同次抽樣的 TP 加總,最後取 99 次的中位數。兩個 IoU 的合法邊來自原生 MOT 全部框的逐幀一對一配對與所有同方向跨越。選定事件後重新求最大配對;結果表列官方 IoU 0.5 與附帶 IoU 0.3。E3 先對同片、同方向的三個 seed 計數取平均,再計算與 GT 的差,只產生計數。預設 ByteTrack 的 E3 整片誤差是 65.70%。

## 從完整凍結檔重做

[完整預測清單](../../docs/study109_masks.json.gz)保存六份 P0 MOT 的指紋、全部預測事件的幀與座標、兩個支持者的配對 ID、每個主偵測器的保留及隨機清單。[完整標註配對圖](../../docs/study109_labels.json.gz)保存兩種 IoU 的合法邊及 GT 檔案指紋。下列指令先還原凍結當時的 JSON 位元組,再重播配對及成績。沿用上方的環境設定及 `$out`。

```powershell
$maskSha = '3be646a2092890ecf56abef9508f9a36c281a2c44da7388a2fc7298990aa3ea8'
@'
import gzip
import sys
from pathlib import Path
out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
for part in ("masks", "labels"):
    data = gzip.decompress(Path("docs", f"study109_{part}.json.gz").read_bytes())
    dest = out / f"white_{part}.json"
    if dest.exists():
        if dest.read_bytes() != data:
            raise SystemExit(f"Existing file has a different fingerprint: {dest}")
    else:
        dest.write_bytes(data)
'@ | uv run --offline --no-sync python - $out
uv run --offline --no-sync python tools/study109/freeze109.py replay "$out/white_masks.json"
uv run --offline --no-sync python tools/study109/score109.py score "$out/white_masks.json" "$out/white_labels.json" --mask-sha $maskSha --out "$out/white_results.json" --bundle "$out/white_replay.json.gz"
```

## 從原始追蹤框與 GT 重建

這一層需要 `runs/trackers/val/q_lk_loho_noW_s{0,1,2}/data/` 的兩段原生 MOT、`runs/gt/val/BEE24-{10,15}/gt/gt.txt` 與 `beetrack/zones.json`。預設 ByteTrack 對照使用相同目錄下的 `q_bt_loho_noW_s{0,1,2}`。先記錄預測指紋才開啟 GT,標註程式會核對六個輸入 MOT 的指紋。請將 `$out` 指向本次輸出位置。

```powershell
$prereg = '037ce592d12d80433414efd4219e96de3d2d7620'
$maskSha = '3be646a2092890ecf56abef9508f9a36c281a2c44da7388a2fc7298990aa3ea8'
uv run --offline --no-sync python tools/study109/freeze109.py freeze "$out/white_masks.json" --prereg $prereg
uv run --offline --no-sync python tools/study109/labels109.py "$out/white_masks.json" "$out/white_labels.json" --mask-sha $maskSha
uv run --offline --no-sync python tools/study109/score109.py score "$out/white_masks.json" "$out/white_labels.json" --mask-sha $maskSha --out "$out/white_results.json" --bundle "$out/white_replay.json.gz"
uv run --offline --no-sync python tools/study109/bytetrack109.py predict "$out/bytetrack_predictions.json"
uv run --offline --no-sync python tools/study109/bytetrack109.py score "$out/bytetrack_predictions.json" "$out/white_labels.json" "$out/bytetrack_results.json" --mask-sha $maskSha
```

| 檔案 | SHA-256 |
|---|---|
| 完整預測 JSON,解壓後 | `3be646a2092890ecf56abef9508f9a36c281a2c44da7388a2fc7298990aa3ea8` |
| 完整標註配對圖 JSON,解壓後 | `bc4e2bc2ac7d060ae48f157d89aa9a544cfcd24d2073e331ccb120be6e76a233` |
| 公開事件重播包 `study109_white.json.gz`,壓縮位元組 | `f3cc3cb0498b21eb22673c99957037f50a3b00d4ba3c6ad14ad7b1abc3fa1faa` |
| 完整重算成績 `white_results.json` | `ce6afe71ab55ee61bfa42fe5fdf7315574f758f5508111f70af57a9005f31135` |
| 預設 ByteTrack 對照 `study109_bytetrack.json` | `61bb44edf2311fa39c9405d5f5962965d446a8699ed97a28e22d707cb18d2cad` |
