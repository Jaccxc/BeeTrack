# §186 固定端點的計數下界

預登記 01fb9c21a3565c16df7884f66a03d2dc1b9e4073。只用 §178 C 的 33-head、35-head,各 3950 幀,seed 0–2。這是已知開發場景的必要條件診斷,沒有部署新模型。

每段先按官方 >25 幀缺口切開。K 是段數,a / b 是內側首 / 尾數;內側配內側的數目 s 在 max(0,a+b−K) 到 min(a,b)。任意重配後的進 / 出總數是 b−s / a−s。枚舉 s 得到全片分向 L1 的精確最小值,除以 GT 事件數就是 30 起點 W 的樂觀下界。只用進減出的 signed 下界另列。時間、距離、影像和真身分全被放寬,低於門檻不表示可達。

單路平均先對每個 seed 算 W 再平均;其下界也逐 seed 平均。E3 先平均三個 seed 每窗的進出計數再算 W,必須使用三個 s 的整數和求自己的下界,不能套單路平均。沒有三條軌跡合成的物理事件,不報 E3 F1。

evidence.json.gz 包含整數事件、30 起點分向窗、端點數、官方實框事件配對統計及來源指紋。標準函式庫即可重播數值:

```powershell
uv run python <package>/replay186.py <inflated-evidence.json>
```

完整重跑需要既有 pl 偵測快取、修正 GT、入口多邊形和 m_dev_strict.json。從 final_project 執行 run186.py --help;輸出用全新的 D: 目錄。所有六份原生浮點 MOT 與 freeze.json 必須先封存才開始 GT 評分。permutation_fixture.py 以實際 scorer 檢查窮舉及反例。

從 final_project 執行 `uv run python tools/study186/check186.py D:/beetrack_runs/endpoints186` 可從封存原生 MOT 獨立重算切段、事件、窗、官方事件最大配對及精確分數下界,其獨立數值紀錄在 independent.json.gz。需要原生 MOT 才能重做此檢查;攜帶的整數 evidence 以 replay186.py 重播即可。執行 `uv run python tools/study186/check186.py` 可跑合成反例。

manifest.json 列本包內容的 SHA-256;外部封存的 release-v2.sha256.json 也鎖定 manifest 本身。原生 MOT 留在私有現場,不隨套件發布。
