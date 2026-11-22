# §209 跨部位運動關聯模型

從 BeeTrack repo root 執行,使用既有離線環境:

```powershell
uv run --offline --no-sync python tools/study209/replay_release209.py tools/study209/evidence.json.gz tools/study209/window_support.json --out D:/beetrack_runs/study209.replay.json
```

已實跑的同一入口重擬合四折×三臂=12次source-only logistic,核對來源順序、類別/蜂箱權重、標準化、係數及分數,重算固定slate和合軌預算下的六態選擇。再由匿名有號事件重建全部30相位分向窗,從兩IoU合法圖重算最大TP與F1,池化白板/灰板兩片後算W/F1,核全部36態及投資線。另核代表日弱標的原分數AP、true/sham共同交集與獨立運動歸因條件。

`evidence.json.gz` 只有匿名候選/軌端鍵、幾何與true/sham特徵、弱S/D/U、來源擬合順序/權重、模型、動作、事件、整數窗及合法配對邊。不含自然影像、局部紋理裁片、原始MOT、GT矩形或個別GT身分。重播不能從這些數值重新跑LK或重做GT矩形關聯。主原生獨核另檢查全36完整MOT及兩IoU官方配對;影像獨核只在五個有候選的片各抽一對重做遮罩/GFTT/LK,其餘重播封存點跡,不是全部像素重抽。

`window_support.json` 是在原36態產生後另登記的固定slate/k補充,與主包共用K/GT窗,只另存匿名slate及非零稀疏δ。枚舉active子集、核零δ填額及二項式實體選擇重數、去重完整窗向量,並核U/G/F/X的位置。active型態比例不是模型成功率或所有實體k子集的比例。它不涵蓋J,不改原門檻。最小見證只有窗數的原生/獨立物理核對;F1未重評,零δ也不代表框或事件圖不變。

附原生catalog/features/labels/fit/physical及獨核原碼,portable副本只調整對既有public study203/study208的程式引用;不複製整包208。原始來源與出刊來源SHA分列於`release_manifest.json`,執行來源快照在`verification_sources.zip`。外部原生重建須另備原封存study208 K/input bundle及參照score、允許的六片train原影像/標註和repo zones,不是從此小包取出原素材。各入口接受明確repo/input/output參數,保留fresh D:輸出限制與歷史seal檢查;新的上游重建若seal不同不能靜默當成原封存。此出刊未宣稱重新執行完整像素原生重建。

六片都是既用開發片,每片一個固定成員,不是三人成員E3或新蜂箱盲測。上游CS見過未標註目標畫面。S/D/U只是代表日標註框相容的弱身分訊號;anyD=false不是同蜂證明。G用幾何,F再加兩塊平移差,X再加跨塊/底板/塊內RMS;U是同預算幾何排序,J是本次全幾何互斥參照,不是A的後續F去重或m15重現。詳見`results.md`。
