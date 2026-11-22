# §210 交錯觀測偏移關聯:數值重播

從BeeTrack repo root執行,使用既有離線環境:

```powershell
uv run --offline --no-sync python tools/study210/release210.py tools/study210/evidence210.json.gz tools/study210/all_disjoint210.json --repo . --out D:/beetrack_runs/study210.replay.json
```

此入口已在相同public layout實跑。作者權威schema及主replay是`replay210.py`,投影器`pack210.py`引用同一schema。數值包有六片245個匿名候選、G6/X2/反號X2、弱S/D/U/co-visible、來源order/weights、模型及36態匿名事件/整數窗/兩IoU合法圖。可重算四折G擬合、full來源校正status/零化、分數、slate/固定k與N/G/X/M/ALL,並沿public study209的既有核心重算事件→全部30相位、最大TP/F1及W。W/G先池化整數再計分,不是三成員E3。此run實際為4個G擬合及4個缺類別零校正,不是8個有效校正擬合,也不是§209的12次數字。

`all_disjoint210.json`是另一份登記的全圖摘要,以主包SHA綁定同一K/GT窗及δ,不複製原軌。主replay核全部eligible非零δ的ID集合=active slate,且active互不共run,再枚舉所有active子集及各active預算/去重向量。惰性邊可刪去不改計數,反之每個active子集可行,故這是原凍結eligible圖一次性互斥合軌的完整計數支援。它與原固定k容量分開,不擴張至共軌多步、重開候選或改框,更不給F1界。

兩head最小count見證另有原raw重建/全原框保留/官方事件與W獨核,沒有新增GT配對或F1評分。35最佳選集等於主36態ALL,本入口核其選集與分子一致,所以可明確引用原ALL的F1;這不是對新最優見證重評F1。33最小count見證未另評F1,不從δ推得。W/G以K達此族最小W,不表示所有同窗配置的事件圖都相同。

公開重播不含自然影像、GT矩形/個別GT身分或原MOT。它不能從G6/X2重新抽取原框支架/offset或指派GT owner。弱標沿§208已核可靠owner,主原生獨核另核本節標籤規則與36份完整MOT,以有序原框cache重建官方雙IoU圖/最大配對及事件窗。原始與portable來源SHA分列在`release_manifest.json`,原執行source/receipt快照在`verification_sources.zip`。

附catalog/feature/labels/fit/physical/capacity/global/independent-source程式;portable副本只改到既有public study203/208/209的code引用,不搬舊大型證據。外部原生重建需要另備原封存§208 input/可靠owner、六片原標註、zones及上游K快取。歷史seal、fresh D:輸出及顯式repo/input旗標保留;新的source/code seal不能靜默當原封存。本出刊未重跑完整原生pipeline。六片皆為既用開發片,每片單一固定成員,上游CS見過未標註目標畫面,不是端到端新蜂箱盲測。
