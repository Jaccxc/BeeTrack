"""Rebuild §115 reference tables solely from already-public §103, §108 and §111 archives."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
from statistics import mean
import sys

HERE = Path(__file__).resolve().parent
DEFAULT_REPO = HERE.parents[1]
TRACKERS = ('P0', 'ByteTrack', 'calibrated ByteTrack', 'TOPICTrack selected')
TAUS = ('0.5', '0.3')


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def dump(path, record):
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def expected(role):
    for fold in (33, 35):
        for part in (('head', 'tail') if role == 'source' else ('head',)):
            for base in (0, 3, 6):
                yield f'{fold}_{base}_{part}', fold, (68-fold if role == 'source' else fold), part, base


def build(repo):
    import numpy as np
    sys.path.insert(0, str(repo))
    from tools.study103.replay import bins, error
    from tools.study104.event_overlap import replay_arm

    hashes = {}
    def read(filename):
        raw = (repo / 'docs' / filename).read_bytes()
        hashes[filename] = digest(raw)
        return json.loads(gzip.decompress(raw) if filename.endswith('.gz') else raw)

    old = read('study103_results.json')
    tracker_graphs = read('study111_tracker_graphs.json.gz')
    rows = {(r['hive'], r['seed'], r['method']): r for r in old['cases']}
    graphs = {(r['hive'], r['seed'], r['method']): r for r in tracker_graphs['cases']}
    if len(rows) != 60 or len(graphs) != 24:
        raise ValueError('Historical target coverage differs')

    def event_stats(pred, gt, edges):
        arm = replay_arm(dict(predictions=pred, gt=gt, edges=edges))
        tp = len(arm.matches)
        return dict(tp=tp, fp=len(pred)-tp, fn=len(gt)-tp, f1=2*tp/max(1, len(pred)+len(gt)))

    def close(a, b, tolerance=1e-8):
        if abs(a-b) > tolerance:
            raise ValueError(f'Historical W mismatch {a} vs {b}')

    all_cases = {}
    for role in ('source','target'):
        archive = read(f'study108_{role}.json.gz')
        for key, fold, hive, part, base in expected(role):
            group = dict(clip=f'BEE24-{hive}-{part}', fold=fold, seeds=list(range(base,base+3)), methods={})
            for method in TRACKERS:
                if role=='source' and method!='P0' or method=='TOPICTrack selected' and base!=0:
                    continue
                counts, native = [], []
                for seed in group['seeds']:
                    seedkey = f'{fold}_{seed}_{part}'
                    label = archive['labels'][seedkey]
                    if label['clip'] != group['clip'] or label['tag'] != f'loho_no{fold}_s{seed}':
                        raise ValueError('Study108 source/target alignment differs')
                    gt = label['gt']
                    n = label['nframes']
                    gtframes = [f for _,f in gt]
                    truth = bins(gtframes,n)
                    group['gt_in_out'] = [sum(f>0 for f in gtframes),sum(f<0 for f in gtframes)]
                    if method=='P0':
                        orig = archive['masks']['predictions'][seedkey]
                        pred = orig['predictions']
                        stream = [r[1] for r in pred]
                        edges = {'0.5':label['edges05'],'0.3':label['edges03']}
                        item = dict(seed=seed, mot_sha256=orig['mot_sha256'])
                        cell = bins(stream,n)
                    else:
                        original_method = 'ByteTrack' if method=='calibrated ByteTrack' else method
                        gra = graphs[hive,seed,original_method]
                        pred, edges = gra['predictions'],gra['edges']
                        stream = [f for _,f in pred]
                        if tracker_graphs['clips'][str(hive)]['gt'] != gt:
                            raise ValueError('§111 and §108 GT event ledgers differ')
                        if Counter(stream) != Counter(rows[hive,seed,original_method]['events']):
                            raise ValueError('§111 and §103 signed event streams differ')
                        item = dict(seed=seed, tracker=gra['tracker'],mot_sha256=gra['mot_sha256'])
                        cell = bins(stream,n)
                        if method=='calibrated ByteTrack':
                            ratio=rows[hive,seed,method]['scale_ratio']
                            item['scale_ratio_in_out']=ratio
                            cell = cell * [a/b for a,b in ratio]
                    if role=='target':
                        if Counter(old['clips'][str(hive)]['gt']) != Counter(gtframes) or n!=old['clips'][str(hive)]['frames']:
                            raise ValueError('§103 and §108 signed GT or frame count differs')
                        if method=='P0' and Counter(rows[hive,seed,method]['events'])!=Counter(stream):
                            raise ValueError('§103 and §108 native streams differ')
                    item['w_percent']=error(cell,truth)
                    item['counts_in_out']=(cell.sum(axis=0)/30).tolist()
                    if role=='target':
                        close(item['w_percent'],rows[hive,seed,method]['phase']['original'],0.000006)
                        close(item['w_percent'],label['refs'][method],0.0051)
                    if method!='calibrated ByteTrack':
                        item['event']={t:event_stats(pred,gt,edges[t]) for t in TAUS}
                    counts.append(cell)
                    native.append(item)
                average=np.mean(counts,axis=0)
                e3=error(average,truth)
                if method in label['group_refs']:
                    close(e3,label['group_refs'][method])
                group['methods'][method]=dict(seeds=native,e3_w_percent=e3,
                    mean_seed_w_percent=mean(v['w_percent'] for v in native),
                    e3_counts_in_out=(average.sum(axis=0)/30).tolist())
            all_cases[role+'/'+key]=group
    return dict(archives_sha256=hashes,cases=all_cases)


def render(record):
    lines=['# §115 可比的歷史對照','',
           '資料源為 §103 的分向事件與分向校正比例、§108 的來源/目標 P0 事件圖、§111 的目標 ByteTrack 及 TOPICTrack 兩層 IoU 事件圖。只使用既有公開輸出。所有 W 按 25 fps、750 幀窗、30 個整秒起點、部分窗及進/出方向重算。W 越小越好。', '',
           '每列先列三個單模型的 W（按 seed 順序），隨後括號列三份分向窗計數先平均、再對 GT 算出的 E3 W，單位 %。TOPICTrack 的三路平均由公開事件在此逐窗導出。三路平均本身無可供配對的事件身分。', '',
           '|留出折・實評片・seed|P0|ByteTrack 預設|ByteTrack 校正|TOPICTrack 來源選參數|',
           '|---|---|---|---|---|']
    def label(r):
        return f"no{r['fold']} / {r['clip']} / {''.join(map(str,r['seeds']))}"
    def show_w(m):
        return '、'.join(f"{s['w_percent']:.2f}" for s in m['seeds'])+f" ({m['e3_w_percent']:.2f})"
    for k,r in record['cases'].items():
        if k.startswith('target/'):
            lines.append('|'+label(r)+'|'+'|'.join(show_w(r['methods'][m]) if m in r['methods'] else '—' for m in TRACKERS)+'|')
    lines+=['','### 留出 head 事件 F1','','每格為三個單模型的 F1 等權平均（IoU 0.5 / 0.3）。F1 為 0–1。可回查 JSON 中每個 seed 的分子、FP、FN 與 MOT 指紋。校正只縮放兩方向計數，無獨立可配對的校正事件。','','|留出折・實評片・seed|P0|ByteTrack 預設|TOPICTrack 來源選參數|','|---|---|---|---|']
    def show_f1(m):
        return ' / '.join(f"{mean(s['event'][t]['f1'] for s in m['seeds']):.4f}" for t in TAUS)
    for k,r in record['cases'].items():
        if k.startswith('target/'):
            lines.append('|'+label(r)+'|'+'|'.join(show_f1(r['methods'][m]) if m in r['methods'] else '—' for m in ('P0','ByteTrack','TOPICTrack selected'))+'|')
    lines+=['','### 已見來源片','','同一實拍 head 在兩折的角色和偵測器不同。來源列用排除另一蜂箱所訓練的模型。','','|留出折・實評片・seed|GT 進/出|P0 各路 W (E3 W)|P0 平均 F1，0.5 / 0.3|','|---|---|---|---|']
    for k,r in record['cases'].items():
        if k.startswith('source/'):
            m=r['methods']['P0']
            lines.append('|'+label(r)+'|'+'/'.join(map(str,r['gt_in_out']))+'|'+show_w(m)+'|'+show_f1(m)+'|')
    lines+=['','### 比較範圍','',
        '- 三種原生 tracker 以同折、同 seed 的 640 偵測比較。§115 的整數輸出需三份偵測及追蹤。表中每個單 seed 基準各用一份；括號內的 E3 計數各用三份。不同基準的實際 CPU/GPU 耗時未在此測量。',
        '- §95 ByteTrack 的兩個比例按 seed 在已見蜂箱 head+tail 及四支 val（10、15、24、28）估計，分別乘入每個窗的進/出計數。§111 官方事件圖對應未縮放的同一份 ByteTrack MOT。不能把反算的小數計數當作新增事件。',
        '- §99 的 TOPICTrack 參數按每個偵測器在已見蜂箱 val 選取。六個實作名稱記於 JSON，僅 no35 seed 0 為 t30，其餘為 t20，均為 f2/h1 post，即修正 ReID 的第二設定、min_hits 1、TOPIC 開關關閉。seed 3–8 未有此來源選參數的配對結果。不得用目標最佳參數、不同解析度、白板、整片相對誤差或全體合併數取代。',
        '- 這些公開包收錄來源 P0 圖與目標三法圖。來源表的其他 tracker 沒有在這些包中提供同折同片逐 seed 的已核對圖；不以相同蜂箱的另一折目標成績代填。兩片留出 head 和以往已見資料限制外推，各 seed/trio 亦非獨立蜂箱樣本。','',
        '### 公開檔案 SHA-256','']
    lines += [f'- `docs/{k}`　`{v}`' for k,v in record['archives_sha256'].items()]
    return '\n'.join(lines)+'\n'


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--repo',type=Path,default=DEFAULT_REPO)
    cli.add_argument('--out',type=Path,required=True)
    args=cli.parse_args()
    result=build(args.repo)
    args.out.mkdir(parents=True,exist_ok=True)
    dump(args.out/'historical115.json',result)
    (args.out/'historical115.md').write_text(render(result), encoding='utf-8')
