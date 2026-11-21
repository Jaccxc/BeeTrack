"""Registered CPU-only driver: seal all 12 streams before any GT consumer."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import subprocess
import sys

from freeze190 import PREREG, source, sha, save


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--repo', required=True, type=Path)
    ap.add_argument('--out', required=True, type=Path)
    args = ap.parse_args()
    repo, out = args.repo.resolve(), args.out.resolve()
    if out.drive.upper() != 'D:':
        raise ValueError('D: output required')
    out.mkdir(parents=True, exist_ok=False)
    lock = out / 'RUNNING.lock'
    with lock.open('x') as f:
        f.write(str(os.getpid()))
    script = Path(__file__).with_name('freeze190.py')
    codes = {p.name: sha(p, code=True) for p in (script, script.with_name('candidates190.py'))}
    def run(h, k):
        log = out / f'h{h}_s{k}.log'
        command = [sys.executable, str(script), '--repo', str(repo), '--out', str(out),
                   '--prereg', PREREG, '--hive', str(h), '--seed', str(k)]
        with log.open('wb') as handle:
            p = subprocess.run(command, cwd=repo, stdout=handle, stderr=subprocess.STDOUT)
        if p.returncode:
            raise RuntimeError(f'h{h}_s{k} exited {p.returncode}; see {log.name}')
        print(f'FROZEN h{h}_s{k}', flush=True)
        return h, k
    with ThreadPoolExecutor(max_workers=3) as workers:
        futures = [workers.submit(run, h, k) for h in (33, 35) for k in range(3, 9)]
        failures = []
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as exc:
                failures.append(str(exc))
        if failures:
            raise RuntimeError('; '.join(failures))
    records = []
    for h in (33, 35):
        for k in range(3, 9):
            folder = out / f'h{h}_s{k}'
            receipt = folder / 'receipt.json'
            r = json.loads(receipt.read_text())
            spec = source(h, k)
            assert r['prereg'] == PREREG and all(r[name] == value for name, value in spec.items())
            assert (folder / 'SEALED.sha256').read_text().strip() == sha(receipt)
            assert all(sha(folder / name) == digest for name, digest in r['files'].items())
            assert sha(repo / r['source_det_path']) == r['source_det_sha256']
            assert all(r['code_sha256'][name] == digest for name, digest in codes.items())
            records.append(dict(hive=h, seed=k, directory=folder.name, receipt_sha256=sha(receipt),
                                candidates=r['candidates'], native_mot_rows=r['native_mot_rows'],
                                source_det_sha256=r['source_det_sha256']))
    assert all(sha(script.with_name(name), code=True) == digest for name, digest in codes.items())
    save(out / 'manifest.tmp', dict(schema='emerge190-all-frozen-v1', prereg=PREREG,
                                   complete=True, streams=records, code_sha256=codes))
    (out / 'manifest.tmp').rename(out / 'manifest.json')
    (out / 'ALL_FROZEN.sha256').write_text(sha(out / 'manifest.json') + '\n')
    lock.unlink()
    print('ALL12 FROZEN ' + sha(out / 'manifest.json'), flush=True)


if __name__ == '__main__':
    main()
