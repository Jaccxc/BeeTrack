"""Exact-list deterministic ZIP_STORED transport for numeric gzip shards."""
import json
from pathlib import Path
import re
import shutil
import tempfile
import zipfile
from stream202 import file_sha


def allowed_files(manifest):
    names = ['manifest.json']+[e['file'] for e in manifest['shards']]
    assert len(names) == len(set(names))
    assert all(re.fullmatch('[0-9a-f]{64}\\.json\\.gz', name) for name in names[1:])
    return sorted(names)


def make_zip(root, target):
    root, target = Path(root), Path(target)
    assert not target.exists()
    manifest = json.loads((root/'manifest.json').read_text())
    names = allowed_files(manifest)
    by_name = {e['file']: e for e in manifest['shards']}
    with zipfile.ZipFile(target, 'x', compression=zipfile.ZIP_STORED, allowZip64=True) as out:
        for name in names:
            source = root/name
            if name != 'manifest.json': assert file_sha(source) == by_name[name]['sha256']
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED; info.create_system = 3
            info.external_attr = 0o100644 << 16
            with source.open('rb') as stream, out.open(info, 'w', force_zip64=True) as dest:
                shutil.copyfileobj(stream, dest, 65536)
    with zipfile.ZipFile(target) as check:
        assert check.namelist() == names and check.testzip() is None
        assert all(i.compress_type == zipfile.ZIP_STORED and i.date_time == (1980,1,1,0,0,0) for i in check.infolist())
    return dict(sha256=file_sha(target), bytes=target.stat().st_size, files=len(names))


def replay_zip(path, code_dir, repo, work_dir=None):
    from replay_sharded202 import run
    with zipfile.ZipFile(path) as archive:
        infos = archive.infolist()
        assert len({i.filename for i in infos}) == len(infos)
        assert 'manifest.json' in archive.namelist() and archive.getinfo('manifest.json').file_size < 5*1024*1024
        manifest = json.loads(archive.read('manifest.json'))
        names = allowed_files(manifest)
        assert sorted(archive.namelist()) == names
        assert all(i.compress_type == zipfile.ZIP_STORED and not i.is_dir() and not (i.flag_bits & 1) for i in infos)
        with tempfile.TemporaryDirectory(dir=work_dir, prefix='numeric202_zip_') as temp:
            root = Path(temp)
            for name in names:
                with archive.open(name) as source, (root/name).open('xb') as out:
                    shutil.copyfileobj(source, out, 65536)
            return run(root, code_dir, repo, work_dir)
