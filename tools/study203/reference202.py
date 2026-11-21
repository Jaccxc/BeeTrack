"""Read only a requested202 component from the pinned public ZIP."""
import gzip
import io
import json
from pathlib import Path
import zipfile
from tools.study202.stream202 import file_sha

SHA202='ef5fd41a7ebadf1425fa840cc73b1c412f5b5f76d049c7c58f0f4c66196d6788'


class Reference202:
    def __init__(self,path):
        assert file_sha(path)==SHA202
        self.zip=zipfile.ZipFile(path)
        manifest=json.loads(self.zip.read('manifest.json'))
        self.sources=manifest['sources']
        self.index={(r['role'],r['key']):r for r in manifest['shards']}
        self.unit=None;self.clip=None
    def get(self,role,key):
        entry=self.index[role,key]
        with self.zip.open(entry['file']) as packed, gzip.GzipFile(fileobj=packed) as inflated:
            return json.load(io.TextIOWrapper(inflated,encoding='utf-8'))
    def baseline(self,unit,member):
        if self.unit!=unit:
            self.clip=self.get('native_clip','K|'+unit)['clip'];self.unit=unit
        role=next(m for m in self.clip['members'] if str(m['member'])==str(member))
        return role,self.get('physical_metric',role['physical_key'])
