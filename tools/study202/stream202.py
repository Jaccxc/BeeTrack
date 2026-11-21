"""Bounded JSON reader and deterministic per-unit transport; no field discovery."""
import gzip
import hashlib
import json
import re
from pathlib import Path


class Reader:
    def __init__(self, stream, chunk=65536):
        self.stream, self.chunk = stream, chunk
        self.buf, self.pos, self.eof = '', 0, False
        self.decoder = json.JSONDecoder(object_pairs_hook=self.unique,
            parse_constant=lambda _: (_ for _ in ()).throw(ValueError('nonfinite JSON')))

    @staticmethod
    def unique(pairs):
        result = {}
        for k, v in pairs:
            if k in result: raise ValueError('duplicate JSON key')
            result[k] = v
        return result

    def fill(self):
        self.buf = self.buf[self.pos:]; self.pos = 0
        part = self.stream.read(self.chunk)
        self.eof = not part
        self.buf += part

    def peek(self):
        while True:
            while self.pos < len(self.buf) and self.buf[self.pos].isspace(): self.pos += 1
            if self.pos < len(self.buf): return self.buf[self.pos]
            if self.eof: return ''
            self.fill()

    def expect(self, char):
        if self.peek() != char: raise ValueError('JSON structure mismatch')
        self.pos += 1

    def value(self):
        self.peek()
        while True:
            try:
                value, end = self.decoder.raw_decode(self.buf, self.pos)
                if end == len(self.buf) and not self.eof:
                    self.fill(); continue
                if end < len(self.buf) and self.buf[end] not in ',:]} \t\r\n':
                    raise ValueError('incomplete JSON token')
                self.pos = end
                return value
            except (json.JSONDecodeError, ValueError):
                if self.eof: raise
                self.fill()

    def object(self):
        self.expect('{'); seen = set(); first = True
        while self.peek() != '}':
            if not first: self.expect(',')
            key = self.value()
            if not isinstance(key, str) or key in seen: raise ValueError('duplicate/nonstring object key')
            seen.add(key); self.expect(':'); first = False
            yield key
        self.expect('}')

    def array(self):
        self.expect('['); first = True
        while self.peek() != ']':
            if not first: self.expect(',')
            first = False
            yield None
        self.expect(']')

    def skip_value(self):
        """Skip an unused, source-hash-verified subtree without quadratic raw_decode retries."""
        first = self.peek()
        if first not in '[{':
            self.value(); return
        depth, quoted, escaped = 0, False, False
        while True:
            if self.pos == len(self.buf):
                if self.eof: raise ValueError('unterminated skipped JSON')
                self.fill()
            char = self.buf[self.pos]; self.pos += 1
            if quoted:
                if escaped: escaped = False
                elif char == '\\': escaped = True
                elif char == '"': quoted = False
            elif char == '"': quoted = True
            elif char in '[{': depth += 1
            elif char in ']}':
                depth -= 1
                if depth == 0: return


def sparse_action(action):
    if 'delta_windows' not in action: return action
    matrix = action.pop('delta_windows')
    indices, values = [], []
    for j, row in enumerate(matrix):
        assert len(row) == 2
        for k, value in enumerate(row):
            assert type(value) is int
            if value: indices.append(2*j+k); values.append(value)
    action['v_indices'], action['v_values'] = indices, values
    return action


def read_capacity_unit(reader):
    unit = {}
    def action_value():
        action = sparse_action(reader.value())
        fields = ('action_id', 'parent_key', 'orphan_key', 'member_index', 'kind',
                  'parent_canonical_id', 'cut_id', 'candidate_id', 'edge_id',
                  'v_indices', 'v_values', 'delta_events')
        compact = {k: action[k] for k in fields if k in action}
        if 'labels' in action: compact['labels'] = {t: {'class': x['class']} for t, x in action['labels'].items()}
        return compact
    for key in reader.object():
        if key == 'actions':
            actions = []
            for _ in reader.array():
                actions.append(action_value())
            unit[key] = actions
        elif key == 'members':
            actions, slots = [], []
            for _ in reader.array():
                member_actions, slot = [], None
                for field in reader.object():
                    if field in ('cuts', 'edges'):
                        for _ in reader.array(): member_actions.append(action_value())
                    elif field == 'member_index': slot = reader.value()
                    else: reader.skip_value()
                assert type(slot) is int
                slots.append(slot)
                for action in member_actions: action['member_index'] = slot
                actions.extend(member_actions)
            unit['actions'], unit['member_indices'] = actions, slots
        elif key in ('unit', 'hive', 'group', 'nf', 'pred_sum', 'gt_bins', 'member_indices'):
            unit[key] = reader.value()
        else:
            # Other source-only metadata is pinned by the complete original-file digest.
            reader.skip_value()
    return unit


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''): h.update(chunk)
    return h.hexdigest()


class Writer:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=False)
        self.entries = []

    def write(self, role, key, value):
        self.check_strings(value)
        chunks = json.JSONEncoder(sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(',', ':')).iterencode(value)
        return self._write(role, key, chunks)

    def write_array(self, role, key, values):
        def chunks():
            yield '['
            first = True
            encoder = json.JSONEncoder(sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(',', ':'))
            for value in values:
                self.check_strings(value)
                if not first: yield ','
                yield from encoder.iterencode(value)
                first = False
            yield ']'
        return self._write(role, key, chunks())

    @staticmethod
    def check_strings(value):
        if isinstance(value, str):
            if re.search(r'(?i)(?:[a-z]:[\\/]|file://|\\\\[^\\]+\\)', value):
                raise ValueError('absolute machine path in decoded numeric shard')
        elif isinstance(value, list):
            for item in value: Writer.check_strings(item)
        elif isinstance(value, dict):
            for key, item in value.items():
                Writer.check_strings(key); Writer.check_strings(item)

    def _write(self, role, key, chunks):
        token = hashlib.sha256((role+'|'+key).encode()).hexdigest()
        name = token+'.json.gz'; target = self.root/name
        assert not target.exists()
        raw_hash, raw_bytes = hashlib.sha256(), 0
        # Inflated private file supports the existing exact privacy scanner.
        inflated = self.root/(token+'.inflated.private.json')
        with target.open('xb') as out, inflated.open('xb') as plain:
            with gzip.GzipFile(filename='', mode='wb', fileobj=out, mtime=0) as archive:
                pending = bytearray()
                for text in chunks:
                    pending.extend(text.encode())
                    if len(pending) >= 65536:
                        raw_hash.update(pending); raw_bytes += len(pending)
                        archive.write(pending); plain.write(pending); pending.clear()
                if pending:
                    raw_hash.update(pending); raw_bytes += len(pending)
                    archive.write(pending); plain.write(pending)
        entry = dict(role=role, key=key, file=name, sha256=file_sha(target), inflated_sha256=raw_hash.hexdigest(), inflated_bytes=raw_bytes)
        self.entries.append(entry)
        return entry

    def finish(self, sources, references=None):
        # Only manifest-listed .gz and manifest.json ship; inflated files stay private.
        manifest = dict(schema='orphan-continuation-sharded-v1', sources=sources, references=references or {}, shards=self.entries)
        (self.root/'manifest.json').write_text(json.dumps(manifest, sort_keys=True, indent=2)+'\n', encoding='utf-8')
        return manifest


def shard_capacity_input(source, writer):
    with Path(source).open(encoding='utf-8') as stream:
        reader = Reader(stream)
        metadata = {}
        for key in reader.object():
            if key == 'units':
                for _ in reader.array():
                    unit = read_capacity_unit(reader)
                    writer.write('capacity_input', unit['unit'], unit)
                    del unit
            elif key == 'warm_starts':
                # Warm starts are provenance, not part of the proof; retain them per unit.
                for unit in reader.object(): writer.write('capacity_warm_starts', unit, reader.value())
            else: metadata[key] = reader.value()
        assert not reader.peek()
        writer.write('capacity_metadata', 'shared', metadata)


def shards(root, role):
    """Yield one decoded shard at a time after packed and inflated integrity checks."""
    root = Path(root)
    manifest = json.loads((root/'manifest.json').read_text())
    for entry in manifest['shards']:
        if entry['role'] != role: continue
        path = (root/entry['file']).resolve()
        assert path.is_relative_to(root.resolve()) and file_sha(path) == entry['sha256']
        # Hash first, then parse; never retain inflated bytes alongside decoded object.
        h, size = hashlib.sha256(), 0
        with gzip.open(path, 'rb') as stream:
            for chunk in iter(lambda: stream.read(65536), b''): h.update(chunk); size += len(chunk)
        assert h.hexdigest() == entry['inflated_sha256'] and size == entry['inflated_bytes']
        with gzip.open(path, 'rt', encoding='utf-8') as stream: value = json.load(stream)
        yield entry['key'], value
        del value


def verified_path(root, entry):
    root = Path(root); path = (root/entry['file']).resolve()
    assert path.is_relative_to(root.resolve()) and file_sha(path) == entry['sha256']
    h, size = hashlib.sha256(), 0
    with gzip.open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(65536), b''): h.update(chunk); size += len(chunk)
    assert h.hexdigest() == entry['inflated_sha256'] and size == entry['inflated_bytes']
    return path
