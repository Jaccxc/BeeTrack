"""Transactional per-frame detector output and native two-frame history for §117."""

from collections import deque
import hashlib
import io
import json
from pathlib import Path
import sqlite3
import tempfile
import time
import zlib

from filelock import FileLock
import numpy as np
import torch


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                      separators=(",", ":")).encode("utf-8")


def frame_record(blob, digest):
    raw = zlib.decompress(blob)
    if hashlib.sha256(raw).digest() != digest:
        raise ValueError("Frame record failed its SHA-256 check")
    return json.loads(raw)


def history_blob(values):
    arrays = []
    for value in values:
        arr = value.detach().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)
        if arr.dtype != np.float32:
            raise ValueError("Store native float32 history without dtype conversion")
        arrays.append(np.ascontiguousarray(arr))
    if len(arrays) != 3:
        raise ValueError("Expected P3, logits and decoded native boxes")
    buf = io.BytesIO()
    np.savez(buf, p3=arrays[0], logits=arrays[1], boxes=arrays[2])
    return buf.getvalue()


def iter_records(path):
    """Yield frame records in original order from an already sealed case."""
    uri = Path(path).resolve().as_uri() + "?mode=ro"
    db = sqlite3.connect(uri, uri=True)
    try:
        if db.execute("SELECT sealed FROM state WHERE id=1").fetchone() != (1,):
            raise ValueError("Case is not sealed")
        for blob, digest in db.execute("SELECT record,sha256 FROM frames ORDER BY frame"):
            yield frame_record(blob, digest)
    finally:
        db.close()


class Store:
    """One exclusive writer per case. Call begin(f,f) immediately before inference."""

    def __init__(self, path, config):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = FileLock(str(self.path)+".lock", timeout=0)
        self.lock.acquire()
        self.db = None
        try:
            self.db = sqlite3.connect(self.path)
            self.db.execute("PRAGMA synchronous=FULL")
            self.db.execute("CREATE TABLE IF NOT EXISTS state "
                            "(id INTEGER PRIMARY KEY,config BLOB NOT NULL,last INTEGER NOT NULL,"
                            "pending INTEGER,sealed INTEGER NOT NULL)")
            cfg = canonical(config)
            state = self.db.execute("SELECT config,last,pending,sealed FROM state WHERE id=1").fetchone()
            if state is None:
                self.db.executescript("""
                    CREATE TABLE IF NOT EXISTS frames(frame INTEGER PRIMARY KEY, record BLOB NOT NULL, sha256 BLOB NOT NULL);
                    CREATE TABLE IF NOT EXISTS history(frame INTEGER PRIMARY KEY, arrays BLOB NOT NULL, sha256 BLOB NOT NULL);
                    CREATE TABLE IF NOT EXISTS retries(frame INTEGER NOT NULL, utc TEXT NOT NULL, reason TEXT NOT NULL);
                """)
                with self.db:
                    self.db.execute("INSERT INTO state VALUES (1,?,0,NULL,0)", (cfg,))
                state = cfg, 0, None, 0
            if state[0] != cfg:
                raise ValueError("Stored case has a different configuration or source fingerprint")
            self.last, self.pending, self.sealed = state[1], state[2], state[3] == 1
            if state[3] == 2:
                self._compact()
            if not self.sealed:
                self.db.execute("PRAGMA journal_mode=WAL")
                self.db.execute("PRAGMA wal_autocheckpoint=2048")
            self.active, self.record = None, None
        except BaseException:
            self.close()
            raise

    def close(self):
        if self.db is not None:
            self.db.close()
            self.db = None
        self.lock.release()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def history(self):
        if self.sealed:
            return []
        result = []
        for f, blob, dig in self.db.execute("SELECT frame,arrays,sha256 FROM history ORDER BY frame"):
            if hashlib.sha256(blob).digest() != dig:
                raise ValueError(f"Corrupt native history at frame {f}")
            with np.load(io.BytesIO(blob), allow_pickle=False) as a:
                result.append((f, *(torch.from_numpy(a[k]) for k in ("p3", "logits", "boxes"))))
        if [r[0] for r in result] != list(range(max(1, self.last-1), self.last+1)):
            raise ValueError("The native two-frame history is incomplete")
        return result

    def begin(self, first, last):
        if self.pending is not None:
            raise RuntimeError(f"Frame {self.pending} was started but not committed; forward count is uncertain. "
                               "Inspect the original run before any explicitly recorded retry.")
        if self.sealed or first != self.last+1 or last != first:
            raise ValueError("Expected the single next frame in an unsealed case")
        with self.db:
            self.db.execute("UPDATE state SET pending=? WHERE id=1", (first,))
        self.pending = self.active = first

    def authorize_retry(self, frame, reason):
        """Record an explicit exception to exact-once inference for one interrupted frame."""
        if self.active is not None or self.pending != frame or not isinstance(reason, str) or not reason.strip():
            raise ValueError("Expected a previously interrupted frame and a retry reason")
        with self.db:
            self.db.execute("INSERT INTO retries VALUES (?,strftime('%Y-%m-%dT%H:%M:%fZ','now'),?)",
                            (frame, reason))
            self.db.execute("UPDATE state SET pending=NULL WHERE id=1")
        self.pending = None

    def append(self, frame, record):
        if self.active != frame or self.record is not None:
            raise ValueError("Append one result for the active frame")
        raw = canonical(record)
        self.record = (zlib.compress(raw, level=1), hashlib.sha256(raw).digest())

    def commit(self, history):
        f = self.active
        if f is None or self.record is None:
            raise ValueError("Begin and append the next frame before committing")
        entries = list(history)
        if [entry[0] for entry in entries] != list(range(max(1, f-1), f+1)):
            raise ValueError("Expected the latest two consecutive native frames")
        # The preceding observation is already durable. Persist only the new one.
        blob = history_blob(entries[-1][1:])
        with self.db:
            self.db.execute("INSERT INTO frames VALUES (?,?,?)", (f, *self.record))
            self.db.execute("DELETE FROM history WHERE frame<?", (f-1,))
            self.db.execute("INSERT INTO history VALUES (?,?,?)", (f, blob, hashlib.sha256(blob).digest()))
            self.db.execute("UPDATE state SET last=?,pending=NULL WHERE id=1", (f,))
        self.last, self.pending, self.active, self.record = f, None, None, None

    def read(self, frame):
        row = self.db.execute("SELECT record,sha256 FROM frames WHERE frame=?", (frame,)).fetchone()
        if row is None:
            raise KeyError(frame)
        return frame_record(*row)

    def finalize(self, expected=1000):
        if self.pending is not None or self.last != expected:
            raise ValueError("All expected frames must be committed before sealing")
        if self.sealed:
            return
        if self.db.execute("SELECT COUNT(*),MIN(frame),MAX(frame) FROM frames").fetchone() != (expected,1,expected):
            raise ValueError("Frame archive is incomplete")
        with self.db:
            self.db.execute("DELETE FROM history")
            self.db.execute("UPDATE state SET sealed=2 WHERE id=1")
        self._compact()

    def _compact(self):
        self.db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        self.db.execute("PRAGMA journal_mode=DELETE")
        self.db.execute("VACUUM")
        with self.db:
            self.db.execute("UPDATE state SET sealed=1 WHERE id=1")
        self.sealed = True


def self_check():
    native = torch.zeros((1,128,64,80), dtype=torch.float32)
    logits = torch.zeros((1,1,64,80), dtype=torch.float32)
    boxes = torch.zeros((5120,4), dtype=torch.float32)
    hist = deque(maxlen=2)
    cfg = {"study":117,"synthetic":True}
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp)/"synthetic.sqlite"
        with Store(path, cfg) as s:
            begin = time.perf_counter()
            for f in range(1,13):
                s.begin(f,f)
                hist.append((f,native+f,logits+f,boxes+f))
                s.append(f,{"f":f})
                s.commit(hist)
            elapsed = time.perf_counter()-begin
            s.begin(13,13)
        with Store(path, cfg) as s:
            assert s.last == 12 and s.pending == 13
            try:
                s.begin(13,13)
            except RuntimeError:
                pass
            else:
                raise AssertionError("Repeated in-flight forward was not detected")
            for a,b in zip(s.history(),hist):
                assert a[0]==b[0] and all(torch.equal(x,y) for x,y in zip(a[1:],b[1:]))
            s.authorize_retry(13,"Synthetic interruption before inference")
            s.begin(13,13)
            hist.append((13,native,logits,boxes))
            s.append(13,{"f":13})
            s.commit(hist)
            s.finalize(13)
        assert list(iter_records(path))==[{"f":f} for f in range(1,14)]
        sealed_sha = hashlib.sha256(path.read_bytes()).hexdigest()
        with Store(path,cfg) as s:
            assert s.sealed and s.history()==[] and s.read(3)=={"f":3}
            s.finalize(13)
        assert hashlib.sha256(path.read_bytes()).hexdigest()==sealed_sha
        print(f"13 synthetic frames passed, {elapsed*1000/12:.1f} ms/frame for 128x64x80 FP32, "
              f"sealed SQLite {path.stat().st_size} bytes")


if __name__ == "__main__":
    torch.set_num_threads(1)
    self_check()
