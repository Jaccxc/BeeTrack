"""Lossless little-endian scientific tables for study 114 geometry and GT overlaps.

Each named table records column labels, row count, dtype and the SHA-256 of the
uncompressed bytes. Its row-major payload uses zlib compression and Base64.
"""
from array import array
from base64 import b64decode, b64encode
import hashlib
import json
import math
import sys
import zlib


# Rows are stored in row-major order. Field names, type and row count remain visible in JSON.
# Signed 32-bit integers cover native decimal centipixels, row indices and resolved track IDs.
GEOMETRY = {
    "observations": ("<i4", ["frame","x100","y100","w100","h100","n_raw_refs"]),
    "raw_refs": ("<i4", ["raw_index","frame_local_index"]),
    "candidates": ("<i4", ["before_obs","after_obs","direction","n_native_ids","n_raw_pairs"]),
    "native_ids": ("<i4", ["track_id"]),
    "raw_pairs": ("<i4", ["before_raw_index","after_raw_index"]),
    "native_events": ("<i4", ["track_id","signed_frame","candidate_index"]),
}
OVERLAPS = {
    "observations": ("<i4", ["observation_index","n_overlaps"]),
    "gt_side": ("<i4", ["gt_id","inside"]),
    "iou": ("<f8", ["iou"]),
}


def _buffer(dtype):
    code, length = {"<i4": ("i",4), "<f8": ("d",8)}[dtype]
    buf = array(code)
    if buf.itemsize != length:
        raise RuntimeError("Platform does not provide the required fixed-width native array")
    return buf


def _table_pack(schema, buffers):
    tables = {}
    for name, (dtype, cols) in schema.items():
        values = buffers[name]
        nrow, rem = divmod(len(values), len(cols))
        if rem:
            raise ValueError("Incomplete row in " + name)
        if sys.byteorder != "little":
            values.byteswap()
        raw = memoryview(values).cast("B")
        tables[name] = dict(dtype=dtype, columns=cols.copy(), rows=nrow,
                            sha256=hashlib.sha256(raw).hexdigest(),
                            zlib_base64=b64encode(zlib.compress(raw, level=4)).decode("ascii"))
    return tables


def _table_read(schema, tables):
    if set(schema) != set(tables):
        raise ValueError("Unexpected scientific data tables")
    out = {}
    for name, (dtype, cols) in schema.items():
        p = tables[name]
        if (set(p) != {"dtype","columns","rows","sha256","zlib_base64"} or
                p["dtype"] != dtype or p["columns"] != cols or type(p["rows"]) is not int or p["rows"] < 0):
            raise ValueError("Table schema differs for " + name)
        values = _buffer(dtype)
        nbyte = p["rows"]*len(cols)*values.itemsize
        z = zlib.decompressobj()
        raw = z.decompress(b64decode(p["zlib_base64"], validate=True), nbyte+1)
        if (not z.eof or z.unused_data or z.unconsumed_tail or len(raw) != nbyte
                or hashlib.sha256(raw).hexdigest() != p["sha256"]):
            raise ValueError("Table length, compressed stream or SHA-256 differs for " + name)
        values.frombytes(raw)
        if sys.byteorder != "little":
            values.byteswap()
        out[name] = values
    return out


def decoded_tables(encoded, kind):
    """Expose verified, flat primitive arrays for decoded-content inspection.

    Yield (table name, dtype, column names, row count, array). Geometry accepts
    the encoded geometry object, proof accepts its complete encoded proof object.
    """
    if kind == "geometry":
        schema, tables = GEOMETRY, encoded
    elif kind == "proof":
        schema, tables = OVERLAPS, encoded["overlaps"]
    else:
        raise ValueError("Expected geometry or proof")
    for name, values in _table_read(schema, tables).items():
        dtype, cols = schema[name]
        yield name, dtype, tuple(cols), tables[name]["rows"], values


def _integer_row(values, width, dest):
    if len(values) != width or any(type(x) is not int for x in values):
        raise ValueError("Expected a fixed-width integer row")
    dest.extend(values)


def pack_geometry(geometry):
    """Pack the complete observation, crossing-candidate and native-event lists."""
    if set(geometry) != {"observations","candidates","native_events"}:
        raise ValueError("Unexpected geometry schema")
    b = {key:_buffer(dtype) for key,(dtype,_) in GEOMETRY.items()}
    for row in geometry["observations"]:
        f,x,y,w,h,refs = row
        _integer_row((f,x,y,w,h,len(refs)), 6, b["observations"])
        for pair in refs:
            _integer_row(pair, 2, b["raw_refs"])
    for before,after,direction,ids,pairs in geometry["candidates"]:
        _integer_row((before,after,direction,len(ids),len(pairs)), 5, b["candidates"])
        _integer_row(ids, len(ids), b["native_ids"])
        for pair in pairs:
            _integer_row(pair, 2, b["raw_pairs"])
    for event in geometry["native_events"]:
        _integer_row(event, 3, b["native_events"])
    return _table_pack(GEOMETRY, b)


def _rows(values, width):
    return zip(*[iter(values)]*width)


def _grouped(values, width, counts):
    """Consume a flat table exactly once and restore the original variable group sizes."""
    it = iter(values)
    if min(counts, default=0) < 0 or sum(counts)*width != len(values):
        raise ValueError("Group sizes do not partition a scientific data table")
    if width == 1:
        return ([next(it) for _ in range(n)] for n in counts)
    rows = _rows(it, width)
    return ([list(next(rows)) for _ in range(n)] for n in counts)


def unpack_geometry(encoded):
    b = _table_read(GEOMETRY, encoded)
    obs, cand = b["observations"], b["candidates"]
    refs = _grouped(b["raw_refs"], 2, obs[5::6])
    ids = _grouped(b["native_ids"], 1, cand[3::5])
    pairs = _grouped(b["raw_pairs"], 2, cand[4::5])
    return {"observations":[[f,x,y,w,h,group] for (f,x,y,w,h,_),group in zip(_rows(obs,6),refs)],
            "candidates":[[a,z,d,native,raw] for (a,z,d,_,_),native,raw in zip(_rows(cand,5),ids,pairs)],
            "native_events":[list(row) for row in _rows(b["native_events"],3)]}


def pack_proof(proof):
    """Pack sparse per-observation frame overlaps, preserving polygon, turns and metadata."""
    b = {key:_buffer(dtype) for key,(dtype,_) in OVERLAPS.items()}
    for idx, options in proof["overlaps"]:
        _integer_row((idx,len(options)), 2, b["observations"])
        for gid,iou,side in options:
            _integer_row((gid,side), 2, b["gt_side"])
            if type(iou) is not float or not math.isfinite(iou):
                raise ValueError("Expected a finite binary64 IoU")
            b["iou"].append(iou)
    return dict(proof, overlaps=_table_pack(OVERLAPS, b))


def unpack_proof(encoded):
    b = _table_read(OVERLAPS, encoded["overlaps"])
    obs, floats = b["observations"], b["iou"]
    if len(floats)*2 != len(b["gt_side"]) or not all(map(math.isfinite,floats)):
        raise ValueError("IoU table differs from the GT/side pairs")
    groups = _grouped(b["gt_side"], 2, obs[1::2])
    ious = iter(floats)
    overlaps = [[idx,[[gid,next(ious),side] for gid,side in group]]
                for (idx,_),group in zip(_rows(obs,2),groups)]
    return dict(encoded, overlaps=overlaps)


def self_check():
    def wire(record):
        return json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()

    geometry = {"observations":[[3,-120,51,3010,1500,[[5,0],[9,1]]],[5,100,49,3000,1490,[]]],
                "candidates":[[0,1,-1,[8,11],[[5,21]]],[1,0,1,[],[]]],
                "native_events":[[8,-5,0]]}
    proof = {"polygon":[[0.,1.],[0.,2.],[3.,4.]], "turns":[[21,5,-1]],
             "overlaps":[[0,[[3,0.30000000000000004,1],[5,1.,0]]],[1,[]]]}
    for record, encoder, decoder in ((geometry,pack_geometry,unpack_geometry),(proof,pack_proof,unpack_proof)):
        assert wire(decoder(json.loads(wire(encoder(record))))) == wire(record)
    empty = {"observations":[],"candidates":[],"native_events":[]}
    assert unpack_geometry(pack_geometry(empty)) == empty
    assert list(next(iter(decoded_tables(pack_proof(proof),"proof")))[-1]) == [0,2,1,0]
    damaged = pack_geometry(geometry)
    damaged["raw_pairs"]["rows"] += 1
    try:
        unpack_geometry(damaged)
    except ValueError:
        pass
    else:
        raise ValueError("Truncated table accepted")
    print("Scientific tables verified.")


if __name__ == "__main__":
    self_check()
