"""Download BEE24.zip (2.2 GB, Google Drive) into data/ and unzip it.
run: poetry run python courses/dlia/final_project/tools/fetch_bee24.py
Source: https://holmescao.github.io/datasets/BEE24 (Cao et al., IEEE TIP 2025)."""
import html, pathlib, re, urllib.parse, urllib.request, zipfile

FILE_ID = "1KvqQTZWWhyDsQuEIqQY3XLM_8QlYjDqi"
data = pathlib.Path(__file__).resolve().parent.parent / "data"
data.mkdir(exist_ok=True)
zpath = data / "BEE24.zip"

if not zpath.exists():
    base = "https://drive.usercontent.google.com/download"
    page = urllib.request.urlopen(f"{base}?id={FILE_ID}&export=download").read().decode("utf-8", "replace")
    # large files get a virus-scan warning page; its form carries the confirm/uuid fields
    fields = dict(re.findall(r'name="([^"]+)" value="([^"]*)"', page))
    assert "confirm" in fields, "Drive page changed: no confirm field"
    url = base + "?" + urllib.parse.urlencode({k: html.unescape(v) for k, v in fields.items()})
    tmp = zpath.with_suffix(".part")
    with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
        assert "html" not in r.headers.get("Content-Type", ""), "got an HTML page instead of the zip"
        while chunk := r.read(1 << 20):
            f.write(chunk)
    tmp.rename(zpath)
print("zip:", zpath, zpath.stat().st_size // 2**20, "MB")

with zipfile.ZipFile(zpath) as z:
    assert z.testzip() is None, "corrupt zip"
    z.extractall(data)
print("extracted into", data)
