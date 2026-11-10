"""Dataset statistics + proposal figures from the real BEE24 files (ground truth only, no model yet).
run: python courses/dlia/final_project/proposal/make_figs.py      (needs Pillow; data from tools/fetch_bee24.py)
writes proposal/stats.json and proposal/img/{hive_raw,gt_boxes,gt_traj,demo_mock}.jpg"""
import collections, configparser, json, pathlib, statistics
from PIL import Image, ImageDraw, ImageFont

here = pathlib.Path(__file__).resolve().parent
root = here.parent / "data" / "BEE24"
out = here / "img"; out.mkdir(exist_ok=True)

def read_seq(d):
    rows = [l.split(",") for l in (d / "gt" / "gt.txt").read_text().split()]
    gt = [(int(r[0]), int(r[1]), *map(float, r[2:6])) for r in rows]   # frame, id, x, y, w, h
    frames = sorted((d / "img1").glob("*.jpg"))
    w, h = Image.open(frames[0]).size
    ini = configparser.ConfigParser(); ini.read(d / "seqinfo.ini")   # some test sequences ship without seqinfo.ini
    fps = int(ini["Sequence"]["frameRate"]) if ini.has_section("Sequence") else None
    return dict(dir=d, w=w, h=h, fps=fps, n=len(frames), gt=gt)

seqs = {split: [read_seq(d) for d in sorted((root / split).iterdir())] for split in ("train", "test")}
allseq = seqs["train"] + seqs["test"]
boxes = sum(len(s["gt"]) for s in allseq)
tracks = sum(len({g[1] for g in s["gt"]}) for s in allseq)
per_frame = [c for s in allseq for c in collections.Counter(g[0] for g in s["gt"]).values()]
res = collections.Counter(f'{s["w"]}x{s["h"]}' for s in allseq)
assert boxes > 0 and len(allseq) == len(seqs["train"]) + len(seqs["test"])

# figure sequence: the training sequence with the most bees per frame (figures never use test data)
fig = max(seqs["train"], key=lambda s: len(s["gt"]) / s["n"])
frame = fig["n"] // 2
by_frame = collections.defaultdict(list)
for g in fig["gt"]:
    by_frame[g[0]].append(g)
img = Image.open(fig["dir"] / "img1" / f"{frame:06d}.jpg").convert("RGB")
W, H = img.size
font = ImageFont.truetype("C:/Windows/Fonts/arialbd.ttf", max(12, W // 60))
PALETTE = [(230, 159, 0), (86, 180, 233), (0, 158, 115), (240, 228, 66), (0, 114, 178), (213, 94, 0), (204, 121, 167)]
col = lambda i: PALETTE[i % len(PALETTE)]

def save(im, name):
    im.save(out / name, quality=90)

def boxes_on(im, items):
    d = ImageDraw.Draw(im)
    for _, i, x, y, w, h in items:
        d.rectangle([x, y, x + w, y + h], outline=col(i), width=3)
        d.text((x + 2, y - font.size - 2), str(i), fill=col(i), font=font, stroke_width=2, stroke_fill=(0, 0, 0))
    return im

def trails(im, upto, span):
    d = ImageDraw.Draw(im)
    pts = collections.defaultdict(list)
    for f in range(max(1, upto - span), upto + 1):
        for _, i, x, y, w, h in by_frame[f]:
            pts[i].append((x + w / 2, y + h / 2))
    for i, p in pts.items():
        if len(p) > 1:
            d.line(p, fill=col(i), width=3)
    return pts

line_y = H // 2
def draw_line(im):
    ImageDraw.Draw(im).line([(0, line_y), (W, line_y)], fill=(255, 255, 255), width=4)

def crossings(start, upto):
    """GT counting protocol: centre crosses line_y; moving up = in, down = out (direction convention for the illustration)."""
    last, n_in, n_out = {}, 0, 0
    for f in range(start, upto + 1):
        for _, i, x, y, w, h in by_frame[f]:
            cy = y + h / 2
            if i in last and (last[i] - line_y) * (cy - line_y) < 0:
                n_in, n_out = (n_in + 1, n_out) if cy < line_y else (n_in, n_out + 1)
            last[i] = cy
    return n_in, n_out

save(img.copy(), "hive_raw.jpg")
# portrait crop for the title slide's right half (6.333 x 7.5 in), centred on the bees
cw = round(H * 6.333 / 7.5)
cx = statistics.mean(x + w / 2 for _, _, x, _, w, _ in by_frame[frame])
x0 = int(min(max(cx - cw / 2, 0), W - cw))
save(img.crop((x0, 0, x0 + cw, H)), "hive_title.jpg")
save(boxes_on(img.copy(), by_frame[frame]), "gt_boxes.jpg")
t = img.copy(); trails(t, frame, 60); draw_line(t); save(t, "gt_traj.jpg")
m = img.copy(); trails(m, frame, 30); draw_line(m); boxes_on(m, by_frame[frame])
win = 10 * (fig["fps"] or 25)
n_in, n_out = crossings(frame - win, frame)
big = ImageFont.truetype("C:/Windows/Fonts/arialbd.ttf", max(20, W // 28))   # readable once shrunk onto the slide
ImageDraw.Draw(m).text((14, 10), f"last 10 s:  IN {n_in}   OUT {n_out}", fill=(255, 255, 255), font=big, stroke_width=4, stroke_fill=(0, 0, 0))
save(m, "demo_mock.jpg")

stats = dict(
    videos=len(allseq), trainSeq=len(seqs["train"]), testSeq=len(seqs["test"]), boxes=boxes, tracks=tracks,
    frames=sum(s["n"] for s in allseq), avgPerFrame=statistics.mean(per_frame), maxPerFrame=max(per_frame),
    res="≈" + res.most_common(1)[0][0].replace("x", "×") + " px · " + "/".join(str(f) for f in sorted({s["fps"] for s in allseq if s["fps"]})) + " fps",
    fps=sorted({s["fps"] for s in allseq if s["fps"]}), noSeqinfo=[s["dir"].name for s in allseq if not s["fps"]], rawSeq=fig["dir"].name, rawFrame=frame, imgRatio=H / W,
)
(here / "stats.json").write_text(json.dumps(stats, indent=1), encoding="utf-8")
print(json.dumps(stats, indent=1))
