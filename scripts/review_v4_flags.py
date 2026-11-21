
from pathlib import Path
import csv
import random
from PIL import Image, ImageDraw, ImageOps

ROOT = Path(r"E:\LungLens_Data\processed\yolo_v4")
CSV = ROOT / "review" / "quality_screen.csv"
OUT = ROOT / "review"

rng = random.Random(42)

with CSV.open("r", encoding="utf-8-sig", newline="") as f:
    rows = list(csv.DictReader(f))


def choose_cases(items, count=12):
    """尽量选不同病例，避免连续切片重复。"""
    rng.shuffle(items)
    selected = []
    seen = set()

    for row in items:
        uid = row["image"].rsplit("_z", 1)[0]
        if uid in seen:
            continue
        seen.add(uid)
        selected.append(row)
        if len(selected) >= count:
            break

    return selected


def load_image(row):
    return Image.open(
        ROOT / "images" / row["split"] / row["image"]
    ).convert("RGB")


def load_boxes(row):
    stem = Path(row["image"]).stem
    label = ROOT / "labels" / row["split"] / (stem + ".txt")

    boxes = []
    for line in label.read_text(encoding="utf-8").splitlines():
        _, cx, cy, bw, bh = map(float, line.split())
        boxes.append((
            (cx - bw / 2) * 512,
            (cy - bh / 2) * 512,
            (cx + bw / 2) * 512,
            (cy + bh / 2) * 512
        ))
    return boxes


def make_tile(row, tiny=False):
    im = load_image(row)
    canvas = Image.new("RGB", (640, 340), "white")
    draw = ImageDraw.Draw(canvas)

    boxes = load_boxes(row)

    full = im.copy()
    d = ImageDraw.Draw(full)

    if tiny:
        for box in boxes:
            d.rectangle(box, outline="red", width=3)

    full = full.resize((300, 300))
    canvas.paste(full, (10, 30))

    if tiny and boxes:
        # 优先放大小于4px的目标
        box = min(
            boxes,
            key=lambda b: min(b[2] - b[0], b[3] - b[1])
        )
        x1, y1, x2, y2 = box
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2

        radius = 40
        left = round(cx - radius)
        top = round(cy - radius)

        crop = im.crop((
            left, top, left + 80, top + 80
        ))
        cd = ImageDraw.Draw(crop)

        cd.rectangle(
            (
                x1 - left, y1 - top,
                x2 - left, y2 - top
            ),
            outline="red",
            width=1
        )

        right = crop.resize((300, 300))
        size = min(x2 - x1, y2 - y1)
        info = f"box={size:.1f}px"
    else:
        # 阴性样本放大中心区域
        right = im.crop((128, 128, 384, 384))
        right = right.resize((300, 300))
        info = f"gray_ratio={row['low_density_ratio']}"

    canvas.paste(right, (330, 30))

    draw.text(
        (10, 8),
        f"{row['split']} | {row['image'][-22:]} | {info}",
        fill="black"
    )

    return canvas


def save_sheet(items, filename, tiny=False):
    if not items:
        print(filename, "没有候选图片")
        return

    cols = 2
    rows_n = (len(items) + 1) // 2

    sheet = Image.new(
        "RGB",
        (cols * 640, rows_n * 340),
        "#dddddd"
    )

    for i, row in enumerate(items):
        tile = make_tile(row, tiny)
        sheet.paste(
            tile,
            ((i % cols) * 640, (i // cols) * 340)
        )

    path = OUT / filename
    sheet.save(path, quality=95)
    print("已生成:", path)


tiny_rows = [
    r for r in rows
    if int(r["boxes_under_4px"]) > 0
]

low_lung_rows = [
    r for r in rows
    if r["review_flag"] == "LOW_LUNG_CANDIDATE"
]

save_sheet(
    choose_cases(tiny_rows),
    "tiny_boxes_review.jpg",
    tiny=True
)

save_sheet(
    choose_cases(low_lung_rows),
    "low_lung_review.jpg"
)

print("\n复核图片生成完成。")
print("没有修改V4原始图片或标签。")
