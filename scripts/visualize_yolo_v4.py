
from pathlib import Path
from PIL import Image, ImageDraw
import random
import math

ROOT = Path(r"E:\LungLens_Data\processed\yolo_v4")
OUT = ROOT / "review"
OUT.mkdir(exist_ok=True)
rng = random.Random(42)

def read_boxes(txt, w, h):
    boxes = []
    for line in txt.read_text(encoding="utf-8").splitlines():
        cls, cx, cy, bw, bh = map(float, line.split())
        boxes.append((
            (cx - bw / 2) * w,
            (cy - bh / 2) * h,
            (cx + bw / 2) * w,
            (cy + bh / 2) * h,
        ))
    return boxes

def make_tile(image_path, boxes, positive):
    im = Image.open(image_path).convert("RGB")
    w, h = im.size

    full = im.copy()
    draw = ImageDraw.Draw(full)

    for box in boxes:
        draw.rectangle(box, outline="red", width=3)

    full.thumbnail((300, 300))
    canvas = Image.new("RGB", (610, 345), "white")
    canvas.paste(full, (0, 30))

    if positive and boxes:
        # 放大第一个结节附近的区域
        x1, y1, x2, y2 = boxes[0]
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

        radius = 64
        crop = im.crop((
            int(cx - radius),
            int(cy - radius),
            int(cx + radius),
            int(cy + radius),
        ))

        crop_draw = ImageDraw.Draw(crop)
        crop_draw.rectangle(
            (
                x1 - (cx - radius),
                y1 - (cy - radius),
                x2 - (cx - radius),
                y2 - (cy - radius),
            ),
            outline="red",
            width=2,
        )

        crop = crop.resize((300, 300))
        canvas.paste(crop, (310, 30))
    else:
        # 阴性样本右侧显示另一种放大视图
        crop = im.crop((
            w // 4, h // 4,
            3 * w // 4, 3 * h // 4,
        ))
        crop = crop.resize((300, 300))
        canvas.paste(crop, (310, 30))

    draw = ImageDraw.Draw(canvas)
    short_name = image_path.stem[-18:]
    kind = "POS" if positive else "UNLABELED"
    draw.text(
        (8, 8),
        f"{kind}  {short_name}  boxes={len(boxes)}",
        fill="black"
    )
    return canvas

def save_sheet(split, kind, records):
    if not records:
        print(f"{split} {kind}: 没有样本")
        return

    tiles = [
        make_tile(path, boxes, kind == "positive")
        for path, boxes in records
    ]

    sheet = Image.new(
        "RGB",
        (610 * 2, 345 * math.ceil(len(tiles) / 2)),
        "#dddddd"
    )

    for i, tile in enumerate(tiles):
        sheet.paste(
            tile,
            ((i % 2) * 610, (i // 2) * 345)
        )

    output = OUT / f"{split}_{kind}.jpg"
    sheet.save(output, quality=92)
    print(f"已保存: {output}")

for split in ("train", "val"):
    positives = []
    negatives = []

    for path in sorted((ROOT / "images" / split).glob("*.png")):
        label = ROOT / "labels" / split / f"{path.stem}.txt"

        with Image.open(path) as im:
            w, h = im.size

        boxes = read_boxes(label, w, h)
        record = (path, boxes)

        if boxes:
            positives.append(record)
        else:
            negatives.append(record)

    # 重点查看小框、大框、多结节及随机样本
    def smallest(r):
        return min(
            min(x2 - x1, y2 - y1)
            for x1, y1, x2, y2 in r[1]
        )

    selected = []
    seen = set()

    candidates = (
        sorted(positives, key=smallest)[:3]
        + sorted(positives, key=smallest, reverse=True)[:2]
        + sorted(positives, key=lambda r: len(r[1]), reverse=True)[:2]
        + rng.sample(positives, min(8, len(positives)))
    )

    for record in candidates:
        if record[0] not in seen:
            selected.append(record)
            seen.add(record[0])
        if len(selected) >= 12:
            break

    save_sheet(split, "positive", selected)
    save_sheet(
        split,
        "negative",
        rng.sample(negatives, min(8, len(negatives)))
    )

print("\n可视化完成，V4原始图像和标签未修改。")
