
from pathlib import Path
from PIL import Image, ImageDraw

DATA = Path(r"E:\LungLens_Data\processed\yolo_v1")
OUT = Path(r"D:\肺结节模型训练\evaluation\nodule_zoom")
OUT.mkdir(parents=True, exist_ok=True)

for split in ["train", "val"]:
    for image_path in sorted((DATA / "images" / split).glob("*.png")):
        label_path = DATA / "labels" / split / (image_path.stem + ".txt")

        if not label_path.exists():
            continue

        image = Image.open(image_path).convert("RGB")
        w, h = image.size

        for idx, line in enumerate(label_path.read_text().splitlines()):
            cls, cx, cy, bw, bh = map(float, line.split())

            x = cx * w
            y = cy * h

            # 以结节中心为中心，裁剪 96×96 像素区域
            half = 48
            left = int(round(x - half))
            top = int(round(y - half))

            # 即使靠近图像边界，也保持固定裁剪尺寸
            crop = Image.new("RGB", (96, 96), (0, 0, 0))
            src_box = (
                max(0, left),
                max(0, top),
                min(w, left + 96),
                min(h, top + 96),
            )
            if src_box[2] <= src_box[0] or src_box[3] <= src_box[1]:
                continue

            part = image.crop(src_box)
            crop.paste(
                part,
                (src_box[0] - left, src_box[1] - top)
            )

            # 将标签框转换为裁剪图内的坐标
            x1 = (cx - bw / 2) * w - left
            y1 = (cy - bh / 2) * h - top
            x2 = (cx + bw / 2) * w - left
            y2 = (cy + bh / 2) * h - top

            draw = ImageDraw.Draw(crop)
            draw.rectangle((x1, y1, x2, y2), outline="red", width=2)

            # 放大4倍，便于查看结节结构
            crop = crop.resize((384, 384), Image.Resampling.NEAREST)

            name = f"{split}_{image_path.stem}_n{idx}.png"
            crop.save(OUT / name)

            print(name, "标签框尺寸:",
                  round(bw * w, 2), "x", round(bh * h, 2))

print("\n局部放大图已保存:", OUT)
