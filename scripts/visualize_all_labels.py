
from pathlib import Path
from PIL import Image, ImageDraw

DATA = Path(r"E:\LungLens_Data\processed\yolo_v1")
OUT = Path(r"D:\肺结节模型训练\evaluation\all_labels")
OUT.mkdir(parents=True, exist_ok=True)

for split in ["train", "val"]:
    image_dir = DATA / "images" / split
    label_dir = DATA / "labels" / split

    for image_path in sorted(image_dir.glob("*.png")):
        image = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(image)
        w, h = image.size

        label_path = label_dir / f"{image_path.stem}.txt"

        if not label_path.exists():
            print("缺少标签:", label_path)
            continue

        lines = label_path.read_text(encoding="utf-8").splitlines()

        for line in lines:
            cls, cx, cy, bw, bh = map(float, line.split())

            x1 = (cx - bw / 2) * w
            y1 = (cy - bh / 2) * h
            x2 = (cx + bw / 2) * w
            y2 = (cy + bh / 2) * h

            draw.rectangle((x1, y1, x2, y2), outline="red", width=3)

        output_path = OUT / f"{split}_{image_path.name}"
        image.save(output_path)

        print(split, image_path.name, "结节数:", len(lines))

print("\n全部标签检查图已生成")
print("输出目录:", OUT)
