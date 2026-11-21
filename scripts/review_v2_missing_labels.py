
from pathlib import Path
from collections import defaultdict
import csv
import math

import SimpleITK as sitk
from PIL import Image, ImageDraw

ROOT = Path(r"E:\LungLens_Data")
RAW = ROOT / "raw" / "subset0"
V2 = ROOT / "processed" / "yolo_v2"
ANN = ROOT / "annotations" / "annotations.csv"
OUT = Path(r"D:\肺结节模型训练\evaluation\v2_review")
OUT.mkdir(parents=True, exist_ok=True)

annotations = defaultdict(list)
with ANN.open("r", encoding="utf-8-sig", newline="") as f:
    for row in csv.DictReader(f):
        annotations[row["seriesuid"]].append(row)

warning_count = 0

for mhd in sorted(RAW.glob("*.mhd")):
    uid = mhd.stem
    if uid not in annotations:
        continue

    image = sitk.ReadImage(str(mhd))
    w, h, depth = image.GetSize()
    sx, sy, sz = image.GetSpacing()

    nodules = []
    for row in annotations[uid]:
        world = tuple(float(row[k]) for k in
                      ("coordX", "coordY", "coordZ"))
        x, y, z = image.TransformPhysicalPointToContinuousIndex(world)
        diameter = float(row["diameter_mm"])
        center_slice = max(
            0, min(depth - 1, int(math.floor(z + 0.5)))
        )
        nodules.append((x, y, z, diameter, center_slice))

    for split in ("train", "val"):
        for img_path in sorted(
            (V2 / "images" / split).glob(f"{uid}_z*.png")
        ):
            zi = int(img_path.stem.rsplit("_z", 1)[1])

            # 找出中心不在本层、但可能仍出现在本层的结节
            suspects = []
            for j, (x, y, z, diameter, center) in enumerate(nodules):
                distance_mm = abs(zi - z) * sz
                if center != zi and distance_mm <= diameter / 2 + sz / 2:
                    suspects.append((j, x, y, diameter, distance_mm))

            if not suspects:
                continue

            canvas = Image.open(img_path).convert("RGB")
            draw = ImageDraw.Draw(canvas)

            # 红色：本来已有的YOLO标签
            label_path = V2 / "labels" / split / f"{img_path.stem}.txt"
            for line in label_path.read_text(encoding="utf-8").splitlines():
                cls, cx, cy, bw, bh = map(float, line.split())
                draw.rectangle(
                    ((cx-bw/2)*w, (cy-bh/2)*h,
                     (cx+bw/2)*w, (cy+bh/2)*h),
                    outline="red", width=3
                )

            # 黄色：需要人工复核的候选位置
            for j, x, y, diameter, distance_mm in suspects:
                radius_x = diameter / (2 * sx)
                radius_y = diameter / (2 * sy)

                draw.rectangle(
                    (x-radius_x, y-radius_y,
                     x+radius_x, y+radius_y),
                    outline="yellow", width=3
                )

                warning_count += 1
                print(
                    uid[-12:], "z", zi,
                    "候选编号", j,
                    "距离", round(distance_mm, 2), "mm"
                )

            canvas.save(OUT / f"{split}_{img_path.name}")

print("\n待复核候选数:", warning_count)
print("检查图片保存于:", OUT)
print("红框=已有标签，黄框=可能遗漏的结节")
