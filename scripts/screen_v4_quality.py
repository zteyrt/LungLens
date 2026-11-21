
from pathlib import Path
from collections import Counter
import numpy as np
from PIL import Image
import csv

ROOT = Path(r"E:\LungLens_Data\processed\yolo_v4")
OUT = ROOT / "review" / "quality_screen.csv"
OUT.parent.mkdir(parents=True, exist_ok=True)

records = []
stats = Counter()

for split in ("train", "val"):
    image_dir = ROOT / "images" / split
    label_dir = ROOT / "labels" / split

    for img_path in sorted(image_dir.glob("*.png")):
        txt = label_dir / (img_path.stem + ".txt")

        with Image.open(img_path) as im:
            gray = np.array(im.convert("L"))
            h, w = gray.shape

        lines = txt.read_text(
            encoding="utf-8"
        ).splitlines()

        # 统计所有标签框的原始像素尺寸
        tiny4 = 0
        tiny8 = 0

        for line in lines:
            values = line.split()
            bw = float(values[3]) * w
            bh = float(values[4]) * h

            if min(bw, bh) < 4:
                tiny4 += 1

            if min(bw, bh) < 8:
                tiny8 += 1

        # 对负样本进行粗略肺野筛查
        # 只检查图像中间区域，避免部分外围背景
        roi = gray[
            int(h * 0.25):int(h * 0.75),
            int(w * 0.20):int(w * 0.80)
        ]

        # 依据现有WL=-600、WW=1500的映射：
        # 灰度约144对应HU=-500
        low_density_ratio = float(
            np.mean(roi < 144)
        )

        if lines:
            kind = "positive"
            if tiny4 > 0:
                flag = "TINY_BOX_REVIEW"
            else:
                flag = "NONE"

            stats[f"{split}_positive"] += 1
            stats[f"{split}_tiny4"] += tiny4
            stats[f"{split}_tiny8"] += tiny8

        else:
            kind = "unlabeled"

            # 仅为自动筛查阈值，不是肺分割结果
            if low_density_ratio < 0.12:
                flag = "LOW_LUNG_CANDIDATE"
                stats[f"{split}_low_lung"] += 1
            else:
                flag = "NONE"

            stats[f"{split}_unlabeled"] += 1

        records.append([
            split,
            img_path.name,
            kind,
            len(lines),
            tiny4,
            tiny8,
            round(low_density_ratio, 4),
            flag
        ])

with OUT.open(
    "w", encoding="utf-8-sig", newline=""
) as f:
    writer = csv.writer(f)
    writer.writerow([
        "split",
        "image",
        "sample_type",
        "box_count",
        "boxes_under_4px",
        "boxes_under_8px",
        "low_density_ratio",
        "review_flag"
    ])
    writer.writerows(records)

print("========== V4 质量筛查 ==========")

for split in ("train", "val"):
    print(f"\n{split.upper()}")
    print("正样本图片:", stats[f"{split}_positive"])
    print("无标签图片:", stats[f"{split}_unlabeled"])
    print("小于4px的框:", stats[f"{split}_tiny4"])
    print("小于8px的框:", stats[f"{split}_tiny8"])
    print(
        "疑似肺野不足的无标签图片:",
        stats[f"{split}_low_lung"]
    )

print("\n需要复核的部分图片:")

flagged = [
    row for row in records
    if row[-1] != "NONE"
]

for row in flagged[:20]:
    print(row[0], row[1][-28:], row[-1])

print("\n完整筛查结果:", OUT)
print("注意：低密度比例只是粗筛，不能代替肺野分割。")
print("原始V4数据未修改。")
