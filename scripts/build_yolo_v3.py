
from pathlib import Path
from collections import defaultdict, Counter
import csv
import random
import math

import numpy as np
import SimpleITK as sitk
from PIL import Image

ROOT = Path(r"E:\LungLens_Data")
RAW = ROOT / "raw"
ANN = ROOT / "annotations" / "annotations.csv"
V2 = ROOT / "processed" / "yolo_v2"
OUT = ROOT / "processed" / "yolo_v3"

# 防止覆盖旧数据
if OUT.exists():
    raise RuntimeError(f"输出目录已存在：{OUT}")

# 1. 读取参考结节
annotations = defaultdict(list)

with ANN.open("r", encoding="utf-8-sig", newline="") as f:
    for row in csv.DictReader(f):
        annotations[row["seriesuid"]].append(row)

# 2. 收集全部subset的CT
cases = {}

for mhd in sorted(RAW.glob("subset*/*.mhd")):
    uid = mhd.stem

    if uid in cases:
        raise RuntimeError(f"重复病例UID：{uid}")

    if not mhd.with_suffix(".raw").exists():
        raise FileNotFoundError(mhd.with_suffix(".raw"))

    cases[uid] = mhd

print("发现CT病例:", len(cases))
print(
    "对应参考结节:",
    sum(len(annotations[u]) for u in cases)
)

# 3. 锁定旧版病例归属
split_map = {}

for split in ("train", "val"):
    for p in (V2 / "images" / split).glob("*.png"):
        uid = p.stem.rsplit("_z", 1)[0]

        if uid in split_map and split_map[uid] != split:
            raise RuntimeError("发现病例划分冲突")

        split_map[uid] = split

# 4. 将新增有参考结节的病例划分
new_positive = sorted(
    u for u in cases
    if u not in split_map and len(annotations[u]) > 0
)

random.Random(42).shuffle(new_positive)

n = len(new_positive)
n_test = round(n * 0.15)
n_val = round(n * 0.15)

for i, uid in enumerate(new_positive):
    if i < n_test:
        split_map[uid] = "test"
    elif i < n_test + n_val:
        split_map[uid] = "val"
    else:
        split_map[uid] = "train"

# 5. 创建目录
for split in ("train", "val", "test"):
    for kind in ("images", "labels"):
        (OUT / kind / split).mkdir(
            parents=True, exist_ok=True
        )

stats = Counter()

def lung_window(pixels):
    low = -1350.0
    high = 150.0
    clipped = np.clip(pixels, low, high)

    return (
        (clipped - low) / (high - low) * 255
    ).astype(np.uint8)

# 6. 构建结节中心切片
for uid, split in sorted(split_map.items()):
    image = sitk.ReadImage(str(cases[uid]))
    w, h, depth = image.GetSize()
    sx, sy, sz = image.GetSpacing()

    nodules = []

    for row in annotations[uid]:
        world = tuple(
            float(row[key])
            for key in ("coordX", "coordY", "coordZ")
        )

        x, y, z = image.TransformPhysicalPointToContinuousIndex(
            world
        )

        diameter = float(row["diameter_mm"])
        zi = max(
            0,
            min(depth - 1, math.floor(z + 0.5))
        )

        nodules.append((x, y, z, zi, diameter))

    # 同一切片的中心结节合并
    groups = defaultdict(list)

    for index, item in enumerate(nodules):
        groups[item[3]].append(index)

    # 只对需要的中心切片转换图像
    volume = sitk.GetArrayFromImage(image)

    for zi, center_indices in sorted(groups.items()):
        # 保守检查潜在的其他可见结节
        ambiguous = False

        for j, item in enumerate(nodules):
            if j in center_indices:
                continue

            x, y, z, center, diameter = item
            distance = abs(zi - z) * sz

            if distance <= diameter / 2 + sz / 2:
                ambiguous = True
                break

        if ambiguous:
            stats["skipped_images"] += 1
            continue

        name = f"{uid}_z{zi:03d}"
        pixels = lung_window(volume[zi])

        lines = []

        for index in center_indices:
            x, y, z, center, diameter = nodules[index]

            # 近似二维弱标注框
            rx = diameter / (2 * sx)
            ry = diameter / (2 * sy)

            x1 = max(0.0, x - rx)
            y1 = max(0.0, y - ry)
            x2 = min(float(w), x + rx)
            y2 = min(float(h), y + ry)

            if x2 <= x1 or y2 <= y1:
                raise ValueError(f"无效检测框：{uid}")

            cx = (x1 + x2) / (2 * w)
            cy = (y1 + y2) / (2 * h)
            bw = (x2 - x1) / w
            bh = (y2 - y1) / h

            lines.append(
                f"0 {cx:.6f} {cy:.6f} "
                f"{bw:.6f} {bh:.6f}"
            )

        Image.fromarray(pixels).save(
            OUT / "images" / split / f"{name}.png"
        )

        (OUT / "labels" / split / f"{name}.txt").write_text(
            "\n".join(lines) + "\n",
            encoding="utf-8"
        )

        stats[f"{split}_images"] += 1
        stats[f"{split}_labels"] += len(lines)

    print(
        f"{uid[-12:]} | {split} | "
        f"参考结节 {len(nodules)}"
    )

# 7. YOLO配置
yaml = (
    f"path: '{OUT.as_posix()}'\n"
    "train: images/train\n"
    "val: images/val\n"
    "test: images/test\n"
    "nc: 1\n"
    "names:\n"
    "  0: lung_nodule\n"
)

(OUT / "dataset.yaml").write_text(
    yaml, encoding="utf-8"
)

# 保存分组，后续训练保持不变
with (OUT / "case_split.csv").open(
    "w", encoding="utf-8", newline=""
) as f:
    writer = csv.writer(f)
    writer.writerow(["seriesuid", "split"])

    for uid, split in sorted(split_map.items()):
        writer.writerow([uid, split])

print("\n========== YOLO V3 结果 ==========")

for split in ("train", "val", "test"):
    print(
        split,
        "病例:", sum(s == split for s in split_map.values()),
        "图片:", stats[f"{split}_images"],
        "标签:", stats[f"{split}_labels"]
    )

print("跳过的待复核切片:", stats["skipped_images"])
print("数据集位置:", OUT)
print("原始CT和旧版YOLO数据均未覆盖")
