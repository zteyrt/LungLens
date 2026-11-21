
from pathlib import Path
from collections import defaultdict
import csv
import random

import numpy as np
import SimpleITK as sitk
from PIL import Image

ROOT = Path(r"E:\LungLens_Data")
RAW = ROOT / "raw" / "subset0"
ANN = ROOT / "annotations" / "annotations.csv"
V1 = ROOT / "processed" / "yolo_v1"
V2 = ROOT / "processed" / "yolo_v2"

# 1. 读取全部结节标注
annotations = defaultdict(list)

with ANN.open("r", encoding="utf-8-sig", newline="") as f:
    for row in csv.DictReader(f):
        annotations[row["seriesuid"]].append(row)

uids = sorted(
    p.stem for p in RAW.glob("*.mhd")
    if p.stem in annotations
)

# 2. 保留旧版病例的 Train / Val 归属
old_split = {}

for split in ("train", "val"):
    for image_path in (V1 / "images" / split).glob("*.png"):
        uid = image_path.stem.split("_z")[0]

        if uid in old_split and old_split[uid] != split:
            raise ValueError("旧数据集存在病例泄漏: " + uid)

        old_split[uid] = split

if not old_split:
    raise RuntimeError("未找到 yolo_v1 划分信息")

# 3. 新病例按固定随机种子划分
new_uids = [uid for uid in uids if uid not in old_split]
random.Random(42).shuffle(new_uids)

# 原有6例预计4 train + 2 val
# 新增7例，安排6 train + 1 val
for i, uid in enumerate(new_uids):
    old_split[uid] = "val" if i == 0 else "train"

print("Train病例:", sum(old_split[u] == "train" for u in uids))
print("Val病例:", sum(old_split[u] == "val" for u in uids))

# 4. 创建输出目录
# 首次构建时要求目标为空，防止残留旧标签
for split in ("train", "val"):
    for kind in ("images", "labels"):
        folder = V2 / kind / split
        folder.mkdir(parents=True, exist_ok=True)

        if any(folder.iterdir()):
            raise RuntimeError(
                f"输出目录不是空的，请先检查: {folder}"
            )


def lung_window(ct):
    """肺窗：WL=-600, WW=1500。"""
    low, high = -1350.0, 150.0
    pixels = np.clip(ct, low, high)
    return ((pixels - low) / (high - low) * 255).astype(np.uint8)


image_count = 0
label_count = 0
review_count = 0

# 5. 逐例生成中心切片与YOLO标签
for uid in uids:
    image = sitk.ReadImage(str(RAW / f"{uid}.mhd"))
    volume = sitk.GetArrayFromImage(image)

    width, height, depth = image.GetSize()
    sx, sy, sz = image.GetSpacing()

    # 每个结节的连续体素中心、中心切片和直径
    nodules = []

    for row in annotations[uid]:
        world = (
            float(row["coordX"]),
            float(row["coordY"]),
            float(row["coordZ"]),
        )

        x, y, z = image.TransformPhysicalPointToContinuousIndex(world)
        diameter = float(row["diameter_mm"])

        zi = int(np.floor(z + 0.5))
        zi = max(0, min(depth - 1, zi))

        nodules.append((x, y, z, zi, diameter))

    # 同一中心切片上的多个结节合并成一张图片
    slice_groups = defaultdict(list)

    for index, nodule in enumerate(nodules):
        slice_groups[nodule[3]].append(index)

    split = old_split[uid]

    for zi, indices in sorted(slice_groups.items()):
        pixels = lung_window(volume[zi])
        output_name = f"{uid}_z{zi:03d}"

        image_path = V2 / "images" / split / f"{output_name}.png"
        label_path = V2 / "labels" / split / f"{output_name}.txt"

        Image.fromarray(pixels).save(image_path)

        lines = []

        for index in indices:
            x, y, z, center_z, diameter = nodules[index]

            bw = diameter / sx / width
            bh = diameter / sy / height

            # 转换成YOLO归一化格式
            cx = x / width
            cy = y / height

            # 将方框限制在图片范围内
            x1 = max(0.0, cx - bw / 2)
            y1 = max(0.0, cy - bh / 2)
            x2 = min(1.0, cx + bw / 2)
            y2 = min(1.0, cy + bh / 2)

            if x2 <= x1 or y2 <= y1:
                raise ValueError("无效的结节框")

            lines.append(
                f"0 {(x1+x2)/2:.6f} {(y1+y2)/2:.6f} "
                f"{x2-x1:.6f} {y2-y1:.6f}"
            )

        label_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

        # 检查同张切片是否可能存在其他非中心结节
        for j, nodule in enumerate(nodules):
            if j in indices:
                continue

            _, _, z, _, diameter = nodule

            distance_mm = abs(zi - z) * sz

            if distance_mm <= diameter / 2 + sz / 2:
                review_count += 1
                print(
                    "需人工复核其他结节是否可见:",
                    uid[-12:], "切片", zi,
                    "距离(mm)", round(distance_mm, 2)
                )

        image_count += 1
        label_count += len(lines)

    print(
        "完成:", uid[-12:],
        "划分:", split,
        "结节:", len(nodules),
        "中心切片:", len(slice_groups)
    )

# 6. 编写数据集配置文件
yaml_text = (
    f"path: '{V2.as_posix()}'\n"
    "train: images/train\n"
    "val: images/val\n"
    "nc: 1\n"
    "names:\n"
    "  0: lung_nodule\n"
)

(V2 / "dataset.yaml").write_text(yaml_text, encoding="utf-8")

print("\n========== YOLO v2 构建完成 ==========")
print("病例数量:", len(uids))
print("图片数量:", image_count)
print("结节标签数量:", label_count)
print("需复核的其他结节情况:", review_count)
print("数据集目录:", V2)


