from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
from PIL import Image

# 数据路径
DATA_ROOT = Path(r"E:\LungLens_Data")
RAW_DIR = DATA_ROOT / "raw" / "subset0"
CSV_PATH = DATA_ROOT / "annotations" / "annotations.csv"

# 本次仅制作实验样本，不创建正式训练/验证划分
OUTPUT = DATA_ROOT / "processed" / "demo"
IMAGE_DIR = OUTPUT / "images"
LABEL_DIR = OUTPUT / "labels"

IMAGE_DIR.mkdir(parents=True, exist_ok=True)
LABEL_DIR.mkdir(parents=True, exist_ok=True)

# 1. 寻找已有 CT
ct_files = sorted(RAW_DIR.glob("*.mhd"))

if not ct_files:
    raise FileNotFoundError("未找到 .mhd 文件")

ct_path = ct_files[0]
uid = ct_path.stem

print("CT:", uid)

# 2. 读取 CT 与标注
image = sitk.ReadImage(str(ct_path))
ct = sitk.GetArrayFromImage(image)
df = pd.read_csv(CSV_PATH)

nodules = df[df["seriesuid"].astype(str) == uid]

if nodules.empty:
    raise RuntimeError("当前病例没有对应结节标注")

spacing_x, spacing_y, _ = image.GetSpacing()

# 3. 肺窗处理函数
def lung_window(arr, width=1500, level=-600):
    hu_min = level - width / 2
    hu_max = level + width / 2

    clipped = np.clip(arr, hu_min, hu_max)
    normalized = (clipped - hu_min) / width

    return np.round(normalized * 255).astype(np.uint8)

# 4. 按中心切片收集结节标签
slice_labels = {}

for _, row in nodules.iterrows():
    world = (
        float(row["coordX"]),
        float(row["coordY"]),
        float(row["coordZ"])
    )

    x, y, z = image.TransformPhysicalPointToContinuousIndex(world)
    z_idx = int(round(z))

    if not 0 <= z_idx < ct.shape[0]:
        continue

    h, w = ct[z_idx].shape
    diameter = float(row["diameter_mm"])

    # 将标注直径从毫米转换为像素
    box_w = diameter / spacing_x
    box_h = diameter / spacing_y

    # 先计算像素边界，再裁剪到图像范围
    x1 = max(0.0, x - box_w / 2)
    y1 = max(0.0, y - box_h / 2)
    x2 = min(float(w), x + box_w / 2)
    y2 = min(float(h), y + box_h / 2)

    if x2 <= x1 or y2 <= y1:
        continue

    # YOLO 归一化格式
    xc = (x1 + x2) / (2 * w)
    yc = (y1 + y2) / (2 * h)
    bw = (x2 - x1) / w
    bh = (y2 - y1) / h

    label = f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}"

    slice_labels.setdefault(z_idx, []).append(label)

# 5. 保存 CT 切片与对应标签
for z_idx, labels in sorted(slice_labels.items()):
    slice_data = ct[z_idx]

    image_uint8 = lung_window(slice_data)

    name = f"{uid}_z{z_idx:03d}"

    image_path = IMAGE_DIR / f"{name}.png"
    label_path = LABEL_DIR / f"{name}.txt"

    Image.fromarray(image_uint8, mode="L").save(image_path)

    label_path.write_text(
        "\n".join(labels) + "\n",
        encoding="utf-8"
    )

    print("\n保存图像:", image_path)
    print("保存标签:", label_path)
    print("标签内容:")
    print("\n".join(labels))

print("\nYOLO 实验样本生成成功！")
