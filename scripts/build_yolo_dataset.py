
from pathlib import Path
import json
import numpy as np
import pandas as pd
import SimpleITK as sitk
from PIL import Image

# ========== 1. 项目路径 ==========
ROOT = Path(r"E:\LungLens_Data")
RAW = ROOT / "raw" / "subset0"
CSV = ROOT / "annotations" / "annotations.csv"
SPLIT = ROOT / "processed" / "dataset_split.json"
OUT = ROOT / "processed" / "yolo_v1"

df = pd.read_csv(CSV)
split_info = json.loads(SPLIT.read_text(encoding="utf-8"))

# ========== 2. 肺窗处理 ==========
def lung_window(array, width=1500, level=-600):
    lower = level - width / 2
    clipped = np.clip(array, lower, lower + width)
    normalized = (clipped - lower) / width

    return np.round(normalized * 255).astype(np.uint8)


# ========== 3. 生成一个病例的样本 ==========
def process_case(uid, subset):
    ct_path = RAW / f"{uid}.mhd"

    if not ct_path.exists():
        raise FileNotFoundError(ct_path)

    image = sitk.ReadImage(str(ct_path))
    ct = sitk.GetArrayFromImage(image)

    sx, sy, _ = image.GetSpacing()

    nodules = df[df["seriesuid"].astype(str) == uid]
    if nodules.empty:
        raise ValueError(f"病例缺少结节标注: {uid}")

    # 同一切片可能有多个结节
    labels_by_slice = {}

    for _, row in nodules.iterrows():
        point = (
            float(row["coordX"]),
            float(row["coordY"]),
            float(row["coordZ"])
        )

        x, y, z = image.TransformPhysicalPointToContinuousIndex(
            point
        )

        z_idx = int(round(z))

        if not 0 <= z_idx < ct.shape[0]:
            raise ValueError(f"结节超出切片范围: {uid}")

        height, width = ct[z_idx].shape

        diameter = float(row["diameter_mm"])

        # 直径(mm) -> 像素长度
        box_w = diameter / sx
        box_h = diameter / sy

        # 裁剪到图像边界
        x1 = max(0.0, x - box_w / 2)
        y1 = max(0.0, y - box_h / 2)
        x2 = min(float(width), x + box_w / 2)
        y2 = min(float(height), y + box_h / 2)

        if x2 <= x1 or y2 <= y1:
            raise ValueError(f"无效结节框: {uid}")

        # 生成 YOLO 归一化标签
        xc = (x1 + x2) / (2 * width)
        yc = (y1 + y2) / (2 * height)
        bw = (x2 - x1) / width
        bh = (y2 - y1) / height

        label = f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}"

        labels_by_slice.setdefault(z_idx, []).append(label)

    image_dir = OUT / "images" / subset
    label_dir = OUT / "labels" / subset

    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    # ========== 4. 保存 PNG 和 TXT ==========
    for z_idx, labels in sorted(labels_by_slice.items()):
        array = lung_window(ct[z_idx])

        name = f"{uid}_z{z_idx:03d}"

        image_path = image_dir / f"{name}.png"
        label_path = label_dir / f"{name}.txt"

        Image.fromarray(array).save(image_path)

        label_path.write_text(
            "\n".join(labels) + "\n",
            encoding="utf-8"
        )

    print(
        f"[{subset}] {uid[-12:]} "
        f"图片:{len(labels_by_slice)} "
        f"结节:{len(nodules)}"
    )

    return len(labels_by_slice), len(nodules)


# ========== 5. 读取已保存的病例划分 ==========
train_cases = split_info["train"]
val_cases = split_info["val"]

assert set(train_cases).isdisjoint(set(val_cases))

stats = {}

for subset, cases in [
    ("train", train_cases),
    ("val", val_cases)
]:
    images = 0
    nodules = 0

    for uid in cases:
        n_images, n_nodules = process_case(uid, subset)
        images += n_images
        nodules += n_nodules

    stats[subset] = (images, nodules)


# ========== 6. 生成 YOLO 配置 ==========
yaml_path = OUT / "dataset.yaml"

yaml_content = (
    f"path: '{OUT.as_posix()}'\n"
    "train: images/train\n"
    "val: images/val\n"
    "names:\n"
    "  0: lung_nodule\n"
)

yaml_path.write_text(yaml_content, encoding="utf-8")


# ========== 7. 验证标签 ==========
for subset in ("train", "val"):
    image_dir = OUT / "images" / subset
    label_dir = OUT / "labels" / subset

    for image_path in image_dir.glob("*.png"):
        label_path = label_dir / (image_path.stem + ".txt")

        assert label_path.exists(), f"标签缺失: {image_path}"

        for line in label_path.read_text(
            encoding="utf-8"
        ).splitlines():
            parts = line.split()

            assert len(parts) == 5
            assert parts[0] == "0"

            values = [float(v) for v in parts[1:]]

            assert all(0 <= v <= 1 for v in values)
            assert values[2] > 0 and values[3] > 0

            x, y, w, h = values

            assert x - w / 2 >= -1e-5
            assert y - h / 2 >= -1e-5
            assert x + w / 2 <= 1 + 1e-5
            assert y + h / 2 <= 1 + 1e-5


print("\n========== 数据集构建完成 ==========")

for subset, (images, nodules) in stats.items():
    print(f"{subset}: 图片 {images} 张，结节 {nodules} 个")

print("配置文件:", yaml_path)
print("标签格式检查通过")
