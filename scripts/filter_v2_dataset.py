
from pathlib import Path
import shutil

ROOT = Path(r"E:\LungLens_Data\processed")
SRC = ROOT / "yolo_v2"
DST = ROOT / "yolo_v2_filtered"

REVIEW = Path(
    r"D:\肺结节模型训练\evaluation\v2_review"
)

# 读取需要排除的图片文件名
exclude = {
    p.name.removeprefix("train_")
    for p in REVIEW.glob("train_*.png")
}

print("需要排除的训练图片:", len(exclude))

if len(exclude) != 8:
    raise RuntimeError("复核图片数量不是8，请先检查")

# 防止误覆盖已有数据集
if DST.exists():
    raise RuntimeError(
        f"目标目录已存在，请先检查: {DST}"
    )

# 检查文件匹配关系
for filename in exclude:
    image_path = SRC / "images/train" / filename
    label_path = SRC / "labels/train" / (
        Path(filename).stem + ".txt"
    )

    if not image_path.exists() or not label_path.exists():
        raise FileNotFoundError(filename)

counts = {}

# 复制安全样本，原始v2不变
for split in ("train", "val"):
    (DST / "images" / split).mkdir(
        parents=True, exist_ok=True
    )
    (DST / "labels" / split).mkdir(
        parents=True, exist_ok=True
    )

    image_count = 0
    label_count = 0

    for image_path in sorted(
        (SRC / "images" / split).glob("*.png")
    ):
        if split == "train" and image_path.name in exclude:
            print("暂时排除:", image_path.name)
            continue

        label_path = SRC / "labels" / split / (
            image_path.stem + ".txt"
        )

        if not label_path.exists():
            raise FileNotFoundError(label_path)

        shutil.copy2(
            image_path, DST / "images" / split / image_path.name
        )
        shutil.copy2(
            label_path, DST / "labels" / split / label_path.name
        )

        image_count += 1
        label_count += len(
            label_path.read_text(encoding="utf-8").splitlines()
        )

    counts[split] = (image_count, label_count)

# 写入独立的数据集配置
yaml_text = (
    f"path: '{DST.as_posix()}'\n"
    "train: images/train\n"
    "val: images/val\n"
    "nc: 1\n"
    "names:\n"
    "  0: lung_nodule\n"
)

(DST / "dataset.yaml").write_text(
    yaml_text, encoding="utf-8"
)

print("\n========== 过滤结果 ==========")
for split, (images, labels) in counts.items():
    print(
        f"{split}: 图片 {images} 张，"
        f"结节标签 {labels} 个"
    )

print("新数据集:", DST)
print("原始 yolo_v2 保持不变")
