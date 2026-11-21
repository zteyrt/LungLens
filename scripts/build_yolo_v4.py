
from pathlib import Path
from collections import defaultdict, Counter
import csv
import math
import random

import numpy as np
import SimpleITK as sitk
from PIL import Image


# =====================================================
# 1. 路径与参数
# =====================================================

ROOT = Path(r"E:\LungLens_Data")
RAW = ROOT / "raw"
ANN = ROOT / "annotations" / "annotations.csv"

V3 = ROOT / "processed" / "yolo_v3"
OUT = ROOT / "processed" / "yolo_v4"

# 每个有结节病例最多选取的阴性切片
NEGATIVE_PER_POSITIVE_CASE = 2

# 没有参考结节的病例最多选取的阴性切片
NEGATIVE_PER_NEGATIVE_CASE = 3

# 只保留估算截面直径至少为1mm的标签
MIN_CROSS_SECTION_DIAMETER_MM = 1.0

# 阴性样本避开肺尖、肺底的一部分区域
CENTRAL_Z_START = 0.20
CENTRAL_Z_END = 0.80

RANDOM_SEED = 42

if OUT.exists():
    raise RuntimeError(
        f"V4输出目录已经存在：{OUT}\n"
        "为避免覆盖旧数据，请先检查目录，不要直接删除。"
    )

if not ANN.exists():
    raise FileNotFoundError(ANN)

if not (V3 / "case_split.csv").exists():
    raise FileNotFoundError(V3 / "case_split.csv")


# =====================================================
# 2. 读取参考结节
# =====================================================

annotations = defaultdict(list)

with ANN.open("r", encoding="utf-8-sig", newline="") as f:
    for row in csv.DictReader(f):
        annotations[row["seriesuid"]].append(row)


# =====================================================
# 3. 收集CT病例
# =====================================================

cases = {}

for mhd in sorted(RAW.glob("subset*/*.mhd")):

    uid = mhd.stem
    raw_file = mhd.with_suffix(".raw")

    if uid in cases:
        raise RuntimeError(f"重复病例UID：{uid}")

    if not raw_file.exists():
        raise FileNotFoundError(raw_file)

    cases[uid] = mhd


print("========== LungLens V4 数据构建 ==========")
print("CT病例:", len(cases))
print(
    "参考结节:",
    sum(len(annotations[uid]) for uid in cases)
)


# =====================================================
# 4. 沿用V3病例划分
# =====================================================

with (V3 / "case_split.csv").open(
    "r", encoding="utf-8-sig", newline=""
) as f:

    split_map = {
        row["seriesuid"]: row["split"]
        for row in csv.DictReader(f)
    }

valid_splits = {"train", "val", "test"}

for uid, split in split_map.items():

    if split not in valid_splits:
        raise ValueError(f"非法数据集划分：{uid} {split}")

    if uid not in cases:
        raise FileNotFoundError(f"划分表中找不到CT：{uid}")


# 有参考结节但没有划分的病例不能直接随机补入
unassigned_positive = [
    uid for uid in cases
    if len(annotations[uid]) > 0 and uid not in split_map
]

if unassigned_positive:
    raise RuntimeError(
        "存在有参考结节但没有病例划分的CT："
        + str(unassigned_positive[:10])
    )


# 为无参考结节的病例分配独立病例划分
unassigned_negative = [
    uid for uid in cases
    if len(annotations[uid]) == 0 and uid not in split_map
]

existing_counts = Counter(split_map.values())
existing_total = sum(existing_counts.values())

rng = random.Random(RANDOM_SEED)
rng.shuffle(unassigned_negative)

n_new = len(unassigned_negative)

n_test = round(
    n_new * existing_counts["test"] / existing_total
)

n_val = round(
    n_new * existing_counts["val"] / existing_total
)

for i, uid in enumerate(unassigned_negative):

    if i < n_test:
        split_map[uid] = "test"

    elif i < n_test + n_val:
        split_map[uid] = "val"

    else:
        split_map[uid] = "train"


if set(split_map) != set(cases):
    raise RuntimeError("病例划分没有覆盖全部CT病例")


print("\n病例划分：")

for split in ("train", "val", "test"):
    print(
        split,
        sum(s == split for s in split_map.values()),
        "例"
    )


# =====================================================
# 5. 创建输出目录
# =====================================================

for split in ("train", "val", "test"):

    for kind in ("images", "labels"):

        (OUT / kind / split).mkdir(
            parents=True,
            exist_ok=True
        )


# =====================================================
# 6. 工具函数
# =====================================================

def lung_window(pixels):

    low = -1350.0
    high = 150.0

    clipped = np.clip(pixels, low, high)

    return (
        (clipped - low) /
        (high - low) * 255
    ).astype(np.uint8)


def select_evenly(values, count):

    values = sorted(set(values))

    if count <= 0 or not values:
        return []

    if len(values) <= count:
        return values

    indices = np.linspace(
        0,
        len(values) - 1,
        count
    )

    indices = sorted(set(
        int(round(i)) for i in indices
    ))

    return [values[i] for i in indices]


def get_cross_section(nodule, zi, sz, sx, sy, width, height):

    """
    根据球形近似计算结节在当前横断面的截面框。

    注意：这是弱标注近似，不是真实病灶轮廓。
    """

    x = nodule["x"]
    y = nodule["y"]
    z = nodule["z"]
    diameter = nodule["diameter"]

    radius = diameter / 2.0

    dz_mm = abs(zi - z) * sz

    if dz_mm > radius:
        return None

    radius_sq = radius ** 2 - dz_mm ** 2

    cross_radius = math.sqrt(
        max(0.0, radius_sq)
    )

    cross_diameter = cross_radius * 2

    if cross_diameter < MIN_CROSS_SECTION_DIAMETER_MM:
        return None

    rx = cross_radius / sx
    ry = cross_radius / sy

    x1 = max(0.0, x - rx)
    y1 = max(0.0, y - ry)

    x2 = min(float(width), x + rx)
    y2 = min(float(height), y + ry)

    if x2 <= x1 or y2 <= y1:
        return None

    cx = (x1 + x2) / (2 * width)
    cy = (y1 + y2) / (2 * height)

    bw = (x2 - x1) / width
    bh = (y2 - y1) / height

    return (
        f"0 {cx:.6f} {cy:.6f} "
        f"{bw:.6f} {bh:.6f}"
    )


def is_safe_negative(zi, nodules, sz):

    """
    阴性切片必须避开所有参考结节的可能Z轴范围，
    并额外留出半个切片间距。
    """

    for nodule in nodules:

        dz_mm = abs(zi - nodule["z"]) * sz

        margin = (
            nodule["diameter"] / 2.0
            + sz / 2.0
        )

        if dz_mm <= margin:
            return False

    return True


# =====================================================
# 7. 构建图片、标签和审计清单
# =====================================================

stats = Counter()

manifest_rows = []
nodule_audit_rows = []
case_report_rows = []


for uid, split in sorted(split_map.items()):

    print(f"\n处理病例：{uid[-12:]} | {split}")

    image = sitk.ReadImage(str(cases[uid]))

    width, height, depth = image.GetSize()

    sx, sy, sz = image.GetSpacing()

    if min(sx, sy, sz) <= 0:
        raise ValueError(f"非法体素间距：{uid}")

    volume = sitk.GetArrayFromImage(image)

    if volume.shape != (depth, height, width):
        raise ValueError(f"CT数组维度异常：{uid}")


    # ---------------------------------------------
    # 7.1 转换结节物理坐标
    # ---------------------------------------------

    nodules = []

    for index, row in enumerate(annotations[uid]):

        world = tuple(
            float(row[key])
            for key in ("coordX", "coordY", "coordZ")
        )

        x, y, z = image.TransformPhysicalPointToContinuousIndex(
            world
        )

        diameter = float(row["diameter_mm"])

        if diameter <= 0:
            raise ValueError(f"结节直径异常：{uid}")

        if not (
            0 <= x < width
            and 0 <= y < height
            and 0 <= z < depth
        ):
            raise ValueError(
                f"结节中心超出CT范围：{uid} annotation={index}"
            )

        nodules.append({
            "id": index,
            "x": x,
            "y": y,
            "z": z,
            "diameter": diameter
        })


    # ---------------------------------------------
    # 7.2 生成所有可能包含参考结节的切片标签
    # ---------------------------------------------

    slice_labels = defaultdict(list)

    included_nodules = set()

    for zi in range(depth):

        for nodule in nodules:

            line = get_cross_section(
                nodule,
                zi,
                sz,
                sx,
                sy,
                width,
                height
            )

            if line is None:
                continue

            slice_labels[zi].append(
                (nodule["id"], line)
            )

            included_nodules.add(nodule["id"])


    positive_slices = set(slice_labels.keys())


    # ---------------------------------------------
    # 7.3 选择阴性切片
    # ---------------------------------------------

    z_start = int(depth * CENTRAL_Z_START)
    z_end = int(depth * CENTRAL_Z_END)

    negative_candidates = []

    for zi in range(z_start, z_end):

        if zi in positive_slices:
            continue

        if is_safe_negative(zi, nodules, sz):
            negative_candidates.append(zi)


    if nodules:
        negative_target = NEGATIVE_PER_POSITIVE_CASE
    else:
        negative_target = NEGATIVE_PER_NEGATIVE_CASE


    negative_slices = select_evenly(
        negative_candidates,
        negative_target
    )


    # ---------------------------------------------
    # 7.4 保存图片和标签
    # ---------------------------------------------

    selected_slices = sorted(
        positive_slices | set(negative_slices)
    )

    case_positive_images = 0
    case_negative_images = 0
    case_label_count = 0

    for zi in selected_slices:

        name = f"{uid}_z{zi:03d}"

        pixels = lung_window(volume[zi])

        image_path = (
            OUT / "images" / split / f"{name}.png"
        )

        label_path = (
            OUT / "labels" / split / f"{name}.txt"
        )

        Image.fromarray(pixels).save(image_path)


        entries = slice_labels.get(zi, [])

        lines = [
            line for nodule_id, line in entries
        ]

        label_path.write_text(
            "\n".join(lines) + ("\n" if lines else ""),
            encoding="utf-8"
        )

        if lines:

            sample_type = "positive"
            case_positive_images += 1
            case_label_count += len(lines)

        else:

            sample_type = "negative"
            case_negative_images += 1


        manifest_rows.append([
            uid,
            split,
            zi,
            sample_type,
            len(lines)
        ])


    # ---------------------------------------------
    # 7.5 每个参考结节单独记录是否生成标签
    # ---------------------------------------------

    for nodule in nodules:

        included = nodule["id"] in included_nodules

        nodule_audit_rows.append([
            uid,
            split,
            nodule["id"],
            round(nodule["diameter"], 3),
            int(included)
        ])


    # ---------------------------------------------
    # 7.6 统计病例信息
    # ---------------------------------------------

    case_missing = len(nodules) - len(included_nodules)

    stats[f"{split}_cases"] += 1
    stats[f"{split}_positive_images"] += case_positive_images
    stats[f"{split}_negative_images"] += case_negative_images
    stats[f"{split}_labels"] += case_label_count
    stats["included_unique_nodules"] += len(included_nodules)
    stats["missing_unique_nodules"] += case_missing

    case_report_rows.append([
        uid,
        split,
        len(nodules),
        len(included_nodules),
        case_missing,
        case_positive_images,
        case_negative_images
    ])

    print(
        f"参考结节 {len(nodules)} | "
        f"保留结节 {len(included_nodules)} | "
        f"正样本图片 {case_positive_images} | "
        f"负样本图片 {case_negative_images} | "
        f"未生成结节标签 {case_missing}"
    )


# =====================================================
# 8. 保存数据集划分表
# =====================================================

with (OUT / "case_split.csv").open(
    "w", encoding="utf-8-sig", newline=""
) as f:

    writer = csv.writer(f)

    writer.writerow(["seriesuid", "split"])

    for uid, split in sorted(split_map.items()):
        writer.writerow([uid, split])


# =====================================================
# 9. 保存图片清单
# =====================================================

with (OUT / "sample_manifest.csv").open(
    "w", encoding="utf-8-sig", newline=""
) as f:

    writer = csv.writer(f)

    writer.writerow([
        "seriesuid",
        "split",
        "z_index",
        "sample_type",
        "label_count"
    ])

    writer.writerows(manifest_rows)


# =====================================================
# 10. 保存结节审计表
# =====================================================

with (OUT / "nodule_audit.csv").open(
    "w", encoding="utf-8-sig", newline=""
) as f:

    writer = csv.writer(f)

    writer.writerow([
        "seriesuid",
        "split",
        "annotation_index",
        "diameter_mm",
        "label_generated"
    ])

    writer.writerows(nodule_audit_rows)


# =====================================================
# 11. 保存病例统计表
# =====================================================

with (OUT / "case_report.csv").open(
    "w", encoding="utf-8-sig", newline=""
) as f:

    writer = csv.writer(f)

    writer.writerow([
        "seriesuid",
        "split",
        "reference_nodules",
        "included_nodules",
        "missing_nodules",
        "positive_images",
        "negative_images"
    ])

    writer.writerows(case_report_rows)


# =====================================================
# 12. YOLO数据集配置
# =====================================================

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
    yaml,
    encoding="utf-8"
)


# =====================================================
# 13. 输出最终结果
# =====================================================

print("\n========== LungLens V4 构建完成 ==========")

for split in ("train", "val", "test"):

    print(f"\n{split.upper()}")

    print("病例:", stats[f"{split}_cases"])
    print("正样本图片:", stats[f"{split}_positive_images"])
    print("负样本图片:", stats[f"{split}_negative_images"])
    print("生成标签:", stats[f"{split}_labels"])


print("\n========== 结节保留情况 ==========")

print(
    "已生成标签的独立参考结节:",
    stats["included_unique_nodules"]
)

print(
    "未生成标签的独立参考结节:",
    stats["missing_unique_nodules"]
)

print("\n数据集目录:", OUT)
print("病例划分: case_split.csv")
print("图片清单: sample_manifest.csv")
print("结节审计: nodule_audit.csv")
print("病例报告: case_report.csv")

print("\n原始CT、V3数据集和V3模型均未修改。")
print("注意：当前结节框采用球形截面近似，仍需人工复核。")
print("阴性样本表示没有已知参考结节，不等于临床确认无结节。")
