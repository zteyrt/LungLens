
from pathlib import Path
from collections import Counter
import csv
from PIL import Image

DATA = Path(r"E:\LungLens_Data\processed\yolo_v4")

SPLITS = ("train", "val", "test")

# 读取病例划分
with (DATA / "case_split.csv").open(
    "r", encoding="utf-8-sig", newline=""
) as f:
    split_map = {
        row["seriesuid"]: row["split"]
        for row in csv.DictReader(f)
    }

# 读取图片清单
manifest = {}

with (DATA / "sample_manifest.csv").open(
    "r", encoding="utf-8-sig", newline=""
) as f:
    for row in csv.DictReader(f):
        uid = row["seriesuid"]
        zi = int(row["z_index"])
        key = f"{uid}_z{zi:03d}"

        if key in manifest:
            raise RuntimeError(f"图片清单重复：{key}")

        manifest[key] = row

# 读取结节审计
with (DATA / "nodule_audit.csv").open(
    "r", encoding="utf-8-sig", newline=""
) as f:
    nodule_rows = list(csv.DictReader(f))

nodule_keys = set()

for row in nodule_rows:
    key = (
        row["seriesuid"],
        row["annotation_index"]
    )

    if key in nodule_keys:
        raise RuntimeError(f"重复结节记录：{key}")

    nodule_keys.add(key)

    if row["label_generated"] != "1":
        raise RuntimeError(f"结节没有生成标签：{key}")

    if split_map.get(row["seriesuid"]) != row["split"]:
        raise RuntimeError(f"结节病例划分错误：{key}")


print("========== LungLens V4 数据审计 ==========")

all_image_stems = set()
all_case_splits = {}
all_counts = Counter()

for split in SPLITS:

    image_dir = DATA / "images" / split
    label_dir = DATA / "labels" / split

    images = sorted(image_dir.glob("*.png"))
    labels = sorted(label_dir.glob("*.txt"))

    image_stems = {p.stem for p in images}
    label_stems = {p.stem for p in labels}

    if len(image_stems) != len(images):
        raise RuntimeError(f"{split} 存在重复图片文件名")

    if image_stems != label_stems:
        raise RuntimeError(
            f"{split} 图片和标签文件不匹配："
            f"缺少标签 {len(image_stems-label_stems)}，"
            f"多余标签 {len(label_stems-image_stems)}"
        )

    counts = Counter()

    for image_path in images:

        stem = image_path.stem
        uid = stem.rsplit("_z", 1)[0]

        # 检查病例划分
        if split_map.get(uid) != split:
            raise RuntimeError(
                f"病例划分错误：{uid} 当前目录 {split}"
            )

        if uid in all_case_splits:
            if all_case_splits[uid] != split:
                raise RuntimeError(
                    f"病例跨数据集泄漏：{uid}"
                )

        all_case_splits[uid] = split

        # 检查图片清单
        if stem not in manifest:
            raise RuntimeError(f"图片不在清单中：{stem}")

        record = manifest[stem]

        if record["split"] != split:
            raise RuntimeError(f"清单划分错误：{stem}")

        # 检查图片
        with Image.open(image_path) as image:
            if image.size != (512, 512):
                raise RuntimeError(
                    f"图片尺寸异常：{stem} {image.size}"
                )

            if image.mode != "L":
                raise RuntimeError(
                    f"图片不是灰度图：{stem} {image.mode}"
                )

        # 检查标签
        label_path = label_dir / f"{stem}.txt"

        lines = label_path.read_text(
            encoding="utf-8"
        ).splitlines()

        if len(lines) != int(record["label_count"]):
            raise RuntimeError(
                f"标签数量与清单不一致：{stem}"
            )

        if lines:
            counts["positive"] += 1
        else:
            counts["negative"] += 1

        for line in lines:

            values = line.split()

            if len(values) != 5:
                raise RuntimeError(
                    f"标签格式错误：{stem}"
                )

            cls, cx, cy, bw, bh = map(float, values)

            if not all(
                __import__("math").isfinite(v)
                for v in (cls, cx, cy, bw, bh)
            ):
                raise RuntimeError(f"标签包含非有限数值：{stem}")

            if cls != 0:
                raise RuntimeError(f"类别编号错误：{stem}")

            if not (
                0 <= cx <= 1
                and 0 <= cy <= 1
                and 0 < bw <= 1
                and 0 < bh <= 1
                and cx - bw / 2 >= -1e-5
                and cy - bh / 2 >= -1e-5
                and cx + bw / 2 <= 1.00001
                and cy + bh / 2 <= 1.00001
            ):
                raise RuntimeError(
                    f"标签坐标越界：{stem} {line}"
                )

            counts["boxes"] += 1

        all_image_stems.add(stem)

    # 检查清单是否与实际图片一致
    expected_stems = {
        stem for stem, row in manifest.items()
        if row["split"] == split
    }

    if image_stems != expected_stems:
        raise RuntimeError(
            f"{split} 图片清单与实际文件不一致"
        )

    print(f"\n{split.upper()}")
    print("图片总数:", len(images))
    print("正样本图片:", counts["positive"])
    print("负样本图片:", counts["negative"])
    print("标签框数量:", counts["boxes"])
    print("病例数量:", len({
        p.stem.rsplit("_z", 1)[0] for p in images
    }))

    all_counts.update(counts)


# 检查所有病例是否都被覆盖
if set(split_map) != set(all_case_splits):
    missing = set(split_map) - set(all_case_splits)
    raise RuntimeError(
        f"存在没有生成任何图片的病例：{missing}"
    )

# 检查结节审计
if len(nodule_rows) != 293:
    raise RuntimeError(
        f"结节审计记录数量异常：{len(nodule_rows)}"
    )

if len(manifest) != len(all_image_stems):
    raise RuntimeError("图片清单总数异常")

print("\n========== V4 最终审计 ==========")
print("病例总数:", len(all_case_splits))
print("图片总数:", len(all_image_stems))
print("正样本图片:", all_counts["positive"])
print("负样本图片:", all_counts["negative"])
print("标签框总数:", all_counts["boxes"])
print("独立参考结节审计记录:", len(nodule_rows))

print("\n所有结构、文件、标签坐标及病例划分检查通过。")
print("注意：通过程序审计不等于医学标注已通过人工验证。")
