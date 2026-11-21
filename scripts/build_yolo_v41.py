
from pathlib import Path
from collections import Counter
import csv
import shutil

from PIL import Image


# ==========================================
# 1. 路径和配置
# ==========================================

ROOT = Path(r"E:\LungLens_Data\processed")
SOURCE = ROOT / "yolo_v4"
OUTPUT = ROOT / "yolo_v41"

QUALITY_CSV = SOURCE / "review" / "quality_screen.csv"

SPLITS = ("train", "val", "test")
MIN_BOX_PX = 3.0

# False：测试集不清洗，保留独立评估条件
FILTER_TEST = False


# ==========================================
# 2. 基础检查
# ==========================================

def check_paths():
    if not SOURCE.exists():
        raise FileNotFoundError(
            f"找不到V4数据集: {SOURCE}"
        )

    if OUTPUT.exists():
        raise RuntimeError(
            f"输出目录已经存在: {OUTPUT}\n"
            "为防止覆盖，请不要直接删除旧文件。"
        )

    for split in SPLITS:
        for folder in ("images", "labels"):
            path = SOURCE / folder / split

            if not path.is_dir():
                raise FileNotFoundError(path)

    if not QUALITY_CSV.is_file():
        raise FileNotFoundError(
            f"请先运行 screen_v4_quality.py:\n{QUALITY_CSV}"
        )

    if not (SOURCE / "case_split.csv").is_file():
        raise FileNotFoundError(
            SOURCE / "case_split.csv"
        )


# ==========================================
# 3. 加载V4筛查结果
# ==========================================

def load_flags():
    flags = {}

    with QUALITY_CSV.open(
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as f:
        for row in csv.DictReader(f):
            key = (row["split"], row["image"])

            if key in flags:
                raise RuntimeError(
                    f"筛查记录重复: {key}"
                )

            flags[key] = row["review_flag"]

    return flags


# ==========================================
# 4. 读取并检查YOLO标签
# ==========================================

def read_boxes(label_path, width, height):
    lines = label_path.read_text(
        encoding="utf-8-sig"
    ).splitlines()

    box_sizes = []

    for line in lines:
        parts = line.split()

        if len(parts) != 5:
            raise ValueError(
                f"标签格式错误: {label_path}"
            )

        cls, cx, cy, bw, bh = map(float, parts)

        if cls != 0:
            raise ValueError(
                f"类别错误: {label_path}"
            )

        if not (
            0 <= cx <= 1 and 0 <= cy <= 1
            and 0 < bw <= 1 and 0 < bh <= 1
            and cx - bw / 2 >= -0.00001
            and cy - bh / 2 >= -0.00001
            and cx + bw / 2 <= 1.00001
            and cy + bh / 2 <= 1.00001
        ):
            raise ValueError(
                f"标签坐标异常: {label_path}"
            )

        box_width = bw * width
        box_height = bh * height

        box_sizes.append(
            min(box_width, box_height)
        )

    return lines, box_sizes


# ==========================================
# 5. 判断是否保留图片
# ==========================================

def decide_keep(split, filename, box_sizes, flags):
    # 独立测试集必须保留
    if split == "test" and not FILTER_TEST:
        return True, "KEEP_TEST"

    # 正样本：只要存在极小弱框，
    # 就排除整张图片，避免删除框后产生漏标
    if box_sizes:
        if min(box_sizes) < MIN_BOX_PX:
            return False, "TINY_BOX"

        return True, "KEEP_POSITIVE"

    # 无标签图：排除筛查发现的低肺野候选
    key = (split, filename)

    if key not in flags:
        raise RuntimeError(
            f"缺少质量筛查记录: {key}"
        )

    if flags[key] == "LOW_LUNG_CANDIDATE":
        return False, "LOW_LUNG"

    return True, "KEEP_UNLABELED"


# ==========================================
# 6. 主程序
# ==========================================

def main():
    check_paths()
    flags = load_flags()

    print("========== LungLens V4.1 清洗 ==========")
    print("来源:", SOURCE)
    print("输出:", OUTPUT)
    print("极小弱框阈值:", MIN_BOX_PX, "px")
    print("测试集保持不变")

    for split in SPLITS:
        for kind in ("images", "labels"):
            (OUTPUT / kind / split).mkdir(
                parents=True,
                exist_ok=True
            )

    stats = Counter()
    manifest = []
    filter_log = []
    kept_cases = set()

    # 读取原有病例分组
    with (SOURCE / "case_split.csv").open(
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as f:
        original_splits = {
            row["seriesuid"]: row["split"]
            for row in csv.DictReader(f)
        }

    # ======================================
    # 逐张筛选
    # ======================================

    for split in SPLITS:
        image_dir = SOURCE / "images" / split
        label_dir = SOURCE / "labels" / split

        images = sorted(image_dir.glob("*.png"))

        print(f"\n正在处理 {split.upper()}")

        for image_path in images:
            uid = image_path.stem.rsplit("_z", 1)[0]

            if original_splits.get(uid) != split:
                raise RuntimeError(
                    f"病例划分不匹配: {uid}"
                )

            label_path = (
                label_dir / f"{image_path.stem}.txt"
            )

            if not label_path.is_file():
                raise FileNotFoundError(label_path)

            with Image.open(image_path) as im:
                width, height = im.size

            lines, sizes = read_boxes(
                label_path, width, height
            )

            keep, reason = decide_keep(
                split,
                image_path.name,
                sizes,
                flags
            )

            stats[f"{split}_original"] += 1

            if keep:
                dst_image = (
                    OUTPUT / "images" /
                    split / image_path.name
                )

                dst_label = (
                    OUTPUT / "labels" /
                    split / label_path.name
                )

                shutil.copy2(image_path, dst_image)
                shutil.copy2(label_path, dst_label)

                kept_cases.add(uid)

                kind = (
                    "positive" if lines
                    else "unlabeled"
                )

                stats[f"{split}_kept"] += 1
                stats[f"{split}_{kind}"] += 1
                stats[f"{split}_boxes"] += len(lines)

                zi = int(
                    image_path.stem.rsplit("_z", 1)[1]
                )

                manifest.append([
                    uid,
                    split,
                    zi,
                    kind,
                    len(lines)
                ])

            else:
                stats[f"{split}_removed"] += 1
                stats[f"{split}_{reason}"] += 1

            filter_log.append([
                split,
                image_path.name,
                "KEEP" if keep else "REMOVE",
                reason,
                len(lines),
                round(min(sizes), 3) if sizes else ""
            ])

        print(
            "原图片:",
            stats[f"{split}_original"]
        )
        print(
            "保留:",
            stats[f"{split}_kept"]
        )
        print(
            "排除:",
            stats[f"{split}_removed"]
        )
        print(
            "极小弱框排除:",
            stats[f"{split}_TINY_BOX"]
        )
        print(
            "低肺野候选排除:",
            stats[f"{split}_LOW_LUNG"]
        )
        print(
            "保留正样本:",
            stats[f"{split}_positive"]
        )
        print(
            "保留无标签图:",
            stats[f"{split}_unlabeled"]
        )
        print(
            "保留标签框:",
            stats[f"{split}_boxes"]
        )

    # ======================================
    # 7. 检查病例覆盖
    # ======================================

    missing_cases = (
        set(original_splits) - kept_cases
    )

    if missing_cases:
        raise RuntimeError(
            "清洗后有病例没有保留任何图片:\n"
            + "\n".join(sorted(missing_cases))
        )

    # ======================================
    # 8. 保存病例划分
    # ======================================

    shutil.copy2(
        SOURCE / "case_split.csv",
        OUTPUT / "case_split.csv"
    )

    # ======================================
    # 9. 保存样本清单
    # ======================================

    with (OUTPUT / "sample_manifest.csv").open(
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

        writer.writerows(manifest)

    # ======================================
    # 10. 保存清洗记录
    # ======================================

    with (OUTPUT / "filter_log.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as f:
        writer = csv.writer(f)

        writer.writerow([
            "split",
            "image",
            "action",
            "reason",
            "original_boxes",
            "min_box_px"
        ])

        writer.writerows(filter_log)

    # ======================================
    # 11. 保存YOLO配置
    # ======================================

    yaml = (
        f"path: '{OUTPUT.as_posix()}'\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "nc: 1\n"
        "names:\n"
        "  0: lung_nodule\n"
    )

    (OUTPUT / "dataset.yaml").write_text(
        yaml,
        encoding="utf-8"
    )

    # ======================================
    # 12. 最终汇总
    # ======================================

    print("\n========== V4.1 构建完成 ==========")

    for split in SPLITS:
        print(
            f"{split.upper()}: "
            f"{stats[f'{split}_original']} -> "
            f"{stats[f'{split}_kept']} 张"
        )

    print(
        "\n总保留图片:",
        sum(stats[f"{s}_kept"] for s in SPLITS)
    )
    print(
        "总保留标签框:",
        sum(stats[f"{s}_boxes"] for s in SPLITS)
    )

    print("\n保存路径:", OUTPUT)
    print("清洗日志: filter_log.csv")
    print("样本清单: sample_manifest.csv")
    print("病例划分: case_split.csv")
    print("数据配置: dataset.yaml")
    print("\n原始V4数据未修改。")
    print("注意：V4.1需重新进行标签质量审计。")


if __name__ == "__main__":
    main()
