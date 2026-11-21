
from pathlib import Path
from collections import Counter
import csv
import random

import numpy as np
from PIL import Image, ImageDraw


ROOT = Path(r"E:\LungLens_Data\processed")
DATA = ROOT / "fp_reduction_v2"
SPLITS = ROOT / "yolo_v41/case_split.csv"

OUT = DATA / "review"
EXPECTED_SHAPE = (9, 64, 64)


def load_csv(path):
    with path.open(
        "r", encoding="utf-8-sig", newline=""
    ) as f:
        return list(csv.DictReader(f))


def make_review(rows, label, name):
    random.seed(42)

    selected = random.sample(
        rows, min(12, len(rows))
    )

    tiles = []

    for row in selected:
        folder = "positive" if label == 1 else "negative"
        path = DATA / folder / row["filename"]

        with np.load(path) as npz:
            patch = npz["image"].astype(np.float32)

        # 展示同一个3D图块的Z=2、4、6切片
        tile = Image.new("RGB", (600, 225), "white")
        draw = ImageDraw.Draw(tile)

        draw.text(
            (8, 5),
            f"{row['filename']} "
            f"label={label} "
            f"conf={row['yolo_confidence'] or '-'}",
            fill="black"
        )

        for i, z in enumerate((2, 4, 6)):
            image = Image.fromarray(
                np.clip(patch[z] * 255, 0, 255)
                .astype(np.uint8)
            ).convert("RGB")

            image = image.resize((190, 190))

            # 中心十字，仅标示图块裁剪中心
            d = ImageDraw.Draw(image)
            d.line((89, 95, 101, 95), fill="red", width=1)
            d.line((95, 89, 95, 101), fill="red", width=1)

            tile.paste(image, (i * 200, 30))

        tiles.append(tile)

    if not tiles:
        return

    sheet = Image.new(
        "RGB",
        (1200, ((len(tiles) + 1) // 2) * 225),
        "#dddddd"
    )

    for i, tile in enumerate(tiles):
        sheet.paste(
            tile,
            ((i % 2) * 600, (i // 2) * 225)
        )

    destination = OUT / name
    sheet.save(destination, quality=94)
    print("复核图:", destination)


def main():
    manifest = load_csv(DATA / "manifest.csv")
    case_split = {
        r["seriesuid"]: r["split"]
        for r in load_csv(SPLITS)
    }

    print("========== 3D数据审计 ==========")

    stats = Counter()
    seen = set()
    positives = []
    negatives = []

    for row in manifest:
        filename = row["filename"]
        uid = row["seriesuid"]
        label = int(row["label"])

        if filename in seen:
            raise RuntimeError(f"重复样本: {filename}")

        seen.add(filename)

        if case_split.get(uid) != "train":
            raise RuntimeError(
                f"非Train病例混入: {uid}"
            )

        if row["split"] != "train":
            raise RuntimeError(
                f"样本split错误: {filename}"
            )

        if label not in (0, 1):
            raise RuntimeError(
                f"标签异常: {filename}"
            )

        folder = "positive" if label == 1 else "negative"
        path = DATA / folder / filename

        if not path.is_file():
            raise FileNotFoundError(path)

        with np.load(path, allow_pickle=False) as npz:
            if set(npz.files) != {"image"}:
                raise RuntimeError(
                    f"NPZ字段异常: {filename}"
                )
            patch = npz["image"]

        if patch.shape != EXPECTED_SHAPE:
            raise RuntimeError(
                f"维度异常: {filename}: {patch.shape}"
            )

        if not np.issubdtype(patch.dtype, np.number):
            raise RuntimeError(
                f"数据类型异常: {filename}"
            )

        if not np.all(np.isfinite(patch)):
            raise RuntimeError(
                f"包含NaN/Inf: {filename}"
            )

        if patch.min() < 0 or patch.max() > 1:
            raise RuntimeError(
                f"像素值越界: {filename}"
            )

        stats["total"] += 1
        stats[folder] += 1

        if label == 1:
            positives.append(row)
        else:
            negatives.append(row)

    # 检查是否有清单之外的额外文件
    disk_files = {
        p.name
        for folder in ("positive", "negative")
        for p in (DATA / folder).glob("*.npz")
    }

    if disk_files != seen:
        raise RuntimeError(
            "磁盘文件和manifest.csv不一致"
        )

    print("审计样本:", stats["total"])
    print("正样本:", stats["positive"])
    print("困难负样本:", stats["negative"])
    print("全部样本尺寸:", EXPECTED_SHAPE)
    print("文件完整性、数值范围、病例划分检查通过")

    OUT.mkdir(parents=True, exist_ok=True)

    make_review(
        positives, 1, "positive_review.jpg"
    )

    make_review(
        negatives, 0, "negative_review.jpg"
    )

    print("\n审计和可视化完成")
    print("原始样本未修改")


if __name__ == "__main__":
    main()

