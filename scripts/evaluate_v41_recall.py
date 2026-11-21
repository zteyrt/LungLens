
from pathlib import Path
from collections import Counter
import csv

from PIL import Image
from ultralytics import YOLO

PROJECT = Path(r"D:\肺结节模型训练")
DATA = Path(r"E:\LungLens_Data\processed\yolo_v41")
WEIGHT = (
    PROJECT / "runs" / "yolo_v41_baseline"
    / "weights" / "best.pt"
)

THRESHOLDS = [0.05, 0.10, 0.15, 0.20, 0.30]
MATCH_IOU = 0.50


def calculate_iou(a, b):
    left = max(a[0], b[0])
    top = max(a[1], b[1])
    right = min(a[2], b[2])
    bottom = min(a[3], b[3])

    intersection = (
        max(0.0, right - left)
        * max(0.0, bottom - top)
    )

    area_a = (
        max(0.0, a[2] - a[0])
        * max(0.0, a[3] - a[1])
    )
    area_b = (
        max(0.0, b[2] - b[0])
        * max(0.0, b[3] - b[1])
    )

    return intersection / (
        area_a + area_b - intersection + 1e-9
    )


def read_labels(image_path):
    label_path = (
        DATA / "labels" / "val"
        / (image_path.stem + ".txt")
    )

    with Image.open(image_path) as image:
        width, height = image.size

    boxes = []

    for line in label_path.read_text(
        encoding="utf-8-sig"
    ).splitlines():
        cls, cx, cy, bw, bh = map(
            float, line.split()
        )

        boxes.append({
            "box": [
                (cx - bw / 2) * width,
                (cy - bh / 2) * height,
                (cx + bw / 2) * width,
                (cy + bh / 2) * height,
            ],
            "size": min(
                bw * width, bh * height
            )
        })

    return boxes


def size_group(size):
    if size < 4:
        return "under4"
    if size < 8:
        return "4to8"
    return "over8"


def evaluate(all_data, threshold):
    stats = Counter()
    size_stats = Counter()
    details = []

    for image_name, gts, preds in all_data:
        selected = [
            p for p in preds
            if p["conf"] >= threshold
        ]

        selected.sort(
            key=lambda p: p["conf"],
            reverse=True
        )

        matched = set()
        tp = 0
        fp = 0

        for pred in selected:
            best_iou = 0.0
            best_index = -1

            for i, gt in enumerate(gts):
                if i in matched:
                    continue

                iou = calculate_iou(
                    pred["box"], gt["box"]
                )

                if iou > best_iou:
                    best_iou = iou
                    best_index = i

            if (
                best_index >= 0
                and best_iou >= MATCH_IOU
            ):
                matched.add(best_index)
                tp += 1
            else:
                fp += 1

        fn = len(gts) - tp

        stats["TP"] += tp
        stats["FP"] += fp
        stats["FN"] += fn
        stats["images"] += 1

        if not gts:
            stats["unlabeled_images"] += 1
            stats["unlabeled_fp"] += fp

        for i, gt in enumerate(gts):
            group = size_group(gt["size"])
            size_stats[f"{group}_total"] += 1

            if i in matched:
                size_stats[f"{group}_hit"] += 1

        details.append([
            image_name,
            len(gts),
            len(selected),
            tp,
            fp,
            fn
        ])

    tp = stats["TP"]
    fp = stats["FP"]
    fn = stats["FN"]

    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0
    fp_per_image = fp / stats["images"]

    return (
        stats, size_stats, details,
        precision, recall, fp_per_image
    )


def main():
    if not WEIGHT.is_file():
        raise FileNotFoundError(WEIGHT)

    images = sorted(
        (DATA / "images" / "val").glob("*.png")
    )

    if not images:
        raise RuntimeError("验证集为空")

    print("========== V4.1 固定阈值评估 ==========")
    print("验证图片:", len(images))
    print("IoU匹配标准:", MATCH_IOU)

    model = YOLO(str(WEIGHT))

    # 一次获取低阈值预测，后面直接统计
    results = model.predict(
        source=[str(p) for p in images],
        imgsz=640,
        batch=1,
        conf=0.01,
        iou=0.7,
        max_det=300,
        device=0,
        stream=True,
        verbose=False
    )

    all_data = []

    for image_path, result in zip(images, results):
        gts = read_labels(image_path)
        preds = []

        for box in result.boxes:
            preds.append({
                "box": box.xyxy[0].cpu().tolist(),
                "conf": float(box.conf[0].item())
            })

        all_data.append(
            (image_path.name, gts, preds)
        )

    if len(all_data) != len(images):
        raise RuntimeError("预测数量与图片数量不一致")

    print(
        "参考框:",
        sum(len(gts) for _, gts, _ in all_data)
    )

    summary = []

    print("\n阈值    TP   FP   FN    P       R      FP/图")

    for threshold in THRESHOLDS:
        (
            stats, sizes, details,
            precision, recall, fp_per_image
        ) = evaluate(all_data, threshold)

        print(
            f"{threshold:.2f}   "
            f"{stats['TP']:3d}  "
            f"{stats['FP']:3d}  "
            f"{stats['FN']:3d}  "
            f"{precision:6.2%}  "
            f"{recall:6.2%}  "
            f"{fp_per_image:.2f}"
        )

        summary.append([
            threshold,
            stats["TP"],
            stats["FP"],
            stats["FN"],
            round(precision, 4),
            round(recall, 4),
            round(fp_per_image, 4)
        ])

        if threshold in (0.10, 0.20):
            print("  按真实框尺寸统计:")

            for key, label in [
                ("under4", "<4px"),
                ("4to8", "4-8px"),
                ("over8", ">=8px"),
            ]:
                total = sizes[f"{key}_total"]
                hit = sizes[f"{key}_hit"]
                ratio = hit / total if total else 0

                print(
                    f"    {label}: "
                    f"{hit}/{total} = {ratio:.2%}"
                )

            print(
                "  无标签图片产生的FP:",
                stats["unlabeled_fp"]
            )

            output = (
                PROJECT / "runs"
                / f"v41_errors_conf_{int(threshold*100):02d}.csv"
            )

            with output.open(
                "w",
                encoding="utf-8-sig",
                newline=""
            ) as f:
                writer = csv.writer(f)
                writer.writerow([
                    "image", "GT", "Pred",
                    "TP", "FP", "FN"
                ])
                writer.writerows(details)

            print("  错误明细:", output)

    output = PROJECT / "runs" / "v41_threshold_summary.csv"

    with output.open(
        "w", encoding="utf-8-sig", newline=""
    ) as f:
        writer = csv.writer(f)
        writer.writerow([
            "threshold", "TP", "FP", "FN",
            "precision", "recall", "fp_per_image"
        ])
        writer.writerows(summary)

    print("\n评估完成:", output)
    print("这是二维框级评估，不是完整CT的FROC。")


if __name__ == "__main__":
    main()

