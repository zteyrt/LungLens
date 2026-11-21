
from pathlib import Path
from collections import Counter
from time import perf_counter
import csv
import numpy as np
from PIL import Image
from ultralytics import YOLO


# =====================================
# 1. 实验配置
# =====================================

PROJECT = Path(r"D:\肺结节模型训练")
DATA = Path(r"E:\LungLens_Data\processed\yolo_v41")
WEIGHT = (
    PROJECT / "runs/yolo_v41_baseline/weights/best.pt"
)

OUT = PROJECT / "runs/patch_v41_experiment"

PATCH_SIZE = 256
STRIDE = 128
IMGSZ = 640

# 推理统一使用较低阈值生成候选
MIN_CONF = 0.05

# 最终比较的固定阈值
THRESHOLDS = (0.10, 0.20)

# 重复预测框合并阈值
NMS_IOU = 0.50

# 与真实框匹配的阈值
MATCH_IOU = 0.50

# 仅使用验证集，不使用测试集
SPLIT = "val"


# =====================================
# 2. IoU 计算
# =====================================

def iou(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])

    intersection = (
        max(0.0, x2 - x1)
        * max(0.0, y2 - y1)
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


# =====================================
# 3. 跨Patch的NMS
# =====================================

def global_nms(predictions, threshold=0.50):
    """
    输入:
      [(box_xyxy, confidence), ...]

    输出:
      合并重复框后的预测列表

    这里用简单NMS，避免依赖torchvision扩展。
    """
    predictions = sorted(
        predictions,
        key=lambda p: p[1],
        reverse=True
    )

    kept = []

    while predictions:
        best = predictions.pop(0)
        kept.append(best)

        predictions = [
            p for p in predictions
            if iou(best[0], p[0]) < threshold
        ]

    return kept


# =====================================
# 4. 读取真实标签
# =====================================

def read_gt(image_path):
    label_path = (
        DATA / "labels" / SPLIT
        / (image_path.stem + ".txt")
    )

    with Image.open(image_path) as im:
        width, height = im.size

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


# =====================================
# 5. 模型单次预测
# =====================================

def predict_boxes(model, source):
    result = model.predict(
        source=source,
        imgsz=IMGSZ,
        conf=MIN_CONF,
        iou=0.7,
        max_det=150,
        batch=1,
        device=0,
        verbose=False
    )[0]

    predictions = []

    for box in result.boxes:
        xyxy = box.xyxy[0].cpu().tolist()
        conf = float(box.conf[0].item())

        predictions.append((xyxy, conf))

    return predictions


# =====================================
# 6. 普通整图推理
# =====================================

def infer_full(model, image_path):
    return predict_boxes(model, str(image_path))


# =====================================
# 7. Patch滑窗推理
# =====================================

def get_positions(length):
    if length <= PATCH_SIZE:
        return [0]

    positions = list(
        range(0, length - PATCH_SIZE + 1, STRIDE)
    )

    last = length - PATCH_SIZE

    if positions[-1] != last:
        positions.append(last)

    return positions


def infer_patch(model, image_path):
    with Image.open(image_path) as original:
        image = original.convert("RGB")

    width, height = image.size

    # 当前实验要求原始图片不少于Patch
    if width < PATCH_SIZE or height < PATCH_SIZE:
        raise ValueError(
            f"图像尺寸小于Patch: {image_path}"
        )

    xs = get_positions(width)
    ys = get_positions(height)

    all_predictions = []

    for y0 in ys:
        for x0 in xs:

            # 裁剪局部图像
            patch = image.crop((
                x0,
                y0,
                x0 + PATCH_SIZE,
                y0 + PATCH_SIZE
            ))

            # YOLO会自动将Patch缩放到模型输入尺寸
            patch_predictions = predict_boxes(
                model, patch
            )

            # 将Patch坐标还原到原图坐标
            for box, conf in patch_predictions:
                x1, y1, x2, y2 = box

                mapped_box = [
                    max(0.0, x1 + x0),
                    max(0.0, y1 + y0),
                    min(float(width), x2 + x0),
                    min(float(height), y2 + y0),
                ]

                if (
                    mapped_box[2] <= mapped_box[0]
                    or mapped_box[3] <= mapped_box[1]
                ):
                    continue

                all_predictions.append(
                    (mapped_box, conf)
                )

    # 合并不同Patch间的重复框
    merged = global_nms(
        all_predictions,
        threshold=NMS_IOU
    )

    return merged


# =====================================
# 8. 按结节尺寸分类
# =====================================

def size_group(size):
    if size < 4:
        return "<4px"
    if size < 8:
        return "4-8px"
    return ">=8px"


# =====================================
# 9. 固定阈值评估
# =====================================

def evaluate(records, threshold):
    stats = Counter()
    size_stats = Counter()
    details = []

    for image_name, gts, predictions in records:

        selected = sorted(
            [
                p for p in predictions
                if p[1] >= threshold
            ],
            key=lambda p: p[1],
            reverse=True
        )

        matched = set()
        tp = 0
        fp = 0

        for pred_box, conf in selected:
            best_iou = 0.0
            best_idx = -1

            for idx, gt in enumerate(gts):
                if idx in matched:
                    continue

                score = iou(
                    pred_box,
                    gt["box"]
                )

                if score > best_iou:
                    best_iou = score
                    best_idx = idx

            if (
                best_idx >= 0
                and best_iou >= MATCH_IOU
            ):
                matched.add(best_idx)
                tp += 1
            else:
                fp += 1

        fn = len(gts) - tp

        stats["TP"] += tp
        stats["FP"] += fp
        stats["FN"] += fn
        stats["images"] += 1

        if not gts:
            stats["unlabeled_fp"] += fp

        for idx, gt in enumerate(gts):
            group = size_group(gt["size"])

            size_stats[(group, "total")] += 1

            if idx in matched:
                size_stats[(group, "hit")] += 1

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

    precision = (
        tp / (tp + fp)
        if tp + fp else 0.0
    )
    recall = (
        tp / (tp + fn)
        if tp + fn else 0.0
    )

    return stats, size_stats, details, precision, recall


# =====================================
# 10. 保存CSV
# =====================================

def save_csv(path, header, rows):
    with path.open(
        "w",
        encoding="utf-8-sig",
        newline=""
    ) as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


# =====================================
# 11. 主程序
# =====================================

def main():

    if not WEIGHT.is_file():
        raise FileNotFoundError(WEIGHT)

    image_dir = DATA / "images" / SPLIT

    images = sorted(
        image_dir.glob("*.png")
    )

    if not images:
        raise RuntimeError(
            "验证集图片不存在"
        )

    OUT.mkdir(parents=True, exist_ok=True)

    print("========== LungLens Patch实验 ==========")
    print("验证图片:", len(images))
    print("Patch:", PATCH_SIZE)
    print("Stride:", STRIDE)
    print("模型输入:", IMGSZ)
    print("最小推理置信度:", MIN_CONF)
    print("跨Patch NMS IoU:", NMS_IOU)
    print("真实框匹配IoU:", MATCH_IOU)

    model = YOLO(str(WEIGHT))

    full_records = []
    patch_records = []

    start_time = perf_counter()
    full_seconds = 0.0
    patch_seconds = 0.0

    for idx, image_path in enumerate(images, 1):
        gts = read_gt(image_path)

        t0 = perf_counter()
        full_preds = infer_full(model, image_path)
        full_seconds += perf_counter() - t0

        t0 = perf_counter()
        patch_preds = infer_patch(model, image_path)
        patch_seconds += perf_counter() - t0

        full_records.append((
            image_path.name,
            gts,
            full_preds
        ))

        patch_records.append((
            image_path.name,
            gts,
            patch_preds
        ))

        if idx % 25 == 0 or idx == len(images):
            print(
                f"已处理: {idx}/{len(images)}"
            )

    summary_rows = []

    for mode, records in [
        ("FULL", full_records),
        ("PATCH", patch_records),
    ]:
        print(f"\n========== {mode} ==========")

        for threshold in THRESHOLDS:
            (
                stats,
                size_stats,
                details,
                precision,
                recall
            ) = evaluate(records, threshold)

            fp_per_image = (
                stats["FP"] / stats["images"]
            )

            print(
                f"\nconf={threshold:.2f}"
            )
            print(
                f"TP={stats['TP']} "
                f"FP={stats['FP']} "
                f"FN={stats['FN']}"
            )
            print(
                f"Precision={precision:.2%}"
            )
            print(
                f"Recall={recall:.2%}"
            )
            print(
                f"FP/图={fp_per_image:.3f}"
            )
            print(
                "无标签图片上的FP:",
                stats["unlabeled_fp"]
            )

            print("不同尺寸召回率:")

            for group in (
                "<4px", "4-8px", ">=8px"
            ):
                total = size_stats[
                    (group, "total")
                ]
                hit = size_stats[
                    (group, "hit")
                ]

                rate = (
                    hit / total if total else 0
                )

                print(
                    f"  {group}: "
                    f"{hit}/{total} "
                    f"({rate:.2%})"
                )

            summary_rows.append([
                mode,
                threshold,
                stats["TP"],
                stats["FP"],
                stats["FN"],
                round(precision, 5),
                round(recall, 5),
                round(fp_per_image, 5),
                stats["unlabeled_fp"],
                size_stats[("<4px", "hit")],
                size_stats[("4-8px", "hit")],
                size_stats[(">=8px", "hit")],
            ])

            save_csv(
                OUT / (
                    f"{mode.lower()}_conf_"
                    f"{int(threshold*100):02d}.csv"
                ),
                [
                    "image", "GT", "Pred",
                    "TP", "FP", "FN"
                ],
                details
            )

    save_csv(
        OUT / "comparison.csv",
        [
            "mode", "threshold", "TP", "FP", "FN",
            "precision", "recall", "fp_per_image",
            "unlabeled_fp",
            "hit_under4",
            "hit_4to8",
            "hit_over8"
        ],
        summary_rows
    )

    total_seconds = perf_counter() - start_time

    print("\n========== 实验完成 ==========")
    print(
        f"FULL平均耗时: "
        f"{full_seconds/len(images):.3f}秒/图"
    )
    print(
        f"PATCH平均耗时: "
        f"{patch_seconds/len(images):.3f}秒/图"
    )
    print(
        f"总计耗时: {total_seconds:.1f}秒"
    )
    print(
        "结果目录:", OUT
    )
    print(
        "汇总文件:", OUT / "comparison.csv"
    )
    print(
        "\n注意：当前是二维切片级评估，"
        "不是完整CT的FROC。"
    )


if __name__ == "__main__":
    main()
