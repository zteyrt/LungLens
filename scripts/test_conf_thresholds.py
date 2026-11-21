
from pathlib import Path
from ultralytics import YOLO

ROOT = Path(r"D:\肺结节模型训练")
DATA = Path(r"E:\LungLens_Data\processed\yolo_v3")

MODELS = {
    "V3": ROOT / "runs/yolo_v3_baseline/weights/best.pt",
    "V41": ROOT / "runs/yolo_v41_baseline/weights/best.pt",
}

THRESHOLDS = [0.05, 0.10, 0.20, 0.30, 0.50]
IOU_THRESHOLD = 0.50


def box_iou(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)

    area_a = max(0, a[2]-a[0]) * max(0, a[3]-a[1])
    area_b = max(0, b[2]-b[0]) * max(0, b[3]-b[1])

    return inter / (area_a + area_b - inter + 1e-9)


def read_gt(image_path):
    label = DATA / "labels/val" / (image_path.stem + ".txt")
    boxes = []

    for line in label.read_text(
        encoding="utf-8"
    ).splitlines():
        cls, cx, cy, bw, bh = map(float, line.split())

        # V3原始图片为512×512
        boxes.append([
            (cx - bw / 2) * 512,
            (cy - bh / 2) * 512,
            (cx + bw / 2) * 512,
            (cy + bh / 2) * 512,
        ])

    return boxes


def evaluate(predictions, gt_boxes, threshold):
    tp = fp = fn = 0

    for preds, gts in zip(predictions, gt_boxes):
        matched = set()

        # 按置信度从高到低匹配，每个真实框最多匹配一次
        for box, conf in preds:
            if conf < threshold:
                continue

            best_iou = 0
            best_idx = None

            for j, gt in enumerate(gts):
                if j in matched:
                    continue

                score = box_iou(box, gt)

                if score > best_iou:
                    best_iou = score
                    best_idx = j

            if best_idx is not None and best_iou >= IOU_THRESHOLD:
                tp += 1
                matched.add(best_idx)
            else:
                fp += 1

        fn += len(gts) - len(matched)

    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0

    return tp, fp, fn, precision, recall


def main():
    images = sorted((DATA / "images/val").glob("*.png"))

    if not images:
        raise RuntimeError("没有找到验证图片")

    gt_boxes = [read_gt(p) for p in images]

    print("========== 置信度阈值测试 ==========")
    print("验证图片:", len(images))
    print("真实标注框:", sum(map(len, gt_boxes)))
    print("IoU匹配阈值:", IOU_THRESHOLD)

    for name, weight in MODELS.items():
        model = YOLO(str(weight))
        predictions = []

        print(f"\n========== {name} ==========")

        for image_path in images:
            result = model.predict(
                source=str(image_path),
                imgsz=640,
                conf=0.01,
                iou=0.7,
                max_det=300,
                device=0,
                verbose=False
            )[0]

            preds = []

            for box in result.boxes:
                coords = box.xyxy[0].cpu().tolist()
                conf = float(box.conf[0].item())
                preds.append((coords, conf))

            preds.sort(key=lambda x: x[1], reverse=True)
            predictions.append(preds)

        print("阈值  TP   FP   FN   Precision  Recall  FP/图")

        for threshold in THRESHOLDS:
            tp, fp, fn, precision, recall = evaluate(
                predictions,
                gt_boxes,
                threshold
            )

            print(
                f"{threshold:0.2f}  "
                f"{tp:3d}  {fp:3d}  {fn:3d}  "
                f"{precision:8.2%}  "
                f"{recall:6.2%}  "
                f"{fp/len(images):.2f}"
            )

    print("\n注意：FP/图不等于FP/CT。")
    print("这是验证集框级指标，不是结节级FROC。")


if __name__ == "__main__":
    main()
