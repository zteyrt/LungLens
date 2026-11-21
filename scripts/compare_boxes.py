
from pathlib import Path
from PIL import Image, ImageDraw
from ultralytics import YOLO

ROOT = Path(r"D:\肺结节模型训练")
DATA = Path(r"E:\LungLens_Data\processed\yolo_v1")
OUT = ROOT / "runs" / "box_comparison"
OUT.mkdir(parents=True, exist_ok=True)

model = YOLO(str(ROOT / "runs/smoke_test/weights/best.pt"))


def read_labels(path, w, h):
    """读取 YOLO 标签，并还原为像素坐标。"""
    boxes = []

    if not path.exists():
        return boxes

    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()

        if len(parts) != 5:
            raise ValueError(f"标签格式错误: {path}")

        _, cx, cy, bw, bh = map(float, parts)

        boxes.append([
            (cx - bw / 2) * w,
            (cy - bh / 2) * h,
            (cx + bw / 2) * w,
            (cy + bh / 2) * h,
        ])

    return boxes


def calculate_iou(a, b):
    """计算两个检测框的交并比。"""
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])

    intersection = max(0, x2-x1) * max(0, y2-y1)

    area_a = max(0, a[2]-a[0]) * max(0, a[3]-a[1])
    area_b = max(0, b[2]-b[0]) * max(0, b[3]-b[1])

    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


val_images = sorted((DATA / "images/val").glob("*.png"))
val_images += sorted((DATA / "images/val").glob("*.jpg"))

for image_path in val_images:
    image = Image.open(image_path).convert("RGB")
    w, h = image.size
    draw = ImageDraw.Draw(image)

    label_path = DATA / "labels/val" / (image_path.stem + ".txt")
    gt_boxes = read_labels(label_path, w, h)

    result = model.predict(
        str(image_path),
        imgsz=512,
        conf=0.01,
        device=0,
        verbose=False
    )[0]

    predictions = []

    if result.boxes is not None:
        for box in result.boxes:
            xyxy = box.xyxy[0].cpu().tolist()
            confidence = float(box.conf[0])
            predictions.append((xyxy, confidence))

    # 只画最高置信度的3个预测框，避免遮挡画面
    for pred, conf in predictions[:3]:
        draw.rectangle(pred, outline="blue", width=2)
        draw.text((pred[0], pred[1]), f"{conf:.2f}", fill="blue")

    # 最后画真实框，保证红框不会被蓝框遮住
    for gt in gt_boxes:
        draw.rectangle(gt, outline="red", width=2)

    output_path = OUT / image_path.name
    image.save(output_path)

    print(f"\n图片: {image_path.name}")
    print(f"真实框数量: {len(gt_boxes)}")
    print(f"预测框数量: {len(predictions)}")

    for gt in gt_boxes:
        if predictions:
            best_iou = max(
                calculate_iou(gt, pred)
                for pred, _ in predictions
            )
        else:
            best_iou = 0.0

        print("真实框:", [round(v, 2) for v in gt])
        print("最大 IoU:", round(best_iou, 4))

    print("对比图已保存:", output_path)

print("\n分析完成")

