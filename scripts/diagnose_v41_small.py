
from pathlib import Path
from collections import Counter
from PIL import Image
from ultralytics import YOLO
import csv

ROOT = Path(r"D:\肺结节模型训练")
DATA = Path(r"E:\LungLens_Data\processed\yolo_v41")
WEIGHT = ROOT / "runs/yolo_v41_baseline/weights/best.pt"

CONF = 0.10


def iou(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])

    inter = max(0, x2-x1) * max(0, y2-y1)
    aa = max(0, a[2]-a[0]) * max(0, a[3]-a[1])
    ab = max(0, b[2]-b[0]) * max(0, b[3]-b[1])
    return inter / (aa + ab - inter + 1e-9)


def center_distance(a, b):
    ax = (a[0] + a[2]) / 2
    ay = (a[1] + a[3]) / 2
    bx = (b[0] + b[2]) / 2
    by = (b[1] + b[3]) / 2
    return ((ax-bx)**2 + (ay-by)**2)**0.5


model = YOLO(str(WEIGHT))
stats = Counter()
records = []

images = sorted((DATA / "images/val").glob("*.png"))

for image_path in images:
    label_path = DATA / "labels/val" / (
        image_path.stem + ".txt"
    )

    with Image.open(image_path) as im:
        w, h = im.size

    gts = []
    for line in label_path.read_text(
        encoding="utf-8-sig"
    ).splitlines():
        _, cx, cy, bw, bh = map(float, line.split())
        gts.append((
            (cx-bw/2)*w, (cy-bh/2)*h,
            (cx+bw/2)*w, (cy+bh/2)*h
        ))

    if not gts:
        continue

    result = model.predict(
        source=str(image_path),
        conf=CONF,
        imgsz=640,
        iou=0.7,
        batch=1,
        device=0,
        verbose=False
    )[0]

    preds = result.boxes.xyxy.cpu().tolist()

    for gt in gts:
        size = min(gt[2]-gt[0], gt[3]-gt[1])
        group = (
            "<4px" if size < 4
            else "4-8px" if size < 8
            else ">=8px"
        )

        best_iou = max(
            (iou(gt, p) for p in preds),
            default=0.0
        )

        nearest = min(
            (center_distance(gt, p) for p in preds),
            default=float("inf")
        )

        # 此处只是逐GT诊断，不是唯一匹配的正式Recall
        if best_iou >= 0.5:
            status = "MATCHED"
        elif best_iou >= 0.1:
            status = "POOR_LOCALIZATION"
        elif nearest <= max(8.0, size):
            status = "NEARBY_PREDICTION"
        else:
            status = "NO_NEARBY_PREDICTION"

        stats[(group, status)] += 1
        stats[(group, "TOTAL")] += 1

        records.append([
            image_path.name,
            group,
            round(size, 2),
            round(best_iou, 3),
            round(nearest, 2) if preds else "",
            status
        ])

print("========== V4.1 小目标诊断 ==========")

statuses = (
    "MATCHED",
    "POOR_LOCALIZATION",
    "NEARBY_PREDICTION",
    "NO_NEARBY_PREDICTION"
)

for group in ("<4px", "4-8px", ">=8px"):
    print(f"\n{group}  总数: {stats[(group, 'TOTAL')]}")
    for status in statuses:
        print(f"  {status}: {stats[(group, status)]}")

output = ROOT / "runs/v41_small_diagnosis.csv"

with output.open(
    "w", encoding="utf-8-sig", newline=""
) as f:
    writer = csv.writer(f)
    writer.writerow([
        "image", "size_group", "size_px",
        "best_iou", "nearest_center_px", "status"
    ])
    writer.writerows(records)

print("\n诊断明细:", output)
print("注意：这是逐GT最近预测诊断，不是正式TP/FP匹配。")
