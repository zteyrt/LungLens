
from pathlib import Path
import csv
import math
import numpy as np
import SimpleITK as sitk
from PIL import Image, ImageDraw
from ultralytics import YOLO

DATA = Path(r"E:\LungLens_Data")
PROJECT = Path.cwd()

UID_SUFFIX = "3382155770295147"
DIAMETER_TARGET = 12.99686424

WEIGHT = (
    PROJECT / "runs/yolo_v41_baseline/weights/best.pt"
)
OUT = PROJECT / "runs/inspect_13mm_nodule"
OUT.mkdir(parents=True, exist_ok=True)

def convert_ct(a):
    a = np.clip(a.astype(np.float32), -1350, 150)
    a = ((a + 1350) / 1500 * 255).astype(np.uint8)
    return Image.fromarray(a).convert("RGB")

def main():
    ann_path = DATA / "annotations/annotations.csv"

    with ann_path.open(
        "r", encoding="utf-8-sig", newline=""
    ) as f:
        annotations = [
            row for row in csv.DictReader(f)
            if row["seriesuid"].endswith(UID_SUFFIX)
        ]

    if not annotations:
        raise RuntimeError("找不到目标病例标注")

    gt = min(
        annotations,
        key=lambda r: abs(
            float(r["diameter_mm"]) - DIAMETER_TARGET
        )
    )

    uid = gt["seriesuid"]

    matches = [
        p for p in (DATA / "raw").rglob("*.mhd")
        if p.stem == uid
    ]

    if len(matches) != 1:
        raise RuntimeError("找不到唯一的CT文件")

    ct = sitk.ReadImage(str(matches[0]))
    volume = sitk.GetArrayFromImage(ct)

    world = (
        float(gt["coordX"]),
        float(gt["coordY"]),
        float(gt["coordZ"])
    )

    cx, cy, cz = (
        ct.TransformPhysicalPointToContinuousIndex(world)
    )

    center_z = round(cz)

    print("========== 13mm结节检查 ==========")
    print("结节直径:", gt["diameter_mm"])
    print("中心Z层:", center_z)
    print("原图像素中心:", round(cx, 2), round(cy, 2))

    model = YOLO(str(WEIGHT))

    offsets = [-4, -2, -1, 0, 1, 2, 4]
    tiles = []

    for dz in offsets:
        z = center_z + dz

        if z < 0 or z >= volume.shape[0]:
            continue

        image = convert_ct(volume[z])

        result = model.predict(
            source=np.array(image),
            imgsz=640,
            conf=0.01,
            iou=0.7,
            max_det=300,
            batch=1,
            device=0,
            verbose=False
        )[0]

        boxes = (
            result.boxes.xyxy.cpu().numpy()
        )
        scores = (
            result.boxes.conf.cpu().numpy()
        )

        # 根据参考结节中心寻找最近的预测框
        nearest = None
        nearest_distance = float("inf")
        near_score = None

        full = image.copy()
        draw = ImageDraw.Draw(full)

        for box, score in zip(boxes, scores):
            x1, y1, x2, y2 = map(float, box)

            px = (x1 + x2) / 2
            py = (y1 + y2) / 2

            distance = math.hypot(
                px - cx, py - cy
            )

            if distance < nearest_distance:
                nearest_distance = distance
                nearest = box
                near_score = float(score)

            if float(score) >= 0.10:
                draw.rectangle(
                    (x1, y1, x2, y2),
                    outline="yellow",
                    width=2
                )

        # 红色十字：参考中心
        draw.line(
            (cx-9, cy, cx+9, cy),
            fill="red", width=2
        )
        draw.line(
            (cx, cy-9, cx, cy+9),
            fill="red", width=2
        )

        radius = 48
        crop = full.crop((
            round(cx-radius),
            round(cy-radius),
            round(cx+radius),
            round(cy+radius)
        ))
        crop = crop.resize((320, 320))

        tile = Image.new(
            "RGB", (320, 350), "white"
        )
        tile.paste(crop, (0, 28))

        if nearest is None:
            info = f"z={z}, no prediction"
        else:
            info = (
                f"z={z}, nearest={nearest_distance:.1f}px, "
                f"conf={near_score:.3f}"
            )

        ImageDraw.Draw(tile).text(
            (6, 7), info, fill="black"
        )

        print(info)
        tiles.append(tile)

    columns = 4
    rows = math.ceil(len(tiles) / columns)

    sheet = Image.new(
        "RGB",
        (columns * 320, rows * 350),
        "#dddddd"
    )

    for i, tile in enumerate(tiles):
        sheet.paste(
            tile,
            (
                (i % columns) * 320,
                (i // columns) * 350
            )
        )

    output = OUT / "13mm_prediction_review.jpg"
    sheet.save(output, quality=95)

    print("\n复核图:", output)
    print("黄色框: conf >= 0.10的模型预测")
    print("红色十字: LUNA16参考结节中心")
    print("模型权重和原始CT均未修改")

if __name__ == "__main__":
    main()
