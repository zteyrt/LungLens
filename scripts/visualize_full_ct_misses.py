
from pathlib import Path
import csv
import math

import numpy as np
import SimpleITK as sitk
from PIL import Image, ImageDraw

ROOT = Path(r"E:\LungLens_Data")
RUNS = Path.cwd() / "runs"

FN_CSV = (
    RUNS / "full_ct_v41_excluded_v1"
    / "fn_diagnosis.csv"
)
ANN_CSV = ROOT / "annotations" / "annotations.csv"
RAW = ROOT / "raw"

OUT = RUNS / "full_ct_v41_miss_review"
OUT.mkdir(parents=True, exist_ok=True)


def read_csv(path):
    with path.open(
        "r", encoding="utf-8-sig", newline=""
    ) as f:
        return list(csv.DictReader(f))


def convert_ct(slice_hu):
    image = np.clip(
        slice_hu.astype(np.float32),
        -1350, 150
    )
    image = ((image + 1350) / 1500 * 255)
    return Image.fromarray(
        image.astype(np.uint8)
    ).convert("RGB")


def get_crop(volume, cx, cy, z, radius=48):
    h, w = volume.shape[1:]

    z = max(0, min(z, volume.shape[0] - 1))

    image = convert_ct(volume[z])

    left = round(cx - radius)
    top = round(cy - radius)

    # 超出图像边缘时自动填充黑色
    crop = Image.new(
        "RGB", (radius * 2, radius * 2),
        "black"
    )

    region = image.crop((
        max(0, left),
        max(0, top),
        min(w, left + 2 * radius),
        min(h, top + 2 * radius)
    ))

    crop.paste(
        region,
        (
            max(0, -left),
            max(0, -top)
        )
    )

    draw = ImageDraw.Draw(crop)

    px = cx - left
    py = cy - top

    # 仅标记参考中心，不代表真实边界
    draw.line(
        (px - 8, py, px + 8, py),
        fill="red", width=1
    )
    draw.line(
        (px, py - 8, px, py + 8),
        fill="red", width=1
    )

    return crop.resize((256, 256))


def main():
    misses = read_csv(FN_CSV)

    annotations = {}

    for row in read_csv(ANN_CSV):
        annotations.setdefault(
            row["seriesuid"], []
        ).append(row)

    ct_paths = {
        p.stem: p
        for p in RAW.rglob("*.mhd")
    }

    selected = [
        r for r in misses
        if r["reason"] == "NO_NEARBY_CANDIDATE"
    ]

    print("========== 漏检结节可视化 ==========")
    print("目标结节数量:", len(selected))

    if not selected:
        return

    sheet = Image.new(
        "RGB",
        (3 * 256, len(selected) * 300),
        "white"
    )

    for row_num, row in enumerate(selected):
        uid = row["seriesuid"]
        index = int(row["annotation_index"])

        gt = annotations[uid][index]

        if uid not in ct_paths:
            raise FileNotFoundError(uid)

        ct = sitk.ReadImage(str(ct_paths[uid]))
        volume = sitk.GetArrayFromImage(ct)

        world = (
            float(gt["coordX"]),
            float(gt["coordY"]),
            float(gt["coordZ"])
        )

        cx, cy, cz = (
            ct.TransformPhysicalPointToContinuousIndex(
                world
            )
        )

        center_z = round(cz)

        print(
            f"{row_num + 1}. "
            f"病例={uid[-16:]} "
            f"直径={gt['diameter_mm']}mm "
            f"中心层={center_z}"
        )

        for col, dz in enumerate((-1, 0, 1)):
            crop = get_crop(
                volume, cx, cy, center_z + dz
            )

            sheet.paste(
                crop,
                (col * 256, row_num * 300 + 36)
            )

        draw = ImageDraw.Draw(sheet)

        label = (
            f"Case {uid[-16:]}  "
            f"Diameter {float(gt['diameter_mm']):.2f}mm"
        )

        draw.text(
            (8, row_num * 300 + 8),
            label,
            fill="black"
        )

    target = OUT / "missing_nodules_5.jpg"
    sheet.save(target, quality=95)

    print("\n可视化已保存:", target)
    print("红色十字表示参考结节中心")
    print("三列依次为Z-1、中心Z、Z+1")
    print("没有修改原始CT或模型")


if __name__ == "__main__":
    main()
