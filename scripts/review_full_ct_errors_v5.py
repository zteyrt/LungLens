
from pathlib import Path
from collections import defaultdict
import csv
import math

import SimpleITK as sitk
import numpy as np
from PIL import Image, ImageDraw


PROJECT = Path(r"D:\肺结节模型训练")
RAW = Path(r"E:\LungLens_Data\raw")
ANN = Path(
    r"E:\LungLens_Data\annotations\annotations.csv"
)
SPLIT = Path(
    r"E:\LungLens_Data\processed\yolo_v41\case_split.csv"
)
SOURCE = PROJECT / "runs/full_ct_v5_val_v1"
OUT = PROJECT / "runs/full_ct_v5_error_review"

CONF = 0.57
MIN_SLICES = 1
MAX_REVIEW_IMAGES = 24


def read_csv(path):
    with path.open(
        "r", encoding="utf-8-sig", newline=""
    ) as f:
        return list(csv.DictReader(f))


def distance(a, b):
    return math.sqrt(
        (a["x"] - b["x"]) ** 2
        + (a["y"] - b["y"]) ** 2
        + (a["z"] - b["z"]) ** 2
    )


def load_data():
    cases = {
        r["seriesuid"]
        for r in read_csv(SPLIT)
        if r["split"] == "val"
    }

    gt_map = defaultdict(list)
    for r in read_csv(ANN):
        uid = r["seriesuid"]
        if uid in cases:
            gt_map[uid].append({
                "x": float(r["coordX"]),
                "y": float(r["coordY"]),
                "z": float(r["coordZ"]),
                "diameter": float(r["diameter_mm"])
            })

    cand_map = defaultdict(list)
    for r in read_csv(SOURCE / "candidates_3d.csv"):
        uid = r["seriesuid"]
        if uid in cases:
            cand_map[uid].append({
                "x": float(r["world_x"]),
                "y": float(r["world_y"]),
                "z": float(r["world_z"]),
                "score": float(r["score"]),
                "slices": int(r["num_slices"]),
                "z_index": int(r["representative_z"])
            })

    return sorted(cases), gt_map, cand_map


def classify(cases, gt_map, cand_map):
    false_positives = []
    false_negatives = []
    tp = 0

    for uid in cases:
        gts = gt_map[uid]

        selected = sorted(
            [
                c for c in cand_map[uid]
                if c["score"] >= CONF
                and c["slices"] >= MIN_SLICES
            ],
            key=lambda c: c["score"],
            reverse=True
        )

        matched = set()

        for c in selected:
            best_idx = None
            best_ratio = float("inf")

            for i, gt in enumerate(gts):
                if i in matched:
                    continue

                radius = gt["diameter"] / 2
                if radius <= 0:
                    continue

                ratio = distance(c, gt) / radius

                if ratio <= 1 and ratio < best_ratio:
                    best_ratio = ratio
                    best_idx = i

            if best_idx is not None:
                matched.add(best_idx)
                tp += 1
            else:
                # 这是未匹配参考结节的候选，
                # 不是医学确认的假阳性
                false_positives.append({
                    "uid": uid,
                    **c
                })

        for i, gt in enumerate(gts):
            if i in matched:
                continue

            # 寻找所有候选中最近的一个
            # 仅用于分析位置误差，不用于重新匹配
            all_cands = cand_map[uid]
            nearest = min(
                all_cands,
                key=lambda c: distance(c, gt),
                default=None
            )

            false_negatives.append({
                "uid": uid,
                "index": i,
                **gt,
                "nearest_distance_mm": (
                    distance(nearest, gt)
                    if nearest else None
                ),
                "nearest_score": (
                    nearest["score"]
                    if nearest else None
                )
            })

    return tp, false_positives, false_negatives


def find_ct_paths():
    paths = {}
    for p in RAW.rglob("*.mhd"):
        if p.stem in paths:
            raise RuntimeError(
                f"CT UID重复: {p.stem}"
            )
        paths[p.stem] = p
    return paths


def ct_to_gray(slice_hu):
    a = np.clip(
        slice_hu.astype(np.float32),
        -1350, 150
    )
    a = ((a + 1350) / 1500 * 255)
    return Image.fromarray(
        a.astype(np.uint8),
        mode="L"
    ).convert("RGB")


def make_review_tile(ct_image, candidate):
    volume = sitk.GetArrayFromImage(ct_image)

    z = candidate["z_index"]
    if z < 0 or z >= volume.shape[0]:
        raise RuntimeError("候选Z索引越界")

    image = ct_to_gray(volume[z])
    width, height = image.size

    index = ct_image.TransformPhysicalPointToContinuousIndex(
        (
            candidate["x"],
            candidate["y"],
            candidate["z"]
        )
    )

    cx, cy = index[0], index[1]

    # 左边：完整CT切片
    full = image.copy()
    draw = ImageDraw.Draw(full)

    r = 8
    draw.ellipse(
        (cx-r, cy-r, cx+r, cy+r),
        outline="red",
        width=3
    )
    draw.line(
        (cx-14, cy, cx+14, cy),
        fill="red", width=2
    )
    draw.line(
        (cx, cy-14, cx, cy+14),
        fill="red", width=2
    )

    full = full.resize((256, 256))

    # 右边：候选附近约96x96像素
    crop_radius = 48
    crop = image.crop((
        round(cx-crop_radius),
        round(cy-crop_radius),
        round(cx+crop_radius),
        round(cy+crop_radius)
    ))

    crop_draw = ImageDraw.Draw(crop)
    crop_draw.ellipse(
        (
            crop_radius-6,
            crop_radius-6,
            crop_radius+6,
            crop_radius+6
        ),
        outline="red",
        width=2
    )

    crop = crop.resize((256, 256))

    tile = Image.new(
        "RGB", (530, 292), "white"
    )

    tile.paste(full, (4, 30))
    tile.paste(crop, (270, 30))

    label = (
        f"{candidate['uid'][-12:]} "
        f"z={z} "
        f"conf={candidate['score']:.2f} "
        f"slices={candidate['slices']}"
    )

    ImageDraw.Draw(tile).text(
        (6, 7), label, fill="black"
    )

    return tile


def save_review_sheet(false_positives, ct_paths):
    selected = sorted(
        false_positives,
        key=lambda c: c["score"],
        reverse=True
    )[:MAX_REVIEW_IMAGES]

    if not selected:
        print("没有需要展示的未匹配候选")
        return

    tiles = []
    current_uid = None
    current_ct = None

    for candidate in selected:
        uid = candidate["uid"]

        if uid not in ct_paths:
            raise FileNotFoundError(
                f"缺少CT: {uid}"
            )

        if uid != current_uid:
            current_ct = sitk.ReadImage(
                str(ct_paths[uid])
            )
            current_uid = uid

        tiles.append(
            make_review_tile(
                current_ct, candidate
            )
        )

    columns = 2
    rows = math.ceil(len(tiles) / columns)

    sheet = Image.new(
        "RGB",
        (columns * 530, rows * 292),
        "#dddddd"
    )

    for i, tile in enumerate(tiles):
        sheet.paste(
            tile,
            (
                (i % columns) * 530,
                (i // columns) * 292
            )
        )

    path = OUT / "fp_review.jpg"
    sheet.save(path, quality=95)
    print("可视化图片:", path)


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    cases, gt_map, cand_map = load_data()

    tp, fps, fns = classify(
        cases, gt_map, cand_map
    )

    print("========== 完整CT误检分析 ==========")
    print("验证病例:", len(cases))
    print("参考结节:", sum(
        len(gt_map[uid]) for uid in cases
    ))
    print("置信度阈值:", CONF)
    print("TP:", tp)
    print("未匹配候选:", len(fps))
    print("漏检参考结节:", len(fns))

    fp_csv = OUT / "false_positives.csv"
    with fp_csv.open(
        "w", encoding="utf-8-sig", newline=""
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "uid", "x", "y", "z",
                "score", "slices", "z_index"
            ]
        )
        writer.writeheader()
        writer.writerows(fps)

    fn_csv = OUT / "false_negatives.csv"
    with fn_csv.open(
        "w", encoding="utf-8-sig", newline=""
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "uid", "index",
                "x", "y", "z", "diameter",
                "nearest_distance_mm",
                "nearest_score"
            ]
        )
        writer.writeheader()
        writer.writerows(fns)

    ct_paths = find_ct_paths()

    save_review_sheet(fps, ct_paths)

    print("\n结果目录:", OUT)
    print("误检明细:", fp_csv)
    print("漏检明细:", fn_csv)
    print("\n原始CT、模型和候选结果均未修改。")


if __name__ == "__main__":
    main()
