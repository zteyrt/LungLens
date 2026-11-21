
from pathlib import Path
from collections import defaultdict, Counter
import csv
import math
import random

import numpy as np
import SimpleITK as sitk
from scipy.ndimage import map_coordinates
from ultralytics import YOLO


# ============================================
# 1. 配置
# ============================================

PROJECT = Path.cwd()
ROOT = Path(r"E:\LungLens_Data")

RAW = ROOT / "raw"
ANNOTATIONS = ROOT / "annotations" / "annotations.csv"
EXCLUDED = ROOT / "annotations" / "annotations_excluded.csv"
SPLITS = ROOT / "processed" / "yolo_v41" / "case_split.csv"

WEIGHT = (
    PROJECT / "runs" / "yolo_v41_baseline"
    / "weights" / "best.pt"
)

OUT = ROOT / "processed" / "fp_reduction_v2"

# 严格只使用 Train 病例生成训练数据
TARGET_SPLIT = "train"

# 3D CNN 输入尺寸: [D,H,W]
PATCH_D = 9
PATCH_H = 64
PATCH_W = 64

# 图块采样的物理间距（毫米）
PATCH_Z_MM = 2.0
PATCH_XY_MM = 1.0

# YOLO用于寻找困难负样本
IMGSZ = 640
YOLO_CONF = 0.10
YOLO_MAX_DET = 100

# 每例CT最多保存多少个困难负样本
NEGATIVE_PER_CASE = 20

# 避免把真实结节或排除结节误当负样本
GT_MARGIN_MM = 3.0
EXCLUDED_MARGIN_MM = 3.0
EXCLUDED_DEFAULT_DIAMETER = 10.0

# 候选去重：中心相近时只保留高分框
DEDUP_XY_MM = 8.0
DEDUP_Z_MM = 6.0

SEED = 42


# ============================================
# 2. 基础读写
# ============================================

def load_csv(path):
    if not path.is_file():
        raise FileNotFoundError(path)

    with path.open(
        "r", encoding="utf-8-sig", newline=""
    ) as f:
        return list(csv.DictReader(f))


def write_csv(path, header, rows):
    with path.open(
        "w", encoding="utf-8-sig", newline=""
    ) as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def get_train_cases():
    cases = sorted(
        row["seriesuid"]
        for row in load_csv(SPLITS)
        if row["split"] == TARGET_SPLIT
    )

    if not cases:
        raise RuntimeError("没有找到Train病例")

    return cases


def get_ct_paths():
    paths = {}

    for path in RAW.rglob("*.mhd"):
        if path.stem in paths:
            raise RuntimeError(
                f"重复CT病例：{path.stem}"
            )
        paths[path.stem] = path

    return paths


def load_nodules(path, cases, excluded=False):
    result = defaultdict(list)
    case_set = set(cases)

    for row in load_csv(path):
        uid = row["seriesuid"]
        if uid not in case_set:
            continue

        diameter = float(row["diameter_mm"])

        if excluded and diameter < 0:
            diameter = EXCLUDED_DEFAULT_DIAMETER

        if diameter <= 0:
            raise ValueError(
                f"结节直径无效: {uid}, {diameter}"
            )

        result[uid].append({
            "world": (
                float(row["coordX"]),
                float(row["coordY"]),
                float(row["coordZ"])
            ),
            "diameter": diameter
        })

    return result


def world_distance(a, b):
    return math.sqrt(
        sum(
            (float(x) - float(y)) ** 2
            for x, y in zip(a, b)
        )
    )


# ============================================
# 3. CT预处理
# ============================================

def ct_to_uint8(slice_hu):
    a = np.clip(
        slice_hu.astype(np.float32),
        -1350, 150
    )

    a = (a + 1350.0) / 1500.0 * 255.0

    return a.astype(np.uint8)


# ============================================
# 4. YOLO候选生成
# ============================================

def predict_candidates(model, ct, volume):
    depth = volume.shape[0]
    candidates = []

    for z in range(depth):
        gray = ct_to_uint8(volume[z])
        rgb = np.repeat(
            gray[:, :, None], 3, axis=2
        )

        result = model.predict(
            source=rgb,
            imgsz=IMGSZ,
            conf=YOLO_CONF,
            iou=0.7,
            max_det=YOLO_MAX_DET,
            batch=1,
            device=0,
            verbose=False
        )[0]

        if result.boxes is None:
            continue

        boxes = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()

        for box, score in zip(boxes, scores):
            x1, y1, x2, y2 = map(float, box)

            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2

            world = (
                ct.TransformContinuousIndexToPhysicalPoint(
                    (float(cx), float(cy), float(z))
                )
            )

            candidates.append({
                "world": tuple(float(v) for v in world),
                "z_index": z,
                "score": float(score)
            })

    return candidates


# ============================================
# 5. 候选去重
# ============================================

def deduplicate_candidates(candidates):
    # 高分优先保留，避免同一位置重复采样
    ordered = sorted(
        candidates,
        key=lambda c: c["score"],
        reverse=True
    )

    kept = []

    for cand in ordered:
        x, y, z = cand["world"]
        duplicate = False

        for prev in kept:
            px, py, pz = prev["world"]

            xy_distance = math.hypot(
                x - px, y - py
            )
            z_distance = abs(z - pz)

            if (
                xy_distance < DEDUP_XY_MM
                and z_distance < DEDUP_Z_MM
            ):
                duplicate = True
                break

        if not duplicate:
            kept.append(cand)

    return kept


# ============================================
# 6. 排除参考结节及排除标注附近候选
# ============================================

def is_safe_negative(candidate, nodules, excluded):
    center = candidate["world"]

    for gt in nodules:
        radius = gt["diameter"] / 2.0

        if world_distance(
            center, gt["world"]
        ) <= radius + GT_MARGIN_MM:
            return False

    for item in excluded:
        radius = item["diameter"] / 2.0

        if world_distance(
            center, item["world"]
        ) <= radius + EXCLUDED_MARGIN_MM:
            return False

    return True


# ============================================
# 7. 三维图块提取
# ============================================

def extract_patch(volume, ct, world_center):
    """
    输出 [9,64,64] float32

    在图像自身的三个物理轴方向上采样：
    Z间距2mm，XY间距1mm。

    map_coordinates进行三线性插值。
    """

    ix, iy, iz = (
        ct.TransformPhysicalPointToContinuousIndex(
            tuple(float(v) for v in world_center)
        )
    )

    sx, sy, sz = ct.GetSpacing()

    z_offsets = (
        np.arange(PATCH_D, dtype=np.float32)
        - (PATCH_D - 1) / 2
    ) * PATCH_Z_MM / sz

    y_offsets = (
        np.arange(PATCH_H, dtype=np.float32)
        - (PATCH_H - 1) / 2
    ) * PATCH_XY_MM / sy

    x_offsets = (
        np.arange(PATCH_W, dtype=np.float32)
        - (PATCH_W - 1) / 2
    ) * PATCH_XY_MM / sx

    zz = iz + z_offsets[:, None, None]
    yy = iy + y_offsets[None, :, None]
    xx = ix + x_offsets[None, None, :]

    coords = np.broadcast_arrays(
        zz, yy, xx
    )

    sampled = map_coordinates(
        volume,
        coords,
        order=1,
        mode="constant",
        cval=-1000.0,
        prefilter=False
    )

    sampled = np.clip(
        sampled.astype(np.float32),
        -1350.0,
        150.0
    )

    # 统一归一化到0~1
    sampled = (
        sampled + 1350.0
    ) / 1500.0

    if sampled.shape != (
        PATCH_D, PATCH_H, PATCH_W
    ):
        raise RuntimeError(
            f"Patch维度异常: {sampled.shape}"
        )

    return sampled.astype(np.float32)


# ============================================
# 8. 样本保存
# ============================================

def save_sample(path, patch):
    if not np.all(np.isfinite(patch)):
        raise RuntimeError(
            f"样本包含NaN/Inf: {path}"
        )

    if not (
        patch.min() >= 0.0
        and patch.max() <= 1.0
    ):
        raise RuntimeError(
            f"归一化异常: {path}"
        )

    # float16可减少硬盘空间
    np.savez_compressed(
        path,
        image=patch.astype(np.float16)
    )


# ============================================
# 9. 主程序
# ============================================

def main():
    random.seed(SEED)

    required = (
        ANNOTATIONS,
        EXCLUDED,
        SPLITS,
        WEIGHT
    )

    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)

    if OUT.exists():
        raise RuntimeError(
            f"输出目录已存在: {OUT}\n"
            "为避免覆盖旧数据，请使用新的输出目录。"
        )

    cases = get_train_cases()
    ct_paths = get_ct_paths()

    missing = [
        uid for uid in cases
        if uid not in ct_paths
    ]

    if missing:
        raise RuntimeError(
            f"缺少CT病例: {missing}"
        )

    gt_map = load_nodules(
        ANNOTATIONS, cases
    )
    excluded_map = load_nodules(
        EXCLUDED, cases, excluded=True
    )

    (OUT / "positive").mkdir(
        parents=True
    )
    (OUT / "negative").mkdir(
        parents=True
    )

    model = YOLO(str(WEIGHT))

    manifest = []
    report = []
    total = Counter()

    print("========== 3D FP Reduction 数据构建 ==========")
    print("Train病例:", len(cases))
    print(
        "参考结节:",
        sum(len(gt_map[c]) for c in cases)
    )
    print("Patch尺寸:", (
        PATCH_D, PATCH_H, PATCH_W
    ))
    print(
        "每例最多困难负样本:",
        NEGATIVE_PER_CASE
    )

    for case_number, uid in enumerate(cases, 1):
        print(
            f"\n[{case_number}/{len(cases)}]"
            f" 病例 {uid[-16:]}"
        )

        ct = sitk.ReadImage(
            str(ct_paths[uid])
        )

        volume = sitk.GetArrayFromImage(
            ct
        ).astype(np.float32)

        nodules = gt_map[uid]
        excluded = excluded_map[uid]

        case_positive = 0
        case_negative = 0

        # ------------------------------------
        # A. 参考结节中心：正样本
        # ------------------------------------

        for index, gt in enumerate(nodules):
            patch = extract_patch(
                volume, ct, gt["world"]
            )

            filename = (
                f"case{case_number:03d}"
                f"_pos{index:03d}.npz"
            )

            save_sample(
                OUT / "positive" / filename,
                patch
            )

            manifest.append([
                filename,
                uid,
                1,
                "reference_nodule",
                *gt["world"],
                gt["diameter"],
                "",
                "train"
            ])

            case_positive += 1

        # ------------------------------------
        # B. YOLO产生困难负样本候选
        # ------------------------------------

        raw_candidates = predict_candidates(
            model, ct, volume
        )

        unique_candidates = deduplicate_candidates(
            raw_candidates
        )

        safe_candidates = [
            c for c in unique_candidates
            if is_safe_negative(
                c, nodules, excluded
            )
        ]

        # 高置信度困难候选优先
        safe_candidates.sort(
            key=lambda c: c["score"],
            reverse=True
        )

        selected = safe_candidates[
            :NEGATIVE_PER_CASE
        ]

        for index, candidate in enumerate(selected):
            patch = extract_patch(
                volume,
                ct,
                candidate["world"]
            )

            filename = (
                f"case{case_number:03d}"
                f"_neg{index:03d}.npz"
            )

            save_sample(
                OUT / "negative" / filename,
                patch
            )

            manifest.append([
                filename,
                uid,
                0,
                "yolo_hard_negative",
                *candidate["world"],
                "",
                candidate["score"],
                "train"
            ])

            case_negative += 1

        total["positive"] += case_positive
        total["negative"] += case_negative

        report.append([
            uid,
            case_positive,
            len(raw_candidates),
            len(unique_candidates),
            len(safe_candidates),
            case_negative
        ])

        print(
            f"  正样本: {case_positive}"
            f" | 原始预测: {len(raw_candidates)}"
            f" | 去重后: {len(unique_candidates)}"
            f" | 安全负候选: {len(safe_candidates)}"
            f" | 保存负样本: {case_negative}"
        )

        del volume, ct

    # ----------------------------------------
    # 10. 保存清单与统计
    # ----------------------------------------

    write_csv(
        OUT / "manifest.csv",
        [
            "filename",
            "seriesuid",
            "label",
            "source",
            "world_x",
            "world_y",
            "world_z",
            "diameter_mm",
            "yolo_confidence",
            "split"
        ],
        manifest
    )

    write_csv(
        OUT / "case_report.csv",
        [
            "seriesuid",
            "positive_count",
            "raw_predictions",
            "unique_candidates",
            "safe_negative_candidates",
            "saved_negatives"
        ],
        report
    )

    # 检查样本数量
    positive_files = list(
        (OUT / "positive").glob("*.npz")
    )
    negative_files = list(
        (OUT / "negative").glob("*.npz")
    )

    if len(positive_files) != total["positive"]:
        raise RuntimeError("正样本文件数量不一致")

    if len(negative_files) != total["negative"]:
        raise RuntimeError("负样本文件数量不一致")

    if len(manifest) != (
        total["positive"] + total["negative"]
    ):
        raise RuntimeError("样本清单数量不一致")

    print("\n========== 3D数据构建完成 ==========")
    print("Train病例:", len(cases))
    print("正样本:", total["positive"])
    print("困难负样本:", total["negative"])
    print("总样本:", len(manifest))
    print("保存目录:", OUT)
    print("清单:", OUT / "manifest.csv")
    print("病例报告:", OUT / "case_report.csv")
    print("\n原始CT、V4.1、YOLO权重未修改。")
    print(
        "注意：负样本是未匹配参考标注的"
        "候选，并非医学确认阴性。"
    )


if __name__ == "__main__":
    main()
