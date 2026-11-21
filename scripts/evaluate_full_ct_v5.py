
from pathlib import Path
from collections import defaultdict
import csv
import math

import numpy as np
import SimpleITK as sitk
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ultralytics import YOLO


# ==================================================
# 1. 配置
# ==================================================

from lunglens_paths import (
    PROJECT_ROOT as PROJECT, RAW_ROOT as RAW,
    ANNOTATIONS_FILE as ANNOTATIONS, LEGACY_EVAL_SPLIT as SPLIT_CSV,
)

WEIGHT = (
    PROJECT / "runs/yolo_v5_finetune_v1/weights/best.pt"
)

OUT = PROJECT / "runs/full_ct_v5_val_v1"

# 只评估验证集，保护独立测试集
SPLIT = "val"

IMGSZ = 640
BATCH = 4

# 从较低置信度开始保留候选，以便计算FROC
MIN_CONF = 0.05

# 完整CT中，每层最多保留的检测框
MAX_DET = 100

# 相邻切片合并设置
MAX_Z_GAP = 2
MAX_XY_DISTANCE_MM = 10.0

# FROC评估两个版本：
# 单层候选保留 / 至少连续两层出现
MIN_SLICE_OPTIONS = (1, 2)

# 输出主要观察的置信度阈值
REPORT_THRESHOLDS = (0.05, 0.10, 0.20, 0.30, 0.50)

# CT窗宽窗位，与先前数据生成保持一致
WINDOW_MIN = -1350.0
WINDOW_MAX = 150.0


# ==================================================
# 2. 通用工具
# ==================================================

def save_csv(path, header, rows):
    with path.open(
        "w", encoding="utf-8-sig", newline=""
    ) as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def load_val_cases():
    cases = []

    with SPLIT_CSV.open(
        "r", encoding="utf-8-sig", newline=""
    ) as f:
        for row in csv.DictReader(f):
            if row["split"] == SPLIT:
                cases.append(row["seriesuid"])

    if not cases:
        raise RuntimeError("没有找到验证集病例")

    return sorted(cases)


def find_ct_files():
    paths = {}

    # 搜索subset0到subset9中的完整MHD文件
    for path in RAW.rglob("*.mhd"):
        uid = path.stem

        if uid in paths:
            raise RuntimeError(
                f"重复的CT seriesuid: {uid}"
            )

        paths[uid] = path

    return paths


def load_annotations():
    annotations = defaultdict(list)

    with ANNOTATIONS.open(
        "r", encoding="utf-8-sig", newline=""
    ) as f:
        for row in csv.DictReader(f):
            annotations[row["seriesuid"]].append({
                "x": float(row["coordX"]),
                "y": float(row["coordY"]),
                "z": float(row["coordZ"]),
                "diameter": float(row["diameter_mm"])
            })

    return annotations


def ct_to_rgb(slice_hu):
    # 保持CT强度顺序，不使用每张图片单独归一化
    clipped = np.clip(
        slice_hu.astype(np.float32),
        WINDOW_MIN,
        WINDOW_MAX
    )

    image = (
        (clipped - WINDOW_MIN)
        / (WINDOW_MAX - WINDOW_MIN)
        * 255.0
    ).clip(0, 255).astype(np.uint8)

    # YOLO预训练网络需要3通道
    return np.repeat(image[:, :, None], 3, axis=2)


# ==================================================
# 3. 完整CT逐层检测
# ==================================================

def predict_volume(model, image, uid):
    volume = sitk.GetArrayFromImage(image)

    depth, height, width = volume.shape

    detections = []

    print(
        f"  CT尺寸: {width}x{height}x{depth}"
    )

    if width != 512 or height != 512:
        raise RuntimeError(
            f"{uid}: 当前脚本要求512x512 CT切片"
        )

    for start in range(0, depth, BATCH):
        end = min(start + BATCH, depth)

        batch_images = [
            ct_to_rgb(volume[z])
            for z in range(start, end)
        ]

        results = model.predict(
            source=batch_images,
            imgsz=IMGSZ,
            conf=MIN_CONF,
            iou=0.7,
            max_det=MAX_DET,
            batch=BATCH,
            device=0,
            verbose=False
        )

        if len(results) != len(batch_images):
            raise RuntimeError(
                f"{uid}: 预测结果数量异常"
            )

        for offset, result in enumerate(results):
            z = start + offset

            if result.boxes is None:
                continue

            boxes = (
                result.boxes.xyxy.cpu().numpy()
            )
            scores = (
                result.boxes.conf.cpu().numpy()
            )

            for box, score in zip(boxes, scores):
                x1, y1, x2, y2 = [
                    float(v) for v in box
                ]

                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0

                world = (
                    image.TransformContinuousIndexToPhysicalPoint(
                        (float(cx), float(cy), float(z))
                    )
                )

                detections.append({
                    "z_index": z,
                    "cx": cx,
                    "cy": cy,
                    "width": x2 - x1,
                    "height": y2 - y1,
                    "wx": float(world[0]),
                    "wy": float(world[1]),
                    "wz": float(world[2]),
                    "confidence": float(score)
                })

        if end % 100 == 0 or end == depth:
            print(
                f"  已处理切片: {end}/{depth}"
            )

    return detections


# ==================================================
# 4. 三维候选合并：并查集
# ==================================================

class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[
                self.parent[x]
            ]
            x = self.parent[x]
        return x

    def union(self, a, b):
        a = self.find(a)
        b = self.find(b)

        if a == b:
            return

        if self.rank[a] < self.rank[b]:
            a, b = b, a

        self.parent[b] = a

        if self.rank[a] == self.rank[b]:
            self.rank[a] += 1


def merge_3d(detections, image):
    if not detections:
        return []

    n = len(detections)
    uf = UnionFind(n)

    by_z = defaultdict(list)

    for i, det in enumerate(detections):
        by_z[det["z_index"]].append(i)

    spacing = image.GetSpacing()
    sx, sy = spacing[0], spacing[1]

    for z, indices in by_z.items():
        for dz in range(1, MAX_Z_GAP + 1):
            other_indices = by_z.get(z + dz, [])

            for i in indices:
                a = detections[i]

                for j in other_indices:
                    b = detections[j]

                    # 使用图像索引和像素间距计算
                    # 横向物理距离，支持常规轴向CT
                    dx_mm = (a["cx"] - b["cx"]) * sx
                    dy_mm = (a["cy"] - b["cy"]) * sy

                    distance = math.hypot(
                        dx_mm, dy_mm
                    )

                    # 兼顾不同大小的预测框
                    diameter_a = max(
                        a["width"] * sx,
                        a["height"] * sy
                    )
                    diameter_b = max(
                        b["width"] * sx,
                        b["height"] * sy
                    )

                    max_distance = min(
                        MAX_XY_DISTANCE_MM,
                        max(
                            3.0,
                            0.5 * (
                                diameter_a + diameter_b
                            )
                        )
                    )

                    if distance <= max_distance:
                        uf.union(i, j)

    clusters = defaultdict(list)

    for i, det in enumerate(detections):
        clusters[uf.find(i)].append(det)

    candidates = []

    for members in clusters.values():
        # 一个三维候选可能跨多张CT切片
        distinct_z = len({
            d["z_index"] for d in members
        })

        # 用置信度最高的框作为候选中心，
        # 防止简单平均使中心漂移
        best = max(
            members,
            key=lambda d: d["confidence"]
        )

        candidates.append({
            "wx": best["wx"],
            "wy": best["wy"],
            "wz": best["wz"],
            "score": best["confidence"],
            "num_slices": distinct_z,
            "num_boxes": len(members),
            "z_index": best["z_index"],
            "cx": best["cx"],
            "cy": best["cy"]
        })

    return candidates


# ==================================================
# 5. 结节级匹配
# ==================================================

def evaluate_case(candidates, nodules, threshold, min_slices):
    selected = [
        c for c in candidates
        if c["score"] >= threshold
        and c["num_slices"] >= min_slices
    ]

    selected.sort(
        key=lambda c: c["score"],
        reverse=True
    )

    matched_gt = set()

    tp = 0
    fp = 0

    # 每个候选最多匹配一个参考结节
    # 每个参考结节最多被检出一次
    for candidate in selected:
        best_idx = None
        best_ratio = float("inf")

        for idx, gt in enumerate(nodules):
            if idx in matched_gt:
                continue

            dx = candidate["wx"] - gt["x"]
            dy = candidate["wy"] - gt["y"]
            dz = candidate["wz"] - gt["z"]

            distance = math.sqrt(
                dx*dx + dy*dy + dz*dz
            )

            radius = gt["diameter"] / 2.0

            if radius <= 0:
                continue

            ratio = distance / radius

            if ratio <= 1.0 and ratio < best_ratio:
                best_ratio = ratio
                best_idx = idx

        if best_idx is not None:
            matched_gt.add(best_idx)
            tp += 1
        else:
            fp += 1

    fn = len(nodules) - tp

    return tp, fp, fn


# ==================================================
# 6. 汇总不同阈值下的FROC
# ==================================================

def evaluate_all(case_results, threshold, min_slices):
    tp = fp = fn = 0

    for case in case_results:
        a, b, c = evaluate_case(
            case["candidates"],
            case["nodules"],
            threshold,
            min_slices
        )

        tp += a
        fp += b
        fn += c

    sensitivity = (
        tp / (tp + fn)
        if tp + fn > 0 else 0.0
    )

    fp_per_scan = fp / len(case_results)

    return {
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "sensitivity": sensitivity,
        "fp_per_scan": fp_per_scan
    }


def create_curve(case_results, min_slices):
    scores = sorted({
        c["score"]
        for case in case_results
        for c in case["candidates"]
        if c["num_slices"] >= min_slices
    }, reverse=True)

    # 包含无预测的起点及所有实际候选阈值
    thresholds = [float("inf")] + scores

    rows = []

    for threshold in thresholds:
        result = evaluate_all(
            case_results,
            threshold,
            min_slices
        )

        rows.append({
            "threshold": threshold,
            **result
        })

    return rows


def plot_froc(curves):
    plt.figure(figsize=(8, 6))

    for min_slices, rows in curves.items():
        # 按FP/scan聚合，避免相同X值反复折返
        points = {}

        for row in rows:
            x = row["fp_per_scan"]
            y = row["sensitivity"]

            points[x] = max(
                points.get(x, 0.0), y
            )

        xs = sorted(points)
        ys = [points[x] for x in xs]

        plt.plot(
            xs, ys,
            marker=".",
            markersize=3,
            label=f"min slices = {min_slices}"
        )

    plt.axvline(
        x=4.0,
        linestyle="--",
        color="gray",
        label="4 FP/scan"
    )

    plt.axhline(
        y=0.85,
        linestyle=":",
        color="gray",
        label="85% sensitivity"
    )

    plt.xlabel("False positives per CT scan")
    plt.ylabel("Nodule-level sensitivity")
    plt.title("LungLens V4.1 Full CT - Preliminary FROC")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    path = OUT / "froc_val.png"
    plt.savefig(path, dpi=180)
    plt.close()

    return path


# ==================================================
# 7. 主程序
# ==================================================

def main():
    for path in (WEIGHT, ANNOTATIONS, SPLIT_CSV):
        if not path.is_file():
            raise FileNotFoundError(path)

    if OUT.exists():
        raise RuntimeError(
            f"结果目录已经存在: {OUT}\n"
            "请使用新目录名称，避免覆盖以前的实验。"
        )

    val_cases = load_val_cases()
    ct_paths = find_ct_files()
    annotations = load_annotations()

    missing = [
        uid for uid in val_cases
        if uid not in ct_paths
    ]

    if missing:
        raise RuntimeError(
            f"缺少验证集CT: {missing}"
        )

    OUT.mkdir(parents=True)

    print("========== LungLens 完整CT评估 ==========")
    print("数据集:", SPLIT)
    print("病例数量:", len(val_cases))
    print("模型:", WEIGHT)
    print(
        "参考结节数:",
        sum(len(annotations[uid]) for uid in val_cases)
    )

    model = YOLO(str(WEIGHT))

    case_results = []
    case_rows = []
    candidate_rows = []

    for case_index, uid in enumerate(val_cases, 1):
        print(
            f"\n[{case_index}/{len(val_cases)}]"
            f" 病例: {uid}"
        )

        image = sitk.ReadImage(
            str(ct_paths[uid])
        )

        nodules = annotations[uid]

        # 1. 完整体数据逐层推理
        detections = predict_volume(
            model, image, uid
        )

        # 2. 关联相邻切片
        candidates = merge_3d(
            detections, image
        )

        print(
            "  原始二维预测框:",
            len(detections)
        )
        print(
            "  三维候选:",
            len(candidates)
        )
        print(
            "  参考结节:",
            len(nodules)
        )

        case_results.append({
            "uid": uid,
            "nodules": nodules,
            "candidates": candidates
        })

        case_rows.append([
            uid,
            image.GetSize()[2],
            len(nodules),
            len(detections),
            len(candidates)
        ])

        for c in candidates:
            candidate_rows.append([
                uid,
                c["wx"],
                c["wy"],
                c["wz"],
                c["score"],
                c["num_slices"],
                c["num_boxes"],
                c["z_index"]
            ])

    # 保存三维候选与病例统计
    save_csv(
        OUT / "cases.csv",
        [
            "seriesuid", "depth",
            "reference_nodules",
            "2d_boxes", "3d_candidates"
        ],
        case_rows
    )

    save_csv(
        OUT / "candidates_3d.csv",
        [
            "seriesuid", "world_x", "world_y",
            "world_z", "score", "num_slices",
            "num_boxes", "representative_z"
        ],
        candidate_rows
    )

    curves = {}

    print("\n========== 结节级FROC结果 ==========")

    for min_slices in MIN_SLICE_OPTIONS:
        print(
            f"\n最少出现层数: {min_slices}"
        )

        print(
            "conf    TP    FP    FN"
            "    Sensitivity    FP/scan"
        )

        for threshold in REPORT_THRESHOLDS:
            result = evaluate_all(
                case_results,
                threshold,
                min_slices
            )

            print(
                f"{threshold:.2f}   "
                f"{result['TP']:3d}   "
                f"{result['FP']:4d}   "
                f"{result['FN']:3d}   "
                f"{result['sensitivity']:8.2%}   "
                f"{result['fp_per_scan']:8.3f}"
            )

        curve = create_curve(
            case_results, min_slices
        )

        curves[min_slices] = curve

        save_csv(
            OUT / f"froc_min{min_slices}.csv",
            [
                "threshold", "TP", "FP",
                "FN", "sensitivity",
                "fp_per_scan"
            ],
            [
                [
                    row["threshold"],
                    row["TP"],
                    row["FP"],
                    row["FN"],
                    row["sensitivity"],
                    row["fp_per_scan"]
                ]
                for row in curve
            ]
        )

        # 在不超过4 FP/scan条件下，
        # 寻找此验证集观察到的最高敏感度
        eligible = [
            row for row in curve
            if row["fp_per_scan"] <= 4.0
        ]

        if eligible:
            best = max(
                eligible,
                key=lambda r: r["sensitivity"]
            )

            print(
                "<=4 FP/scan最高观察敏感度:",
                f"{best['sensitivity']:.2%}"
            )
            print(
                "对应FP/scan:",
                f"{best['fp_per_scan']:.3f}"
            )
            print(
                "对应阈值:",
                best["threshold"]
            )

    figure = plot_froc(curves)

    print("\n========== 评估完成 ==========")
    print("结果目录:", OUT)
    print("FROC图:", figure)
    print("病例汇总: cases.csv")
    print("三维候选: candidates_3d.csv")
    print("FROC表: froc_min1.csv / froc_min2.csv")

    print(
        "\n重要：这是初步验证集FROC，"
        "未实现LUNA16官方排除标注处理。"
    )
    print(
        "不能直接把结果当作正式医学性能结论。"
    )


if __name__ == "__main__":
    main()
