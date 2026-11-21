
from pathlib import Path
from collections import defaultdict, Counter
import csv
import math

PROJECT = Path(r"D:\肺结节模型训练")
BASE = PROJECT / "runs/full_ct_v41_val_v1"
OUT = PROJECT / "runs/full_ct_v41_excluded_v1"

ANN_DIR = Path(r"E:\LungLens_Data\annotations")
ANN = ANN_DIR / "annotations.csv"
EXCLUDED = ANN_DIR / "annotations_excluded.csv"
SPLIT_FILE = Path(
    r"E:\LungLens_Data\processed\yolo_v41\case_split.csv"
)

CANDIDATES_FILE = BASE / "candidates_3d.csv"

# LUNA16: diameter=-1 时采用10mm的替代直径
EXCLUDED_DEFAULT_DIAMETER = 10.0

MIN_SLICES_OPTIONS = (1, 2)
REPORT_CONFS = (0.05, 0.10, 0.20, 0.30, 0.50, 0.57)

# 仅使用验证集
SPLIT = "val"


def read_csv(path):
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(
        "r", encoding="utf-8-sig", newline=""
    ) as f:
        return list(csv.DictReader(f))


def save_csv(path, header, rows):
    with path.open(
        "w", encoding="utf-8-sig", newline=""
    ) as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def get_cases():
    return sorted(
        row["seriesuid"]
        for row in read_csv(SPLIT_FILE)
        if row["split"] == SPLIT
    )


def load_annotations(path, cases, is_excluded=False):
    result = defaultdict(list)
    case_set = set(cases)

    for row in read_csv(path):
        uid = row["seriesuid"]
        if uid not in case_set:
            continue

        diameter = float(row["diameter_mm"])

        if is_excluded and diameter < 0:
            diameter = EXCLUDED_DEFAULT_DIAMETER

        if diameter <= 0:
            raise ValueError(
                f"无效标注直径: {uid} {diameter}"
            )

        result[uid].append({
            "x": float(row["coordX"]),
            "y": float(row["coordY"]),
            "z": float(row["coordZ"]),
            "radius": diameter / 2.0
        })

    return result


def load_candidates(cases):
    result = defaultdict(list)
    case_set = set(cases)

    for row in read_csv(CANDIDATES_FILE):
        uid = row["seriesuid"]

        if uid not in case_set:
            continue

        result[uid].append({
            "x": float(row["world_x"]),
            "y": float(row["world_y"]),
            "z": float(row["world_z"]),
            "score": float(row["score"]),
            "slices": int(row["num_slices"])
        })

    return result


def distance(a, b):
    return math.sqrt(
        (a["x"] - b["x"]) ** 2
        + (a["y"] - b["y"]) ** 2
        + (a["z"] - b["z"]) ** 2
    )


def inside(candidate, annotation):
    # 与参考中心距离小于结节半径
    return distance(candidate, annotation) < annotation["radius"]


def evaluate(cases, gt_map, excluded_map, cand_map,
             confidence, min_slices):
    total = Counter()

    for uid in cases:
        gts = gt_map[uid]
        excluded = excluded_map[uid]

        predictions = sorted(
            [
                c for c in cand_map[uid]
                if c["score"] >= confidence
                and c["slices"] >= min_slices
            ],
            key=lambda c: c["score"],
            reverse=True
        )

        matched_gt = set()

        for candidate in predictions:
            # 优先判断是否落在正式参考结节内
            matches = []

            for i, gt in enumerate(gts):
                if inside(candidate, gt):
                    ratio = (
                        distance(candidate, gt)
                        / gt["radius"]
                    )
                    matches.append((ratio, i))

            if matches:
                matches.sort()
                unmatched = [
                    item for item in matches
                    if item[1] not in matched_gt
                ]

                if unmatched:
                    matched_gt.add(unmatched[0][1])
                    total["TP"] += 1
                else:
                    # 同一结节上的重复预测不重复记TP
                    total["DUPLICATE_IGNORED"] += 1

                continue

            # 其次判断是否命中排除标注
            if any(
                inside(candidate, ann)
                for ann in excluded
            ):
                total["EXCLUDED_IGNORED"] += 1
                continue

            total["FP"] += 1

        total["FN"] += len(gts) - len(matched_gt)

    total_nodules = total["TP"] + total["FN"]

    sensitivity = (
        total["TP"] / total_nodules
        if total_nodules else 0.0
    )

    fp_per_scan = total["FP"] / len(cases)

    return {
        "TP": total["TP"],
        "FP": total["FP"],
        "FN": total["FN"],
        "excluded_ignored": total["EXCLUDED_IGNORED"],
        "duplicate_ignored": total["DUPLICATE_IGNORED"],
        "sensitivity": sensitivity,
        "fp_per_scan": fp_per_scan
    }


def main():
    cases = get_cases()
    if not cases:
        raise RuntimeError("没有找到Val病例")

    gt_map = load_annotations(ANN, cases)
    excluded_map = load_annotations(
        EXCLUDED, cases, is_excluded=True
    )
    cand_map = load_candidates(cases)

    OUT.mkdir(parents=True, exist_ok=True)

    print("========== LUNA16排除标注复评 ==========")
    print("验证CT:", len(cases))
    print(
        "参考结节:",
        sum(len(gt_map[c]) for c in cases)
    )
    print(
        "排除标注:",
        sum(len(excluded_map[c]) for c in cases)
    )
    print(
        "三维候选:",
        sum(len(cand_map[c]) for c in cases)
    )

    all_scores = sorted({
        c["score"]
        for uid in cases
        for c in cand_map[uid]
    }, reverse=True)

    # 覆盖全部不同的候选置信度，便于寻找操作点
    thresholds = [float("inf")] + all_scores

    for min_slices in MIN_SLICES_OPTIONS:
        print(
            f"\n========== 最少{min_slices}层 =========="
        )
        print(
            "conf    TP   FP   FN   "
            "Sensitivity   FP/scan   排除忽略   重复忽略"
        )

        for conf in REPORT_CONFS:
            r = evaluate(
                cases, gt_map, excluded_map,
                cand_map, conf, min_slices
            )

            print(
                f"{conf:.2f}  "
                f"{r['TP']:3d}  "
                f"{r['FP']:4d}  "
                f"{r['FN']:3d}  "
                f"{r['sensitivity']:9.2%}  "
                f"{r['fp_per_scan']:8.3f}  "
                f"{r['excluded_ignored']:5d}  "
                f"{r['duplicate_ignored']:5d}"
            )

        rows = []

        for threshold in thresholds:
            r = evaluate(
                cases, gt_map, excluded_map,
                cand_map, threshold, min_slices
            )
            rows.append({
                "threshold": threshold,
                **r
            })

        save_csv(
            OUT / f"froc_excluded_min{min_slices}.csv",
            [
                "threshold", "TP", "FP", "FN",
                "excluded_ignored", "duplicate_ignored",
                "sensitivity", "fp_per_scan"
            ],
            [
                [
                    row["threshold"],
                    row["TP"],
                    row["FP"],
                    row["FN"],
                    row["excluded_ignored"],
                    row["duplicate_ignored"],
                    row["sensitivity"],
                    row["fp_per_scan"]
                ]
                for row in rows
            ]
        )

        eligible = [
            r for r in rows
            if r["fp_per_scan"] <= 4.0
        ]

        best = max(
            eligible,
            key=lambda r: (
                r["sensitivity"],
                -r["fp_per_scan"]
            )
        )

        print("\n<=4 FP/scan的最佳观察结果")
        print("TP:", best["TP"])
        print("FP:", best["FP"])
        print("FN:", best["FN"])
        print(
            "Sensitivity:",
            f"{best['sensitivity']:.2%}"
        )
        print(
            "FP/scan:",
            f"{best['fp_per_scan']:.3f}"
        )
        print("阈值:", best["threshold"])

    print("\n========== 复评完成 ==========")
    print("结果目录:", OUT)
    print("未重新推理、训练或修改原始数据")
    print(
        "注意：这是参考LUNA16排除逻辑的验证集复评，"
        "并非官方评估脚本的完整复现。"
    )


if __name__ == "__main__":
    main()
