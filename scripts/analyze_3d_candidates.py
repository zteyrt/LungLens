
from pathlib import Path
from collections import defaultdict
import csv
import math

PROJECT = Path(r"D:\肺结节模型训练")
SOURCE = PROJECT / "runs/full_ct_v41_val_v1"
CANDIDATE_FILE = SOURCE / "candidates_3d.csv"

ANNOTATION_FILE = Path(
    r"E:\LungLens_Data\annotations\annotations.csv"
)
SPLIT_FILE = Path(
    r"E:\LungLens_Data\processed\yolo_v41\case_split.csv"
)

OUT = PROJECT / "runs/full_ct_v41_postprocess_analysis"
OUT.mkdir(parents=True, exist_ok=True)

MIN_SLICES = (1, 2, 3, 4, 5)
THRESHOLDS = [round(i / 100, 2) for i in range(5, 96)]


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


def evaluate(cases, gt_map, candidates, conf, min_slices):
    tp = fp = fn = 0

    for uid in cases:
        gts = gt_map.get(uid, [])
        preds = [
            p for p in candidates.get(uid, [])
            if p["score"] >= conf
            and p["num_slices"] >= min_slices
        ]

        preds.sort(
            key=lambda p: p["score"],
            reverse=True
        )

        matched = set()

        for pred in preds:
            best_idx = None
            best_ratio = float("inf")

            for i, gt in enumerate(gts):
                if i in matched:
                    continue

                radius = gt["diameter"] / 2
                if radius <= 0:
                    continue

                ratio = distance(pred, gt) / radius

                if ratio <= 1 and ratio < best_ratio:
                    best_ratio = ratio
                    best_idx = i

            if best_idx is not None:
                matched.add(best_idx)
                tp += 1
            else:
                fp += 1

        fn += len(gts) - len(matched)

    total = tp + fn

    return {
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "sensitivity": tp / total if total else 0,
        "fp_per_scan": fp / len(cases)
    }


def main():
    cases = sorted(
        row["seriesuid"]
        for row in read_csv(SPLIT_FILE)
        if row["split"] == "val"
    )

    if not cases:
        raise RuntimeError("没有找到Val病例")

    gt_map = defaultdict(list)

    for row in read_csv(ANNOTATION_FILE):
        uid = row["seriesuid"]

        if uid not in cases:
            continue

        gt_map[uid].append({
            "x": float(row["coordX"]),
            "y": float(row["coordY"]),
            "z": float(row["coordZ"]),
            "diameter": float(row["diameter_mm"])
        })

    candidates = defaultdict(list)

    for row in read_csv(CANDIDATE_FILE):
        uid = row["seriesuid"]

        if uid not in cases:
            continue

        candidates[uid].append({
            "x": float(row["world_x"]),
            "y": float(row["world_y"]),
            "z": float(row["world_z"]),
            "score": float(row["score"]),
            "num_slices": int(row["num_slices"]),
            "num_boxes": int(row["num_boxes"])
        })

    print("========== 三维候选后处理分析 ==========")
    print("验证病例:", len(cases))
    print(
        "参考结节:",
        sum(len(gt_map[uid]) for uid in cases)
    )
    print(
        "三维候选:",
        sum(len(candidates[uid]) for uid in cases)
    )

    rows = []

    for min_slices in MIN_SLICES:
        for conf in THRESHOLDS:
            result = evaluate(
                cases, gt_map, candidates,
                conf, min_slices
            )

            rows.append({
                "min_slices": min_slices,
                "conf": conf,
                **result
            })

    # 保存完整网格结果
    output = OUT / "postprocess_grid.csv"

    with output.open(
        "w", encoding="utf-8-sig", newline=""
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "min_slices", "conf",
                "TP", "FP", "FN",
                "sensitivity", "fp_per_scan"
            ]
        )
        writer.writeheader()
        writer.writerows(rows)

    print("\n各跨层条件在 <=4 FP/scan 时的最好观察结果")
    print("层数  conf    TP   FP   FN   Sensitivity  FP/scan")

    for min_slices in MIN_SLICES:
        valid = [
            r for r in rows
            if r["min_slices"] == min_slices
            and r["fp_per_scan"] <= 4
        ]

        if not valid:
            print(min_slices, "没有符合条件的组合")
            continue

        best = max(
            valid,
            key=lambda r: (
                r["sensitivity"],
                -r["fp_per_scan"]
            )
        )

        print(
            f"{min_slices:3d}  "
            f"{best['conf']:.2f}  "
            f"{best['TP']:3d}  "
            f"{best['FP']:3d}  "
            f"{best['FN']:3d}  "
            f"{best['sensitivity']:8.2%}  "
            f"{best['fp_per_scan']:7.3f}"
        )

    print("\n完整分析结果:", output)
    print("没有重新推理，也没有修改模型。")
    print(
        "结果仅用于验证集探索，"
        "不能视为独立测试性能。"
    )


if __name__ == "__main__":
    main()
