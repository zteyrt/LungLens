
from pathlib import Path
from collections import defaultdict, Counter
import csv
import math

ROOT = Path(r"E:\LungLens_Data")
RUN = Path(r"D:\肺结节模型训练\runs")
CANDIDATES = RUN / "full_ct_v5_val_v1/candidates_3d.csv"
ANNOTATIONS = ROOT / "annotations/annotations.csv"
SPLITS = ROOT / "processed/yolo_v41/case_split.csv"
OUTPUT = RUN / "full_ct_v5_excluded_v1/fn_diagnosis.csv"

CONF = 0.5487037301063538


def read_csv(path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def dist(a, b):
    return math.sqrt(
        (a["x"]-b["x"])**2
        + (a["y"]-b["y"])**2
        + (a["z"]-b["z"])**2
    )


def main():
    cases = {
        r["seriesuid"] for r in read_csv(SPLITS)
        if r["split"] == "val"
    }

    gt = defaultdict(list)
    preds = defaultdict(list)

    for r in read_csv(ANNOTATIONS):
        if r["seriesuid"] in cases:
            gt[r["seriesuid"]].append({
                "x": float(r["coordX"]),
                "y": float(r["coordY"]),
                "z": float(r["coordZ"]),
                "diameter": float(r["diameter_mm"])
            })

    for r in read_csv(CANDIDATES):
        if r["seriesuid"] in cases:
            preds[r["seriesuid"]].append({
                "x": float(r["world_x"]),
                "y": float(r["world_y"]),
                "z": float(r["world_z"]),
                "score": float(r["score"]),
                "slices": int(r["num_slices"])
            })

    output_rows = []
    stats = Counter()

    for uid in sorted(cases):
        gts = gt[uid]
        all_preds = preds[uid]
        selected = sorted(
            (p for p in all_preds if p["score"] >= CONF),
            key=lambda p: p["score"],
            reverse=True
        )

        matched = set()

        # 与上一版FROC保持一致的贪心匹配
        for p in selected:
            inside = []

            for i, g in enumerate(gts):
                radius = g["diameter"] / 2
                if radius > 0:
                    ratio = dist(p, g) / radius
                    if ratio < 1:
                        inside.append((ratio, i))

            inside.sort()
            for _, i in inside:
                if i not in matched:
                    matched.add(i)
                    break

        for i, g in enumerate(gts):
            if i in matched:
                stats["HIT"] += 1
                continue

            radius = g["diameter"] / 2
            inside = [
                p for p in all_preds
                if dist(p, g) < radius
            ]

            nearest = min(
                all_preds,
                key=lambda p: dist(p, g),
                default=None
            )

            if any(p["score"] >= CONF for p in inside):
                reason = "MATCH_CONFLICT"
            elif inside:
                reason = "LOW_CONFIDENCE"
            elif nearest is not None and dist(nearest, g) <= 10:
                reason = "LOCALIZATION_ERROR"
            else:
                reason = "NO_NEARBY_CANDIDATE"

            stats[reason] += 1

            output_rows.append({
                "seriesuid": uid,
                "annotation_index": i,
                "diameter_mm": round(g["diameter"], 3),
                "reason": reason,
                "best_inside_score": (
                    round(max(p["score"] for p in inside), 4)
                    if inside else ""
                ),
                "nearest_distance_mm": (
                    round(dist(nearest, g), 3)
                    if nearest else ""
                ),
                "nearest_score": (
                    round(nearest["score"], 4)
                    if nearest else ""
                )
            })

    if OUTPUT.exists():
        raise FileExistsError(
            f"结果文件已存在，为避免覆盖已停止: {OUTPUT}"
        )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "seriesuid", "annotation_index",
                "diameter_mm", "reason",
                "best_inside_score",
                "nearest_distance_mm", "nearest_score"
            ]
        )
        writer.writeheader()
        writer.writerows(output_rows)

    print("========== 完整CT漏检原因 ==========")
    print("成功匹配:", stats["HIT"])
    print("置信度不足:", stats["LOW_CONFIDENCE"])
    print("附近候选定位偏差:", stats["LOCALIZATION_ERROR"])
    print("附近无候选:", stats["NO_NEARBY_CANDIDATE"])
    print("匹配竞争:", stats["MATCH_CONFLICT"])
    print("漏检合计:", len(output_rows))
    print("结果文件:", OUTPUT)
    print("注意：分类是基于已有候选和距离规则的诊断。")


if __name__ == "__main__":
    main()
