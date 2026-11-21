
from pathlib import Path
from collections import defaultdict
import csv
import math

import SimpleITK as sitk

ROOT = Path(r"E:\LungLens_Data")
RAW = ROOT / "raw"
ANN = ROOT / "annotations" / "annotations.csv"
V3 = ROOT / "processed" / "yolo_v3"

# 读取参考结节
annotations = defaultdict(list)

with ANN.open("r", encoding="utf-8-sig", newline="") as f:
    for row in csv.DictReader(f):
        annotations[row["seriesuid"]].append(row)

# 读取 V3 病例划分
with (V3 / "case_split.csv").open(
    "r", encoding="utf-8-sig", newline=""
) as f:
    split_map = {
        row["seriesuid"]: row["split"]
        for row in csv.DictReader(f)
    }

mhd_map = {
    p.stem: p for p in RAW.glob("subset*/*.mhd")
}

print("========== V3 跳过切片原因诊断 ==========")

total_slices = 0
old_skipped = 0
skipped_labels = 0
far_xy_slices = 0
near_xy_slices = 0

for uid, split in sorted(split_map.items()):

    if uid not in mhd_map:
        print("缺少CT:", uid)
        continue

    image = sitk.ReadImage(str(mhd_map[uid]))

    w, h, depth = image.GetSize()
    sx, sy, sz = image.GetSpacing()

    nodules = []

    for row in annotations[uid]:

        world = tuple(
            float(row[k])
            for k in ("coordX", "coordY", "coordZ")
        )

        x, y, z = image.TransformPhysicalPointToContinuousIndex(
            world
        )

        diameter = float(row["diameter_mm"])

        zi = max(
            0,
            min(depth - 1, math.floor(z + 0.5))
        )

        nodules.append((x, y, z, zi, diameter))

    groups = defaultdict(list)

    for i, item in enumerate(nodules):
        groups[item[3]].append(i)

    case_skipped = 0
    case_far = 0
    case_dropped = 0

    for zi, center_indices in sorted(groups.items()):

        total_slices += 1
        triggers = []

        for j, other in enumerate(nodules):

            if j in center_indices:
                continue

            x, y, z, center, diameter = other

            dz = abs(zi - z) * sz

            # 复现 V3 原来的判断条件
            if dz <= diameter / 2 + sz / 2:

                gaps = []

                for i in center_indices:
                    cx, cy, cz, cc, cd = nodules[i]

                    dx = (x - cx) * sx
                    dy = (y - cy) * sy

                    xy_distance = math.hypot(dx, dy)

                    # 两个结节近似圆形投影之间的间距
                    gap = xy_distance - (
                        diameter / 2 + cd / 2
                    )

                    gaps.append(gap)

                best_gap = min(gaps)

                triggers.append((dz, best_gap))

        if triggers:

            old_skipped += 1
            case_skipped += 1
            dropped = len(center_indices)

            skipped_labels += dropped
            case_dropped += dropped

            if any(gap > 0 for dz, gap in triggers):
                far_xy_slices += 1
                case_far += 1
            else:
                near_xy_slices += 1

            if old_skipped <= 20:
                print(
                    f"\n病例 {uid[-12:]} | {split} "
                    f"| Z={zi} | 当前标签={dropped}"
                )

                for dz, gap in triggers:
                    print(
                        f"  触发条件: dz={dz:.2f}mm "
                        f"| XY间距={gap:.2f}mm"
                    )

    if case_skipped:
        print(
            f"\n病例汇总 {uid[-12:]}: "
            f"跳过切片 {case_skipped}, "
            f"丢失中心标签 {case_dropped}, "
            f"XY明确分离的切片 {case_far}"
        )

print("\n========== 最终诊断 ==========")
print("中心切片总数:", total_slices)
print("原逻辑跳过切片:", old_skipped)
print("对应未生成中心标签:", skipped_labels)
print("存在XY明确分离结节的跳过切片:", far_xy_slices)
print("XY投影可能接近的跳过切片:", near_xy_slices)

print("\n诊断完成，未修改任何数据。")
