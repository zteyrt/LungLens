
from pathlib import Path
import pandas as pd
import SimpleITK as sitk

ROOT = Path(r"E:\LungLens_Data")

RAW = ROOT / "raw" / "subset0"
CSV = ROOT / "annotations" / "annotations.csv"

df = pd.read_csv(CSV)

# 距离已标注结节中心至少 10mm
# 注意：这只是初步候选过滤
SAFE_DISTANCE_MM = 10.0

for ct_path in sorted(RAW.glob("*.mhd")):
    uid = ct_path.stem

    image = sitk.ReadImage(str(ct_path))
    spacing_z = image.GetSpacing()[2]

    total_slices = image.GetSize()[2]

    nodules = df[df["seriesuid"].astype(str) == uid]

    # 结节中心在 Z 方向的连续体素坐标
    centers_z = []

    for _, row in nodules.iterrows():
        world = (
            float(row["coordX"]),
            float(row["coordY"]),
            float(row["coordZ"])
        )

        _, _, z = image.TransformPhysicalPointToContinuousIndex(world)

        centers_z.append(z)

    candidates = []

    for z_idx in range(total_slices):
        # 当前切片与所有标注中心的最小距离(mm)
        min_distance = min(
            abs(z_idx - center_z) * spacing_z
            for center_z in centers_z
        )

        if min_distance >= SAFE_DISTANCE_MM:
            candidates.append(z_idx)

    print(
        f"病例 {uid[-12:]} | "
        f"总切片 {total_slices} | "
        f"背景候选 {len(candidates)}"
    )
