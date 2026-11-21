from pathlib import Path
import SimpleITK as sitk
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
import pandas as pd

raw_dir = Path(r"E:\LungLens_Data\raw\subset0")
csv_path = Path(r"E:\LungLens_Data\annotations\annotations.csv")
output_dir = Path(r"D:\肺结节模型训练\evaluation")

ct_path = next(raw_dir.glob("*.mhd"))
image = sitk.ReadImage(str(ct_path))
ct = sitk.GetArrayFromImage(image)

df = pd.read_csv(csv_path)
row = df[df["seriesuid"] == ct_path.stem].iloc[0]

world = (
    float(row["coordX"]),
    float(row["coordY"]),
    float(row["coordZ"])
)

x, y, z = image.TransformPhysicalPointToContinuousIndex(world)

center_z = round(z)

print("CT shape:", ct.shape)
print("Nodule center:", (x, y, z))
print("Center slice:", center_z)

fig, axes = plt.subplots(1, 5, figsize=(15, 4))

for i, offset in enumerate(range(-2, 3)):
    slice_idx = center_z + offset

    ax = axes[i]

    if not 0 <= slice_idx < ct.shape[0]:
        ax.axis("off")
        continue

    slice_data = ct[slice_idx]

    ax.imshow(
        slice_data,
        cmap="gray",
        vmin=-1350,
        vmax=150
    )

    # 标记同一个XY中心，方便观察各切片
    ax.scatter(x, y, c="yellow", s=15)

    ax.set_xlim(x - 35, x + 35)
    ax.set_ylim(y + 35, y - 35)
    ax.set_title(f"Slice {slice_idx}")
    ax.axis("off")

plt.tight_layout()

save_path = output_dir / "nodule_5_slices.png"
plt.savefig(save_path, dpi=150)
plt.close()

print("保存成功:", save_path)
