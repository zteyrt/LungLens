from pathlib import Path
import pandas as pd
import SimpleITK as sitk
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

ROOT = Path(r"E:\LungLens_Data")
RAW = ROOT / "raw" / "subset0"
CSV = ROOT / "annotations" / "annotations.csv"
OUTPUT = Path(r"D:\肺结节模型训练\evaluation")
OUTPUT.mkdir(parents=True, exist_ok=True)

# 读取结节标注
df = pd.read_csv(CSV)

# 找到已下载的CT
ct_files = list(RAW.glob("*.mhd"))

if not ct_files:
    raise FileNotFoundError("没有找到CT文件")

ct_path = ct_files[0]
seriesuid = ct_path.stem

# 获取对应结节
nodules = df[df["seriesuid"].astype(str) == seriesuid]

if nodules.empty:
    raise RuntimeError("当前CT没有对应结节标注")

# 读取CT
image = sitk.ReadImage(str(ct_path))
ct = sitk.GetArrayFromImage(image)
spacing = image.GetSpacing()

print("CT形状:", ct.shape)
print("像素间距:", spacing)

# 绘制所有已标注结节
for idx, row in nodules.iterrows():

    world_point = (
        float(row["coordX"]),
        float(row["coordY"]),
        float(row["coordZ"])
    )

    # 物理坐标(mm) -> 连续体素坐标(x,y,z)
    voxel = image.TransformPhysicalPointToContinuousIndex(
        world_point
    )

    x, y, z = voxel
    slice_idx = int(round(z))

    if not (0 <= slice_idx < ct.shape[0]):
        print("结节位置超出CT范围，跳过:", world_point)
        continue

    diameter_mm = float(row["diameter_mm"])

    # XY像素尺寸可能不同
    radius_x = diameter_mm / (2 * spacing[0])
    radius_y = diameter_mm / (2 * spacing[1])

    fig, ax = plt.subplots(figsize=(8, 8))

    ax.imshow(
        ct[slice_idx],
        cmap="gray",
        vmin=-1350,
        vmax=150
    )

    circle = Circle(
        (x, y),
        radius=max(radius_x, radius_y),
        fill=False,
        edgecolor="red",
        linewidth=2
    )

    ax.add_patch(circle)

    ax.scatter(x, y, c="yellow", s=12)
    ax.set_title(
        f"Slice {slice_idx} | Diameter {diameter_mm:.2f} mm"
    )
    ax.axis("off")

    out_path = OUTPUT / f"nodule_{idx}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("结节切片:", slice_idx)
    print("结节像素坐标:", (round(x, 2), round(y, 2)))
    print("标注直径(mm):", round(diameter_mm, 2))
    print("可视化结果:", out_path)

print("全部结节可视化完成！")
