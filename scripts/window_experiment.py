from pathlib import Path
import numpy as np
import SimpleITK as sitk
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

# 1. 数据路径
raw_dir = Path(r"E:\LungLens_Data\raw\subset0")
output_dir = Path(r"D:\肺结节模型训练\evaluation")
output_dir.mkdir(parents=True, exist_ok=True)

# 2. 读取CT
ct_files = list(raw_dir.glob("*.mhd"))

if not ct_files:
    raise FileNotFoundError("没有找到CT文件")

ct_path = ct_files[0]

image = sitk.ReadImage(str(ct_path))
ct = sitk.GetArrayFromImage(image)

# 3. 选择此前确认的结节切片
slice_idx = 78

if not 0 <= slice_idx < ct.shape[0]:
    raise ValueError("切片编号超出范围")

ct_slice = ct[slice_idx].astype(np.float32)

print("CT形状:", ct.shape)
print("当前切片:", slice_idx)
print("原始HU最小值:", ct_slice.min())
print("原始HU最大值:", ct_slice.max())

# 4. 肺窗转换函数
def apply_window(slice_data, width, level):
    hu_min = level - width / 2
    hu_max = level + width / 2

    clipped = np.clip(slice_data, hu_min, hu_max)

    normalized = (clipped - hu_min) / width

    return (normalized * 255).astype(np.uint8)

# 5. 不同窗口
lung_window = apply_window(ct_slice, 1500, -600)
soft_window = apply_window(ct_slice, 400, 40)

# 6. 绘图对比
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(ct_slice, cmap="gray")
axes[0].set_title("Original HU (Auto Display)")

axes[1].imshow(lung_window, cmap="gray", vmin=0, vmax=255)
axes[1].set_title("Lung Window: WW1500 WL-600")

axes[2].imshow(soft_window, cmap="gray", vmin=0, vmax=255)
axes[2].set_title("Soft Tissue: WW400 WL40")

for ax in axes:
    ax.axis("off")

plt.tight_layout()

output_path = output_dir / "window_comparison.png"

plt.savefig(output_path, dpi=150)
plt.close(fig)

# 7. 保存肺窗切片
plt.imsave(
    output_dir / "lung_window_slice78.png",
    lung_window,
    cmap="gray",
    vmin=0,
    vmax=255
)

print("肺窗图像范围:", lung_window.min(), lung_window.max())
print("图像对比已保存:", output_path)
print("肺窗切片已保存！")
