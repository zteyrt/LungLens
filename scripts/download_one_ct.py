from pathlib import Path
import os

root = Path(r"E:\LungLens_Data")
os.environ["HF_HOME"] = str(root / "hf_cache")

import pandas as pd
import SimpleITK as sitk
from huggingface_hub import HfApi, hf_hub_download

repo = "Yousra0x0/LUNA16_RAW"
raw_dir = root / "raw"
raw_dir.mkdir(parents=True, exist_ok=True)

csv_path = root / "annotations" / "annotations.csv"

if not csv_path.exists():
    raise FileNotFoundError("找不到 annotations.csv，请检查 E 盘数据目录")

df = pd.read_csv(csv_path)
annotated = set(df["seriesuid"].astype(str))

print("正在查询 CT 数据列表...")

api = HfApi()
files = list(api.list_repo_tree(
    repo_id=repo,
    repo_type="dataset",
    path_in_repo="subset0"
))

paths = {f.path: f for f in files}
candidates = []

for path, info in paths.items():
    if not path.endswith(".raw"):
        continue

    uid = Path(path).stem
    mhd_path = f"subset0/{uid}.mhd"
    size = info.size

    if (
        uid in annotated
        and mhd_path in paths
        and size is not None
        and size <= 300 * 1024 * 1024
    ):
        candidates.append((size, uid))

if not candidates:
    raise RuntimeError("没有找到符合条件的 CT 病例")

candidates.sort()
size, uid = candidates[0]

print("选择 CT:", uid)
print("CT 大小约:", round(size / 1024**2, 2), "MB")

for ext in [".mhd", ".raw"]:
    filename = f"subset0/{uid}{ext}"

    print("正在下载:", filename)

    hf_hub_download(
        repo_id=repo,
        repo_type="dataset",
        filename=filename,
        local_dir=str(raw_dir)
    )

ct_path = raw_dir / "subset0" / f"{uid}.mhd"

image = sitk.ReadImage(str(ct_path))
array = sitk.GetArrayFromImage(image)

nodules = df[df["seriesuid"].astype(str) == uid]

print("\n========== 检查结果 ==========")
print("CT 文件:", ct_path)
print("CT 形状 [Z,Y,X]:", array.shape)
print("像素间距 [X,Y,Z]:", image.GetSpacing())
print("结节标注数量:", len(nodules))
print(nodules[["coordX", "coordY", "coordZ", "diameter_mm"]])
print("CT 读取成功！")
