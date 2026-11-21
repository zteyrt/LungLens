from pathlib import Path
import os
import pandas as pd

#必须在导入数据集之前设置缓存路径
ROOT = Path(r"E:\LungLens_Data")
os.environ["HF_HOME"] = str(ROOT / "hf_cache")

from huggingface_hub import HfApi,hf_hub_download

REPO = "Yousra0x0/LUNA16_RAW"
RAW_DIR = ROOT / "raw"
CSV_PATH = ROOT / "annotations" / "annotations.csv"


TARGET_CASES = 6
MAX_CASE_MB = 150
MAX_NEW_MB = 800

df = pd.read_csv(CSV_PATH)
annotated = set(df["seriesuid"].astype(str))

# 已下载的完整病例
local_dir = RAW_DIR / "subset0"

existing = {
    p.stem
    for p in local_dir.glob("*.mhd")
    if p.with_suffix(".raw").exists()
}

print("已有病例:", len(existing))

# 查询远程数据
api = HfApi()
files = list(api.list_repo_tree(
    repo_id=REPO,
    repo_type="dataset",
    path_in_repo="subset0"
))

file_map = {f.path: f for f in files}
candidates = []

for path, info in file_map.items():
    if not path.endswith(".raw"):
        continue

    uid = Path(path).stem
    mhd_path = f"subset0/{uid}.mhd"

    if uid not in annotated or uid in existing:
        continue

    if mhd_path not in file_map or info.size is None:
        continue

    size_mb = info.size / (1024 ** 2)

    if size_mb <= MAX_CASE_MB:
        candidates.append((size_mb, uid))

candidates.sort()

# 严格控制新增下载量
selected = []
total_mb = 0

for size_mb, uid in candidates:
    if len(selected) + len(existing) >= TARGET_CASES:
        break

    if total_mb + size_mb > MAX_NEW_MB:
        continue

    selected.append((size_mb, uid))
    total_mb += size_mb

print("计划新增病例:", len(selected))
print("预计新增原始数据:", round(total_mb, 2), "MB")

for i, (size_mb, uid) in enumerate(selected, 1):
    print(f"\n[{i}/{len(selected)}] 下载病例 {uid}")

    for ext in [".mhd", ".raw"]:
        hf_hub_download(
            repo_id=REPO,
            repo_type="dataset",
            filename=f"subset0/{uid}{ext}",
            local_dir=str(RAW_DIR)
        )

    print("完成，大小约:", round(size_mb, 2), "MB")

print("\n下载任务完成")
print("新增病例数量:", len(selected))
