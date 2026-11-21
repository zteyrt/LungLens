
from pathlib import Path
from collections import Counter
import csv

from huggingface_hub import HfApi, hf_hub_download

ROOT = Path(r"E:\LungLens_Data")
RAW_DIR = ROOT / "raw" / "subset0"
CACHE_DIR = ROOT / "hf_cache"
CSV_FILE = ROOT / "annotations" / "annotations.csv"

REPO = "Yousra0x0/LUNA16_RAW"

# 本次最多下载6例，控制新增数据量
MAX_CASES = 6
MAX_TOTAL_MB = 800
MAX_SINGLE_MB = 150

RAW_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# 1. 统计有参考结节的病例
with CSV_FILE.open("r", encoding="utf-8-sig", newline="") as f:
    rows = list(csv.DictReader(f))

nodule_counts = Counter(row["seriesuid"] for row in rows)

# 2. 读取远程文件清单，不下载CT
api = HfApi()
files = list(api.list_repo_tree(
    repo_id=REPO,
    repo_type="dataset",
    path_in_repo="subset0",
    recursive=False,
    expand=True
))

remote = {item.path: item for item in files}

# 3. 筛选病例
candidates = []

for uid, count in nodule_counts.items():
    mhd_name = f"subset0/{uid}.mhd"
    raw_name = f"subset0/{uid}.raw"

    if mhd_name not in remote or raw_name not in remote:
        continue

    # 已经完整存在的病例不重复下载
    if (RAW_DIR / f"{uid}.mhd").exists() and \
       (RAW_DIR / f"{uid}.raw").exists():
        continue

    raw_size = getattr(remote[raw_name], "size", None)
    if raw_size is None:
        continue

    size_mb = raw_size / 1024**2

    if size_mb > MAX_SINGLE_MB:
        continue

    candidates.append((uid, count, size_mb))

# 结节数量多的优先；相同数量时选文件小的
candidates.sort(key=lambda x: (-x[1], x[2], x[0]))

# 4. 控制下载预算
selected = []
total_mb = 0.0

for uid, count, size_mb in candidates:
    if len(selected) >= MAX_CASES:
        break

    if total_mb + size_mb > MAX_TOTAL_MB:
        continue

    selected.append((uid, count, size_mb))
    total_mb += size_mb

print("========== 本次下载计划 ==========")
print("可用候选病例:", len(candidates))
print("计划新增病例:", len(selected))
print("预计RAW总大小:", round(total_mb, 1), "MB")

for uid, count, size_mb in selected:
    print(uid[-12:], "结节:", count,
          "大小:", round(size_mb, 1), "MB")

# 5. 下载选中的病例
success = 0

for index, (uid, count, size_mb) in enumerate(selected, 1):
    print(f"\n[{index}/{len(selected)}] 下载病例 {uid[-12:]}")

    try:
        for ext in [".mhd", ".raw"]:
            hf_hub_download(
                repo_id=REPO,
                repo_type="dataset",
                filename=f"subset0/{uid}{ext}",
                local_dir=str(ROOT / "raw"),
                cache_dir=str(CACHE_DIR)
            )

        mhd_path = RAW_DIR / f"{uid}.mhd"
        raw_path = RAW_DIR / f"{uid}.raw"

        header = mhd_path.read_text(
            encoding="utf-8", errors="replace"
        )

        raw_file = None
        for line in header.splitlines():
            if line.strip().startswith("ElementDataFile"):
                raw_file = line.split("=", 1)[1].strip()
                break

        if raw_file != raw_path.name:
            raise ValueError("MHD与RAW文件名不匹配")

        if raw_path.stat().st_size == 0:
            raise ValueError("RAW文件为空")

        success += 1
        print("下载完成，文件初步检查通过")

    except Exception as e:
        print("此病例失败:", e)
        print("保留现有文件，稍后可以重试")

print("\n========== 下载汇总 ==========")
print("本次完成病例:", success)
print("原有完整病例:", 6)
print("请重新运行数据清单检查以确认最终数量")

