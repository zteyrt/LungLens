
from pathlib import Path
from collections import Counter
import csv
import shutil

ROOT = Path(r"E:\LungLens_Data")
CT_DIR = ROOT / "raw" / "subset0"
CSV_PATH = ROOT / "annotations" / "annotations.csv"

# 1. 读取结节标注
if not CSV_PATH.exists():
    raise FileNotFoundError(CSV_PATH)

with CSV_PATH.open("r", encoding="utf-8-sig", newline="") as f:
    annotations = list(csv.DictReader(f))

nodule_counts = Counter(
    row["seriesuid"] for row in annotations
)

# 2. 检查 CT 文件
mhd_files = sorted(CT_DIR.glob("*.mhd"))

print("========== LungLens 数据检查 ==========")
print("CT 目录:", CT_DIR)
print("CT 病例数:", len(mhd_files))

total_nodules = 0

for mhd in mhd_files:
    # 从 MHD 元数据中寻找对应的 RAW 文件
    header = mhd.read_text(
        encoding="utf-8", errors="replace"
    )

    raw_name = None
    for line in header.splitlines():
        if line.strip().startswith("ElementDataFile"):
            raw_name = line.split("=", 1)[1].strip()
            break

    if not raw_name:
        raw_ok = False
        size_mb = 0
    else:
        raw_path = mhd.parent / raw_name
        raw_ok = raw_path.is_file()
        size_mb = (
            raw_path.stat().st_size / (1024 ** 2)
            if raw_ok else 0
        )

    count = nodule_counts.get(mhd.stem, 0)
    total_nodules += count

    print(
        f"病例 {mhd.stem[-12:]} | "
        f"结节 {count} | "
        f"RAW {'正常' if raw_ok else '缺失'} | "
        f"{size_mb:.1f} MB"
    )

# 3. 检查磁盘空间
total, used, free = shutil.disk_usage(ROOT)

print("\n========== 汇总 ==========")
print("CT 病例数:", len(mhd_files))
print("参考标注结节数:", total_nodules)
print("E 盘剩余空间:", round(free / 1024**3, 2), "GB")

print("\n数据检查完成")
