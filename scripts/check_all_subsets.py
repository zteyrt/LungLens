
from pathlib import Path
from collections import Counter
import csv

import SimpleITK as sitk

ROOT = Path(r"E:\LungLens_Data")
RAW = ROOT / "raw"
ANN = ROOT / "annotations" / "annotations.csv"

with ANN.open("r", encoding="utf-8-sig", newline="") as f:
    rows = list(csv.DictReader(f))

counts = Counter(row["seriesuid"] for row in rows)

total_cases = 0
total_nodules = 0
valid_cases = 0
failed = []

print("========== LungLens 全子集检查 ==========")

for subset in sorted(RAW.glob("subset*")):
    if not subset.is_dir():
        continue

    mhd_files = sorted(subset.glob("*.mhd"))

    if not mhd_files:
        continue

    print(f"\n{subset.name}: {len(mhd_files)} 例")

    for mhd in mhd_files:
        uid = mhd.stem
        total_cases += 1

        try:
            # 从MHD读取真正的RAW文件名
            raw_name = None

            for line in mhd.read_text(
                encoding="utf-8",
                errors="replace"
            ).splitlines():
                if line.strip().startswith("ElementDataFile"):
                    raw_name = line.split("=", 1)[1].strip()
                    break

            if not raw_name:
                raise ValueError("找不到ElementDataFile")

            raw_path = subset / raw_name

            if not raw_path.is_file():
                raise FileNotFoundError(raw_path)

            if raw_path.stat().st_size == 0:
                raise ValueError("RAW文件为空")

            # 使用SimpleITK读取三维CT
            image = sitk.ReadImage(str(mhd))

            if image.GetDimension() != 3:
                raise ValueError("不是三维影像")

            size = image.GetSize()
            spacing = image.GetSpacing()

            if any(x <= 0 for x in size + spacing):
                raise ValueError("尺寸或像素间距无效")

            valid_cases += 1
            total_nodules += counts.get(uid, 0)

            print(
                f"  {uid[-12:]} | "
                f"结节 {counts.get(uid, 0)} | "
                f"RAW {raw_path.stat().st_size / 1024**2:.1f}MB | "
                "正常"
            )

        except Exception as e:
            failed.append((uid, str(e)))
            print(f"  {uid[-12:]} | 失败: {e}")

print("\n========== 最终汇总 ==========")
print("发现CT文件记录:", total_cases)
print("成功解码CT:", valid_cases)
print("正常病例对应参考结节:", total_nodules)
print("异常病例:", len(failed))

for uid, error in failed:
    print("异常:", uid[-12:], error)
