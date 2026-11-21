
from pathlib import Path
from collections import defaultdict
import csv
import SimpleITK as sitk

ROOT = Path(r"E:\LungLens_Data")
CT_DIR = ROOT / "raw" / "subset0"
CSV_FILE = ROOT / "annotations" / "annotations.csv"

# 读取结节标注
annotations = defaultdict(list)

with CSV_FILE.open("r", encoding="utf-8-sig", newline="") as f:
    for row in csv.DictReader(f):
        annotations[row["seriesuid"]].append(row)

valid_cases = 0
valid_nodules = 0
invalid_cases = []

for mhd_path in sorted(CT_DIR.glob("*.mhd")):
    uid = mhd_path.stem

    print(f"\n检查病例: {uid[-12:]}")

    try:
        # 读取CT三维影像
        image = sitk.ReadImage(str(mhd_path))

        size = image.GetSize()
        spacing = image.GetSpacing()

        print("影像尺寸:", size)
        print("体素间距:", spacing)

        if any(s <= 0 for s in spacing):
            raise ValueError("体素间距无效")

        case_valid_nodules = 0

        # 检查结节坐标
        for row in annotations.get(uid, []):
            world = (
                float(row["coordX"]),
                float(row["coordY"]),
                float(row["coordZ"])
            )

            # 将毫米坐标转换为连续体素索引
            voxel = image.TransformPhysicalPointToContinuousIndex(world)

            inside = all(
                0 <= voxel[i] < size[i]
                for i in range(3)
            )

            if not inside:
                raise ValueError(
                    f"结节坐标超出影像范围: {voxel}"
                )

            diameter = float(row["diameter_mm"])

            if diameter <= 0:
                raise ValueError("结节直径无效")

            case_valid_nodules += 1

        print("结节坐标检查通过:", case_valid_nodules)

        valid_cases += 1
        valid_nodules += case_valid_nodules

    except Exception as e:
        invalid_cases.append((uid, str(e)))
        print("检查失败:", e)

print("\n========== 最终结果 ==========")
print("正常病例:", valid_cases)
print("坐标检查通过的结节:", valid_nodules)
print("异常病例:", len(invalid_cases))

for uid, error in invalid_cases:
    print("异常:", uid[-12:], error)
