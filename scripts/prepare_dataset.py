
from pathlib import Path
import json
import random

import pandas as pd
import SimpleITK as sitk

# 1. 设置路径
DATA_ROOT = Path(r"E:\LungLens_Data")
RAW_DIR = DATA_ROOT / "raw" / "subset0"
CSV_PATH = DATA_ROOT / "annotations" / "annotations.csv"
OUTPUT_DIR = DATA_ROOT / "processed"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 2. 读取结节标注
df = pd.read_csv(CSV_PATH)

# 3. 找到所有已下载的CT
ct_files = sorted(RAW_DIR.glob("*.mhd"))

print("发现CT文件数:", len(ct_files))

valid_cases = []

# 4. 验证CT文件与标注是否匹配
for ct_path in ct_files:

    uid = ct_path.stem
    raw_path = ct_path.with_suffix(".raw")

    if not raw_path.exists():
        print("缺少RAW文件:", uid)
        continue

    nodules = df[df["seriesuid"].astype(str) == uid]

    if nodules.empty:
        print("没有匹配的结节标注:", uid)
        continue

    # 只读取图像头信息，不加载完整三维数组
    reader = sitk.ImageFileReader()
    reader.SetFileName(str(ct_path))

    try:
        reader.ReadImageInformation()
        size = reader.GetSize()
    except Exception as e:
        print("CT头信息读取失败:", uid, e)
        continue

    valid_cases.append(uid)

    print(
        "有效病例:",
        uid[-12:],
        "切片数:",
        size[2],
        "结节数:",
        len(nodules)
    )

# 5. 按病例划分训练集和验证集
random.seed(42)
random.shuffle(valid_cases)

if len(valid_cases) < 2:
    raise RuntimeError("至少需要2例有效CT才能划分训练与验证集")

val_count = max(1, round(len(valid_cases) / 3))

train_cases = valid_cases[:-val_count]
val_cases = valid_cases[-val_count:]

# 6. 检查没有病例重复
assert set(train_cases).isdisjoint(set(val_cases))

# 7. 保存划分结果
split_info = {
    "train": train_cases,
    "val": val_cases
}

output_path = OUTPUT_DIR / "dataset_split.json"

output_path.write_text(
    json.dumps(split_info, indent=4, ensure_ascii=False),
    encoding="utf-8"
)

print("\n========== 数据集划分结果 ==========")
print("训练病例:", len(train_cases))
print("验证病例:", len(val_cases))
print("是否存在重复:", bool(set(train_cases) & set(val_cases)))
print("划分结果保存至:", output_path)
