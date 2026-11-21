from pathlib import Path
from collections import Counter
import csv

root = Path(r"E:\LungLens_Data")
raw = root / "raw"
old = root / "processed/yolo_v41/case_split.csv"
out = root / "processed/yolo_v5/case_split.csv"

with old.open("r", encoding="utf-8-sig", newline="") as f:
    original = list(csv.DictReader(f))

old_split = {}

for row in original:
    uid = row["seriesuid"]
    split = row["split"].lower()

    if uid in old_split:
        if old_split[uid] != split:
            raise RuntimeError(f"病例划分冲突: {uid}")
        continue

    if split not in ("train", "val", "test"):
        raise RuntimeError(f"未知划分: {split}")

    old_split[uid] = split

current = set()

for mhd in raw.glob("subset*/*.mhd"):
    r = mhd.with_suffix(".raw")
    if r.is_file() and r.stat().st_size > 0:
        if mhd.stem in current:
            raise RuntimeError(f"重复病例UID: {mhd.stem}")
        current.add(mhd.stem)

missing = set(old_split) - current

if missing:
    raise RuntimeError(
        f"原有病例缺失 {len(missing)} 例，停止生成划分"
    )

new_cases = current - set(old_split)

split_map = dict(old_split)

for uid in new_cases:
    split_map[uid] = "train"

counts = Counter(split_map.values())

print("========== V5 病例划分 ==========")
print("CT总数:", len(current))
print("原始病例:", len(old_split))
print("新增Train病例:", len(new_cases))
print("Train:", counts["train"])
print("Val:", counts["val"])
print("Test:", counts["test"])

if len(current) != 118:
    raise RuntimeError("病例数量不是118，请核查")

if counts["val"] != 11 or counts["test"] != 7:
    raise RuntimeError("原验证或测试病例数量异常")

if out.exists():
    raise RuntimeError(
        f"目标文件已存在，为避免覆盖已停止: {out}"
    )

out.parent.mkdir(parents=True, exist_ok=True)

with out.open("w", encoding="utf-8-sig", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["seriesuid", "split"])

    for uid in sorted(split_map):
        writer.writerow([uid, split_map[uid]])

print("划分文件:", out)
print("V4.1原始划分未修改")
print("========== 完成 ==========")
