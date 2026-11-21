
from pathlib import Path
from collections import Counter
import csv

ROOT = Path(r"E:\LungLens_Data")
DATA = ROOT / "processed" / "yolo_v3"
ANN = ROOT / "annotations" / "annotations.csv"

with ANN.open("r", encoding="utf-8-sig", newline="") as f:
    reference = Counter(
        row["seriesuid"] for row in csv.DictReader(f)
    )

with (DATA / "case_split.csv").open(
    "r", encoding="utf-8-sig", newline=""
) as f:
    split_map = {
        row["seriesuid"]: row["split"]
        for row in csv.DictReader(f)
    }

print("========== YOLO V3 数据审计 ==========")

total_saved = 0

for split in ("train", "val", "test"):
    images = sorted((DATA / "images" / split).glob("*.png"))
    saved = Counter()

    for image in images:
        uid = image.stem.rsplit("_z", 1)[0]

        if split_map.get(uid) != split:
            raise RuntimeError(f"病例划分冲突: {uid}")

        label = DATA / "labels" / split / (
            image.stem + ".txt"
        )

        if not label.exists():
            raise FileNotFoundError(label)

        lines = label.read_text(
            encoding="utf-8"
        ).splitlines()

        for line in lines:
            values = line.split()
            if len(values) != 5:
                raise ValueError(f"标签格式错误: {label}")

            cls, cx, cy, bw, bh = map(float, values)

            if cls != 0 or not (
                0 <= cx <= 1 and 0 <= cy <= 1
                and 0 < bw <= 1 and 0 < bh <= 1
                and cx - bw / 2 >= -1e-5
                and cy - bh / 2 >= -1e-5
                and cx + bw / 2 <= 1.00001
                and cy + bh / 2 <= 1.00001
            ):
                raise ValueError(f"标签坐标异常: {label}")

        saved[uid] += len(lines)

    ref_count = sum(
        reference[uid]
        for uid, assigned in split_map.items()
        if assigned == split
    )

    saved_count = sum(saved.values())
    total_saved += saved_count

    print(f"\n{split.upper()}")
    print("病例数:", sum(s == split for s in split_map.values()))
    print("图片数:", len(images))
    print("参考结节数:", ref_count)
    print("已生成标签数:", saved_count)
    print("未保留的参考标签:", ref_count - saved_count)

    for uid, assigned in sorted(split_map.items()):
        if assigned != split:
            continue

        missing = reference[uid] - saved[uid]
        if missing > 0:
            print(
                "  待复核病例:",
                uid[-12:],
                "未保留:", missing
            )

print("\n========== 汇总 ==========")
print("已生成标签:", total_saved)
print("全数据源参考结节:", sum(reference.values()))
print("注意：本审计不判断病灶是否实际可见")
print("数据审计完成")
