
from pathlib import Path
from collections import Counter
import argparse
import csv
import shutil

import SimpleITK as sitk
from huggingface_hub import HfApi, hf_hub_download

ROOT = Path(r"E:\LungLens_Data")
RAW_ROOT = ROOT / "raw"
ANN = ROOT / "annotations" / "annotations.csv"

REPO = "Yousra0x0/LUNA16_RAW"
TARGET = 200
MAX_FILE_MB = 220
MIN_FREE_GB = 15

parser = argparse.ArgumentParser()
parser.add_argument(
    "--download", action="store_true",
    help="实际下载选中的CT"
)
parser.add_argument(
    "--batch", type=int, default=10,
    help="每批最多下载多少例"
)
parser.add_argument(
    "--max-gb", type=float, default=2.0,
    help="每批RAW文件大小上限(GB)"
)
args = parser.parse_args()

if args.batch < 1 or args.max_gb <= 0:
    raise ValueError("下载批次和大小必须大于0")

# 1. 读取参考结节
with ANN.open(
    "r", encoding="utf-8-sig", newline=""
) as f:
    counts = Counter(
        row["seriesuid"] for row in csv.DictReader(f)
    )

# 2. 检查本地已有病例
def local_complete(mhd):
    raw = mhd.with_suffix(".raw")
    return (
        raw.exists()
        and raw.stat().st_size > 0
    )

existing = {
    p.stem
    for p in RAW_ROOT.glob("subset*/*.mhd")
    if local_complete(p)
}

print("本地已有完整文件对:", len(existing))

# 3. 查询远程仓库
api = HfApi()

print("\n正在检查远程子集...")

available = {}
remote_cases = []

for subset_id in range(10):
    subset = f"subset{subset_id}"

    try:
        items = list(api.list_repo_tree(
            repo_id=REPO,
            repo_type="dataset",
            path_in_repo=subset,
            recursive=False,
            expand=True
        ))
    except Exception as e:
        print(f"{subset}: 无法读取 - {e}")
        continue

    files = {
        item.path: item
        for item in items
        if getattr(item, "size", None) is not None
    }

    available[subset] = files
    found = 0

    for path, item in files.items():
        if not path.endswith(".mhd"):
            continue

        uid = Path(path).stem
        raw_path = f"{subset}/{uid}.raw"

        if raw_path not in files:
            continue

        raw_size = files[raw_path].size

        if raw_size is None or raw_size <= 0:
            continue

        size_mb = raw_size / 1024**2

        remote_cases.append(
            (subset, uid, counts.get(uid, 0), size_mb)
        )
        found += 1

    print(f"{subset}: 可配对CT {found} 例")

# 4. 跳过已经下载的病例
candidates = [
    item for item in remote_cases
    if item[1] not in existing
    and item[3] <= MAX_FILE_MB
]

# 有结节和没有参考结节的病例分开处理
positive = sorted(
    [c for c in candidates if c[2] > 0],
    key=lambda c: (-c[2], c[3], c[1])
)

unannotated = sorted(
    [c for c in candidates if c[2] == 0],
    key=lambda c: (c[3], c[1])
)

# 每批优先约80%参考结节病例，20%其他病例
positive_limit = max(1, round(args.batch * 0.8))

ordered = (
    positive[:positive_limit]
    + unannotated[:args.batch - positive_limit]
    + positive[positive_limit:]
    + unannotated[args.batch - positive_limit:]
)

# 5. 限制每批数量与下载大小
selected = []
total_mb = 0.0
remaining = max(0, TARGET - len(existing))

for item in ordered:
    if len(selected) >= min(args.batch, remaining):
        break

    if total_mb + item[3] > args.max_gb * 1024:
        continue

    selected.append(item)
    total_mb += item[3]

print("\n========== 下载计划 ==========")
print("目标CT数量:", TARGET)
print("本地已有:", len(existing))
print("远程可配对病例:", len(remote_cases))
print("满足大小限制的新候选:", len(candidates))
print("本批计划:", len(selected))
print("本批RAW预计大小:", round(total_mb, 1), "MB")

for subset, uid, nodules, mb in selected:
    print(
        subset,
        uid[-12:],
        "参考结节:", nodules,
        "大小:", round(mb, 1), "MB"
    )

if not args.download:
    print("\n当前只是预览，未下载任何文件。")
    raise SystemExit(0)

# 6. 实际下载
success = 0

for index, (subset, uid, nodules, mb) in enumerate(
    selected, 1
):
    free_gb = shutil.disk_usage(ROOT).free / 1024**3

    if free_gb < MIN_FREE_GB + args.max_gb:
        print("剩余空间不足，停止本批下载")
        break

    print(
        f"\n[{index}/{len(selected)}]"
        f" 下载 {subset}/{uid[-12:]}"
    )

    try:
        for ext in (".mhd", ".raw"):
            filename = f"{subset}/{uid}{ext}"

            hf_hub_download(
                repo_id=REPO,
                repo_type="dataset",
                filename=filename,
                local_dir=str(RAW_ROOT)
            )

        mhd_path = RAW_ROOT / subset / f"{uid}.mhd"
        raw_path = RAW_ROOT / subset / f"{uid}.raw"

        # 检查元数据引用的RAW文件名
        header = mhd_path.read_text(
            encoding="utf-8", errors="replace"
        )

        raw_name = None
        for line in header.splitlines():
            if line.strip().startswith("ElementDataFile"):
                raw_name = line.split("=", 1)[1].strip()
                break

        if raw_name != raw_path.name:
            raise ValueError("MHD引用的RAW文件名异常")

        # 检查实际文件大小
        remote_size = available[subset][
            f"{subset}/{uid}.raw"
        ].size

        if raw_path.stat().st_size != remote_size:
            raise ValueError("RAW文件大小不匹配")

        # 解码CT，检查元数据及参考结节坐标
        image = sitk.ReadImage(str(mhd_path))
        size = image.GetSize()
        spacing = image.GetSpacing()

        if any(s <= 0 for s in size + spacing):
            raise ValueError("影像尺寸或间距无效")

        if image.GetDimension() != 3:
            raise ValueError("不是三维CT")

        print(
            "校验通过 | 尺寸:", size,
            "| 间距:", tuple(round(s, 3) for s in spacing)
        )
        success += 1

    except Exception as e:
        print("下载或校验失败:", e)
        print("保留文件，稍后排查或重试")

print("\n========== 本批结果 ==========")
print("本批成功:", success)
print("本批计划:", len(selected))
print("已有文件未主动删除或修改")
print("请运行数据清单检查，确认最新总数")

