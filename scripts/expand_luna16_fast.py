
from pathlib import Path
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import argparse
import csv
import os
import random
import shutil
import time

# 必须在导入 huggingface_hub 前设置
ROOT = Path(r"E:\LungLens_Data")
RAW_ROOT = ROOT / "raw"
HF_HOME = ROOT / "hf_cache"
CACHE_DIR = HF_HOME / "hub"

os.environ["HF_HOME"] = str(HF_HOME)
os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "180"
os.environ["HF_HUB_ETAG_TIMEOUT"] = "60"

import SimpleITK as sitk
from huggingface_hub import HfApi, hf_hub_download

REPO = "Yousra0x0/LUNA16_RAW"
ANN = ROOT / "annotations" / "annotations.csv"

MANIFEST = ROOT / "download_manifest_fast.csv"
FAILURES = ROOT / "download_failures_fast.csv"

MAX_FILE_MB = 220
MIN_FREE_GB = 15


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--target", type=int, default=165)
    parser.add_argument("--batch", type=int, default=10)
    parser.add_argument("--max-gb", type=float, default=2.0)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--download", action="store_true")

    return parser.parse_args()


def load_annotations():
    with ANN.open("r", encoding="utf-8-sig", newline="") as f:
        return Counter(
            row["seriesuid"]
            for row in csv.DictReader(f)
        )


def read_raw_reference(mhd):
    with mhd.open("r", encoding="ascii", errors="replace") as f:
        for line in f:
            if line.strip().startswith("ElementDataFile"):
                return Path(
                    line.split("=", 1)[1].strip().strip('"')
                ).name

    return None


def local_pair_valid(case):
    folder = RAW_ROOT / case["subset"]

    mhd = folder / f'{case["uid"]}.mhd'
    raw = folder / f'{case["uid"]}.raw'

    if not mhd.is_file() or not raw.is_file():
        return False

    if mhd.stat().st_size != case["mhd_size"]:
        return False

    if raw.stat().st_size != case["raw_size"]:
        return False

    return (
        read_raw_reference(mhd)
        == raw.name
    )


def query_remote():
    print("正在查询 LUNA16 全部远程子集...")

    api = HfApi()
    cases = []

    for i in range(10):
        subset = f"subset{i}"

        try:
            items = list(api.list_repo_tree(
                repo_id=REPO,
                repo_type="dataset",
                path_in_repo=subset,
                recursive=False,
                expand=True
            ))

        except Exception as e:
            raise RuntimeError(
                f"{subset} 远程查询失败，停止以避免错误统计: {e}"
            )

        files = {
            item.path: item
            for item in items
            if getattr(item, "size", None) is not None
        }

        count = 0

        for path, item in files.items():
            if not path.endswith(".mhd"):
                continue

            uid = Path(path).stem
            raw_path = f"{subset}/{uid}.raw"

            if raw_path not in files:
                continue

            raw_size = files[raw_path].size
            mhd_size = item.size

            if not raw_size or not mhd_size:
                continue

            cases.append({
                "subset": subset,
                "uid": uid,
                "mhd_size": mhd_size,
                "raw_size": raw_size,
                "nodule_count": 0
            })

            count += 1

        print(f"{subset}: {count} 例")

    uids = [c["uid"] for c in cases]

    if len(uids) != len(set(uids)):
        raise RuntimeError("远程仓库存在重复UID，停止下载")

    return cases


def append_csv(path, row):
    exists = path.exists()

    with path.open(
        "a", encoding="utf-8", newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(row.keys())
        )

        if not exists:
            writer.writeheader()

        writer.writerow(row)


def download_file(remote_path, expected_size, destination, retries):
    destination.parent.mkdir(parents=True, exist_ok=True)

    # 已有正确大小的文件不重复下载
    if (
        destination.is_file()
        and destination.stat().st_size == expected_size
    ):
        return

    temp = destination.with_name(
        destination.name + ".part"
    )

    for attempt in range(1, retries + 1):

        try:
            cached = Path(hf_hub_download(
                repo_id=REPO,
                repo_type="dataset",
                filename=remote_path,
                cache_dir=str(CACHE_DIR),
                etag_timeout=60
            ))

            if cached.stat().st_size != expected_size:
                raise RuntimeError("缓存文件大小与远程记录不一致")

            if (
                destination.is_file()
                and destination.stat().st_size == expected_size
            ):
                return

            shutil.copyfile(cached, temp)

            if temp.stat().st_size != expected_size:
                raise RuntimeError("本地复制后的文件大小不一致")

            # 原子替换，避免留下半成品
            os.replace(temp, destination)

            return

        except Exception as e:

            if temp.exists():
                temp.unlink()

            if attempt == retries:
                raise RuntimeError(
                    f"{remote_path} 连续失败 {retries} 次: {e}"
                )

            wait = min(3 * (2 ** (attempt - 1)), 30)

            print(
                f"下载失败，第 {attempt}/{retries} 次重试，"
                f"{wait} 秒后继续：{e}"
            )

            time.sleep(wait)


def download_case(case, retries):
    subset = case["subset"]
    uid = case["uid"]

    folder = RAW_ROOT / subset

    download_file(
        f"{subset}/{uid}.mhd",
        case["mhd_size"],
        folder / f"{uid}.mhd",
        retries
    )

    download_file(
        f"{subset}/{uid}.raw",
        case["raw_size"],
        folder / f"{uid}.raw",
        retries
    )

    if not local_pair_valid(case):
        raise RuntimeError("MHD/RAW配对或文件大小校验失败")

    return case


def validate_ct(case):
    mhd = (
        RAW_ROOT / case["subset"] /
        f'{case["uid"]}.mhd'
    )

    image = sitk.ReadImage(str(mhd))

    if image.GetDimension() != 3:
        raise RuntimeError("CT不是三维影像")

    size = image.GetSize()
    spacing = image.GetSpacing()

    if any(x <= 0 for x in size):
        raise RuntimeError("CT尺寸异常")

    if any(x <= 0 for x in spacing):
        raise RuntimeError("CT间距异常")

    return size, spacing


def select_batch(candidates, limit, max_gb, batch_number):
    positive = [
        c for c in candidates
        if c["nodule_count"] > 0
    ]

    negative = [
        c for c in candidates
        if c["nodule_count"] == 0
    ]

    rng = random.Random(42 + batch_number)
    rng.shuffle(positive)
    rng.shuffle(negative)

    positive_limit = round(limit * 0.8)

    ordered = (
        positive[:positive_limit]
        + negative[:limit - positive_limit]
        + positive[positive_limit:]
        + negative[limit - positive_limit:]
    )

    selected = []
    total_mb = 0.0

    for case in ordered:

        if len(selected) >= limit:
            break

        size_mb = (
            case["raw_size"] + case["mhd_size"]
        ) / 1024**2

        if total_mb + size_mb > max_gb * 1024:
            continue

        selected.append(case)
        total_mb += size_mb

    return selected, total_mb


def get_verified(cases):
    return {
        c["uid"]
        for c in cases
        if local_pair_valid(c)
    }


def main():
    args = parse_args()

    if args.target < 1 or args.batch < 1:
        raise ValueError("目标病例数和批次必须大于0")

    if not 1 <= args.workers <= 4:
        raise ValueError("workers 建议设置为1～4")

    if args.retries < 1 or args.max_gb <= 0:
        raise ValueError("重试次数和下载预算无效")

    if not ANN.is_file():
        raise FileNotFoundError(ANN)

    RAW_ROOT.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    counts = load_annotations()
    remote_cases = query_remote()

    for case in remote_cases:
        case["nodule_count"] = counts.get(case["uid"], 0)

    verified = get_verified(remote_cases)

    print("\n========== LungLens 高速下载 ==========")
    print("目标病例:", args.target)
    print("本地完整病例:", len(verified))
    print("待新增病例:", max(0, args.target - len(verified)))
    print("并发数:", args.workers)
    print("每批上限:", args.batch)
    print("每批RAW预算:", args.max_gb, "GB")
    print("自动重试:", args.retries)

    if len(verified) >= args.target:
        print("已经达到目标，无需下载")
        return

    attempted = set()
    batch_number = 0

    while len(verified) < args.target:

        candidates = [
            c for c in remote_cases
            if c["uid"] not in verified
            and c["uid"] not in attempted
            and c["raw_size"] <= MAX_FILE_MB * 1024**2
        ]

        if not candidates:
            print("没有更多可用候选病例")
            break

        limit = min(
            args.batch,
            args.target - len(verified)
        )

        batch_number += 1

        selected, total_mb = select_batch(
            candidates,
            limit,
            args.max_gb,
            batch_number
        )

        if not selected:
            print("本批没有符合大小预算的病例")
            break

        print(f"\n========== 第 {batch_number} 批 ==========")
        print("计划下载:", len(selected))
        print("预计大小:", round(total_mb, 1), "MB")

        for c in selected:
            print(
                c["subset"],
                c["uid"][-12:],
                "结节:", c["nodule_count"]
            )

        if not args.download:
            print("\n预览模式：未下载文件")
            break

        free_gb = shutil.disk_usage(ROOT).free / 1024**3
        required_gb = total_mb / 1024 + MIN_FREE_GB

        if free_gb < required_gb:
            print("磁盘空间不足，停止下载")
            break

        batch_success = 0

        with ThreadPoolExecutor(
            max_workers=args.workers
        ) as pool:

            future_map = {
                pool.submit(
                    download_case,
                    case,
                    args.retries
                ): case
                for case in selected
            }

            for future in as_completed(future_map):

                case = future_map[future]
                uid = case["uid"]

                try:
                    future.result()

                    size, spacing = validate_ct(case)

                    append_csv(MANIFEST, {
                        "time": datetime.now().isoformat(timespec="seconds"),
                        "uid": uid,
                        "subset": case["subset"],
                        "nodule_count": case["nodule_count"],
                        "raw_size": case["raw_size"],
                        "x": size[0],
                        "y": size[1],
                        "z": size[2],
                        "spacing": str(spacing),
                        "status": "success"
                    })

                    print(
                        f"[成功] {uid[-12:]} "
                        f"| CT尺寸 {size}"
                    )

                    batch_success += 1

                except Exception as e:

                    attempted.add(uid)

                    append_csv(FAILURES, {
                        "time": datetime.now().isoformat(timespec="seconds"),
                        "uid": uid,
                        "subset": case["subset"],
                        "error": str(e)
                    })

                    print(
                        f"[失败] {uid[-12:]}: {e}"
                    )

        verified = get_verified(remote_cases)

        print("\n本批成功:", batch_success)
        print("目前完整病例:", len(verified), "/", args.target)
        print(
            "E盘剩余:",
            round(shutil.disk_usage(ROOT).free / 1024**3, 2),
            "GB"
        )

    print("\n========== 下载任务结束 ==========")
    print("完整CT病例:", len(verified))
    print("目标病例:", args.target)
    print("新增病例:", max(0, len(verified) - 0))
    print("成功清单:", MANIFEST)
    print("失败清单:", FAILURES)
    print("原有文件未主动删除")


if __name__ == "__main__":
    main()
