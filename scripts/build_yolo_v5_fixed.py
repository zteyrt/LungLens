from pathlib import Path
from collections import defaultdict, Counter
import csv
import math
import time
import os
import shutil

import numpy as np
import SimpleITK as sitk
from PIL import Image


# ============================================================
# LungLens V5 数据集构建
# 输入：118例LUNA16 CT + annotations.csv + V5 case_split.csv
# 输出：E:\LungLens_Data\processed\yolo_v5_dataset
#
# 不修改原始CT、V4、V4.1或原始V5病例划分。
# 不套用仅针对V4生成的quality_screen.csv。
# 结节框为球形截面近似弱标注，不是真实病灶轮廓。
# ============================================================

ROOT = Path(r"E:\LungLens_Data")
RAW_ROOT = ROOT / "raw"
ANN_FILE = ROOT / "annotations" / "annotations.csv"
SPLIT_FILE = ROOT / "processed" / "yolo_v5" / "case_split.csv"

OUTPUT = ROOT / "processed" / "yolo_v5_dataset"
STAGING = ROOT / "processed" / "yolo_v5_dataset_building"

# 下载缺失RAW时使用本地HF缓存，避免重复下载已缓存的数据。
os.environ.setdefault("HF_HOME", str(ROOT / "hf_cache"))
from huggingface_hub import HfApi, hf_hub_download

HF_REPO = "Yousra0x0/LUNA16_RAW"

SPLITS = ("train", "val", "test")
EXPECTED_CASES = {"train": 100, "val": 11, "test": 7}

NEGATIVE_PER_POSITIVE_CASE = 2
NEGATIVE_PER_NEGATIVE_CASE = 3
MIN_CROSS_SECTION_DIAMETER_MM = 1.0
CENTRAL_Z_START = 0.20
CENTRAL_Z_END = 0.80


def fail(message):
    raise RuntimeError(message)


def write_csv(path, header, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def lung_window(pixels):
    low, high = -1350.0, 150.0
    clipped = np.clip(pixels, low, high)
    return ((clipped - low) / (high - low) * 255).astype(np.uint8)


def select_evenly(values, count):
    values = sorted(set(values))
    if count <= 0 or not values:
        return []
    if len(values) <= count:
        return values
    indices = np.linspace(0, len(values) - 1, count)
    indices = sorted(set(int(round(i)) for i in indices))
    return [values[i] for i in indices]


def read_element_data_file(mhd_path):
    text = mhd_path.read_text(encoding="utf-8", errors="replace")
    for line in text.splitlines():
        if line.strip().lower().startswith("elementdatafile"):
            return line.split("=", 1)[1].strip()
    return None


def load_annotations():
    if not ANN_FILE.is_file():
        raise FileNotFoundError(ANN_FILE)

    required = {"seriesuid", "coordX", "coordY", "coordZ", "diameter_mm"}
    result = defaultdict(list)

    with ANN_FILE.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            fail("annotations.csv缺少必要列：" + ", ".join(sorted(required)))
        for row in reader:
            uid = row["seriesuid"].strip()
            if uid:
                result[uid].append(row)
    return result


def load_split():
    if not SPLIT_FILE.is_file():
        raise FileNotFoundError(SPLIT_FILE)

    split_map = {}
    with SPLIT_FILE.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or not {"seriesuid", "split"}.issubset(reader.fieldnames):
            fail("case_split.csv必须包含seriesuid和split两列。")

        for row in reader:
            uid = row["seriesuid"].strip()
            split = row["split"].strip().lower()
            if not uid:
                fail("case_split.csv中存在空seriesuid。")
            if split not in SPLITS:
                fail(f"非法数据集划分：{uid} -> {split}")
            if uid in split_map:
                fail(f"病例划分表中存在重复UID：{uid}")
            split_map[uid] = split

    counts = Counter(split_map.values())
    actual = {s: counts[s] for s in SPLITS}
    print("V5病例划分：", actual)
    if actual != EXPECTED_CASES:
        fail(f"病例划分数量不符合预期。预期={EXPECTED_CASES}，实际={actual}")
    return split_map


def collect_cts(split_map):
    """只收集V5划分中的病例；缺少RAW时从原始HF仓库补齐并校验。"""
    mhd_by_uid = {}
    for mhd in sorted(RAW_ROOT.glob("subset*/*.mhd")):
        uid = mhd.stem
        if uid in mhd_by_uid:
            fail(f"发现重复病例UID：{uid}\n{mhd_by_uid[uid]}\n{mhd}")
        mhd_by_uid[uid] = mhd

    missing_mhd = sorted(set(split_map) - set(mhd_by_uid))
    if missing_mhd:
        fail(
            "V5病例划分中的MHD文件缺失，无法构建完整118例数据集：\n"
            + "\n".join(missing_mhd[:30])
        )

    api = None
    subset_listing_cache = {}
    cases = {}

    def get_remote_size(subset, remote_path):
        nonlocal api
        if api is None:
            api = HfApi()
        if subset not in subset_listing_cache:
            print(f"正在查询远程文件信息：{subset}")
            entries = list(api.list_repo_tree(
                repo_id=HF_REPO,
                repo_type="dataset",
                path_in_repo=subset,
                recursive=False,
                expand=True,
            ))
            subset_listing_cache[subset] = {
                item.path.replace("\\", "/"): item
                for item in entries
            }
        item = subset_listing_cache[subset].get(remote_path)
        if item is None:
            return None
        return getattr(item, "size", None)

    for index, uid in enumerate(sorted(split_map), 1):
        mhd = mhd_by_uid[uid]
        subset = mhd.parent.name
        referenced = read_element_data_file(mhd)
        if not referenced or referenced.strip().upper() == "LOCAL":
            fail(f"MHD中没有有效的ElementDataFile：{mhd}")

        ref_norm = referenced.strip().replace("\\", "/")
        ref_path = Path(ref_norm)
        if ref_path.is_absolute() or ".." in ref_path.parts:
            fail(f"MHD引用了不安全或异常的RAW路径：{mhd}\n引用={referenced}")

        # MetaImage通常只写RAW文件名；若写了subset路径，也兼容处理。
        if ref_norm.lower().startswith((subset + "/").lower()):
            target_raw = RAW_ROOT / Path(ref_norm)
            remote_path = ref_norm
        else:
            target_raw = mhd.parent / Path(ref_norm)
            remote_path = f"{subset}/{ref_norm}"

        expected_size = None
        local_ok = target_raw.is_file() and target_raw.stat().st_size > 0

        # 若存在零字节残留文件，先改名保留，避免下载器误判为已有完整文件。
        if target_raw.exists() and not local_ok:
            backup = target_raw.with_name(
                target_raw.name + f".empty_backup_{int(time.time())}"
            )
            target_raw.rename(backup)
            print("发现空RAW残留，已保留备份：", backup)

        if local_ok:
            # 本地已有非空RAW时先不访问网络；SimpleITK后续还会实际解码CT。
            pass
        else:
            print(f"\n[{index}/{len(split_map)}] 发现RAW缺失：{uid[-12:]}")
            expected_size = get_remote_size(subset, remote_path)
            if expected_size is None or expected_size <= 0:
                fail(
                    f"远程仓库中找不到有效RAW文件或无法取得文件大小：{remote_path}\n"
                    "请检查仓库文件清单和网络连接。"
                )

            target_raw.parent.mkdir(parents=True, exist_ok=True)
            print("远程文件：", remote_path)
            print("预计大小：", round(expected_size / 1024**2, 2), "MB")
            repaired = False
            last_error = None

            for attempt in range(1, 4):
                try:
                    print(f"下载/恢复尝试 {attempt}/3 ...")
                    downloaded = Path(hf_hub_download(
                        repo_id=HF_REPO,
                        repo_type="dataset",
                        filename=remote_path,
                        local_dir=str(RAW_ROOT),
                        force_download=(attempt > 1),
                    ))

                    if downloaded.resolve() != target_raw.resolve():
                        shutil.copy2(downloaded, target_raw)

                    if not target_raw.is_file():
                        raise RuntimeError("下载后目标RAW文件仍不存在")
                    actual_size = target_raw.stat().st_size
                    if actual_size != expected_size:
                        raise RuntimeError(
                            f"RAW大小校验失败：实际{actual_size}字节，"
                            f"远程{expected_size}字节"
                        )

                    repaired = True
                    print("RAW补齐并通过大小校验：", target_raw)
                    break
                except Exception as exc:
                    last_error = exc
                    print("本次尝试失败：", repr(exc))
                    if attempt < 3:
                        time.sleep(3)

            if not repaired:
                fail(
                    f"RAW自动补齐失败：{target_raw}\n"
                    f"最后错误：{last_error}\n"
                    "请先修复该病例的RAW文件，再重新运行构建脚本。"
                )

        # 若MHD引用的是不同文件名，尊重MHD的真实引用，不强行改名。
        if not target_raw.is_file() or target_raw.stat().st_size <= 0:
            fail(f"RAW文件缺失或为空：{target_raw}")

        print(f"CT文件对检查 {index}/{len(split_map)}：{uid[-12:]} OK")
        cases[uid] = mhd

    return cases

def get_cross_section(nodule, zi, sz, sx, sy, width, height):
    """用球形近似计算当前横断面YOLO框，返回(label, min_side_px)。"""
    radius = nodule["diameter"] / 2.0
    dz_mm = abs(zi - nodule["z"]) * sz
    if dz_mm > radius:
        return None

    cross_radius = math.sqrt(max(0.0, radius * radius - dz_mm * dz_mm))
    if cross_radius * 2.0 < MIN_CROSS_SECTION_DIAMETER_MM:
        return None

    rx = cross_radius / sx
    ry = cross_radius / sy

    x1 = max(0.0, nodule["x"] - rx)
    y1 = max(0.0, nodule["y"] - ry)
    x2 = min(float(width), nodule["x"] + rx)
    y2 = min(float(height), nodule["y"] + ry)

    if x2 <= x1 or y2 <= y1:
        return None

    box_w = x2 - x1
    box_h = y2 - y1
    cx = (x1 + x2) / (2.0 * width)
    cy = (y1 + y2) / (2.0 * height)
    bw = box_w / width
    bh = box_h / height

    if not (
        0 <= cx <= 1 and 0 <= cy <= 1
        and 0 < bw <= 1 and 0 < bh <= 1
    ):
        return None

    return (
        f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}",
        min(box_w, box_h),
    )


def is_safe_negative(zi, nodules, sz):
    for nodule in nodules:
        dz_mm = abs(zi - nodule["z"]) * sz
        margin = nodule["diameter"] / 2.0 + sz / 2.0
        if dz_mm <= margin:
            return False
    return True


def main():
    start_time = time.time()

    print("========== LungLens V5 数据构建 ==========")
    print("CT目录：", RAW_ROOT)
    print("参考标注：", ANN_FILE)
    print("V5病例划分：", SPLIT_FILE)
    print("输出目录：", OUTPUT)

    if OUTPUT.exists():
        fail(f"输出目录已存在，为防止覆盖停止：{OUTPUT}")
    if STAGING.exists():
        fail(
            f"临时构建目录已存在：{STAGING}\n"
            "这可能是上次中断留下的文件，请先检查，不要直接覆盖。"
        )

    annotations = load_annotations()
    split_map = load_split()
    cases = collect_cts(split_map)

    print("\n========== 输入数据核对 ==========")
    print("本地完整CT文件对：", len(cases))
    print("V5划分病例数：", len(split_map))
    print("本地CT对应参考结节：", sum(len(annotations.get(u, [])) for u in cases))

    missing = sorted(set(split_map) - set(cases))
    extra = sorted(set(cases) - set(split_map))
    if missing:
        fail("划分表中的CT本地缺失：\n" + "\n".join(missing[:20]))
    if extra:
        fail("本地存在未纳入V5划分的CT：\n" + "\n".join(extra[:20]))
    if len(cases) != 118:
        fail(f"当前完整CT为{len(cases)}例，不是预期的118例。")

    # 先写入临时目录；全部通过后才改名为正式输出目录。
    for split in SPLITS:
        for kind in ("images", "labels"):
            (STAGING / kind / split).mkdir(parents=True, exist_ok=True)

    stats = Counter()
    manifest = []
    nodule_audit = []
    case_report = []
    small_boxes = []

    for case_no, (uid, mhd_path) in enumerate(sorted(cases.items()), 1):
        split = split_map[uid]
        rows = annotations.get(uid, [])

        print(
            f"\n[{case_no}/{len(cases)}] {uid[-12:]} | {split} | "
            f"参考结节={len(rows)}"
        )

        image = sitk.ReadImage(str(mhd_path))
        width, height, depth = image.GetSize()
        sx, sy, sz = image.GetSpacing()

        if image.GetDimension() != 3:
            fail(f"不是三维CT：{uid}")
        if min(width, height, depth) <= 0 or min(sx, sy, sz) <= 0:
            fail(f"CT尺寸或体素间距异常：{uid}")

        volume = sitk.GetArrayFromImage(image)
        if volume.shape != (depth, height, width):
            fail(f"CT数组维度异常：{uid}，shape={volume.shape}")

        # 将物理坐标转换为连续体素坐标。
        nodules = []
        for ann_idx, row in enumerate(rows):
            try:
                world = (
                    float(row["coordX"]),
                    float(row["coordY"]),
                    float(row["coordZ"]),
                )
                diameter = float(row["diameter_mm"])
            except (TypeError, ValueError) as exc:
                fail(f"标注数值无法解析：{uid}，第{ann_idx}条：{exc}")

            if not math.isfinite(diameter) or diameter <= 0:
                fail(f"结节直径非法：{uid}，第{ann_idx}条")

            x, y, z = image.TransformPhysicalPointToContinuousIndex(world)
            if not all(math.isfinite(v) for v in (x, y, z)):
                fail(f"坐标转换出现非有限值：{uid}，第{ann_idx}条")

            if not (
                -0.5 <= x <= width - 0.5
                and -0.5 <= y <= height - 0.5
                and -0.5 <= z <= depth - 0.5
            ):
                fail(
                    f"结节中心超出CT范围：{uid}，第{ann_idx}条，"
                    f"index={(x, y, z)}，size={(width, height, depth)}"
                )

            nodules.append({
                "id": ann_idx,
                "x": x, "y": y, "z": z,
                "diameter": diameter,
            })

        # 生成所有满足截面直径阈值的正样本切片标签。
        slice_labels = defaultdict(list)
        nodule_sizes = defaultdict(list)

        for zi in range(depth):
            for nodule in nodules:
                result = get_cross_section(
                    nodule, zi, sz, sx, sy, width, height
                )
                if result is None:
                    continue
                label, min_side = result
                slice_labels[zi].append((nodule["id"], label, min_side))
                nodule_sizes[nodule["id"]].append(min_side)

        positive_slices = set(slice_labels)
        included_nodules = set(nodule_sizes)

        # 安全负样本：避开所有参考结节的Z轴范围。
        z_start = max(0, int(depth * CENTRAL_Z_START))
        z_end = min(depth, int(depth * CENTRAL_Z_END))
        negative_candidates = []

        for zi in range(z_start, z_end):
            if zi not in positive_slices and is_safe_negative(zi, nodules, sz):
                negative_candidates.append(zi)

        negative_target = (
            NEGATIVE_PER_POSITIVE_CASE if nodules
            else NEGATIVE_PER_NEGATIVE_CASE
        )
        negative_slices = select_evenly(negative_candidates, negative_target)
        selected_slices = sorted(positive_slices | set(negative_slices))

        if not selected_slices:
            fail(f"病例没有生成任何切片：{uid}")

        pos_images = 0
        neg_images = 0
        label_count = 0

        for zi in selected_slices:
            stem = f"{uid}_z{zi:03d}"
            filename = stem + ".png"

            pixels = lung_window(volume[zi])
            Image.fromarray(pixels).save(
                STAGING / "images" / split / filename
            )

            entries = slice_labels.get(zi, [])
            lines = [entry[1] for entry in entries]
            (STAGING / "labels" / split / (stem + ".txt")).write_text(
                "\n".join(lines) + ("\n" if lines else ""),
                encoding="utf-8",
            )

            if lines:
                sample_type = "positive"
                pos_images += 1
                label_count += len(lines)
            else:
                sample_type = "negative"
                neg_images += 1

            manifest.append([
                uid, split, zi, sample_type, len(lines), filename
            ])

            # 小于3像素的框只记录，不自动删除整张正样本图。
            for nodule_id, label, min_side in entries:
                if min_side < 3.0:
                    small_boxes.append([
                        uid, split, zi, nodule_id,
                        round(min_side, 3), label
                    ])

        missing_nodules = len(nodules) - len(included_nodules)

        for nodule in nodules:
            sizes = nodule_sizes.get(nodule["id"], [])
            nodule_audit.append([
                uid, split, nodule["id"],
                round(nodule["diameter"], 4),
                round(nodule["x"], 3),
                round(nodule["y"], 3),
                round(nodule["z"], 3),
                int(bool(sizes)),
                len(sizes),
                round(min(sizes), 3) if sizes else "",
                round(max(sizes), 3) if sizes else "",
            ])

        stats[f"{split}_cases"] += 1
        stats[f"{split}_positive"] += pos_images
        stats[f"{split}_negative"] += neg_images
        stats[f"{split}_boxes"] += label_count
        stats["reference_nodules"] += len(nodules)
        stats["included_nodules"] += len(included_nodules)
        stats["missing_nodules"] += missing_nodules

        case_report.append([
            uid, split, len(nodules), len(included_nodules),
            missing_nodules, pos_images, neg_images,
            label_count, len(negative_candidates),
        ])

        print(
            f"  正样本={pos_images} | 负样本={neg_images} | "
            f"标签框={label_count} | "
            f"已生成标签结节={len(included_nodules)}/{len(nodules)}"
        )

    # 保存审计文件。
    write_csv(
        STAGING / "case_split.csv",
        ["seriesuid", "split"],
        [[uid, split] for uid, split in sorted(split_map.items())],
    )
    write_csv(
        STAGING / "sample_manifest.csv",
        ["seriesuid", "split", "z_index", "sample_type", "label_count", "filename"],
        manifest,
    )
    write_csv(
        STAGING / "nodule_audit.csv",
        [
            "seriesuid", "split", "annotation_index", "diameter_mm",
            "index_x", "index_y", "index_z", "label_generated",
            "positive_slice_count", "min_box_side_px", "max_box_side_px",
        ],
        nodule_audit,
    )
    write_csv(
        STAGING / "case_report.csv",
        [
            "seriesuid", "split", "reference_nodules", "included_nodules",
            "missing_nodules", "positive_images", "negative_images",
            "label_count", "safe_negative_candidates",
        ],
        case_report,
    )
    write_csv(
        STAGING / "small_boxes_review.csv",
        [
            "seriesuid", "split", "z_index", "annotation_index",
            "min_box_side_px", "yolo_label",
        ],
        small_boxes,
    )

    yaml = (
        f'path: "{OUTPUT.as_posix()}"\n'
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "nc: 1\n"
        "names:\n"
        "  0: lung_nodule\n"
    )
    (STAGING / "dataset.yaml").write_text(yaml, encoding="utf-8")

    # 图片与标签一一对应、标签坐标合法性检查。
    print("\n========== 构建后完整性审计 ==========")
    for split in SPLITS:
        image_dir = STAGING / "images" / split
        label_dir = STAGING / "labels" / split
        image_files = sorted(image_dir.glob("*.png"))
        label_files = sorted(label_dir.glob("*.txt"))

        if {p.stem for p in image_files} != {p.stem for p in label_files}:
            fail(f"{split}图片与标签文件不一一对应。")

        for image_path in image_files:
            with Image.open(image_path) as im:
                if im.mode != "L" or min(im.size) <= 0:
                    fail(f"图片模式或尺寸异常：{image_path}")

            label_path = label_dir / (image_path.stem + ".txt")
            for line_no, line in enumerate(
                label_path.read_text(encoding="utf-8").splitlines(), 1
            ):
                parts = line.split()
                if len(parts) != 5:
                    fail(f"标签格式错误：{label_path} 第{line_no}行")
                cls, cx, cy, bw, bh = map(float, parts)
                if cls != 0 or not (
                    0 <= cx <= 1 and 0 <= cy <= 1
                    and 0 < bw <= 1 and 0 < bh <= 1
                    and cx - bw / 2 >= -1e-5
                    and cy - bh / 2 >= -1e-5
                    and cx + bw / 2 <= 1.00001
                    and cy + bh / 2 <= 1.00001
                ):
                    fail(f"YOLO标签越界或类别错误：{label_path} 第{line_no}行")

        print(
            f"{split}: 图片={len(image_files)}，"
            f"标签文件={len(label_files)}，检查通过"
        )

    # 病例级数据泄漏检查。
    case_splits_seen = defaultdict(set)
    for row in manifest:
        case_splits_seen[row[0]].add(row[1])
    leakage = {
        uid: values for uid, values in case_splits_seen.items()
        if len(values) != 1
    }
    if leakage:
        fail(f"发现病例跨集合泄漏：{list(leakage.items())[:10]}")

    elapsed = time.time() - start_time
    total_pos = sum(stats[f"{s}_positive"] for s in SPLITS)
    total_neg = sum(stats[f"{s}_negative"] for s in SPLITS)
    total_boxes = sum(stats[f"{s}_boxes"] for s in SPLITS)

    summary = [
        "LungLens V5 dataset build summary",
        f"CT cases: {len(cases)}",
        f"Reference nodules: {stats['reference_nodules']}",
        f"Generated nodules: {stats['included_nodules']}",
        f"Not generated nodules: {stats['missing_nodules']}",
        f"Positive images: {total_pos}",
        f"Negative images: {total_neg}",
        f"Total images: {total_pos + total_neg}",
        f"YOLO boxes: {total_boxes}",
        f"Small box records (<3 px): {len(small_boxes)}",
        f"Elapsed seconds: {elapsed:.1f}",
    ]
    for split in SPLITS:
        summary.append(
            f"{split}: cases={stats[f'{split}_cases']}, "
            f"positive={stats[f'{split}_positive']}, "
            f"negative={stats[f'{split}_negative']}, "
            f"boxes={stats[f'{split}_boxes']}"
        )
    (STAGING / "build_summary.txt").write_text(
        "\n".join(summary) + "\n", encoding="utf-8"
    )

    print("\n========== LungLens V5 构建汇总 ==========")
    print("CT病例：", len(cases))
    print("参考结节：", stats["reference_nodules"])
    print("已生成标签的独立结节：", stats["included_nodules"])
    print("未生成标签的独立结节：", stats["missing_nodules"])
    print("正样本图片：", total_pos)
    print("负样本图片：", total_neg)
    print("图片总数：", total_pos + total_neg)
    print("YOLO标签框：", total_boxes)
    print("小于3px的框记录：", len(small_boxes))
    print("耗时：", round(elapsed, 1), "秒")

    for split in SPLITS:
        print(
            f"{split.upper()}: {stats[f'{split}_cases']}例 | "
            f"正样本{stats[f'{split}_positive']} | "
            f"负样本{stats[f'{split}_negative']} | "
            f"标签框{stats[f'{split}_boxes']}"
        )

    # 只有所有构建和审计均通过，才生成正式目录。
    STAGING.rename(OUTPUT)

    print("\n========== V5数据集构建成功 ==========")
    print("数据集目录：", OUTPUT)
    print("训练配置：", OUTPUT / "dataset.yaml")
    print("病例划分：", OUTPUT / "case_split.csv")
    print("样本清单：", OUTPUT / "sample_manifest.csv")
    print("结节审计：", OUTPUT / "nodule_audit.csv")
    print("病例报告：", OUTPUT / "case_report.csv")
    print("小框复核：", OUTPUT / "small_boxes_review.csv")
    print("汇总报告：", OUTPUT / "build_summary.txt")
    print("\n原始CT、V4、V4.1和原始V5划分文件均未修改。")
    print("注意：球形截面标签是弱标注；无参考标注不等于临床确认阴性。")


if __name__ == "__main__":
    main()
