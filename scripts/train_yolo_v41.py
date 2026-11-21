
from pathlib import Path
from collections import Counter
import csv
import hashlib
import math

from PIL import Image
from ultralytics import YOLO


PROJECT = Path(r"D:\肺结节模型训练")
ROOT = Path(r"E:\LungLens_Data\processed")

DATA = ROOT / "yolo_v41"
V4 = ROOT / "yolo_v4"
WEIGHTS = PROJECT / "yolov8n.pt"

SPLITS = ("train", "val", "test")


def file_hash(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(
            lambda: f.read(1024 * 1024), b""
        ):
            h.update(block)
    return h.hexdigest()


def load_splits():
    with (DATA / "case_split.csv").open(
        "r", encoding="utf-8-sig", newline=""
    ) as f:
        return {
            r["seriesuid"]: r["split"]
            for r in csv.DictReader(f)
        }


def audit():
    print("========== V4.1 训练前检查 ==========")

    if not WEIGHTS.is_file():
        raise FileNotFoundError(WEIGHTS)

    if not (DATA / "dataset.yaml").is_file():
        raise FileNotFoundError(DATA / "dataset.yaml")

    split_map = load_splits()
    counts = Counter()

    # 比较V4与V4.1病例划分
    if file_hash(DATA / "case_split.csv") != file_hash(
        V4 / "case_split.csv"
    ):
        raise RuntimeError("病例划分与V4不一致")

    for split in SPLITS:
        images = sorted(
            (DATA / "images" / split).glob("*.png")
        )
        labels = sorted(
            (DATA / "labels" / split).glob("*.txt")
        )

        if {p.stem for p in images} != {
            p.stem for p in labels
        }:
            raise RuntimeError(
                f"{split} 图片与标签不匹配"
            )

        cases = set()

        for image_path in images:
            uid = image_path.stem.rsplit("_z", 1)[0]

            if split_map.get(uid) != split:
                raise RuntimeError(
                    f"病例划分错误: {image_path.name}"
                )

            cases.add(uid)

            with Image.open(image_path) as im:
                width, height = im.size

            label = (
                DATA / "labels" / split /
                (image_path.stem + ".txt")
            )

            lines = label.read_text(
                encoding="utf-8-sig"
            ).splitlines()

            for line in lines:
                parts = line.split()

                if len(parts) != 5:
                    raise ValueError(
                        f"标签格式错误: {label}"
                    )

                cls, cx, cy, bw, bh = map(
                    float, parts
                )

                if not all(map(
                    math.isfinite,
                    (cls, cx, cy, bw, bh)
                )):
                    raise ValueError(
                        f"非法数值: {label}"
                    )

                if cls != 0 or not (
                    0 <= cx <= 1
                    and 0 <= cy <= 1
                    and bw > 0
                    and bh > 0
                    and cx - bw / 2 >= -1e-5
                    and cy - bh / 2 >= -1e-5
                    and cx + bw / 2 <= 1.00001
                    and cy + bh / 2 <= 1.00001
                ):
                    raise ValueError(
                        f"坐标异常: {label}"
                    )

                counts[f"{split}_boxes"] += 1

                if split != "test":
                    if min(
                        bw * width, bh * height
                    ) < 3.0 - 0.001:
                        raise RuntimeError(
                            f"仍有小于3px框: {label}"
                        )

            counts[f"{split}_images"] += 1
            counts[
                f"{split}_positive"
                if lines else f"{split}_negative"
            ] += 1

        expected_cases = {
            uid for uid, s in split_map.items()
            if s == split
        }

        if cases != expected_cases:
            raise RuntimeError(
                f"{split} 病例覆盖不完整"
            )

        print(
            f"{split.upper()}: "
            f"{len(cases)}例, "
            f"{counts[f'{split}_images']}张图, "
            f"{counts[f'{split}_positive']}张阳性, "
            f"{counts[f'{split}_negative']}张无标签, "
            f"{counts[f'{split}_boxes']}个框"
        )

    # 独立测试集必须与V4完全一致
    for kind, suffix in (
        ("images", "*.png"),
        ("labels", "*.txt")
    ):
        a = DATA / kind / "test"
        b = V4 / kind / "test"

        files_a = sorted(p.name for p in a.glob(suffix))
        files_b = sorted(p.name for p in b.glob(suffix))

        if files_a != files_b:
            raise RuntimeError(
                f"TEST {kind} 文件列表改变"
            )

        for name in files_a:
            if file_hash(a / name) != file_hash(
                b / name
            ):
                raise RuntimeError(
                    f"TEST文件被修改: {name}"
                )

    print("独立TEST与V4逐文件一致")
    print("训练前检查通过\n")


def main():
    audit()

    print("========== YOLO V4.1 训练 ==========")

    model = YOLO(str(WEIGHTS))

    model.train(
        data=str(DATA / "dataset.yaml"),
        epochs=80,
        patience=20,
        imgsz=640,
        batch=4,
        nbs=4,
        device=0,
        workers=0,
        amp=True,
        cache=False,
        optimizer="AdamW",
        lr0=0.0005,
        lrf=0.1,
        warmup_epochs=3,
        hsv_h=0,
        hsv_s=0,
        hsv_v=0,
        mosaic=0,
        mixup=0,
        scale=0.05,
        translate=0.02,
        degrees=0,
        shear=0,
        perspective=0,
        fliplr=0.5,
        flipud=0,
        seed=42,
        deterministic=True,
        plots=True,
        save=True,
        project=str(PROJECT / "runs"),
        name="yolo_v41_baseline",
        exist_ok=False
    )

    print("\nV4.1训练完成")
    print("请查看 runs 下新建的 yolo_v41_baseline 实验目录")


if __name__ == "__main__":
    main()
