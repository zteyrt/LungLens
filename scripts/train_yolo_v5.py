"""LungLens YOLO V5: fine-tune V4.1 checkpoint on the 118-CT V5 dataset.

Run from any cwd:
    python scripts/train_yolo_v5.py

Only trains on V5 train split, validates on V5 val split. Does not touch test.
All old experiments are protected; this script refuses an existing run directory.
"""
from __future__ import annotations

import csv
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = Path(r"E:\LungLens_Data\processed\yolo_v5_dataset")
CANONICAL_SPLIT = Path(r"E:\LungLens_Data\processed\yolo_v5\case_split.csv")
DATA_YAML = DATA_ROOT / "dataset.yaml"
PRETRAINED = PROJECT_ROOT / "runs" / "yolo_v41_baseline" / "weights" / "best.pt"
RUN_PROJECT = PROJECT_ROOT / "runs"
RUN_NAME = "yolo_v5_finetune_v1"
RUN_DIR = RUN_PROJECT / RUN_NAME

SEED = 42
IMGSZ = 512
EPOCHS = 60
BATCH = 8
PATIENCE = 15
WORKERS = 0  # Windows: safer; avoid DataLoader multiprocessing issues.


def read_splits(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise FileNotFoundError(path)
    mapping: dict[str, str] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or not {"seriesuid", "split"}.issubset(reader.fieldnames):
            raise RuntimeError(f"病例划分缺少必要字段: {path}")
        for row in reader:
            uid = row["seriesuid"].strip()
            split = row["split"].strip().lower()
            if not uid or split not in ("train", "val", "test") or uid in mapping:
                raise RuntimeError(f"病例划分错误或重复: {path}: {row}")
            mapping[uid] = split
    return mapping


def check_data() -> dict:
    if RUN_DIR.exists():
        raise RuntimeError(f"实验目录已存在，不覆盖旧实验: {RUN_DIR}\n若是新实验，请修改 RUN_NAME。")
    for p in (DATA_YAML, PRETRAINED, DATA_ROOT / "case_split.csv", CANONICAL_SPLIT):
        if not p.is_file():
            raise FileNotFoundError(p)

    current = read_splits(DATA_ROOT / "case_split.csv")
    official = read_splits(CANONICAL_SPLIT)
    if current != official:
        raise RuntimeError("V5数据集病例划分与原始V5病例划分不一致。")
    counts = Counter(current.values())
    if any(counts[k] != n for k, n in {"train": 100, "val": 11, "test": 7}.items()):
        raise RuntimeError(f"病例数量异常: {counts}")

    # Ensure no accidental path misdirection in dataset.yaml.
    yaml_content = DATA_YAML.read_text(encoding="utf-8-sig")
    normalized = yaml_content.replace("\\", "/").replace("'", "").replace('"', "")
    if DATA_ROOT.as_posix().lower() not in normalized.lower():
        raise RuntimeError(f"dataset.yaml 似乎指向了其他数据集: {DATA_YAML}")

    image_counts = {}
    box_counts = {}
    for split in ("train", "val", "test"):
        images = DATA_ROOT / "images" / split
        labels = DATA_ROOT / "labels" / split
        if not images.is_dir() or not labels.is_dir():
            raise FileNotFoundError(f"图像或标签目录缺失: {split}")
        image_files = sorted(images.glob("*.png"))
        label_files = sorted(labels.glob("*.txt"))
        if {p.stem for p in image_files} != {p.stem for p in label_files}:
            raise RuntimeError(f"{split} 图像标签不一一对应。")
        if not image_files:
            raise RuntimeError(f"{split} 无数据。")

        n_boxes = 0
        for img in image_files:
            if "_z" not in img.stem:
                raise RuntimeError(f"文件名缺少切片索引: {img}")
            uid = img.stem.rsplit("_z", 1)[0]
            if current.get(uid) != split:
                raise RuntimeError(f"跨病例划分泄漏: {img.name} -> {split}")
            label_path = labels / (img.stem + ".txt")
            lines = label_path.read_text(encoding="utf-8-sig").splitlines()
            for line in lines:
                vals = line.split()
                if len(vals) != 5:
                    raise RuntimeError(f"非法标签: {label_path}")
                cls, x, y, w, h = [float(v) for v in vals]
                if not (cls == 0 and 0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
                    raise RuntimeError(f"标签数值非法: {label_path}")
            n_boxes += len(lines)

        image_counts[split] = len(image_files)
        box_counts[split] = n_boxes

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用：当前不启动GPU训练。请先检查PyTorch环境。")

    return {
        "cases": dict(counts),
        "images": image_counts,
        "boxes": box_counts,
        "gpu": torch.cuda.get_device_name(0),
        "source_weights": str(PRETRAINED),
        "dataset": str(DATA_YAML),
    }


def main() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    overview = check_data()
    print("========== LungLens YOLO V5 微调 ==========")
    print("GPU:", overview["gpu"])
    print("病例:", overview["cases"])
    print("图片:", overview["images"])
    print("标签框:", overview["boxes"])
    print("初始权重:", PRETRAINED)
    print("数据集:", DATA_YAML)
    print("输出目录:", RUN_DIR)
    print("说明: 仅训练train，使用val早停，test保持未触碰。")

    RUN_PROJECT.mkdir(parents=True, exist_ok=True)
    # Do not pre-create RUN_DIR; Ultralytics owns its experiment directory.
    model = YOLO(str(PRETRAINED))
    model.train(
        data=str(DATA_YAML),
        project=str(RUN_PROJECT),
        name=RUN_NAME,
        exist_ok=False,
        epochs=EPOCHS,
        patience=PATIENCE,
        imgsz=IMGSZ,
        batch=BATCH,
        device=0,
        workers=WORKERS,
        seed=SEED,
        deterministic=True,
        optimizer="AdamW",
        lr0=0.0003,
        lrf=0.01,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        cos_lr=True,
        amp=True,
        # Medical boxes have physical meaning: avoid aggressive geometric transforms.
        mosaic=0.2,
        close_mosaic=10,
        mixup=0.0,
        degrees=0.0,
        translate=0.03,
        scale=0.10,
        shear=0.0,
        perspective=0.0,
        fliplr=0.5,
        flipud=0.0,
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.0,
        plots=True,
        save=True,
        verbose=True,
    )

    best = RUN_DIR / "weights" / "best.pt"
    if not best.is_file():
        raise RuntimeError(f"训练未生成best.pt，请检查训练日志: {RUN_DIR}")
    (RUN_DIR / "v5_experiment_info.json").write_text(
        json.dumps({
            **overview,
            "experiment": RUN_NAME,
            "seed": SEED,
            "epochs_max": EPOCHS,
            "batch": BATCH,
            "image_size": IMGSZ,
            "note": "2D slice validation mAP is not lesion-level full-CT FROC."
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("\n========== V5 微调完成 ==========")
    print("最佳权重:", best)
    print("训练记录:", RUN_DIR / "results.csv")
    print("后续必须用同一11例Val整例CT进行FROC对照。")
    print("独立Test尚未用于调参。")

if __name__ == "__main__":
    main()
