
from pathlib import Path
import shutil

from ultralytics import YOLO

PROJECT = Path(r"D:\肺结节模型训练")
SOURCE = Path(
    r"E:\LungLens_Data\processed\yolo_v3"
)
TEST = Path(
    r"E:\LungLens_Data\processed\overfit_debug_v3"
)

def main():
    # 1. 选择2张已标注的训练图片
    images = sorted(
        (SOURCE / "images" / "train").glob("*.png")
    )

    if len(images) < 2:
        raise RuntimeError("训练图片不足2张")

    selected = images[:2]

    # 2. 创建独立的调试数据集
    for split in ("train", "val"):
        image_dir = TEST / "images" / split
        label_dir = TEST / "labels" / split

        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)

        for image in selected:
            label = SOURCE / "labels" / "train" / (
                image.stem + ".txt"
            )

            if not label.exists():
                raise FileNotFoundError(label)

            # 保留原图和原标签，不修改原始数据
            shutil.copy2(image, image_dir / image.name)
            shutil.copy2(label, label_dir / label.name)

    # 3. 创建YOLO数据配置
    yaml_path = TEST / "dataset.yaml"

    yaml_path.write_text(
        f"path: '{TEST.as_posix()}'\n"
        "train: images/train\n"
        "val: images/val\n"
        "nc: 1\n"
        "names: ['lung_nodule']\n",
        encoding="utf-8"
    )

    print("========== 小样本过拟合测试 ==========")
    print("训练图片:", len(selected))
    print("验证图片:", len(selected))
    print("数据集:", TEST)

    # 4. 加载YOLOv8n通用预训练模型
    model = YOLO(str(PROJECT / "yolov8n.pt"))

    # 5. 启动调试训练
    model.train(
        data=str(yaml_path),
        epochs=200,
        patience=0,
        imgsz=512,
        batch=2,
        nbs=2,
        device=0,
        workers=0,
        amp=True,
        cache=False,

        optimizer="AdamW",
        lr0=0.001,
        warmup_epochs=0,

        # 关闭大部分随机数据增强
        mosaic=0.0,
        mixup=0.0,
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.0,
        scale=0.0,
        translate=0.0,
        degrees=0.0,
        shear=0.0,
        perspective=0.0,
        fliplr=0.0,
        flipud=0.0,

        seed=42,
        plots=True,
        project=str(PROJECT / "runs"),
        name="overfit_debug_v3",
        exist_ok=False
    )

    print("\n小样本过拟合测试完成")
    print("注意：训练集和验证集相同，仅用于调试。")


if __name__ == "__main__":
    main()
