
from pathlib import Path
import torch
from ultralytics import YOLO

PROJECT = Path(r"D:\肺结节模型训练")
DATA = Path(
    r"E:\LungLens_Data\processed\yolo_v2_filtered"
)
YAML = DATA / "dataset.yaml"

def main():
    # 1. 检查数据集
    if not YAML.exists():
        raise FileNotFoundError(YAML)

    train_images = sorted((DATA / "images/train").glob("*.png"))
    val_images = sorted((DATA / "images/val").glob("*.png"))

    def count_labels(split, images):
        total = 0
        for image in images:
            label = DATA / "labels" / split / (
                image.stem + ".txt"
            )
            if not label.exists():
                raise FileNotFoundError(label)
            total += len(label.read_text().splitlines())
        return total

    train_uids = {
        p.stem.rsplit("_z", 1)[0] for p in train_images
    }
    val_uids = {
        p.stem.rsplit("_z", 1)[0] for p in val_images
    }

    if train_uids & val_uids:
        raise RuntimeError("训练和验证存在相同CT病例！")

    print("训练图片:", len(train_images))
    print("验证图片:", len(val_images))
    print("训练标签:", count_labels("train", train_images))
    print("验证标签:", count_labels("val", val_images))
    print("训练/验证病例交集: 0")

    # 2. 检查 GPU
    assert torch.cuda.is_available(), "CUDA不可用"
    print("训练GPU:", torch.cuda.get_device_name(0))

    # 3. 加载预训练模型
    model = YOLO(str(PROJECT / "yolov8n.pt"))

    # 4. 第二次训练
    model.train(
        data=str(YAML),
        epochs=30,
        imgsz=512,
        batch=2,
        device=0,
        workers=0,
        amp=True,
        cache=False,

        optimizer="AdamW",
        lr0=0.001,

        # 调整数据增强
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.0,
        mosaic=0.0,
        mixup=0.0,
        scale=0.1,
        translate=0.05,
        fliplr=0.5,

        seed=42,
        patience=15,
        val=True,
        plots=True,

        project=str(PROJECT / "runs"),
        name="yolo_v2_exp01",
        exist_ok=False
    )

    print("\n第二次训练结束！")

if __name__ == "__main__":
    main()
