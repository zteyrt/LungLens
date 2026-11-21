from pathlib import Path
from ultralytics import YOLO

root = Path(r"D:\肺结节模型训练")
data = r"E:\LungLens_Data\processed\yolo_v5_dataset\dataset.yaml"

models = {
    "V41": root / "runs/yolo_v41_baseline/weights/best.pt",
    "V5": root / "runs/yolo_v5_finetune_v1/weights/best.pt",
}

for name, weights in models.items():
    print(f"\n========== {name} Batch=1 复评 ==========")

    if not weights.is_file():
        raise FileNotFoundError(weights)

    output_name = f"v41_v5_batch1_{name.lower()}"
    output_dir = root / "runs" / output_name

    if output_dir.exists():
        raise RuntimeError(
            f"结果目录已存在，为避免覆盖已停止: {output_dir}"
        )

    result = YOLO(str(weights)).val(
        data=data,
        split="val",
        imgsz=512,
        batch=1,
        device=0,
        workers=0,
        plots=False,
        project=str(root / "runs"),
        name=output_name,
        exist_ok=False,
        verbose=False,
    )

    print("Precision:", round(result.box.mp, 4))
    print("Recall:", round(result.box.mr, 4))
    print("mAP50:", round(result.box.map50, 4))
    print("mAP50-95:", round(result.box.map, 4))

print("\n同集复评完成")
