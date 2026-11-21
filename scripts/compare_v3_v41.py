from pathlib import Path
from ultralytics import YOLO

project = Path(r"D:\肺结节模型训练")
data = r"E:\LungLens_Data\processed\yolo_v3\dataset.yaml"

models = {
    "V3": project / "runs/yolo_v3_baseline/weights/best.pt",
    "V41": project / "runs/yolo_v41_baseline/weights/best.pt",
}

print("========== 统一验证集对比 ==========")

for name, weight in models.items():
    print(f"\n正在评估 {name}")

    model = YOLO(str(weight))

    metrics = model.val(
        data=data,
        split="val",
        imgsz=640,
        batch=4,
        device=0,
        workers=0,
        plots=False,
        verbose=False,
        project=str(project / "runs"),
        name=f"compare_{name.lower()}_on_v3val",
        exist_ok=True,
    )

    print(f"\n{name} 结果")
    print(f"Precision: {metrics.box.mp:.4f}")
    print(f"Recall:    {metrics.box.mr:.4f}")
    print(f"mAP50:     {metrics.box.map50:.4f}")
    print(f"mAP50-95:  {metrics.box.map:.4f}")

print("\n统一验证集对比完成")
