from pathlib import Path
from ultralytics import YOLO

root = Path(r"D:\肺结节模型训练")
model = YOLO(str(
    root / "runs/yolo_v41_baseline/weights/best.pt"
))
data = r"E:\LungLens_Data\processed\yolo_v41\dataset.yaml"

for threshold in (0.10, 0.15, 0.20):
    print(f"\n========== 阈值 {threshold:.2f} ==========")

    result = model.val(
        data=data,
        split="val",
        imgsz=640,
        batch=4,
        conf=threshold,
        iou=0.7,
        device=0,
        workers=0,
        plots=False,
        verbose=False,
        project=str(root / "runs"),
        name=f"v41_threshold_{int(threshold * 100):02d}",
        exist_ok=True
    )

    print(f"Precision: {result.box.mp:.4f}")
    print(f"Recall:    {result.box.mr:.4f}")
    print(f"mAP50:     {result.box.map50:.4f}")
    print(f"mAP50-95:  {result.box.map:.4f}")

print("\n全部评估完成")
