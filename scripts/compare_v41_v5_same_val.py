from ultralytics import YOLO

data = r"E:\LungLens_Data\processed\yolo_v5_dataset\dataset.yaml"

models = {
    "V41": r"D:\肺结节模型训练\runs\yolo_v41_baseline\weights\best.pt",
    "V5": r"D:\肺结节模型训练\runs\yolo_v5_finetune_v1\weights\best.pt",
}

for name, weight in models.items():
    print(f"\n========== {name} 同集评估 ==========")

    model = YOLO(weight)

    result = model.val(
        data=data,
        split="val",
        imgsz=512,
        batch=8,
        device=0,
        workers=0,
        plots=False,
        project=r"D:\肺结节模型训练\runs",
        name=f"same_val_compare_{name.lower()}",
        exist_ok=False
    )

    print("Precision:", round(result.box.mp, 4))
    print("Recall:", round(result.box.mr, 4))
    print("mAP50:", round(result.box.map50, 4))
    print("mAP50-95:", round(result.box.map, 4))
