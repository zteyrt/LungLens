from pathlib import Path
from ultralytics import YOLO
PROJECT = Path(r"D:\肺结节模型训练")
DATA = Path(r"E:\LungLens_Data\processed\yolo_v1")

# 加载刚刚训练完成的模型
model_path = PROJECT / "runs" / "smoke_test" / "weights" / "best.pt"

if not model_path.exists():
    raise FileNotFoundError(model_path)

model = YOLO(str(model_path))

# 验证集图片目录
val_dir = DATA / "images" / "val"

# 将置信度阈值降低到 0.01，用于分析弱预测
results = model.predict(
    source=str(val_dir),
    imgsz=512,
    conf=0.01,
    iou=0.5,
    device=0,
    save=True,
    project=str(PROJECT / "runs"),
    name="prediction_check",
    exist_ok=True,
    verbose=False
)

print("\n========== 预测检查 ==========")

for result in results:
    print("\n图片:", Path(result.path).name)

    boxes = result.boxes

    if boxes is None or len(boxes) == 0:
        print("没有检测到候选框")
        continue

    print("候选框数量:", len(boxes))

    for box in boxes:
        xyxy = box.xyxy[0].cpu().numpy()
        conf = float(box.conf[0])

        print(
            "位置:",
            [round(float(v), 2) for v in xyxy],
            "置信度:",
            round(conf, 4)
        )

print("\n预测检查完成")
