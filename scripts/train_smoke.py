
from pathlib import Path
import torch
from ultralytics import YOLO

# 1. 项目目录
PROJECT_ROOT = Path(r"D:\肺结节模型训练")

# 2. YOLO数据集配置
DATA_YAML = Path(
    r"E:\LungLens_Data\processed\yolo_v1\dataset.yaml"
)

# 3. 检查数据和GPU
if not DATA_YAML.exists():
    raise FileNotFoundError(DATA_YAML)

assert torch.cuda.is_available(), "没有检测到CUDA GPU"

print("GPU:", torch.cuda.get_device_name(0))
print("训练数据:", DATA_YAML)

# 4. 加载YOLOv8n预训练权重
model = YOLO(str(PROJECT_ROOT / "yolov8n.pt"))

# 5. 开始训练
results = model.train(
    data=str(DATA_YAML),
    epochs=3,
    imgsz=512,
    batch=2,
    device=0,
    workers=0,
    amp=True,
    cache=False,
    pretrained=True,
    optimizer="auto",
    seed=42,
    val=True,
    plots=False,
    project=str(PROJECT_ROOT / "runs"),
    name="smoke_test",
    exist_ok=True
)

print("\n第一次训练流程测试完成！")
print("模型保存目录:", PROJECT_ROOT / "runs" / "smoke_test")
