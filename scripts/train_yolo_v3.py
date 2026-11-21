
from pathlib import Path
from ultralytics import YOLO

PROJECT = Path(r"D:\肺结节模型训练")
DATA = Path(
    r"E:\LungLens_Data\processed\yolo_v3\dataset.yaml"
)

WEIGHTS = PROJECT / "yolov8n.pt"

def main():
    if not DATA.is_file():
        raise FileNotFoundError(f"找不到数据集配置: {DATA}")

    if not WEIGHTS.is_file():
        raise FileNotFoundError(f"找不到预训练权重: {WEIGHTS}")

    print("========== LungLens YOLO V3 ==========")
    print("数据集:", DATA)
    print("预训练权重:", WEIGHTS)
    print("训练: 41例 / 158张图片")
    print("验证: 9例 / 39张图片")
    print("独立测试集: 本次不使用")

    model = YOLO(str(WEIGHTS))

    model.train(
        data=str(DATA),

        # 训练规模
        epochs=80,
        patience=20,

        # 尝试提高小目标的输入分辨率
        imgsz=640,

        # 适配8GB显存
        batch=4,
        nbs=4,
        device=0,
        workers=0,

        # 混合精度
        amp=True,
        cache=False,

        # 优化器
        optimizer="AdamW",
        lr0=0.0005,
        lrf=0.1,
        warmup_epochs=3,

        # 医学影像：关闭颜色随机增强
        hsv_h=0,
        hsv_s=0,
        hsv_v=0,

        # 避免大幅改变微小结节形状
        mosaic=0,
        mixup=0,
        scale=0.05,
        translate=0.02,
        degrees=0,
        shear=0,
        perspective=0,

        # 翻转作为轻量增强
        fliplr=0.5,
        flipud=0,

        # 保存结果
        seed=42,
        deterministic=True,
        plots=True,
        save=True,

        project=str(PROJECT / "runs"),
        name="yolo_v3_baseline",
        exist_ok=False
    )

    print("\n========== 训练完成 ==========")
    print("请查看 runs 下最新的 yolo_v3_baseline 实验目录")
    print("重点文件: results.png、weights/best.pt")

if __name__ == "__main__":
    main()
