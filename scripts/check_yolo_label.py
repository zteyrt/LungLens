from pathlib import Path
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

root = Path(r"E:\LungLens_Data\processed\demo")
image_dir = root / "images"
label_dir = root / "labels"

image_files = sorted(image_dir.glob("*.png"))

if not image_files:
    raise FileNotFoundError("没有找到 YOLO 图片")

for image_path in image_files:
    label_path = label_dir / f"{image_path.stem}.txt"

    if not label_path.exists():
        raise FileNotFoundError(label_path)

    img = Image.open(image_path)
    w, h = img.size

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(img, cmap="gray", vmin=0, vmax=255)

    for line in label_path.read_text(encoding="utf-8").splitlines():
        cls, xc, yc, bw, bh = map(float, line.split())

        if not all(0 <= v <= 1 for v in [xc, yc, bw, bh]):
            raise ValueError("YOLO 标签超出归一化范围")

        # 将归一化坐标还原为像素坐标
        center_x = xc * w
        center_y = yc * h
        box_w = bw * w
        box_h = bh * h

        x1 = center_x - box_w / 2
        y1 = center_y - box_h / 2

        rect = Rectangle(
            (x1, y1),
            box_w,
            box_h,
            fill=False,
            edgecolor="red",
            linewidth=2
        )

        ax.add_patch(rect)
        ax.plot(center_x, center_y, "y+")

    ax.set_title("YOLO Label Verification")
    ax.axis("off")

    # 局部放大，便于观察小结节
    ax.set_xlim(center_x - 35, center_x + 35)
    ax.set_ylim(center_y + 35, center_y - 35)

    output = Path(r"D:\肺结节模型训练\evaluation")
    output.mkdir(parents=True, exist_ok=True)

    result = output / f"{image_path.stem}_yolo_check.png"

    fig.savefig(result, dpi=160, bbox_inches="tight")
    plt.close(fig)

    print("标注检查图片:", result)

print("标签可视化验证完成！")
