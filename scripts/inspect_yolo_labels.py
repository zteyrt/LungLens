from pathlib import Path
from PIL import Image
from statistics import mean, median

root = Path(r"E:\LungLens_Data\processed\yolo_v3")

for split in ["train", "val", "test"]:
    image_dir = root / "images" / split
    label_dir = root / "labels" / split

    records = []
    bad = 0
    missing = 0

    for txt in sorted(label_dir.glob("*.txt")):
        candidates = list(image_dir.glob(txt.stem + ".*"))
        if not candidates:
            missing += 1
            continue

        try:
            with Image.open(candidates[0]) as im:
                W, H = im.size

            for line in txt.read_text(encoding="utf-8-sig").splitlines():
                p = line.split()
                if len(p) != 5:
                    bad += 1
                    continue

                cls, xc, yc, bw, bh = map(float, p)
                if not (0 <= xc <= 1 and 0 <= yc <= 1
                        and 0 < bw <= 1 and 0 < bh <= 1):
                    bad += 1
                    continue

                w, h = bw * W, bh * H
                records.append((txt.stem, w, h))

        except Exception:
            bad += 1

    print(f"\n========== {split.upper()} ==========")
    print("有效标签框:", len(records))
    print("缺失图片:", missing)
    print("异常记录:", bad)

    if records:
        ws = [r[1] for r in records]
        hs = [r[2] for r in records]

        print(f"平均宽度: {mean(ws):.2f}px")
        print(f"平均高度: {mean(hs):.2f}px")
        print(f"中位宽度: {median(ws):.2f}px")
        print(f"中位高度: {median(hs):.2f}px")
        print(f"最小尺寸: {min(ws):.2f} x {min(hs):.2f}px")
        print(f"最大尺寸: {max(ws):.2f} x {max(hs):.2f}px")
        print("宽或高小于4px:", sum(w < 4 or h < 4 for w, h in zip(ws, hs)))
        print("宽或高小于8px:", sum(w < 8 or h < 8 for w, h in zip(ws, hs)))

        print("\n前10个标签框:")
        for name, w, h in records[:10]:
            print(f"{name}: {w:.2f} x {h:.2f}px")

print("\n标签尺寸检查完成。")
