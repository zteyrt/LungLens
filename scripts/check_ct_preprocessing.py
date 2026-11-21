
from pathlib import Path
import csv
import numpy as np
import SimpleITK as sitk
from PIL import Image

DATA = Path(r"E:\LungLens_Data")
V41 = DATA / "processed" / "yolo_v41"
RAW = DATA / "raw"

def load_manifest():
    path = V41 / "sample_manifest.csv"
    with path.open(
        "r", encoding="utf-8-sig", newline=""
    ) as f:
        return list(csv.DictReader(f))

def convert_ct(slice_hu):
    arr = np.clip(
        slice_hu.astype(np.float32),
        -1350, 150
    )
    arr = ((arr + 1350) / 1500 * 255)
    return arr.astype(np.uint8)

def main():
    manifest = load_manifest()

    # 只选Val中的图片
    records = [
        r for r in manifest
        if r["split"] == "val"
    ]

    # 找几个不同病例
    selected = []
    seen = set()

    for r in records:
        uid = r["seriesuid"]
        if uid not in seen:
            selected.append(r)
            seen.add(uid)
        if len(selected) >= 5:
            break

    ct_paths = {
        p.stem: p
        for p in RAW.rglob("*.mhd")
    }

    print("========== CT预处理一致性检查 ==========")

    for r in selected:
        uid = r["seriesuid"]
        z = int(r["z_index"])

        ct = sitk.ReadImage(str(ct_paths[uid]))
        volume = sitk.GetArrayFromImage(ct)

        reconstructed = convert_ct(volume[z])

        filename = f"{uid}_z{z:03d}.png"
        image_path = V41 / "images" / "val" / filename

        if not image_path.is_file():
            raise FileNotFoundError(image_path)

        stored = np.array(
            Image.open(image_path).convert("L")
        )

        if stored.shape != reconstructed.shape:
            raise RuntimeError(
                f"图像尺寸不一致: {filename}"
            )

        diff = np.abs(
            stored.astype(np.int16)
            - reconstructed.astype(np.int16)
        )

        print(
            f"\n病例: {uid[-16:]}"
            f"\nZ层: {z}"
            f"\n最大像素差: {diff.max()}"
            f"\n平均像素差: {diff.mean():.5f}"
            f"\n完全一致: {np.array_equal(stored, reconstructed)}"
        )

    print("\n检查结束，没有修改任何数据。")

if __name__ == "__main__":
    main()
