
from pathlib import Path
from collections import defaultdict
import csv
import math

import numpy as np
import torch
import SimpleITK as sitk
from scipy.ndimage import map_coordinates
from torch import nn


# ==================================================
# 1. 路径配置
# ==================================================

PROJECT = Path(__file__).resolve().parent.parent

ROOT = Path(r"E:\LungLens_Data")

CNN_WEIGHT = (
    PROJECT
    / "runs"
    / "fp_reduction_3d_baseline"
    / "best.pt"
)

CANDIDATE_CSV = (
    PROJECT
    / "runs"
    / "full_ct_v41_val_v1"
    / "candidates_3d.csv"
)

SPLIT_CSV = (
    ROOT
    / "processed"
    / "yolo_v41"
    / "case_split.csv"
)

ANNOTATIONS = (
    ROOT
    / "annotations"
    / "annotations.csv"
)

EXCLUDED = (
    ROOT
    / "annotations"
    / "annotations_excluded.csv"
)

RAW = ROOT / "raw"

# 使用新目录，避免覆盖以前的实验
OUT = (
    PROJECT
    / "runs"
    / "fp_reduction_3d_froc_v1"
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

BATCH_SIZE = 16

# 3D CNN 输入大小
PATCH_D = 9
PATCH_H = 64
PATCH_W = 64

PATCH_Z_MM = 2.0
PATCH_XY_MM = 1.0

# CT窗宽窗位
WINDOW_MIN = -1350.0
WINDOW_MAX = 150.0

# 对比的方法
METHODS = ("YOLO", "CNN", "FUSION")

# 最少跨层数
MIN_SLICES_OPTIONS = (1, 2)

# 固定阈值仅用于观察，不用于最终选择最佳点
REPORT_THRESHOLDS = (
    0.05, 0.10, 0.20,
    0.30, 0.50
)

EXCLUDED_DEFAULT_DIAMETER = 10.0


# ==================================================
# 2. CSV工具
# ==================================================

def read_csv(path):
    if not path.is_file():
        raise FileNotFoundError(
            f"文件不存在: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as f:
        return list(csv.DictReader(f))


def save_csv(path, header, rows):
    with path.open(
        "w",
        encoding="utf-8-sig",
        newline=""
    ) as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


# ==================================================
# 3. 读取验证病例
# ==================================================

def load_val_cases():
    rows = read_csv(SPLIT_CSV)

    cases = sorted({
        row["seriesuid"]
        for row in rows
        if row["split"] == "val"
    })

    if not cases:
        raise RuntimeError(
            "没有找到Val病例"
        )

    return cases


# ==================================================
# 4. 读取参考标注和排除标注
# ==================================================

def load_annotations(path, cases, excluded=False):
    result = defaultdict(list)

    case_set = set(cases)

    for row in read_csv(path):
        uid = row["seriesuid"]

        if uid not in case_set:
            continue

        diameter = float(
            row["diameter_mm"]
        )

        # LUNA16排除标注中的-1
        # 使用10mm替代直径
        if excluded and diameter < 0:
            diameter = EXCLUDED_DEFAULT_DIAMETER

        if diameter <= 0:
            raise ValueError(
                f"标注直径异常: {uid}"
            )

        result[uid].append({
            "xyz": (
                float(row["coordX"]),
                float(row["coordY"]),
                float(row["coordZ"])
            ),
            "radius": diameter / 2.0
        })

    return result


# ==================================================
# 5. 读取现有790个三维候选
# ==================================================

def load_candidates(cases):
    result = defaultdict(list)

    case_set = set(cases)

    for row in read_csv(CANDIDATE_CSV):
        uid = row["seriesuid"]

        if uid not in case_set:
            continue

        candidate = {
            "xyz": (
                float(row["world_x"]),
                float(row["world_y"]),
                float(row["world_z"])
            ),
            "yolo_score": float(
                row["score"]
            ),
            "num_slices": int(
                row["num_slices"]
            ),
            "cnn_score": None,
            "fusion_score": None
        }

        result[uid].append(candidate)

    return result


# ==================================================
# 6. 查找CT文件
# ==================================================

def find_ct_paths():
    paths = {}

    for path in RAW.rglob("*.mhd"):
        uid = path.stem

        if uid in paths:
            raise RuntimeError(
                f"CT文件重复: {uid}"
            )

        paths[uid] = path

    return paths


# ==================================================
# 7. 3D CNN模型结构
# 必须与此前训练版本一致
# ==================================================

class FPReduction3DCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv3d(
                1, 16,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool3d((1, 2, 2)),

            nn.Conv3d(
                16, 32,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d((2, 2, 2)),

            nn.Conv3d(
                32, 64,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d((2, 2, 2)),

            nn.Conv3d(
                64, 96,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm3d(96),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool3d(
                (1, 1, 1)
            ),
            nn.Flatten()
        )

        self.classifier = nn.Sequential(
            nn.Linear(96, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)

        return x.squeeze(1)


# ==================================================
# 8. 加载训练好的CNN
# ==================================================

def load_cnn():
    if not CNN_WEIGHT.is_file():
        raise FileNotFoundError(
            CNN_WEIGHT
        )

    model = FPReduction3DCNN().to(
        DEVICE
    )

    checkpoint = torch.load(
        CNN_WEIGHT,
        map_location=DEVICE,
        weights_only=True
    )

    shape = tuple(
        checkpoint["input_shape"]
    )

    if shape != (PATCH_D, PATCH_H, PATCH_W):
        raise RuntimeError(
            f"CNN输入维度不匹配: {shape}"
        )

    model.load_state_dict(
        checkpoint["model_state"]
    )

    model.eval()

    print(
        "加载CNN权重成功，Epoch:",
        checkpoint["epoch"]
    )

    return model


# ==================================================
# 9. 提取与训练完全一致的3D Patch
# ==================================================

def extract_patch(volume, ct, world_center):
    ix, iy, iz = (
        ct.TransformPhysicalPointToContinuousIndex(
            tuple(
                float(v)
                for v in world_center
            )
        )
    )

    sx, sy, sz = ct.GetSpacing()

    z_offsets = (
        np.arange(
            PATCH_D,
            dtype=np.float32
        ) - (PATCH_D - 1) / 2
    ) * PATCH_Z_MM / sz

    y_offsets = (
        np.arange(
            PATCH_H,
            dtype=np.float32
        ) - (PATCH_H - 1) / 2
    ) * PATCH_XY_MM / sy

    x_offsets = (
        np.arange(
            PATCH_W,
            dtype=np.float32
        ) - (PATCH_W - 1) / 2
    ) * PATCH_XY_MM / sx

    zz = iz + z_offsets[
        :, None, None
    ]

    yy = iy + y_offsets[
        None, :, None
    ]

    xx = ix + x_offsets[
        None, None, :
    ]

    coordinates = np.broadcast_arrays(
        zz, yy, xx
    )

    patch = map_coordinates(
        volume,
        coordinates,
        order=1,
        mode="constant",
        cval=-1000.0,
        prefilter=False
    )

    patch = np.clip(
        patch.astype(np.float32),
        WINDOW_MIN,
        WINDOW_MAX
    )

    patch = (
        patch - WINDOW_MIN
    ) / (
        WINDOW_MAX - WINDOW_MIN
    )

    if patch.shape != (
        PATCH_D,
        PATCH_H,
        PATCH_W
    ):
        raise RuntimeError(
            f"Patch尺寸异常: {patch.shape}"
        )

    if not np.all(np.isfinite(patch)):
        raise RuntimeError(
            "Patch包含NaN/Inf"
        )

    return patch.astype(np.float32)


# ==================================================
# 10. CNN批量预测
# ==================================================

@torch.no_grad()
def predict_cnn_scores(model, patches):
    scores = []

    for start in range(
        0,
        len(patches),
        BATCH_SIZE
    ):
        batch = np.stack(
            patches[
                start:start+BATCH_SIZE
            ]
        ).astype(np.float32)

        # [B,1,9,64,64]
        x = torch.from_numpy(
            batch
        ).unsqueeze(1).to(DEVICE)

        logits = model(x)

        probabilities = torch.sigmoid(
            logits
        )

        scores.extend(
            probabilities.cpu().numpy().tolist()
        )

    return scores


# ==================================================
# 11. 为三维候选重新评分
# ==================================================

def rescore_candidates(
    model,
    cases,
    candidates,
    ct_paths
):
    saved_rows = []

    total = len(cases)

    for case_index, uid in enumerate(
        cases, 1
    ):
        case_candidates = candidates[uid]

        if not case_candidates:
            print(
                f"[{case_index}/{total}] "
                "没有候选"
            )
            continue

        if uid not in ct_paths:
            raise FileNotFoundError(
                f"缺少CT: {uid}"
            )

        ct = sitk.ReadImage(
            str(ct_paths[uid])
        )

        volume = (
            sitk.GetArrayFromImage(ct)
            .astype(np.float32)
        )

        patches = [
            extract_patch(
                volume,
                ct,
                c["xyz"]
            )
            for c in case_candidates
        ]

        cnn_scores = predict_cnn_scores(
            model,
            patches
        )

        if len(cnn_scores) != len(
            case_candidates
        ):
            raise RuntimeError(
                f"候选评分数量错误: {uid}"
            )

        for candidate, cnn_score in zip(
            case_candidates,
            cnn_scores
        ):
            candidate["cnn_score"] = float(
                cnn_score
            )

            # 仅作为探索性融合分数
            # 不代表真实校准概率
            candidate["fusion_score"] = math.sqrt(
                max(
                    candidate["yolo_score"],
                    0.0
                )
                *
                max(
                    candidate["cnn_score"],
                    0.0
                )
            )

            saved_rows.append([
                uid,
                *candidate["xyz"],
                candidate["num_slices"],
                candidate["yolo_score"],
                candidate["cnn_score"],
                candidate["fusion_score"]
            ])

        print(
            f"[{case_index}/{total}] "
            f"病例 {uid[-16:]} "
            f"候选数: {len(case_candidates)}"
        )

        del patches, volume, ct

    save_csv(
        OUT / "rescored_candidates.csv",
        [
            "seriesuid",
            "world_x",
            "world_y",
            "world_z",
            "num_slices",
            "yolo_score",
            "cnn_score",
            "fusion_score"
        ],
        saved_rows
    )

    print(
        "\nCNN候选评分完成:",
        len(saved_rows)
    )


# ==================================================
# 12. 三维空间匹配
# ==================================================

def distance_xyz(a, b):
    return math.sqrt(
        sum(
            (x-y)**2
            for x, y in zip(a, b)
        )
    )


def inside_annotation(
    candidate_xyz,
    annotation
):
    d = distance_xyz(
        candidate_xyz,
        annotation["xyz"]
    )

    return d < annotation["radius"]


# ==================================================
# 13. 获取评分
# ==================================================

def get_score(candidate, method):
    if method == "YOLO":
        return candidate["yolo_score"]

    if method == "CNN":
        return candidate["cnn_score"]

    if method == "FUSION":
        return candidate["fusion_score"]

    raise ValueError(
        f"未知评分方法: {method}"
    )


# ==================================================
# 14. 结节级FROC评估
# 已整合距离比例最优匹配修正
# ==================================================

def evaluate(
    cases,
    candidates,
    gt_map,
    excluded_map,
    method,
    threshold,
    min_slices
):
    tp = 0
    fp = 0
    fn = 0

    ignored_excluded = 0
    ignored_duplicate = 0

    for uid in cases:
        gts = gt_map[uid]
        excluded = excluded_map[uid]

        selected = [
            c
            for c in candidates[uid]
            if (
                c["num_slices"] >= min_slices
                and get_score(
                    c, method
                ) >= threshold
            )
        ]

        selected.sort(
            key=lambda c: get_score(
                c, method
            ),
            reverse=True
        )

        matched_gt = set()

        for candidate in selected:
            # =================================
            # 首先匹配正式参考结节
            # =================================

            matching = []

            for i, gt in enumerate(gts):
                d = distance_xyz(
                    candidate["xyz"],
                    gt["xyz"]
                )

                if d < gt["radius"]:
                    ratio = (
                        d / gt["radius"]
                    )

                    matching.append(
                        (ratio, i)
                    )

            if matching:
                # 优先距离相对半径最近者
                matching.sort()

                available = [
                    item
                    for item in matching
                    if item[1] not in matched_gt
                ]

                if available:
                    matched_gt.add(
                        available[0][1]
                    )
                    tp += 1
                else:
                    # 同一参考结节的重复预测
                    ignored_duplicate += 1

                continue

            # =================================
            # 再匹配LUNA16排除标注
            # =================================

            if any(
                inside_annotation(
                    candidate["xyz"],
                    item
                )
                for item in excluded
            ):
                ignored_excluded += 1
                continue

            # =================================
            # 最后计为未匹配候选FP
            # =================================

            fp += 1

        fn += (
            len(gts) - len(matched_gt)
        )

    sensitivity = (
        tp / (tp+fn)
        if tp+fn > 0
        else 0.0
    )

    fp_per_scan = (
        fp / len(cases)
        if cases else 0.0
    )

    return {
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "sensitivity": sensitivity,
        "fp_per_scan": fp_per_scan,
        "excluded_ignored": ignored_excluded,
        "duplicate_ignored": ignored_duplicate
    }


# ==================================================
# 15. FROC比较
# ==================================================

def run_froc(
    cases,
    candidates,
    gt_map,
    excluded_map
):
    summary = []

    for method in METHODS:
        for min_slices in MIN_SLICES_OPTIONS:

            scores = sorted({
                get_score(c, method)
                for uid in cases
                for c in candidates[uid]
                if c["num_slices"] >= min_slices
            }, reverse=True)

            # 包含无检出的起始操作点
            thresholds = [
                float("inf")
            ] + scores

            results = []

            for threshold in thresholds:
                metrics = evaluate(
                    cases,
                    candidates,
                    gt_map,
                    excluded_map,
                    method,
                    threshold,
                    min_slices
                )

                results.append({
                    "threshold": threshold,
                    **metrics
                })

            # 保存所有操作点
            save_csv(
                OUT / (
                    f"froc_{method.lower()}"
                    f"_min{min_slices}.csv"
                ),
                [
                    "threshold",
                    "TP",
                    "FP",
                    "FN",
                    "sensitivity",
                    "fp_per_scan",
                    "excluded_ignored",
                    "duplicate_ignored"
                ],
                [
                    [
                        r["threshold"],
                        r["TP"],
                        r["FP"],
                        r["FN"],
                        r["sensitivity"],
                        r["fp_per_scan"],
                        r["excluded_ignored"],
                        r["duplicate_ignored"]
                    ]
                    for r in results
                ]
            )

            # 找出FP/scan<=4的最佳观察点
            eligible = [
                r
                for r in results
                if r["fp_per_scan"] <= 4.0
            ]

            if not eligible:
                raise RuntimeError(
                    "不存在符合FP约束的操作点"
                )

            best = max(
                eligible,
                key=lambda r: (
                    r["sensitivity"],
                    -r["fp_per_scan"]
                )
            )

            summary.append({
                "method": method,
                "min_slices": min_slices,
                "threshold": best["threshold"],
                **{
                    key: best[key]
                    for key in (
                        "TP",
                        "FP",
                        "FN",
                        "sensitivity",
                        "fp_per_scan"
                    )
                }
            })

    # 保存汇总对比
    save_csv(
        OUT / "comparison.csv",
        [
            "method",
            "min_slices",
            "threshold",
            "TP",
            "FP",
            "FN",
            "sensitivity",
            "fp_per_scan"
        ],
        [
            [
                r["method"],
                r["min_slices"],
                r["threshold"],
                r["TP"],
                r["FP"],
                r["FN"],
                r["sensitivity"],
                r["fp_per_scan"]
            ]
            for r in summary
        ]
    )

    print(
        "\n========== 3D CNN完整CT复评 =========="
    )

    print(
        "Method    Layers   TP   FP   FN   "
        "Sensitivity   FP/scan   Threshold"
    )

    for r in summary:
        print(
            f"{r['method']:<9}"
            f"{r['min_slices']:>4}"
            f"{r['TP']:>6}"
            f"{r['FP']:>5}"
            f"{r['FN']:>5}   "
            f"{r['sensitivity']:>9.2%}   "
            f"{r['fp_per_scan']:>7.3f}   "
            f"{r['threshold']:.4f}"
        )

    # 确认YOLO基线复现
    baseline = next(
        r for r in summary
        if (
            r["method"] == "YOLO"
            and r["min_slices"] == 1
        )
    )

    if not (
        baseline["TP"] == 33
        and baseline["FN"] == 11
        and abs(
            baseline["fp_per_scan"]
            - 35/11
        ) < 1e-6
    ):
        print(
            "\nWARNING: YOLO基线与上一轮不一致，"
            "请先排查匹配逻辑。"
        )
    else:
        print(
            "\nYOLO基线复现成功："
            "33 TP / 35 FP / 11 FN"
        )


# ==================================================
# 16. 主程序
# ==================================================

def main():
    required = (
        CNN_WEIGHT,
        CANDIDATE_CSV,
        SPLIT_CSV,
        ANNOTATIONS,
        EXCLUDED
    )

    for path in required:
        if not path.is_file():
            raise FileNotFoundError(
                path
            )

    # 防止覆盖旧实验
    if OUT.exists():
        raise RuntimeError(
            f"结果目录已存在: {OUT}\n"
            "请修改OUT为新目录，避免覆盖旧数据。"
        )

    cases = load_val_cases()

    gt_map = load_annotations(
        ANNOTATIONS,
        cases
    )

    excluded_map = load_annotations(
        EXCLUDED,
        cases,
        excluded=True
    )

    candidates = load_candidates(
        cases
    )

    ct_paths = find_ct_paths()

    missing_ct = [
        uid
        for uid in cases
        if uid not in ct_paths
    ]

    if missing_ct:
        raise RuntimeError(
            f"缺少CT: {missing_ct}"
        )

    total_gt = sum(
        len(gt_map[uid])
        for uid in cases
    )

    total_candidates = sum(
        len(candidates[uid])
        for uid in cases
    )

    if len(cases) != 11:
        raise RuntimeError(
            f"验证病例数量异常: {len(cases)}"
        )

    if total_gt != 44:
        raise RuntimeError(
            f"参考结节数量异常: {total_gt}"
        )

    if total_candidates != 790:
        raise RuntimeError(
            f"三维候选数量异常: {total_candidates}"
        )

    OUT.mkdir(
        parents=True,
        exist_ok=False
    )

    print(
        "========== 3D CNN候选重评分 =========="
    )

    print("设备:", DEVICE)
    print("验证病例:", len(cases))
    print("参考结节:", total_gt)
    print("三维候选:", total_candidates)
    print("CNN权重:", CNN_WEIGHT)

    model = load_cnn()

    # 第一阶段：生成CNN评分
    rescore_candidates(
        model,
        cases,
        candidates,
        ct_paths
    )

    # 第二阶段：结节级FROC比较
    run_froc(
        cases,
        candidates,
        gt_map,
        excluded_map
    )

    print(
        "\n========== 实验完成 =========="
    )

    print("结果目录:", OUT)
    print(
        "候选评分文件:",
        OUT / "rescored_candidates.csv"
    )
    print(
        "FROC对比文件:",
        OUT / "comparison.csv"
    )

    print(
        "\n注意：这是11例Val上的探索性评估，"
        "不是独立测试集结论。"
    )

    print(
        "原始CT、V4.1模型和3D CNN权重"
        "均未修改。"
    )


if __name__ == "__main__":
    main()
