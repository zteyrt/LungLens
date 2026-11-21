
from pathlib import Path
from collections import Counter, defaultdict
import csv
import random
import math

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader


# =============================================
# 1. 实验配置
# =============================================

PROJECT = Path.cwd()
DATA = Path(r"E:\LungLens_Data\processed\fp_reduction_v2")
OUT = PROJECT / "runs" / "fp_reduction_3d_baseline"

SEED = 42
EPOCHS = 40
PATIENCE = 8
BATCH_SIZE = 16
LR = 0.001
WEIGHT_DECAY = 0.0001

VAL_CASE_RATIO = 0.20
EXPECTED_SHAPE = (9, 64, 64)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# =============================================
# 2. 随机种子
# =============================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =============================================
# 3. 数据集
# =============================================

class CT3DDataset(Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]

        label = int(row["label"])
        folder = (
            "positive" if label == 1
            else "negative"
        )

        path = DATA / folder / row["filename"]

        with np.load(path, allow_pickle=False) as f:
            image = f["image"].astype(np.float32)

        if image.shape != EXPECTED_SHAPE:
            raise ValueError(
                f"图块维度错误: {path}"
            )

        # Conv3d 输入: [C,D,H,W]
        tensor = torch.from_numpy(
            image.copy()
        ).unsqueeze(0)

        target = torch.tensor(
            label, dtype=torch.float32
        )

        return tensor, target


# =============================================
# 4. 轻量 3D CNN
# =============================================

class FPReduction3DCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv3d(
                1, 16, kernel_size=3, padding=1
            ),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool3d((1, 2, 2)),

            nn.Conv3d(
                16, 32, kernel_size=3, padding=1
            ),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d((2, 2, 2)),

            nn.Conv3d(
                32, 64, kernel_size=3, padding=1
            ),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d((2, 2, 2)),

            nn.Conv3d(
                64, 96, kernel_size=3, padding=1
            ),
            nn.BatchNorm3d(96),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool3d((1, 1, 1)),
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


# =============================================
# 5. 读取清单
# =============================================

def load_manifest():
    path = DATA / "manifest.csv"

    with path.open(
        "r", encoding="utf-8-sig", newline=""
    ) as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise RuntimeError("manifest.csv 为空")

    for row in rows:
        if row["split"] != "train":
            raise RuntimeError(
                "发现非Train病例，停止训练"
            )

        if row["label"] not in ("0", "1"):
            raise RuntimeError("标签必须为0或1")

    return rows


# =============================================
# 6. 按CT病例划分训练与内部验证
# =============================================

def split_by_case(rows):
    case_labels = defaultdict(list)

    for row in rows:
        case_labels[
            row["seriesuid"]
        ].append(int(row["label"]))

    positive_cases = sorted(
        uid for uid, labels in case_labels.items()
        if 1 in labels
    )

    negative_only_cases = sorted(
        uid for uid, labels in case_labels.items()
        if 1 not in labels
    )

    rng = random.Random(SEED)

    rng.shuffle(positive_cases)
    rng.shuffle(negative_only_cases)

    def take_validation(cases):
        if len(cases) <= 1:
            return []

        n = max(
            1, round(len(cases) * VAL_CASE_RATIO)
        )

        return cases[:n]

    val_cases = set(
        take_validation(positive_cases)
        + take_validation(negative_only_cases)
    )

    train_cases = set(case_labels) - val_cases

    if not train_cases or not val_cases:
        raise RuntimeError("病例划分失败")

    if train_cases & val_cases:
        raise RuntimeError("病例泄漏")

    train_rows = [
        r for r in rows
        if r["seriesuid"] in train_cases
    ]

    val_rows = [
        r for r in rows
        if r["seriesuid"] in val_cases
    ]

    for subset, name in (
        (train_rows, "train"),
        (val_rows, "val")
    ):
        labels = Counter(
            int(r["label"]) for r in subset
        )

        if labels[0] == 0 or labels[1] == 0:
            raise RuntimeError(
                f"{name} 缺少某一类别"
            )

    return (
        train_rows,
        val_rows,
        train_cases,
        val_cases
    )


# =============================================
# 7. 评价指标
# =============================================

def calculate_auc(labels, scores):
    labels = np.asarray(labels, dtype=np.int32)
    scores = np.asarray(scores, dtype=np.float64)

    positives = scores[labels == 1]
    negatives = scores[labels == 0]

    if not len(positives) or not len(negatives):
        return float("nan")

    # Mann-Whitney U: 与成对比较等价
    total = 0.0

    for s in positives:
        total += np.sum(s > negatives)
        total += 0.5 * np.sum(s == negatives)

    return total / (
        len(positives) * len(negatives)
    )


def calculate_metrics(labels, probs):
    labels = np.asarray(labels, dtype=np.int32)
    probs = np.asarray(probs, dtype=np.float64)

    preds = (probs >= 0.5).astype(np.int32)

    tp = int(np.sum((preds == 1) & (labels == 1)))
    fp = int(np.sum((preds == 1) & (labels == 0)))
    tn = int(np.sum((preds == 0) & (labels == 0)))
    fn = int(np.sum((preds == 0) & (labels == 1)))

    precision = (
        tp / (tp + fp) if tp + fp else 0
    )
    recall = (
        tp / (tp + fn) if tp + fn else 0
    )
    specificity = (
        tn / (tn + fp) if tn + fp else 0
    )

    auc = calculate_auc(labels, probs)

    return {
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "balanced_accuracy": (
            recall + specificity
        ) / 2,
        "auc": auc
    }


# =============================================
# 8. 单轮训练
# =============================================

def train_epoch(model, loader, criterion, optimizer):
    model.train()

    total_loss = 0.0
    total_items = 0

    for images, targets in loader:
        images = images.to(DEVICE)
        targets = targets.to(DEVICE)

        optimizer.zero_grad(set_to_none=True)

        logits = model(images)
        loss = criterion(logits, targets)

        loss.backward()
        optimizer.step()

        n = targets.size(0)
        total_loss += loss.item() * n
        total_items += n

    return total_loss / max(total_items, 1)


# =============================================
# 9. 验证
# =============================================

@torch.no_grad()
def validate(model, loader, criterion):
    model.eval()

    total_loss = 0.0
    total_items = 0
    all_probs = []
    all_labels = []

    for images, targets in loader:
        images = images.to(DEVICE)
        targets = targets.to(DEVICE)

        logits = model(images)
        loss = criterion(logits, targets)

        probabilities = torch.sigmoid(
            logits
        )

        n = targets.size(0)
        total_loss += loss.item() * n
        total_items += n

        all_probs.extend(
            probabilities.cpu().numpy().tolist()
        )
        all_labels.extend(
            targets.cpu().numpy().astype(int).tolist()
        )

    metrics = calculate_metrics(
        all_labels, all_probs
    )

    return (
        total_loss / max(total_items, 1),
        metrics
    )


# =============================================
# 10. 保存CSV
# =============================================

def write_rows(path, header, rows):
    with path.open(
        "w", encoding="utf-8-sig", newline=""
    ) as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


# =============================================
# 11. 主程序
# =============================================

def main():
    set_seed(SEED)

    if OUT.exists():
        raise RuntimeError(
            f"实验目录已经存在: {OUT}\n"
            "为避免覆盖旧实验，请修改OUT名称。"
        )

    rows = load_manifest()

    (
        train_rows, val_rows,
        train_cases, val_cases
    ) = split_by_case(rows)

    train_counts = Counter(
        int(r["label"]) for r in train_rows
    )
    val_counts = Counter(
        int(r["label"]) for r in val_rows
    )

    print("========== 3D CNN Baseline ==========")
    print("设备:", DEVICE)
    print("总样本:", len(rows))
    print("训练病例:", len(train_cases))
    print("内部验证病例:", len(val_cases))

    print(
        "训练正/负:",
        train_counts[1], train_counts[0]
    )
    print(
        "验证正/负:",
        val_counts[1], val_counts[0]
    )

    OUT.mkdir(parents=True)

    write_rows(
        OUT / "internal_split.csv",
        ["seriesuid", "subset"],
        [
            [uid, "train"]
            for uid in sorted(train_cases)
        ] + [
            [uid, "val"]
            for uid in sorted(val_cases)
        ]
    )

    train_loader = DataLoader(
        CT3DDataset(train_rows),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=(DEVICE.type == "cuda")
    )

    val_loader = DataLoader(
        CT3DDataset(val_rows),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=(DEVICE.type == "cuda")
    )

    model = FPReduction3DCNN().to(DEVICE)

    # 正样本更少，适当提高其训练损失权重
    pos_weight = torch.tensor(
        [train_counts[0] / train_counts[1]],
        dtype=torch.float32,
        device=DEVICE
    )

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=pos_weight
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3
    )

    best_loss = float("inf")
    best_epoch = 0
    no_improve = 0
    history = []

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_epoch(
            model, train_loader,
            criterion, optimizer
        )

        val_loss, metrics = validate(
            model, val_loader, criterion
        )

        scheduler.step(val_loss)

        lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"TrainLoss={train_loss:.4f} | "
            f"ValLoss={val_loss:.4f} | "
            f"Recall={metrics['recall']:.2%} | "
            f"Specificity={metrics['specificity']:.2%} | "
            f"AUC={metrics['auc']:.4f}"
        )

        history.append([
            epoch,
            train_loss,
            val_loss,
            metrics["precision"],
            metrics["recall"],
            metrics["specificity"],
            metrics["balanced_accuracy"],
            metrics["auc"],
            lr
        ])

        if val_loss < best_loss - 1e-5:
            best_loss = val_loss
            best_epoch = epoch
            no_improve = 0

            torch.save(
                {
                    "model_state": model.state_dict(),
                    "epoch": epoch,
                    "input_shape": EXPECTED_SHAPE,
                    "seed": SEED,
                    "val_loss": val_loss
                },
                OUT / "best.pt"
            )
        else:
            no_improve += 1

        if no_improve >= PATIENCE:
            print(
                f"\nEarly stopping at epoch {epoch}"
            )
            break

    write_rows(
        OUT / "history.csv",
        [
            "epoch", "train_loss", "val_loss",
            "precision", "recall",
            "specificity", "balanced_accuracy",
            "auc", "learning_rate"
        ],
        history
    )

    # 重新加载最佳权重进行最终内部验证
    checkpoint = torch.load(
        OUT / "best.pt",
        map_location=DEVICE,
        weights_only=True
    )

    model.load_state_dict(
        checkpoint["model_state"]
    )

    final_loss, final = validate(
        model, val_loader, criterion
    )

    print("\n========== 训练完成 ==========")
    print("最佳Epoch:", best_epoch)
    print("内部验证Loss:", round(final_loss, 4))
    print("Precision:", f"{final['precision']:.2%}")
    print("Recall:", f"{final['recall']:.2%}")
    print("Specificity:", f"{final['specificity']:.2%}")
    print("Balanced Accuracy:",
          f"{final['balanced_accuracy']:.2%}")
    print("ROC AUC:", round(final["auc"], 4))

    print(
        "TP/FP/TN/FN:",
        final["TP"],
        final["FP"],
        final["TN"],
        final["FN"]
    )

    print("模型权重:", OUT / "best.pt")
    print("训练记录:", OUT / "history.csv")
    print("病例划分:", OUT / "internal_split.csv")
    print(
        "\n注意：这是二阶段原型的内部验证结果，"
        "不能直接视为完整CT FROC性能。"
    )


if __name__ == "__main__":
    main()
