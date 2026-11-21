# LungLens 代码执行路线

> 以下是**已有实现的导航**，不是可以直接在新机器上一次运行成功的完整一键流程。请先阅读 `setup.md` 并准备外部数据、路径与已有实验产物。

## 1. 数据构建

| 脚本 | 作用 | 前置条件 |
|---|---|---|
| `scripts/expand_luna16_fast.py` | 增量下载 LUNA16 CT；支持命令行参数 | 网络、远程数据源、磁盘空间 |
| `scripts/prepare_v5_split.py` | 保留原 V4.1 验证/测试病例，把新增 CT 分配至 Train | V4.1 划分表与完整本地病例 |
| `scripts/build_yolo_v5_fixed.py` | V5 正式数据构建代码（含缺失 RAW 补全） | V5 划分、CT、`annotations.csv` |
| `scripts/build_yolo_v5.py` | 原始版本；不提供自动补全 RAW | 同上且需要所有 RAW 文件已完整 |

已测试 V5 数据总览：118 CT（Train 100、Val 11、Test 7）、425 个参考结节、2198 张图片、2171 个二维弱标注框。Train/Val/Test 图片数分别为 1750/263/185。

**不要同时运行两个 V5 构建脚本：** 两者目标输出目录相同，且正式构建脚本会防止覆盖现有结果。

## 2. YOLO 微调

```powershell
.\.venv\Scripts\python.exe scripts\train_yolo_v5.py
```

前置条件：已经生成 `E:\LungLens_Data\processed\yolo_v5_dataset\dataset.yaml`，并存在 V4.1 权重 `runs\yolo_v41_baseline\weights\best.pt`。V5 训练使用 V5 train/val，独立 test 不用于超参数选择。

本次基线训练：初始学习率 `0.0003`、batch `8`、输入 `512`、最多 `60` epochs、早停耐心值 `15`；实际在 epoch 25 停止，最佳 epoch 10。

## 3. 二维同集验证

```powershell
.\.venv\Scripts\python.exe scripts\compare_v41_v5_batch1.py
```

在同一 V5 验证图片上分别对 V4.1 和 V5 权重评估。使用 batch=1 避免此前出现的 NMS 超时问题。二维检测的 Precision/Recall/mAP 与完整 CT 结节级 FROC 不能互换。

## 4. 完整 CT 候选生成与 FROC

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_full_ct_v5.py
.\.venv\Scripts\python.exe scripts\evaluate_froc_with_excluded_v5.py
```

第一个脚本读取完整 CT，做逐层 YOLO 推理和跨切片合并，输出 `runs/full_ct_v5_val_v1/candidates_3d.csv`。第二个脚本基于已保存的三维候选，执行参考排除标注规则的复评，输出 `runs/full_ct_v5_excluded_v1/`；它不重新推理。

**注意：** 这两个 V5 脚本目前仍从 `processed/yolo_v41/case_split.csv` 读取验证病例，依赖 V5 保持原验证集不变的历史约定。重构时应先核对划分一致性，再改为 V5 规范化路径，否则可能影响既有结果的可比性。

**注意：** 当前完整 CT 脚本默认 `IMGSZ=640, BATCH=4`，曾发生 NMS 超时警告。正式报告或跨版本比较应在相同设置下消除该警告后复核结果。

## 5. 误检和漏检分析

- `scripts/diagnose_full_ct_fn_v5.py`：基于已有候选的漏检成因初步分类；其默认阈值不同于最佳 FROC 操作点。
- `scripts/review_full_ct_errors_v5_best.py`：使用 `conf≈0.413006`、`min_slices=2` 的最佳操作点检查候选。
- 误检可视化脚本的“未匹配候选”数不等于执行排除标注规则后的正式 FP，不能混用。

## 6. 3D CNN 探索（实验性）

- `scripts/build_fp_reduction_dataset.py`
- `scripts/train_fp_reduction_3d.py`
- `scripts/evaluate_fp_reduction_3d.py`

历史 3D CNN 在内部分类指标上有改善，但接入完整 CT 后并未超过单独 YOLO V4.1。该模型不是当前默认方案，相关结果不可作为已实现的 FROC 性能提升。

## 7. 实验隔离

- 保持患者/CT级的 Train、Val、Test 隔离。
- 不使用 Test 集选阈值、挑 checkpoint 或调整后处理。
- 不覆盖历史 `runs/` 输出；必要时为新实验创建带版本号的目录。
- 若改变阈值、NMS、最少出现层数或排除规则，应另存实验并明确评估口径。
