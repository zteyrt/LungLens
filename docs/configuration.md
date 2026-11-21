# V5 配置与路径迁移

V5 核心脚本已通过 `scripts/lunglens_paths.py` 共享路径配置。默认值保留原有 Windows 实验环境，**不修改训练参数、FROC 匹配逻辑或既有输出目录**。

## 环境变量

| 变量 | 默认值 | 作用 |
|---|---|---|
| `LUNGLENS_PROJECT_ROOT` | 当前仓库根目录 | 训练权重与实验输出 |
| `LUNGLENS_DATA_ROOT` | `E:\LungLens_Data` | `.mhd/.raw`、annotations 和 processed |
| `LUNGLENS_EVAL_SPLIT` | `<data>/processed/yolo_v41/case_split.csv` | V4.1/V5 固定验证病例清单 |

在 PowerShell 中：

```powershell
$env:LUNGLENS_PROJECT_ROOT = 'D:\肺结节模型训练'
$env:LUNGLENS_DATA_ROOT = 'E:\LungLens_Data'

# 检查配置（不会启动训练）
.\.venv\Scripts\python.exe scripts\lunglens_paths.py
```

若要在另一台电脑运行，请把这两个变量设置为新机器的真实目录；变量仅在当前 PowerShell 会话有效。**现有训练和验证输出目录仍受到脚本的防覆盖检查。**

## 适用脚本

- `scripts/prepare_v5_split.py`
- `scripts/build_yolo_v5_fixed.py`
- `scripts/train_yolo_v5.py`
- `scripts/evaluate_full_ct_v5.py`
- `scripts/evaluate_froc_with_excluded_v5.py`

其他历史脚本暂未统一配置，不能假设都支持上述变量。

## 验证病例一致性

现有 V5 与 V4.1 的 11 例 Val、7 例 Test 设计上保持相同。为保证历史 FROC 可比，两个 V5 评估脚本仍默认使用 `yolo_v41/case_split.csv`。在准备迁移到 V5 CSV 前，应逐一核对 Val/Test 的 UID 完全一致。改变评估病例会破坏与此前结果的直接可比性。

## 局限

这是路径可配置性改进，不是端到端复现实验认证。数据与权重并不随仓库提供；训练从 V4.1 `best.pt` 继续，运行前必须单独准备。之前完整 CT 推理存在 NMS 超时警告，未因本次配置调整而自动修复。
