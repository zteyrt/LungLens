# 开发环境与路径准备

## 已测试的本地组合

- Windows，Python 3.12.9
- NVIDIA RTX 5060 Laptop（8 GB）
- PyTorch `2.9.1+cu128`，torchvision `0.24.1+cu128`
- Ultralytics `8.4.204`

这是曾经运行成功的**本机环境记录**，并非对其他系统或后续版本的兼容性保证。

## 安装

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
# 按 https://pytorch.org/get-started/locally/ 选择适合本机的 torch / torchvision
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -c "import torch, SimpleITK, ultralytics; print('torch:', torch.__version__, 'CUDA:', torch.cuda.is_available())"
```

`requirements.txt` 不包含 PyTorch，以避免 `pip` 用不匹配的包覆盖原有 CUDA 环境。若只做 CPU 侧数据构建，不一定需要 GPU。

## 本机目录约定（现有脚本的默认写法）

```text
D:\肺结节模型训练\
  scripts\
  runs\                              # 不提交到 Git

E:\LungLens_Data\
  raw\subset0\<seriesuid>.mhd       # 其他 subset1...subset9 同理
  raw\subset0\<seriesuid>.raw
  annotations\annotations.csv
  annotations\annotations_excluded.csv
  processed\yolo_v41\case_split.csv
  processed\yolo_v5\case_split.csv
  processed\yolo_v5_dataset\dataset.yaml
```

`D:\...` 和 `E:\...` 是作者开发机路径；这些文件不包含在 GitHub 仓库中。更换磁盘或目录时，请检查脚本头部的 `ROOT`、`PROJECT`、`RAW`、`SPLIT_CSV`、`DATA_ROOT`、`OUTPUT` 等常量，以及数据集 YAML 的 `path` 字段。

## 必要数据

1. LUNA16 CT 影像（`.mhd` + 对应 `.raw` 文件对）。
2. 官方参考结节位置文件 `annotations.csv`。
3. 评估时需要 `annotations_excluded.csv`。
4. 当前 V5 是建立在指定的 118 例、425 个参考结节及固定病例划分上的阶段性数据集；单纯下载全部 LUNA16 并不会自动复现同一划分。
5. V5 微调需要先前 V4.1 训练产生的 `runs/yolo_v41_baseline/weights/best.pt`；仓库不分发权重。

数据获取、使用和再分发须遵守数据源许可。项目使用过第三方 Hugging Face 镜像 `Yousra0x0/LUNA16_RAW`，但不能仅凭镜像可访问就推定其具有全部再分发授权。

## 注意事项

- 训练输出会放入 `runs/`，默认被 `.gitignore` 排除。
- 多数训练/构建脚本为避免覆盖旧实验，会拒绝在输出目录已存在时重复运行。请先确认输出目录，不要随意删除历史结果。
- 医学影像可能具有隐私或使用限制；不要将原始 CT、病例识别信息及非公开病人数据提交到公开仓库。
