
from pathlib import Path
from huggingface_hub import hf_hub_download

ROOT = Path(r"E:\LungLens_Data")
CT_DIR = ROOT / "raw" / "subset0"

# 需要补全的病例完整 UID
uid = (
    "1.3.6.1.4.1.14519.5.2.1.6279.6001."
    "248264912111"
)

# 不能仅凭末尾12位推断完整 UID：
# 自动寻找本地已有的 MHD 文件
matches = list(CT_DIR.glob("*248264912111.mhd"))

if len(matches) != 1:
    raise RuntimeError(
        f"找到 {len(matches)} 个匹配的 MHD 文件，"
        "需要检查实际文件名"
    )

mhd_path = matches[0]
uid = mhd_path.stem

print("待修复病例:", uid)

# 从 MHD 读取它真正引用的 RAW 文件名
raw_name = None
for line in mhd_path.read_text(
    encoding="utf-8", errors="replace"
).splitlines():
    if line.strip().startswith("ElementDataFile"):
        raw_name = line.split("=", 1)[1].strip()
        break

if not raw_name:
    raise RuntimeError("MHD 中没有 ElementDataFile")

raw_path = CT_DIR / raw_name

if raw_path.is_file() and raw_path.stat().st_size > 0:
    print("RAW 文件已经存在，无需下载")
else:
    print("开始补下载:", raw_name)

    hf_hub_download(
        repo_id="Yousra0x0/LUNA16_RAW",
        repo_type="dataset",
        filename=f"subset0/{raw_name}",
        local_dir=str(ROOT / "raw"),
        cache_dir=str(ROOT / "hf_cache")
    )

print("RAW 文件是否存在:", raw_path.exists())

if raw_path.exists():
    print(
        "RAW 文件大小:",
        round(raw_path.stat().st_size / 1024**2, 2),
        "MB"
    )
