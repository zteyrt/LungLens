from pathlib import Path

root = Path(r"D:\肺结节模型训练")

source = root / "scripts/review_full_ct_errors_v5.py"
target = root / "scripts/review_full_ct_errors_v5_best.py"

if not source.is_file():
    raise FileNotFoundError(source)

if target.exists():
    raise RuntimeError(f"目标已存在：{target}")

code = source.read_text(encoding="utf-8-sig")

changes = {
    "CONF = 0.57": "CONF = 0.41300588846206665",
    "MIN_SLICES = 1": "MIN_SLICES = 2",
    "full_ct_v5_error_review": "full_ct_v5_best_error_review",
}

for old, new in changes.items():
    if code.count(old) != 1:
        raise RuntimeError(f"预期只匹配1处：{old}")
    code = code.replace(old, new)

compile(code, str(target), "exec")
target.write_text(code, encoding="utf-8")

print("V5最佳操作点分析脚本已生成")
print("阈值：0.41300588846206665")
print("最少层数：2")
print("原始脚本未修改")
