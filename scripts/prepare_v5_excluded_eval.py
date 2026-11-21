from pathlib import Path

root = Path(r"D:\肺结节模型训练")
source = root / "scripts" / "evaluate_froc_with_excluded.py"
target = root / "scripts" / "evaluate_froc_with_excluded_v5.py"

if not source.is_file():
    raise FileNotFoundError(source)

if target.exists():
    raise RuntimeError(f"目标脚本已存在，禁止覆盖：{target}")

code = source.read_text(encoding="utf-8-sig")

changes = {
    "full_ct_v41_val_v1": "full_ct_v5_val_v1",
    "full_ct_v41_excluded_v1": "full_ct_v5_excluded_v1",
}

for old, new in changes.items():
    count = code.count(old)
    if count == 0:
        raise RuntimeError(
            f"原脚本未找到路径标识 {old}，需要检查原代码"
        )
    code = code.replace(old, new)
    print(f"替换 {old} -> {new}，共 {count} 处")

# 防止其他旧版目录引用残留
if "full_ct_v41_" in code:
    raise RuntimeError("脚本仍包含旧版评估目录引用")

compile(code, str(target), "exec")
target.write_text(code, encoding="utf-8")

print("V5排除标注评估脚本已生成：", target)
print("原V4.1脚本未修改")
