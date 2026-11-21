from pathlib import Path

PROJECT = Path(r"D:\肺结节模型训练")
SCRIPTS = PROJECT / "scripts"
RUNS = PROJECT / "runs"

tasks = [
    {
        "source": "diagnose_full_ct_fn.py",
        "target": "diagnose_full_ct_fn_v5.py",
        "changes": {
            "full_ct_v41_val_v1": "full_ct_v5_val_v1",
            "full_ct_v41_excluded_v1": "full_ct_v5_excluded_v1",
        },
    },
    {
        "source": "review_full_ct_errors.py",
        "target": "review_full_ct_errors_v5.py",
        "changes": {
            "full_ct_v41_val_v1": "full_ct_v5_val_v1",
            "full_ct_v41_error_review": "full_ct_v5_error_review",
        },
    },
]

# 先检查全部输入及输出，任何异常都不写入文件
prepared = []

for task in tasks:
    source = SCRIPTS / task["source"]
    target = SCRIPTS / task["target"]

    if not source.is_file():
        raise FileNotFoundError(source)

    if target.exists():
        raise RuntimeError(
            f"目标脚本已存在，禁止覆盖：{target}"
        )

    code = source.read_text(encoding="utf-8-sig")

    for old, new in task["changes"].items():
        count = code.count(old)

        if count == 0:
            raise RuntimeError(
                f"{source.name} 未找到：{old}"
            )

        code = code.replace(old, new)

        print(f"{source.name}: {old} -> {new}")

    if "full_ct_v41" in code:
        raise RuntimeError(
            f"{source.name} 仍存在V4.1路径，请检查"
        )

    compile(code, str(target), "exec")
    prepared.append((target, code))

required = [
    RUNS / "full_ct_v5_val_v1" / "candidates_3d.csv",
    RUNS / "full_ct_v5_val_v1" / "cases.csv",
]

for path in required:
    if not path.is_file():
        raise FileNotFoundError(path)

for target, code in prepared:
    target.write_text(code, encoding="utf-8")
    print("已生成：", target)

print("\nV5分析脚本准备完成")
print("V4.1原始脚本未修改")
