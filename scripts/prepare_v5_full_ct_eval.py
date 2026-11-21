from pathlib import Path

project = Path(r"D:\肺结节模型训练")
source = project / "scripts" / "evaluate_full_ct_v41.py"
target = project / "scripts" / "evaluate_full_ct_v5.py"

old_weight = "yolo_v41_baseline"
new_weight = "yolo_v5_finetune_v1"

old_output = "full_ct_v41_val_v1"
new_output = "full_ct_v5_val_v1"

if target.exists():
    raise RuntimeError(f"文件已经存在，停止覆盖：{target}")

if not source.is_file():
    raise FileNotFoundError(source)

code = source.read_text(encoding="utf-8-sig")

if old_weight not in code:
    raise RuntimeError("没有找到V4.1权重路径，请检查原脚本")

if old_output not in code:
    raise RuntimeError("没有找到V4.1输出目录，请检查原脚本")

code = code.replace(old_weight, new_weight)
code = code.replace(old_output, new_output)

compile(code, str(target), "exec")
target.write_text(code, encoding="utf-8")

print("V5完整CT评估脚本已生成：", target)
print("使用权重：", new_weight)
print("输出目录：", new_output)
print("原V4.1评估脚本未修改")
