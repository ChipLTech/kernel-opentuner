#!/usr/bin/env python3
import os
import sys
import subprocess
import ast
import re
import csv
from datetime import datetime
from pathlib import Path

# 模型名和 dlcutils.py 中对应的方法名映射
MODEL_METHOD_MAP = {
    "llama": "get_llama_kernels",
    "tinyllama": "get_tinyllama_kernels",
    "gemma": "get_gemma_kernels",
    "deepseek_qwen_7b": "get_deepseek_qwen_7b_kernels",
}

if len(sys.argv) != 2:
    print("Usage: python3 update_kernels.py <model_name>")
    sys.exit(1)

model_name = sys.argv[1].lower()
if model_name not in MODEL_METHOD_MAP:
    print(f"Unknown model name: {model_name}")
    sys.exit(1)

method_name = MODEL_METHOD_MAP[model_name]

# 1. 找到最新的 .ansi 文件
try:
    latest_file = subprocess.check_output(
        "ls -1t /tmp/syn/*.ansi | head -n 1", shell=True, text=True
    ).strip()
except subprocess.CalledProcessError:
    print("No ANSI files found in /tmp/syn/")
    latest_file = None

if latest_file:
    print(f"Latest ANSI file: {latest_file}")
else:
    print("Skipping kernel extraction: no ANSI file found.")
    latest_file = None

kernel_dict = {}
total_cycles = None
cycles_value = None

# 2. 运行 tool.py 获取日志（逐行捕获，清理 ANSI）
if latest_file:
    try:
        proc = subprocess.Popen(
            ["python3", "tool.py", latest_file],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=os.getcwd(),  # 可修改为 tool.py 所在目录
        )

        output_lines = []
        for line in proc.stdout:
            # 去掉 ANSI 颜色码和回车覆盖字符
            line_clean = re.sub(r'\x1B\[[0-?]*[ -/]*[@-~]', '', line)
            line_clean = line_clean.replace('\r', '')  # 处理进度条覆盖
            output_lines.append(line_clean)

        proc.wait()
        output_clean = "".join(output_lines)

        # 尝试解析 kernel_to_cycle 和 total_cycles
        kernel_to_cycle_match = re.search(r"kernel_to_cycle:\s*(dict_keys\((.*?)\))", output_clean, re.DOTALL)
        total_cycles_match = re.search(r"total_cycles:\s*([\d,]+)", output_clean)

        if kernel_to_cycle_match:
            # 转成列表
            keys_str = kernel_to_cycle_match.group(2)
            # ast.literal_eval 解析成 list
            kernel_dict = ast.literal_eval(keys_str)
        else:
            print("Warning: Failed to parse kernel_to_cycle from tool.py output")
            print("=== tool.py output ===")
            print(output_clean)

        if total_cycles_match:
            total_cycles = total_cycles_match.group(1).replace(",", "")
        else:
            print("Warning: Failed to parse total_cycles from tool.py output")

    except Exception as e:
        print(f"Error running tool.py: {e}")

# 3. 保存算子到 txt 文件
txt_dir = "/mnt/jfs/ci-dingtalk/autotune"
os.makedirs(txt_dir, exist_ok=True)
kernels_file = os.path.join(txt_dir, f"{model_name}_kernels.txt")
cycles_file = os.path.join(txt_dir, f"{model_name}_cycles.txt")

with open(kernels_file, "w") as f:
    for k in kernel_dict:
        f.write(k + "\n")

if total_cycles:
    try:
        cycles_value = int(total_cycles)
    except ValueError:
        cycles_value = None

    with open(cycles_file, "w") as f:
        f.write(str(total_cycles) + "\n")

    # 追加写入 baseline 历史
    if cycles_value is not None:
        baseline_csv = os.path.join(txt_dir, f"{model_name}_baseline_history.csv")
        date_str = datetime.now().strftime("%Y-%m-%d")
        file_exists = os.path.exists(baseline_csv)
        with open(baseline_csv, "a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            if not file_exists:
                writer.writerow(["date", "cycles"])
            writer.writerow([date_str, cycles_value])

print(f"Saved kernels to {kernels_file}")
if total_cycles:
    print(f"Total cycles: {total_cycles}")
else:
    print(f"No total_cycles available to save to {cycles_file}")

# 4. 更新 dlcutils.py 中对应方法
dlcutils_path = "/home/runner/_work/kernel-opentuner/kernel-opentuner/DLC/dlcutils.py"
if os.path.exists(dlcutils_path) and kernel_dict:
    with open(dlcutils_path, "r") as f:
        dlc_content = f.read()

    pattern = rf"(def {method_name}\(\):\s*return\s*\[).*?(\])"
    new_kernels_str = ",\n    ".join([f'"{k}"' for k in kernel_dict])
    new_content = re.sub(pattern, rf"\1{new_kernels_str}\2", dlc_content, flags=re.DOTALL)

    with open(dlcutils_path, "w") as f:
        f.write(new_content)

    print(f"Updated {method_name} in {dlcutils_path}")
else:
    print(f"Skipping dlcutils.py update: file not found or kernel list empty")

# 5. 清理 /tmp/syn 下遗留的 .ansi 文件，免得占满磁盘
ansi_dir = Path("/tmp/syn")
if ansi_dir.exists():
    removed = 0
    for ansi_file in ansi_dir.glob("*.ansi"):
        try:
            ansi_file.unlink()
            removed += 1
        except OSError as exc:
            print(f"Warning: failed to remove {ansi_file}: {exc}")
    if removed:
        print(f"Removed {removed} ANSI log(s) from {ansi_dir}")
