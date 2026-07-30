#!/usr/bin/env python3
import csv
import os
import re
import sys
from datetime import datetime
from pathlib import Path

from llamatool import diagnose_llama_result


# Model name to the corresponding method in dlcutils.py.
MODEL_METHOD_MAP = {
    "llama": "get_llama_kernels",
    "tinyllama": "get_tinyllama_kernels",
    "gemma": "get_gemma_kernels",
    "deepseek_qwen_7b": "get_deepseek_qwen_7b_kernels",
    "deepseek_llama_8b": "get_deepseek_llama_8b_kernels",
}


def get_latest_ansi(profile_dir):
    ansi_files = list(profile_dir.glob("*.ansi"))
    if not ansi_files:
        raise FileNotFoundError(f"No ANSI files found in {profile_dir}")
    return max(ansi_files, key=lambda path: path.stat().st_mtime)


def parse_profile(profile_path):
    profile_text = profile_path.read_text(errors="ignore")
    kernel_to_cycle, total_cycles = diagnose_llama_result(profile_text)
    if not kernel_to_cycle or total_cycles <= 0:
        raise RuntimeError(f"No kernel cycles parsed from {profile_path}")
    return list(kernel_to_cycle), total_cycles


def update_dlcutils(dlcutils_path, method_name, kernels):
    dlc_content = dlcutils_path.read_text()
    pattern = rf"(def {re.escape(method_name)}\(\):\s*return\s*\[).*?(\])"
    new_kernels = ",\n    ".join(f'"{kernel}"' for kernel in kernels)
    new_content, replacements = re.subn(
        pattern,
        rf"\1{new_kernels}\2",
        dlc_content,
        count=1,
        flags=re.DOTALL,
    )
    if replacements != 1:
        raise RuntimeError(f"Failed to locate {method_name} in {dlcutils_path}")
    dlcutils_path.write_text(new_content)


def main():
    if len(sys.argv) != 2:
        print("Usage: python3 save_kernels.py <model_name>")
        return 1

    model_name = sys.argv[1].lower()
    if model_name not in MODEL_METHOD_MAP:
        print(f"Unknown model name: {model_name}")
        return 1

    profile_dir = Path(os.environ.get("DLC_SYN_LOG_DIR", "/tmp/syn"))
    output_dir = Path(os.environ.get("AUTOTUNE_OUTPUT_DIR", "/home/CI/autotune"))
    dlcutils_path = Path(__file__).resolve().with_name("dlcutils.py")

    try:
        latest_file = get_latest_ansi(profile_dir)
        print(f"Latest ANSI file: {latest_file}")
        kernels, total_cycles = parse_profile(latest_file)

        output_dir.mkdir(parents=True, exist_ok=True)
        kernels_file = output_dir / f"{model_name}_kernels.txt"
        cycles_file = output_dir / f"{model_name}_cycles.txt"
        kernels_file.write_text("".join(f"{kernel}\n" for kernel in kernels))
        cycles_file.write_text(f"{total_cycles}\n")

        baseline_csv = output_dir / f"{model_name}_baseline_history.csv"
        file_exists = baseline_csv.exists()
        with baseline_csv.open("a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            if not file_exists:
                writer.writerow(["date", "cycles"])
            writer.writerow([datetime.now().strftime("%Y-%m-%d"), total_cycles])

        update_dlcutils(
            dlcutils_path,
            MODEL_METHOD_MAP[model_name],
            kernels,
        )
    except Exception as exc:
        print(f"Error: {exc}")
        return 1

    print(f"Saved kernels to {kernels_file}")
    print(f"Total cycles: {total_cycles}")
    print(f"Updated {MODEL_METHOD_MAP[model_name]} in {dlcutils_path}")

    removed = 0
    for ansi_file in profile_dir.glob("*.ansi"):
        try:
            ansi_file.unlink()
            removed += 1
        except OSError as exc:
            print(f"Warning: failed to remove {ansi_file}: {exc}")
    if removed:
        print(f"Removed {removed} ANSI log(s) from {profile_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
