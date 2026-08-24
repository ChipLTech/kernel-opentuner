#!/usr/bin/env python3

import argparse
import json
import os
import subprocess
from pathlib import Path


DEFAULT_CONFIG = Path(
    "/home/runner/_work/vllm-cl/vllm-cl/ci_utils/benchmark_models_list.json"
)


def load_model_config(config_path, model_name):
  with Path(config_path).open() as config_file:
    models = json.load(config_file)

  for model in models:
    if model.get("model_name") == model_name:
      return model
  raise ValueError("Model config not found: " + model_name)


def build_command(model):
  required = ("model_path", "dataset")
  missing = [key for key in required if not model.get(key)]
  if missing:
    raise ValueError("Missing model config fields: " + ", ".join(missing))

  dtype = str(model.get("dtype") or "bfloat16")
  block_size = model.get("block_size")
  if not block_size:
    block_size = "128" if dtype in ("float", "float32") else "256"

  command = [
      "vllm", "bench", "throughput",
      "--model", str(model["model_path"]),
      "--dataset", str(model["dataset"]),
      "--generation-config", "auto",
      "--override-generation-config",
      json.dumps({"temperature": float(model.get("temperature") or 0)}),
      "--enable-chunked-prefill",
      "--max-num-batched-tokens", "1024",
      "--gpu-memory-utilization", "0.95",
      "--enforce-eager",
      "--dtype", dtype,
      "--block-size", str(block_size),
      "--num-prompts", str(model.get("num_prompts") or 32),
      "--output-len", str(model.get("output_len") or 4),
      "--seed", "1024",
  ]

  tpu_num = int(model.get("tpu_num") or 1)
  if tpu_num != 1:
    command.extend(["-tp", str(tpu_num)])
  return command


def run_profile(model_name, config_path, profile_dir, device):
  model = load_model_config(config_path, model_name)
  command = build_command(model)
  profile_dir = Path(profile_dir)
  profile_dir.mkdir(parents=True, exist_ok=True)

  run_env = os.environ.copy()
  run_env.update({
      "DLC_VISIBLE_DEVICES": device,
      "VLLM_USE_DLC_COL_MAJOR_MATMUL": str(model.get("use_col_major") or 1),
      "DLC_SYN_DEBUG": "1",
      "DLC_SYN_VERBOSE": "1",
      "DLC_SYN_PROF_CYCLE": "1",
      "DLC_SYN_LOG_DIR": str(profile_dir),
  })

  print("Model config:", config_path)
  print("Profile model:", model_name)
  print("DLC_VISIBLE_DEVICES:", device)
  print("Profile directory:", profile_dir)
  print("Command:", " ".join(command))

  output = ""
  with subprocess.Popen(
      command,
      stdout=subprocess.PIPE,
      stderr=subprocess.STDOUT,
      bufsize=1,
      universal_newlines=True,
      env=run_env,
  ) as process:
    for line in process.stdout:
      print(line, end="")
      output += line

  if process.returncode != 0:
    raise RuntimeError(
        "Model command failed with exit code " + str(process.returncode)
    )
  return output


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--model-name", required=True)
  parser.add_argument(
      "--config",
      default=os.environ.get("VLLM_MODEL_CONFIG", str(DEFAULT_CONFIG)),
  )
  parser.add_argument("--profile-dir", required=True)
  parser.add_argument(
      "--device",
      default=os.environ.get("DLC_VISIBLE_DEVICES", "0"),
  )
  args = parser.parse_args()

  run_profile(
      model_name=args.model_name,
      config_path=args.config,
      profile_dir=args.profile_dir,
      device=args.device,
  )


if __name__ == "__main__":
  main()
