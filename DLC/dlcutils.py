import os
import re
from llamatool import *
import yaml

class TuneRange:
  def __init__(self, min_value, max_value, is_int=True):
    self.min_value = min_value
    self.max_value = max_value
    self.is_int = is_int

opt_dim = ["MIScheduler", "PostRA-MIScheduler", "MachineSink", "MachineLICM", "RegCoalescer"]
dim_option = {
  "MIScheduler" : ['topdown', 'bottomup', 'bidirectional'],
  "PostRA-MIScheduler" : ['topdown', 'bottomup', 'bidirectional'],
  "MachineSink" : ['pass', 'disable'],
  "MachineLICM" : TuneRange(0, 1, False),
  "RegCoalescer_0" : TuneRange(0, 200),
  "RegCoalescer_1" : TuneRange(0, 512),
}

def get_kernel_path():
  tuner_path = os.path.dirname(os.path.abspath(__file__)) + "/../"
  if os.path.exists(tuner_path + "../DLC_Custom_Kernel/"):
    return tuner_path + "../DLC_Custom_Kernel/"
  else:
    raise SystemError("DLC_Custom_Kernel not found")
  
def get_policy_path():
  kernel_dir = get_kernel_path()
  return kernel_dir + "dlc_src/opt_flag_data/autotune_strategies.csv"

def get_default_policy():
  return ",,,1.0,all"

def get_line_number(file_path, kernel_name):
  with open(file_path, 'r') as file:
    for i, line in enumerate(file):
      if kernel_name == line.split(',')[0]:
        return i, line
  # kernel not seen before
  return -1, ""

def get_flag_dict(flags):
  flag_dict = {}
  for i in range(len(opt_dim)):
    flag_dict[opt_dim[i]] = flags[i]
  return flag_dict

def change_policy_file(line_number, new_line):
  with open(get_policy_path(), 'r') as file:
    data = file.readlines()
  data[line_number] = new_line
  with open(get_policy_path(), 'w') as file:
    file.writelines(data)
    
def diagnose_run_result(lines):
  test_pass = True
  cycle = 0
  result_lines = ""
  for line in lines:
    if "fail" in line:
      test_pass = False
    if "xys0" in line and "xys1" in line:
      result_lines += line + "\n"
      cycle += int(re.findall(r"xys0: \d+", line)[0].split()[-1])
      cycle += int(re.findall(r"xys1: \d+", line)[0].split()[-1])
  return cycle, test_pass, result_lines

def get_most_recent_log_dir(log_dir):
  log_dirs = [log_dir + d for d in os.listdir(log_dir) if os.path.isdir(log_dir + d)]
  if not len(log_dirs):
    return ""
  else:
    log_dirs.sort(key=lambda x: os.path.getmtime(x))
    return log_dirs[-1]
  
def diagnose_llama_result(text):
  kernel_launches_name_body = [get_kernel_info(remove_ansi(kernel)) for kernel in get_kernel_launches(text)]
  kernel_names = [x[0] for x in kernel_launches_name_body]
  kernel_cycles = get_kernel_cycles(text)
  total_cycles = sum(kernel_cycles) + 1

  if len(kernel_cycles) != len(kernel_names):
      print('Warning: kernel names and cycles length mismatch')
      print('kernel length:', len(kernel_names))
      print('cycles length:', len(kernel_cycles))
      kernel_cycles = kernel_cycles[:len(kernel_names)]
      while len(kernel_cycles) < len(kernel_names):
          kernel_cycles.append(0)

  ############################################################################################################################
  kernel_name_cycles = {}
  for name, cycle in zip(kernel_names, kernel_cycles):
      if name not in kernel_name_cycles:
          kernel_name_cycles[name] = 0
      kernel_name_cycles[name] += int(cycle)
      # kernel_name_cycles[name][1] += 1
  return get_kernel_to_cycle(kernel_name_cycles, get_register_name_to_kernel()), total_cycles

def get_register_name_to_kernel():
  with open(get_kernel_path() + 'dlc_src/kernel_info.yaml') as f:
    config = yaml.safe_load(f)
  name_to_kernel = {}
  for item in config:
    if 'name' in item and 'src' in item:
      src = item['src']
      if isinstance(src, list):
        src = src[0].split('.')[0]
      name_to_kernel[item['name']] = src
  return name_to_kernel

def get_kernel_to_cycle(res, name_to_kernel):
  kernel_to_cycle = {}
  for item in res.keys():
    if item not in name_to_kernel.keys():
      kernel = item.split('custom_')[1]
    else:
      kernel = name_to_kernel[item]
    if kernel in kernel_to_cycle.keys():
      kernel_to_cycle[kernel] += res[item]
    else:
      kernel_to_cycle[kernel] = res[item]
  return kernel_to_cycle

def get_llama_path():
  return "/root/llama2-sft/"

def get_llama_kernels():
  return ['embedding_dense', 'arange_int64', 'eq_Scalar_out_int64', 'all_all_out', 'FusedRMSNorm', 'matmul_t', 'dropout_dlc_random',\
    'foreach_mul_scalar', '_foreach_add_tensor', 'permute', 'rotary_pos_emb_f32', 'FusedRoPE', 'scaled_dot_product_efficient_attention',\
    'silu', 'foreach_mul', 'slice', 'log_softmax', 'full', 'reshape_offset_int64', 'nll_loss', 'foreach_div_scalar', 'nll_loss_backward',\
    'log_softmax_backward', 'copy_memory', 'slice_backward', 'scale_masked', 'FusedRMSNormBackward', 'silu_backward', \
    'scaled_dot_product_efficient_attention_backward', 'FusedRoPEBack', 'ne_Tensor_out', 'abs', 'eq_Scalar_out', 'linalg_vector_norm', \
    'copy_stride_smem', 'cat_tensorlist', 'foreach_add_scalar', 'reciprocal', 'clamp_out_scalar', 'fused_adamw', 'mean_dim']