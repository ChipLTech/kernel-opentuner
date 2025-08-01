import os
import re
from llamatool import *
import yaml

class TuneRange:
  def __init__(self, min_value, max_value, is_int=True):
    self.min_value = min_value
    self.max_value = max_value
    self.is_int = is_int

opt_dim = ["MIScheduler", "PostRA-MIScheduler", "MachineSink", "MachineLICM", "RegCoalescer", "rotate", "condcmp"]
dim_option = {
  "MIScheduler" : ['topdown', 'bottomup', 'bidirectional'],
  "PostRA-MIScheduler" : ['topdown', 'bottomup', 'bidirectional'],
  "MachineSink" : ['pass', 'disable'],
  "MachineLICM" : TuneRange(0, 1, False),
  "RegCoalescer_0" : TuneRange(0, 200),
  "RegCoalescer_1" : TuneRange(0, 512),
  "rotate" : ['0.0', '1.0', '-1.0'],
  "condcmp" : ['0.0', '1.0', '-1.0'],
}

def get_kernel_path():
  # tuner_path = os.path.dirname(os.path.abspath(__file__)) + "/../"
  # if os.path.exists(tuner_path + "../DLC_Custom_Kernel/"):
  #   return tuner_path + "../DLC_Custom_Kernel/"
  # else:
  #   raise SystemError("DLC_Custom_Kernel not found")
  return "/home/test_ci/DLC_Custom_Kernel/DLC_Custom_Kernel/"

def get_policy_path():
  kernel_dir = get_kernel_path()
  return kernel_dir + "dlc_src/opt_flag_data/autotune_strategies.csv"

def get_default_policy():
  return ",,,1.0,all,-1.0,-1.0"

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

def get_most_recent_log_dir(log_dir):
  log_dirs = [log_dir + d for d in os.listdir(log_dir) if os.path.isdir(log_dir + d)]
  if not len(log_dirs):
    return ""
  else:
    log_dirs.sort(key=lambda x: os.path.getmtime(x))
    return log_dirs[-1]

def get_llama_path():
    return "/home/test_models/llama2-fine-tune"

def get_tinyllama_kernels():
    return [
        "reshape_offset",
        "matmul_t_pingpong",
        "log_softmax",
        "FusedRMSNormBackward_f32",
        "log_softmax_backward",
        "scaled_dot_product_efficient_attention",
        "dropout_dlc_random",
        "scale_masked",
        "scaled_dot_product_efficient_attention_backward",
        "linalg_vector_norm",
        "permute",
        "foreach_add_tensor",
        "FusedRoPEBack_f32",
        "foreach_mul_scalar",
        "foreach_mul",
        "FusedRoPE_f32",
        "nll_loss_backward",
        "fused_adamw",
        "convert_element_type_32bit",
        "FusedRMSNorm_f32",
        "silu_tensor",
        "full",
        "silu_backward_tensor",
        "slice_backward",
        "slice_tensor",
        "embedding_dense",
        "nll_loss",
        "RotaryPosEmb_f32",
        "foreach_add_scalar",
        "slice_long",
        "cat_tensorlist_pingpong",
        "mean_dim",
        "foreach_div_scalar",
        "ne_Tensor_out",
        "eq_Scalar_out_int64",
        "eq_Scalar_out",
        "clamp_out_scalar",
        "reciprocal",
        "all_all_out",
        "arange_int64",
        "abs",
    ]


def get_gemma_kernels():
    return [
        "matmul_t_pingpong",
        "gelu_backward_tensor",
        "linalg_vector_norm",
        "gelu_tensor",
        "foreach_mul",
        "foreach_add_tensor",
        "log_softmax",
        "foreach_mul_scalar",
        "dropout_dlc_random",
        "log_softmax_backward",
        "permute",
        "scaled_dot_product_efficient_attention",
        "scaled_dot_product_efficient_attention_backward",
        "scale_masked",
        "FusedRoPEBack_f32",
        "fused_adamw",
        "FusedRoPE_f32",
        "mean_dim_reduce",
        "convert_element_type_32bit",
        "sum_intList",
        "pow_tensor_scalar",
        "nll_loss_backward",
        "expand",
        "cos_f32",
        "sin_tensor",
        "full",
        "foreach_div_scalar",
        "slice_backward",
        "slice_tensor",
        "nll_loss",
        "foreach_add_scalar",
        "embedding_dense",
        "cat_tensorlist_pingpong",
        "bmm_f32",
        "rsqrt",
        "convert_element_type_64bit",
        "slice_long",
        "reshape_offset",
        "mean_dim",
        "ne_Tensor_out",
        "eq_Scalar_out_int64",
        "eq_Scalar_out",
        "clamp_out_scalar",
        "reciprocal",
        "all_all_out",
        "arange_int64",
        "abs",
    ]


def get_llama_kernels():
    return [
        "matmul_t_pingpong",
        "scaled_dot_product_efficient_attention",
        "scaled_dot_product_efficient_attention_backward",
        "FusedRMSNormBackward_f32",
        "linalg_vector_norm",
        "foreach_add_tensor",
        "dropout_dlc_random",
        "permute",
        "foreach_mul_scalar",
        "scale_masked",
        "FusedRoPEBack_f32",
        "foreach_mul",
        "silu_tensor",
        "silu_backward_tensor",
        "FusedRoPE_f32",
        "FusedRMSNorm_f32",
        "fused_adamw",
        "log_softmax",
        "log_softmax_backward",
        "convert_element_type_32bit",
        "nll_loss_backward",
        "embedding_dense",
        "full",
        "foreach_add_scalar",
        "slice_backward",
        "slice_tensor",
        "nll_loss",
        "cat_tensorlist_pingpong",
        "RotaryPosEmb_f32",
        "slice_long",
        "reshape_offset",
        "mean_dim",
        "foreach_div_scalar",
        "eq_Scalar_out_int64",
        "ne_Tensor_out",
        "eq_Scalar_out",
        "clamp_out_scalar",
        "reciprocal",
        "all_all_out",
        "arange_int64",
        "abs",
    ]
