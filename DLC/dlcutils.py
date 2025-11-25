import os
import re
from llamatool import *
import yaml

class TuneRange:
  def __init__(self, min_value, max_value, is_int=True):
    self.min_value = min_value
    self.max_value = max_value
    self.is_int = is_int

opt_dim = ["MIScheduler", "PostRA-MIScheduler", "MachineSink", "MachineSink-slot", "MachineSink-chain", "MachineLICM", "RegCoalescer", "rotate", "condcmp"]
dim_option = {
  "MIScheduler" : ['topdown', 'bottomup', 'bidirectional'],
  "PostRA-MIScheduler" : ['topdown', 'bottomup', 'bidirectional'],
  "MachineSink" : ['pass', 'disable'],
  "MachineSink-slot" : ['pass', 'disable'],
  "MachineSink-chain" : ['pass', 'disable'],
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
  return ",,,,,1.0,all,-1.0,-1.0"

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
    return ["reshape_offset_bf16",
    "dropout_dlc_random_bf16",
    "log_softmax",
    "matmul_t_bf16_pingpong",
    "log_softmax_backward",
    "scale_masked_bf16",
    "scaled_dot_product_efficient_attention_backward_bf16",
    "permute_bf16",
    "FusedRoPEBack_bf16",
    "scaled_dot_product_efficient_attention_bf16",
    "foreach_add_tensor_bf16",
    "linalg_vector_norm_bf16",
    "FusedRMSNormBackward_bf16",
    "FusedRoPE_bf16",
    "FusedRMSNorm_bf16",
    "foreach_mul_scalar_bf16",
    "foreach_mul_bf16",
    "convert_element_type_16bit",
    "fused_adamw_bf16",
    "nll_loss_backward",
    "convert_element_type_32bit",
    "nll_loss",
    "embedding_dense_bf16",
    "silu_tensor_bf16",
    "silu_backward_tensor_bf16",
    "RotaryPosEmb_f32",
    "foreach_add_scalar",
    "slice_long",
    "cat_tensorlist_bf16_pingpong",
    "full",
    "copy_memory",
    "mean_dim",
    "strided_set_scalar",
    "foreach_div_f32_i64",
    "foreach_add_tensor",
    "constantPadNd_i64",
    "foreach_div_scalar",
    "sum_intList_bool",
    "ne_Tensor_out",
    "ne_Scalar_out_int64",
    "eq_Scalar_out_int64",
    "all_all_out",
    "foreach_add_scalar_i64",
    "clamp_out_scalar_bf16",
    "isinf",
    "foreach_add_scalar_bf16",
    "reciprocal_bf16",
    "arange_int64"]


def get_gemma_kernels():
    return ["matmul_t_bf16_pingpong",
    "gelu_backward_tensor_bf16",
    "gelu_tensor_bf16",
    "log_softmax",
    "linalg_vector_norm_bf16",
    "foreach_mul_bf16",
    "dropout_dlc_random_bf16",
    "log_softmax_backward",
    "fused_adamw_bf16",
    "foreach_add_tensor_bf16",
    "permute_bf16",
    "scaled_dot_product_efficient_attention_backward_bf16",
    "FusedRoPEBack_bf16",
    "foreach_mul_scalar_bf16",
    "scale_masked_bf16",
    "foreach_mul",
    "scaled_dot_product_efficient_attention_bf16",
    "FusedRoPE_bf16",
    "convert_element_type_16bit",
    "convert_element_type_32bit",
    "mean_dim_reduce",
    "foreach_add_tensor",
    "pow_tensor_scalar",
    "nll_loss_backward",
    "sum_intList",
    "expand",
    "foreach_mul_scalar",
    "foreach_div_scalar",
    "copy_memory",
    "nll_loss",
    "foreach_add_scalar",
    "cat_tensorlist_bf16_pingpong",
    "embedding_dense_bf16",
    "rsqrt",
    "full",
    "cos_f32",
    "sin_tensor",
    "permute",
    "strided_set_scalar",
    "slice_long",
    "mean_dim",
    "foreach_div_f32_i64",
    "bmm_f32",
    "constantPadNd_i64",
    "cat_tensorlist_pingpong",
    "sum_intList_bool",
    "convert_element_type_64bit",
    "ne_Tensor_out",
    "ne_Scalar_out_int64",
    "eq_Scalar_out_int64",
    "all_all_out",
    "foreach_add_scalar_i64",
    "clamp_out_scalar_bf16",
    "isinf",
    "foreach_add_scalar_bf16",
    "reciprocal_bf16",
    "arange_int64"]


def get_llama_kernels():
    return ["matmul_t_bf16_pingpong",
    "dropout_dlc_random_bf16",
    "scaled_dot_product_efficient_attention_backward_bf16",
    "permute_bf16",
    "scaled_dot_product_efficient_attention_bf16",
    "FusedRoPEBack_bf16",
    "foreach_add_tensor_bf16",
    "linalg_vector_norm_bf16",
    "foreach_mul_bf16",
    "scale_masked_bf16",
    "FusedRMSNormBackward_bf16",
    "fused_adamw_bf16",
    "foreach_mul_scalar_bf16",
    "reshape_offset_bf16",
    "silu_tensor_bf16",
    "FusedRoPE_bf16",
    "silu_backward_tensor_bf16",
    "log_softmax",
    "log_softmax_backward",
    "FusedRMSNorm_bf16",
    "convert_element_type_16bit",
    "foreach_add_scalar",
    "nll_loss_backward",
    "embedding_dense_bf16",
    "convert_element_type_32bit",
    "cat_tensorlist_bf16_pingpong",
    "nll_loss",
    "full",
    "RotaryPosEmb_f32",
    "copy_memory",
    "slice_long",
    "strided_set_scalar",
    "mean_dim",
    "foreach_div_f32_i64",
    "foreach_add_tensor",
    "constantPadNd_i64",
    "foreach_div_scalar",
    "sum_intList_bool",
    "ne_Tensor_out",
    "ne_Scalar_out_int64",
    "eq_Scalar_out_int64",
    "all_all_out",
    "foreach_add_scalar_i64",
    "clamp_out_scalar_bf16",
    "isinf",
    "foreach_add_scalar_bf16",
    "reciprocal_bf16",
    "arange_int64"]


def get_deepseek_qwen_7b_kernels():
    return ["matmul_t_bf16_pingpong",
    "sum_intList_bf16",
    "scaled_dot_product_efficient_attention_backward_bf16",
    "dropout_dlc_random_bf16",
    "log_softmax",
    "log_softmax_backward",
    "scaled_dot_product_efficient_attention_bf16",
    "foreach_add_tensor_bf16",
    "foreach_mul_bf16",
    "silu_tensor_bf16",
    "FusedRoPEBack_bf16",
    "scale_masked_bf16",
    "permute_bf16",
    "foreach_mul_scalar_bf16",
    "silu_backward_tensor_bf16",
    "linalg_vector_norm_bf16",
    "foreach_mul",
    "addmm_bf16_pingpong",
    "fused_adamw_bf16",
    "expand_bf16",
    "FusedRoPE_bf16",
    "convert_element_type_16bit",
    "reshape_offset_bf16",
    "mean_dim_reduce",
    "convert_element_type_32bit",
    "pow_tensor_scalar",
    "foreach_add_tensor",
    "copy_memory",
    "expand",
    "sum_intList",
    "foreach_mul_scalar",
    "foreach_div_scalar",
    "nll_loss_backward",
    "nll_loss",
    "full",
    "foreach_add_scalar",
    "embedding_dense_bf16",
    "rsqrt",
    "cat_tensorlist_bf16_pingpong",
    "cos_f32",
    "sin_tensor",
    "slice_long",
    "permute",
    "strided_set_scalar",
    "mean_dim",
    "bmm_f32",
    "cat_tensorlist_pingpong",
    "foreach_div_f32_i64",
    "constantPadNd_i64",
    "sum_intList_bool",
    "ne_Tensor_out",
    "convert_element_type_64bit",
    "ne_Scalar_out_int64",
    "eq_Scalar_out_int64",
    "all_all_out",
    "foreach_add_scalar_i64",
    "clamp_out_scalar_bf16",
    "foreach_add_scalar_bf16",
    "isinf",
    "reciprocal_bf16",
    "arange_int64"]
