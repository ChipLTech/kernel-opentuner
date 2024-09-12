import os
import re

opt_dim = ["MIScheduler", "PostRA-MIScheduler", "MachineSink", "MachineLICM", "RegCoalescer"]
dim_option = [
  ['topdown', 'bottomup', 'bidirectional'],
]

def get_kernel_path():
  tuner_path = os.path.dirname(os.path.abspath(__file__)) + "/../"
  if os.path.exists(tuner_path + "../DLC_Custom_Kernel/"):
    return tuner_path + "../DLC_Custom_Kernel/"
  else:
    raise SystemError("DLC_Custom_Kernel not found")
  
def get_policy_path():
  kernel_dir = get_kernel_path()
  return kernel_dir + "dlc_src/opt_flag_data/autotune_strategies.csv"

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
  for line in lines:
    if "fail" in line:
      test_pass = False
    if "xys0" in line and "xys1" in line:
      print(line)
      cycle += int(re.findall(r"xys0: \d+", line)[0].split()[-1])
      cycle += int(re.findall(r"xys1: \d+", line)[0].split()[-1])
  print("Total cycle: ", cycle)
  return cycle, test_pass