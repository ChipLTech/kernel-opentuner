#!/usr/bin/env python
#
# Optimize blocksize of apps/mmm_block.cpp
#
# This is an extremely simplified version meant only for tutorials
#
from __future__ import print_function

import opentuner
from opentuner import ConfigurationManipulator
from opentuner import IntegerParameter
from opentuner import FloatParameter
from opentuner import MeasurementInterface
from opentuner import EnumParameter
from opentuner import Result
from opentuner.search.driver import SearchDriver

import argparse
from multiprocessing.pool import ThreadPool
import multiprocessing
from dlcutils import *
from ctypes import c_int
import copy
import os
import signal


parser = argparse.ArgumentParser(parents=opentuner.argparsers())
parser.add_argument('--kernel', help='kernel name to tune')

# sync across processes
compile_ready_count = multiprocessing.Value(c_int, 0)
compile_ready_count_lock = multiprocessing.Lock()
test_ready_count = multiprocessing.Value(c_int, 0)
test_ready_count_lock = multiprocessing.Lock()

class KernelFlagsTuner(MeasurementInterface):
  def __init__(self, *pargs, **kwargs):
    super(KernelFlagsTuner, self).__init__(program_name=args.kernel, *pargs,
                                            **kwargs)
    self.kernel_name = pargs[0].kernel
    self.total_kernel = pargs[0].total_kernel
    self.is_executor = pargs[0].is_executor
    self.log_path = pargs[0].log_path
    self.best_res = pargs[0].best_res
    self.best_cycle = 2 ** 32
    self.stragegy_path = get_policy_path()
    self.line_number, content = get_line_number(self.stragegy_path, self.kernel_name)
    info = ""
    if self.line_number == -1:
      compile_ready_count_lock.acquire()
      self.line_number = len(open(self.stragegy_path, 'r').readlines())
      content = self.kernel_name + "," + get_default_policy() + "\n"
      info += "New kernel found, add to the end of the file\n"
      info += "line number: " + str(self.line_number) + "\n"
      with open(self.stragegy_path, 'a') as file:
        file.write(content)
      compile_ready_count_lock.release()
    self.opt_flag = get_flag_dict(content.strip().split(',')[1:])
    # to record if it's tested
    self.option_record = {}
    for key in dim_option.keys():
      self.option_record[key] = {}
      if isinstance(dim_option[key], TuneRange):
        self.option_record[key] = []
      else:
        for option in dim_option[key]:
          self.option_record[key][option] = 0
    info += "Kernel name: " + self.kernel_name + "\n"
    info += "Line number: " + str(self.line_number) + "\n"
    info += "Optimization flags: " + str(self.opt_flag) + "\n"
    print(info)
    # compare to current setting
    self.old_flag = self.opt_flag.copy()
    self.old_performance = 0
    self.old_better = True

  def manipulator(self):
    """
    Define the search space by creating a
    ConfigurationManipulator
    """
    manipulator = ConfigurationManipulator()
    for key in dim_option.keys():
      if isinstance(dim_option[key], TuneRange):
        if dim_option[key].is_int:
          manipulator.add_parameter(
            IntegerParameter(key, dim_option[key].min_value, dim_option[key].max_value))
        else:
          manipulator.add_parameter(
            FloatParameter(key, dim_option[key].min_value, dim_option[key].max_value))
      else:              
        manipulator.add_parameter(
          EnumParameter(key, dim_option[key]))
    return manipulator

  def compile(self, cfg, id):
      """
      Compile a given configuration in parallel
      """
      # print("Compiling with configuration: ", cfg)
      self.set_opt_flag(cfg)
      compile_ready_count_lock.acquire()
      change_policy_file(self.line_number, self.kernel_name + "," + ",".join([str(self.opt_flag[key]) for key in opt_dim]) + "\n")
      compile_ready_count.value += 1
      compile_ready_count_lock.release()
      # only one thread is allowed to compile
      if self.is_executor:
        build_dir = get_kernel_path() + "build/"
        cmake_cmd = 'cmake -G Ninja -S {0} -B {1}'.format(get_kernel_path(), build_dir)
        while compile_ready_count.value != self.total_kernel:
          pass
        cmake_res = self.call_program(cmake_cmd)
        assert cmake_res['returncode'] == 0
        print("CMake finished")
        ninja_cmd = 'ninja -C {0} '.format(build_dir)
        ninja_res = self.call_program(ninja_cmd)
        print("Build finished")
        assert ninja_res['returncode'] == 0
        # inform all threads that the compile is done
        compile_ready_count_lock.acquire()
        compile_ready_count.value = 0
        compile_ready_count_lock.release()
      else:
        while compile_ready_count.value != 0:
          pass
      return {'returncode': 0, 'stdout': '', 'stderr': '', 'timeout': False, 'time': 0.1}

  def run_precompiled(self, desired_result, input, limit, compile_result, id):
    """
    Run a compile_result from compile() sequentially and return performance
    """
    run_cmd = get_kernel_path() + "build/syntests/syntests -t " + self.kernel_name
    print(self.get_prefix(), "Start to run the kernel")
    run_result = self.call_program(run_cmd)
    test_ready_count_lock.acquire()
    test_ready_count.value += 1
    print(self.get_prefix(), "Kernel run finished, total run: ", test_ready_count.value)
    test_ready_count_lock.release()
    cycle, succ, result_lines = diagnose_run_result(run_result['stderr'].decode().split('\n'))
    assert succ
    if self.old_better and cycle < self.old_performance:
      self.old_better = False
    if cycle < self.best_cycle:
      self.best_cycle = cycle
    if result_lines:
      print(result_lines)
    else:
      print(run_result['stderr'].decode())
      cycle = 2 ** 32
    print(self.get_prefix(), "Number of cycles: ", cycle)
    
    # keep another record in log file
    with open(self.log_path, 'a') as f:
      f.write(self.kernel_name + "," + ",".join([str(self.opt_flag[key]) for key in opt_dim]) + "," + str(cycle) + "\n")
      f.write(result_lines)
      f.write(self.get_prefix() + "Total run: " + str(cycle) + " cycles\n")

    return Result(time = cycle)

  def compile_and_run(self, desired_result, input, limit):
    """
    Compile and run a given configuration then
    return performance
    """
    cfg = desired_result.configuration.data
    compile_result = self.compile(cfg, 0)
    return self.run_precompiled(desired_result, input, limit, compile_result, 0)
  
  def extra_convergence_criteria(self, result):
    for res in result:
      self.option_record[opt_dim[0]][res.configuration.data[opt_dim[0]]] = 1
    for key in self.option_record:
      for option in self.option_record[key]:
        if self.option_record[key][option] == 0:
          return False
    return True
  
  def pre_process(self):
    # we need to record the current perfmance
    if self.old_performance == 0:
      print(self.get_prefix() + "Testing current setting")
      self.old_performance = self.run_precompiled(Result(time = 0), None, 0, 0, 0).time
      self.best_cycle = self.old_performance
      if self.is_executor:
        while test_ready_count.value != self.total_kernel:
          pass
        test_ready_count_lock.acquire()
        test_ready_count.value = 0
        test_ready_count_lock.release()
      else:
        while test_ready_count.value != 0:
          pass
      
  def post_process(self):
    if self.is_executor:
      while test_ready_count.value != self.total_kernel:
        pass
      test_ready_count_lock.acquire()
      test_ready_count.value = 0
      test_ready_count_lock.release()
    else:
      while test_ready_count.value != 0:
        pass
  
  def save_final_config(self, configuration):
    """called at the end of tuning"""
    if self.old_better:
      print(self.get_prefix(), "The original setting is better")
      self.opt_flag = self.old_flag.copy()
    else:
      print(self.get_prefix(), "Optimal compiling flag is:", configuration.data)
      for key in configuration.data:
        self.opt_flag[key] = configuration.data[key]
    with open(self.best_res, 'w') as f:
      if self.old_better:
        f.write("Original setting is better\n")
      else:
        f.write("Find a better setting\n")
      f.write("The best setting is: " + ",".join([str(self.opt_flag[key]) for key in opt_dim]) + "\n")
      f.write("The performance is: " + str(self.best_cycle) + "\n")
    compile_ready_count_lock.acquire()
    change_policy_file(self.line_number, self.kernel_name + "," + ",".join([str(self.opt_flag[key]) for key in opt_dim]) + "\n")
    compile_ready_count_lock.release()
    
  def set_opt_flag(self, configuration):
    for key in configuration:
      if key != "MachineLICM" and key != "RegCoalescer_0" and key != "RegCoalescer_1":
        self.opt_flag[key] = configuration[key]
    licm_value = configuration["MachineLICM"]
    if licm_value == 0:
      self.opt_flag["MachineLICM"] = "0.0"
    elif licm_value == 1:
      self.opt_flag["MachineLICM"] = ""
    else:
      self.opt_flag["MachineLICM"] = licm_value
      
    reg_coalescer_0 = configuration["RegCoalescer_0"]
    reg_coalescer_1 = configuration["RegCoalescer_1"]
    if reg_coalescer_0 == 0 or reg_coalescer_1 == 0:
      self.opt_flag["RegCoalescer"] = "no"
    else:
      self.opt_flag["RegCoalescer"] = str(reg_coalescer_0) + "_and_" + str(reg_coalescer_1)
    print(self.get_prefix(), "Optimization flags set to: \n", self.opt_flag)

  def get_prefix(self):
    return "[" + self.kernel_name + "]"


class MultiKernelTuner():
  def __init__(self, *pargs, **kwargs):
    self.kernel_names = pargs[0].kernel.split(',')
    self.thread_pool = ThreadPool(len(self.kernel_names))
    self.kernel_params = []
    self.db_path = pargs[0].database
    print(len(self.kernel_names), "kernels to tune")
      
    for i in range(len(self.kernel_names)):
      kernel_name = self.kernel_names[i]
      single_parg = copy.copy(pargs[0])
      single_parg.kernel = kernel_name
      single_parg.database = self.db_path + "/" + kernel_name + ".db"
      single_parg.log_path = self.db_path + "/" + kernel_name + "_log.txt"
      single_parg.best_res = self.db_path + "/" + kernel_name + "_best.txt"
      single_parg.total_kernel = len(self.kernel_names)
      if i == 0:
        single_parg.is_executor = True
      else:
        single_parg.is_executor = False
      os.system("touch " + single_parg.database)
      os.system("touch " + single_parg.log_path)
      os.system("touch " + single_parg.best_res)
      self.kernel_params.append(single_parg)
      
  def main(self):
    self.thread_pool.map(KernelFlagsTuner.main, self.kernel_params)
    self.thread_pool.close()      
    
def signal_handler(self, sig):
  print(self.get_prefix(), "Caught signal", sig, "write the original setting back")
  with open(get_policy_path(), 'w') as file:
    file.writelines(original_setting)

if __name__ == '__main__':
  args = parser.parse_args()
  args.parallelism = 1
  args.test_limit = 12
  args.stop_after = 3 * 60 # 3min
  with open(get_policy_path(), 'r') as file:
    original_setting = file.readlines()
  signal.signal(signal.SIGINT, signal_handler)
  # KernelFlagsTuner.main(args)
  tuner = MultiKernelTuner(args)
  tuner.main()