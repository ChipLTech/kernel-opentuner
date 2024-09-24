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
from opentuner.tuningrunmain import TuningRunMain

import argparse
from multiprocessing.pool import ThreadPool
from dlcutils import *
import copy
import os

parser = argparse.ArgumentParser(parents=opentuner.argparsers())
parser.add_argument('--kernel', help='kernel name to tune')
max_concurrency = 20

class KernelParameter:
  def __init__(self, name, line_number, opt_flag):
    self.name = name
    self.line_number = line_number
    self.opt_flag = opt_flag
    # compare to current setting
    self.old_flag = self.opt_flag.copy()
    self.old_performance = 0
    self.old_better = True
    # to record if it's tested
    self.option_record = {}
    for key in dim_option.keys():
      self.option_record[key] = {}
      if isinstance(dim_option[key], TuneRange):
        self.option_record[key] = []
      else:
        for option in dim_option[key]:
          self.option_record[key][option] = 0

class KernelFlagsTuner(MeasurementInterface):
  def __init__(self, *pargs, **kwargs):
    super(KernelFlagsTuner, self).__init__(program_name=args.kernel, *pargs,
                                            **kwargs)
    self.kernel_names = pargs[0].kernel.split(',')
    self.strategy_path = get_policy_path()
    self.kernel_params = {}
    for kernel_name in self.kernel_names:
      line_number, content = get_line_number(self.strategy_path, kernel_name)
      opt_flag = get_flag_dict(content.strip().split(',')[1:])
      self.kernel_params[kernel_name] = KernelParameter(kernel_name, line_number, opt_flag)
      print("Kernel name: ", kernel_name)
      print("Line number: ", line_number)
      print("Content: ", content.strip())
      print("Optimization flags: ", opt_flag)
    print("Initialize finished")

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
      # print("MIScheduler set to: ", cfg[opt_dim[0]])
      change_policy_file(self.line_number, self.kernel_name + "," + ",".join([str(self.opt_flag[key]) for key in opt_dim]) + "\n")
      build_dir = get_kernel_path() + "build/"
      cmake_cmd = 'cmake -G Ninja -S {0} -B {1}'.format(get_kernel_path(), build_dir)
      cmake_res = self.call_program(cmake_cmd)
      assert cmake_res['returncode'] == 0
      print("CMake finished")
      ninja_cmd = 'ninja -C {0} '.format(build_dir)
      ninja_res = self.call_program(ninja_cmd)
      print("Build finished")
      assert ninja_res['returncode'] == 0
      return ninja_res

  def run_precompiled(self, desired_result, input, limit, compile_result, id, params):
    """
    Run a compile_result from compile() sequentially and return performance
    """
    run_cmd = get_kernel_path() + "build/syntests/syntests -t " + params.name
    run_result = self.call_program(run_cmd)
    cycle, succ, result_lines = diagnose_run_result(run_result['stderr'].decode().split('\n'))
    assert succ
    if params.old_better and cycle < params.old_performance:
      params.old_better = False
    params.result_lines = result_lines
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
    # for res in result:
    #   self.option_record[opt_dim[0]][res.configuration.data[opt_dim[0]]] = 1
    # for key in self.option_record:
    #   for option in self.option_record[key]:
    #     if self.option_record[key][option] == 0:
    #       return False
    # return True
    return False
  
  def get_current_performance(self, param):
    print("Testing current setting for", param.name)
    result = self.run_precompiled(Result(time = 0), None, 0, 0, 0, param)
    param.old_performance = result.time
  
  def pre_process(self):
    # we need to record the current perfmance
    tasks = []
    for key in self.kernel_params:
      param = self.kernel_params[key]
      if param.old_performance == 0:
        tasks.append(param)
    if tasks:
      thread_pool = ThreadPool(len(tasks))
      thread_pool.map(self.get_current_performance, tasks)
      thread_pool.close()
      for param in tasks:
        print(param.result_lines.strip())
        print("Number of cycles for", param.name, "in current setting:", param.old_performance)

    
  def save_final_config(self, configuration):
    """called at the end of tuning"""
    if self.old_better:
      print("The original setting is better")
      self.opt_flag = self.old_flag.copy()
    else:
      print("Optimal compiling flag is:", configuration.data)
      self.opt_flag["MIScheduler"] = configuration.data[opt_dim[0]]
    change_policy_file(self.line_number, self.kernel_name + "," + ",".join([self.opt_flag[key] for key in opt_dim]) + "\n")
    
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
    print("Optimization flags set to: \n", self.opt_flag)


class MultiKernelTuner():
  def __init__(self, *pargs, **kwargs):
    self.kernel_names = pargs[0].kernel.split(',')
    self.thread_pool = ThreadPool(len(self.kernel_names))
    self.kernel_params = []
    self.db_path = pargs[0].database
      
    for kernel_name in self.kernel_names:
      single_parg = copy.copy(pargs[0])
      single_parg.kernel = kernel_name
      single_parg.database = self.db_path + "/" + kernel_name + ".db"
      os.system("touch " + single_parg.database)
      self.kernel_params.append(single_parg)
      
  def main(self):
    self.thread_pool.map(KernelFlagsTuner.main, self.kernel_params)
    self.thread_pool.close()      
    

if __name__ == '__main__':
  args = parser.parse_args()
  args.parallelism = 1
  args.test_limit = 12
  args.stop_after = 3 * 60 # 3min
  # KernelFlagsTuner.main(args)
  tuner = MultiKernelTuner(args)
  tuner.main()
  