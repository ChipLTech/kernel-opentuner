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
from dlcutils import *

parser = argparse.ArgumentParser(parents=opentuner.argparsers())
parser.add_argument('--kernel', help='kernel name to tune')

class KernelFlagsTuner(MeasurementInterface):
  def __init__(self, *pargs, **kwargs):
    super(KernelFlagsTuner, self).__init__(program_name=args.kernel, *pargs,
                                            **kwargs)
    self.kernel_name = pargs[0].kernel
    self.stragegy_path = get_policy_path()
    self.line_number, content = get_line_number(self.stragegy_path, self.kernel_name)
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
    print("Kernel name: ", self.kernel_name)
    print("Line number: ", self.line_number)
    print("Content: ", content.strip())
    print("Optimization flags: ", self.opt_flag)
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

  def run_precompiled(self, desired_result, input, limit, compile_result, id):
    """
    Run a compile_result from compile() sequentially and return performance
    """
    run_cmd = get_kernel_path() + "build/syntests/syntests -t " + self.kernel_name
    run_result = self.call_program(run_cmd)
    cycle, succ = diagnose_run_result(run_result['stderr'].decode().split('\n'))
    assert succ
    if self.old_better and cycle < self.old_performance:
      self.old_better = False
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
      self.old_performance = self.run_precompiled(Result(time = 0), None, 0, 0, 0).time
      print("Number of cycles in current setting: ", self.old_performance)
  
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


if __name__ == '__main__':
  args = parser.parse_args()
  args.parallelism = 1
  args.test_limit = 12
  args.stop_after = 3 * 60 # 3min
  KernelFlagsTuner.main(args)