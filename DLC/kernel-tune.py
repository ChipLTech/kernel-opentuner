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
    print("Kernel name: ", self.kernel_name)
    print("Line number: ", self.line_number)
    print("Content: ", content.strip())
    print("Optimization flags: ", self.opt_flag)

  def manipulator(self):
    """
    Define the search space by creating a
    ConfigurationManipulator
    """
    manipulator = ConfigurationManipulator()
    manipulator.add_parameter(
      EnumParameter(opt_dim[0],
                    ['topdown', 'bottomup', 'bidirectional']))
    return manipulator

  def compile(self, cfg, id):
      """
      Compile a given configuration in parallel
      """
      self.opt_flag["MIScheduler"] = cfg[opt_dim[0]]
      print("MIScheduler set to: ", cfg[opt_dim[0]])
      change_policy_file(self.line_number, self.kernel_name + "," + ",".join([self.opt_flag[key] for key in opt_dim]))
      build_dir = get_kernel_path() + "build/"
      cmake_cmd = 'cmake -G Ninja -S {0} -B {1}'.format(get_kernel_path(), build_dir)
      cmake_res = self.call_program(cmake_cmd)
      assert cmake_res['returncode'] == 0
      print("CMake finished")
      ninja_cmd = 'ninja -C {0} '.format(build_dir)
      ninja_res = self.call_program(ninja_cmd)
      print("Build finished")
      return ninja_res

  def run_precompiled(self, desired_result, input, limit, compile_result, id):
    """
    Run a compile_result from compile() sequentially and return performance
    """
    assert compile_result['returncode'] == 0

    run_cmd = get_kernel_path() + "build/syntests/syntests -t " + self.kernel_name
    run_result = self.call_program(run_cmd)
    cycle, succ = diagnose_run_result(run_result['stderr'].decode().split('\n'))
    assert succ
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
      print(res.input)
      print(res.input_id)
      print(res.input.data)
    return False
  


if __name__ == '__main__':
  args = parser.parse_args()
  KernelFlagsTuner.main(args)