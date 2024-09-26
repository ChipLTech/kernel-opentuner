from dlcutils import *
import time
import subprocess
import os
import random

log_path = "/wkspc/lanhu/autotune"

def get_diff_files(last_commit, current_commit):
  # if last_commit == "":
  #   print("No last commit found")
  #   changed_files = subprocess.check_output(['git', 'show', '--pretty=\'\'', '--name-only', current_commit]).decode().splitlines()
  # else:
  # print("Last commit:", last_commit)
  changed_files = subprocess.check_output(['git', 'diff', '--name-only', last_commit + ".." + current_commit]).decode().splitlines()
  if not filter_kernel_files(changed_files):
    print("No kernel files changed since last tune")
    prev5_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD~5']).decode().strip()
    print("Look ahead for 5 commits to", prev5_commit)
    changed_files = subprocess.check_output(['git', 'diff', '--name-only', prev5_commit + ".." + current_commit]).decode().splitlines()
  return changed_files

def filter_kernel_files(changed_files):
  kernel_files = []
  for file in changed_files:
    if file.startswith("dlc_kernels/"):
      kernel_files.append(file)
  return kernel_files

def get_commit_hash(log_path):
  try:
    with open(log_path + "/commit_hash.txt", 'r') as f:
      return f.read().strip()
  except FileNotFoundError:
    print("No commit hash found in the log directory")
    return ""
  
def parse_dependency(dep_info):
  kernel_to_dep = {}
  dep_to_kernel = {}
  for line in dep_info:
      line = line.strip()
      if not line:  # Skip empty lines
          continue
      
      # Check if line contains a colon
      if ':' in line and 'dlc_src' in line:
        target, _ = line.split(':', 1)
        target = target.strip().split('/')[-1]
        target = target.split('_dlc')[0]
        kernel_to_dep[target] = []
      elif 'dlc_kernels' in line:
        file_name = line.split()[0].strip().split('/')[-1]
        kernel_to_dep[target].append(file_name)

  # remove all the _xys1 files
  duplicates = []
  for target in kernel_to_dep:
    if "_xys1" in target:
      duplicates.append(target)
  for target in duplicates:
    kernel_to_dep.pop(target)

  for target in kernel_to_dep:
    for dep in kernel_to_dep[target]:
      if dep not in dep_to_kernel:
        dep_to_kernel[dep] = []
      dep_to_kernel[dep].append(target)
  return kernel_to_dep, dep_to_kernel
  
if __name__ == '__main__':
  cur_dir = os.getcwd()
  # check the result we have last time
  most_recent_log_dir = get_most_recent_log_dir(log_path)
  print(most_recent_log_dir)
  os.chdir(get_kernel_path())
  
  # get the commit hash of the most recent commit
  if most_recent_log_dir == "":
    # look ahead for 5 commits
    most_recent_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD~5']).decode().strip()
  else:
    most_recent_commit = get_commit_hash(most_recent_log_dir)
  print("most recent commit:", most_recent_commit)
    
  # pull the newest commit
  # subprocess.run(['git', 'pull'])
  subprocess.run(['rm', '-rf', log_path + '/logs/*'])
  
  # get the commit hash of the newest commit
  current_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip()
  print("current commit:", current_commit)

  # create a new log directory
  new_log_dir = log_path + "/logs/" + str(int(time.time()))
  subprocess.run(['mkdir', new_log_dir])
  subprocess.run(['touch', new_log_dir + "/commit_hash.txt"])
  with open(new_log_dir + "/commit_hash.txt", 'w') as f:
    f.write(current_commit)
    
  # get the changed files
  changed_files = get_diff_files(most_recent_commit, current_commit)
  subprocess.run(['touch', new_log_dir + "/changed_files.txt"])
  with open(new_log_dir + "/changed_files.txt", 'w') as f:
    for file in changed_files:
      f.write(file + "\n")
  changed_files = filter_kernel_files(changed_files)
  changed_files = [file.split('/')[-1] for file in changed_files]
  print("changed files:", changed_files)
    
  # build the dependency files
  build_dir = get_kernel_path() + "build/"
  cmake_cmd = 'cmake -G Ninja -S {0} -B {1}'.format(get_kernel_path(), build_dir)
  subprocess.run(cmake_cmd.split())
  ninja_cmd = 'ninja -C {0} syntests'.format(build_dir)
  subprocess.run(ninja_cmd.split())
  # subprocess.run(['ninja', '-C', build_dir, 'install'])
  
  # get the dependency info
  subprocess.run(['touch', new_log_dir + "/depfiles.txt"])
  dep_info = subprocess.check_output(['ninja', '-C', build_dir, '-t', 'deps'], stderr=subprocess.STDOUT).decode()
  with open(new_log_dir + "/depfiles.txt", 'w') as f:
    f.write(dep_info)
  kernel_to_dep, dep_to_kernel = parse_dependency(dep_info.splitlines())
  
  # take the intersection of the changed files and the dependency files
  candidate_kernel = []
  for file in changed_files:
    for kernel in dep_to_kernel[file]:
      if kernel not in candidate_kernel:
        candidate_kernel.append(kernel)

  print(candidate_kernel, "before filtering")
  available_tests = subprocess.check_output([get_kernel_path() + '/build/syntests/syntests', '-l']).decode().splitlines()
  candidate_kernel = [kernel for kernel in candidate_kernel if kernel in available_tests]
  print(candidate_kernel, "after filtering")
  if len(candidate_kernel) < 10:
    print("Randomly pick kernels to tune")
    ramdom_picked = random.sample(available_tests, 10 - len(candidate_kernel))
    candidate_kernel.extend(ramdom_picked)
    print(candidate_kernel, "after random picking")
  kernel_param = "--kernel=" + ",".join(candidate_kernel)
  subprocess.run(['mkdir', '-p', new_log_dir + "/tunerDB"])
  database_param = "--database=" + new_log_dir + "/tunerDB"
  print("*********** Start to tune ***********")
  subprocess.run(['python3', cur_dir + '/multi-tune.py', kernel_param, database_param])
  subprocess.run(['cp', get_policy_path(), new_log_dir]) 
