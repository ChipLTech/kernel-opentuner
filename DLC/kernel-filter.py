from dlcutils import *
import time
import subprocess
import os

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
  
def parse_dependency_file(dep_file):
  dependencies = {}
  with open(dep_file, 'r') as f:
      for line in f:
          line = line.strip()
          if not line:  # Skip empty lines
              continue
          
          # Check if line contains a colon
          if ':' in line:
            target, deps_str = line.split(':', 1)
            target = target.strip()
            deps = [dep.strip() for dep in deps_str.split() if 'dlc_kernels' in dep]
            # dependencies[target] = deps
          elif 'dlc_kernels' in line:
            file_name = line.split()[0]
            deps.append(file_name)
  dependencies[target] = deps
  return dependencies
  
def have_file_changed(dep_file, changed_files):
  deps = parse_dependency_file(dep_file)
  for target in deps:
    deps[target] = [dep.split('/')[-1] for dep in deps[target]]
    # print(deps)
    for dep in deps[target]:
      if dep in changed_files:
        return True
  return False
  
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
  # cmake_cmd = 'cmake -G Ninja -S {0} -B {1}'.format(get_kernel_path(), build_dir)
  # subprocess.run(cmake_cmd.split())
  # ninja_cmd = 'ninja -C {0} syntests -d keepdepfile'.format(build_dir)
  # subprocess.run(ninja_cmd.split())
  # subprocess.run(['ninja', '-C', build_dir, 'install'])
  
  # copy all the dependency files to the log directory
  dep_file_path = new_log_dir + "/depfiles"
  subprocess.run(['mkdir', dep_file_path])
  dep_files = subprocess.check_output(['ls', build_dir + '/dlc_src/']).decode().splitlines()
  dep_files = [build_dir + '/dlc_src/' + file for file in dep_files if file.endswith('.d')]
  subprocess.run(['cp', *dep_files ,dep_file_path])
  
  # take the intersection of the changed files and the dependency files
  candidate_kernel = []
  dep_files = os.listdir(dep_file_path)
  for file in dep_files:
    if have_file_changed(dep_file_path + "/" + file, changed_files):
      candidate_kernel.append(file.split('.')[0])
  print(candidate_kernel, "before filtering")
  available_tests = subprocess.check_output([get_kernel_path() + '/build/syntests/syntests', '-l']).decode().splitlines()
  candidate_kernel = [kernel for kernel in candidate_kernel if kernel in available_tests]
  print(candidate_kernel, "after filtering")
  kernel_param = "--kernel=" + ",".join(candidate_kernel)
  database_param = "--database=/wkspc/lanhu/tunerDB"
  print("*********** Start to tune ***********")
  print("kernel_param:", kernel_param)
  subprocess.run(['python3', cur_dir + '/multi-tune.py', kernel_param, database_param])
  subprocess.run(['cp', get_policy_path(), new_log_dir]) 
