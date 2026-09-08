"""Fail-fast synchronization helpers for the parallel autotune workers."""

import os
import threading
import time


sync_abort = threading.Event()

try:
  sync_timeout_seconds = float(os.environ.get("AUTOTUNE_SYNC_TIMEOUT", "1800"))
except ValueError:
  sync_timeout_seconds = 1800.0
sync_timeout_seconds = max(1.0, sync_timeout_seconds)


def signal_abort(context, error=None):
  """Wake all workers after a worker failure or synchronization timeout."""
  if error is None:
    print("Autotune abort requested: " + context, flush=True)
  else:
    print("Autotune worker failed in " + context + ": " + repr(error), flush=True)
  sync_abort.set()


def wait_for_counter(counter, expected, phase):
  """Wait for a shared counter without leaving peers in an infinite spin."""
  deadline = time.monotonic() + sync_timeout_seconds
  while counter.value != expected:
    if sync_abort.is_set():
      raise RuntimeError("Autotune synchronization aborted while waiting for " + phase)
    if time.monotonic() >= deadline:
      signal_abort("timeout while waiting for " + phase)
      raise TimeoutError("Autotune synchronization timed out while waiting for " + phase)
    time.sleep(0.01)


def run_worker(task):
  """Run one OpenTuner worker and propagate failures to all peers."""
  worker, worker_args = task
  try:
    return worker(worker_args)
  except BaseException as error:
    worker_name = getattr(worker_args, "kernel", "unknown")
    signal_abort(worker_name, error)
    raise
