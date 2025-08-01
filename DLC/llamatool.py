from __future__ import annotations
import re
import sys
import os
import json

F = 1400

class Metric:

    regexp = re.compile(r'.* cycles: (\d+)  ops: (\d+)  bytes: (\d+)  crt: (\d+)')

    def __init__(self, name: str, body: str):
        self.name = name
        self.body = body
        self.n = 0
        self.cycles = 0
        self.ops = 0
        self.bytes = 0
        self.crt = 0

    def match(self, line: str):
        if r:= self.regexp.match(line):
            self.cycles = int(r.group(1))
            self.ops = int(r.group(2))
            self.bytes = int(r.group(3))
            self.crt = int(r.group(4))
        self.n = 1
        self.cycles = max(self.cycles, 1)
        self.ops = max(self.ops, 1)
        self.bytes = max(self.bytes, 1)
        self.crt = max(self.crt, 1)
        return bool(r)
    
    def __iadd__(self, other: 'Metric') -> 'Metric':
        self.n += other.n
        self.cycles += other.cycles
        self.ops += other.ops
        self.bytes += other.bytes
        self.crt += other.crt
        return self

    def __str__(self, total = None) -> str:
        t_avg = self.cycles / F / self.n if self.n else 0
        crt_avg = self.crt / F / self.n if self.n else 0
        if total is None:
            total = Metric(None, None)
        self.cycles = max(self.cycles, 1)
        self.ops = max(self.ops, 1)
        self.bytes = max(self.bytes, 1)
        self.crt = max(self.crt, 1)
        total.cycles = max(total.cycles, self.cycles)
        total.ops = max(total.ops, self.ops)
        total.bytes = max(total.bytes, self.bytes)
        total.crt = max(total.crt, self.crt)
        return f'n: {self.n:<8}cycles({self.cycles/total.cycles*100:>6.2f}%): {self.cycles:<12}  ops({self.ops/total.ops*100:>6.2f}%):{self.ops:<16}  GFLOPS:{self.ops*F/1000/self.cycles:>9.2f}  BAND:{self.bytes*F/1000/self.cycles:>8.2f} GB/s  t_avg:{t_avg:>9.2f} us  crt_avg:{crt_avg:>5.2f} us  t:{self.cycles/F/1000:>10.2f} ms  crt:{self.crt/F/1000:>8.2f} ms  {self.name}'
    
    def __repr__(self) -> str:
        return str(self)

def remove_ansi(sometext):
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    addr = re.compile(r'addr=0x(\w+) ')
    body = ansi_escape.sub('', sometext)
    body = addr.sub('xxx', body)
    return body

def get_kernel_launches(text: str):
    r1 = re.compile(r'─────────────────────────────────────────────────────────────────────────────────── launch  (\w+) ')
    r2 = re.compile(r'─────────────────────────────────────────────────────────────────────────────────── cycles(.*)\n')

    lines = text.splitlines(keepends=True)
    name: str = ''
    tmp = []
    results_cycle: list[Metric] = []
    results_nocycle: list[Metric] = []
    for line in lines:
        if name:
            if r2.match(line):
                m = Metric(name, remove_ansi(''.join(tmp)))
                if m.match(line):
                    results_cycle.append(m)
                else:
                    results_nocycle.append(m)
                name = ''
                tmp.clear()
            else:
                tmp.append(line)
        elif r:=r1.match(line):
            name = r.group(1)
    # print("results_cycle: \n", results_cycle)
    return results_cycle if results_cycle else results_nocycle

def process_log(in_file, out_file) -> str:
    print(f'{in_file} -> {out_file}')
    with open(in_file, 'r', errors='ignore') as f:
        text = f.read()
    kernel_metrics = get_kernel_launches(text)

    total = Metric("Total", None)
    for m in kernel_metrics:
        total += m
    
    ############################################################################################################################
    kernel_launches = {}  # {(name, body): Metric}
    
    for m in kernel_metrics:
        if (m.name, m.body) not in kernel_launches:
            kernel_launches[(m.name, m.body)] = Metric(m.name, m.body)
        kernel_launches[(m.name, m.body)] += m

    msg = []

    results = list(kernel_launches.values())
    results = sorted(results, key=lambda x: x.cycles, reverse=True)
    for m in results:
        msg.append(f'──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────')
        msg.append(m.__str__(total))
        msg.append(m.body)
    msg.append('\n')

    ############################################################################################################################
    kernel_cycles = {} # {name: Metric}
    for m in kernel_metrics:
        if m.name not in kernel_cycles:
            kernel_cycles[m.name] = Metric(m.name, None)
        kernel_cycles[m.name] += m
        
    results = list(kernel_cycles.values())
    results = sorted(results, key=lambda x: x.cycles, reverse=True)

    for m in results:
        msg.append(m.__str__(total))

    msg.append(f"\n{total}")
    with open(out_file, 'w') as f:
        f.write('\n'.join(msg))


def process_trace(path: str):
    try:
        with open(path, 'r', errors='ignore') as f:
            trace = json.load(f)
        events = trace['traceEvents']
        ts = [e for e in events if 'ts' in e]
        min_t = min(e['ts'] for e in ts)
        for e in ts:
            e['ts'] -= min_t
        with open(path, 'w') as f:
            json.dump(trace, f, indent=4)
    except:
        print(e)
        
def diagnose_llama_result(text):
    kernel_metrics = get_kernel_launches(text)
    kernel_cycles = {} # {name: Metric}
    for m in kernel_metrics:
        if m.name not in kernel_cycles:
            kernel_cycles[m.name] = Metric(m.name, None)
        kernel_cycles[m.name] += m
    results = list(kernel_cycles.values())
    results = sorted(results, key=lambda x: x.cycles, reverse=True)
    # print("results:\n", kernel_metrics[0])
    kernel_to_cycle = {}
    total_cycles = 0
    for item in results:
        kernel_to_cycle[item.name.split('ustom_')[1]] = item.cycles
        total_cycles += item.cycles
    return kernel_to_cycle, total_cycles
