import re

def remove_ansi(sometext):
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    return ansi_escape.sub('', sometext)

def get_kernel_launches(text):
    r = re.compile(r'─────────────────────────────────────────────────────────────────────────────────── ([\s\S]*?)────────────────────────────────────────────────────────────────────────',)
    return r.findall(text)

def get_copy_stride(text):
    r = re.compile(r'([CopyStride] [exec] )', re.DOTALL)
    return r.findall(text)

def get_kernel_info(text):
    r = re.compile(r'(\w+) on.*?\n(.*)', re.DOTALL)
    name, body = r.findall(text)[0]
    addr = re.compile(r'addr=0x(\w+) ')
    body = addr.sub('xxx', body)
    return name, body

def get_kernel_cycles(text):
    # k[0] xys0: 36582818465908 ~ 36582818689383  223475 cycles  xys1: 36582818450796 ~ 36582818675193  224397 cycles   custom_embedding_dense: 14.0186 GFLOPS
    r = re.compile(r'k\[.+ (\d+) cycles.+ (\d+) cycles')
    ret = []
    for item in r.findall(text):
        ret.append(max(int(item[0]), int(item[1])))
    return ret