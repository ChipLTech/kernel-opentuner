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
    r = re.compile(r'(\w+)\s+on.*?\n(.*)', re.DOTALL)
    name, body = r.findall(text)[0]
    addr = re.compile(r'addr=0x(\w+) ')
    body = addr.sub('xxx', body)
    return name, body

def get_kernel_cycles(text):
    # custom_matmul_t_pingpong  xys0_cycle: 33333  xys1_cycle: 33091
    r = re.compile(r'xys0_cycle:\s*(\d+)\s+xys1_cycle:\s*(\d+)')
    ret = []
    for item in r.findall(text):
        ret.append(max(int(item[0]), int(item[1])))
    return ret