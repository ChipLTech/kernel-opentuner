import os
import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# 读取三个数据集（完全保留原始代码）
df_tinyllama = pd.read_csv('/home/CI/autotune/cycles_data_tinyllama.csv', parse_dates=['date'])
df_gemma = pd.read_csv('/home/CI/autotune/cycles_data_gemma.csv', parse_dates=['date'])
df_llama = pd.read_csv('/home/CI/autotune/cycles_data_llama.csv', parse_dates=['date'])

# 新增：读取基准值历史
def read_baseline_history(file_path):
    if not os.path.exists(file_path):
        return pd.DataFrame(columns=['date', 'cycles'])
    try:
        if os.path.getsize(file_path) == 0:
            return pd.DataFrame(columns=['date', 'cycles'])
        return pd.read_csv(file_path, parse_dates=['date'])
    except Exception as exc:
        print(f"无法读取基准值历史文件 {file_path}: {exc}")
        return pd.DataFrame(columns=['date', 'cycles'])

baseline_tinyllama = read_baseline_history('/home/CI/autotune/tinyllama_baseline_history.csv')
baseline_gemma = read_baseline_history('/home/CI/autotune/gemma_baseline_history.csv')
baseline_llama = read_baseline_history('/home/CI/autotune/llama_baseline_history.csv')

# 创建子图布局 (仅添加基准值到模型列表，其他完全保留)
fig = make_subplots(
    rows=3, 
    cols=1,
    shared_xaxes=True,
    vertical_spacing=0.1,
    subplot_titles=("TinyLlama Cycles", "Gemma Cycles", "Llama Cycles")  # 保持原始标题
)

# 为每个数据集创建折线图并添加到子图（仅增加基准值参数）
models = [
    (df_tinyllama, 'TinyLlama', 1, baseline_tinyllama),  # 新增基准值参数
    (df_gemma, 'Gemma', 2, baseline_gemma),              # 新增基准值参数
    (df_llama, 'Llama', 3, baseline_llama)               # 新增基准值参数
]

for df, name, row, baseline_df in models:  # 接收基准值参数
    # 原始折线图代码完全不变
    trace = go.Scatter(
        x=df['date'],
        y=df['cycles'],
        mode='lines+markers',
        name=f"{name} Tune",
        hovertemplate="<b>Date:</b> %{x|%Y-%m-%d}<br><b>Cycles:</b> %{y}<extra></extra>"
    )
    fig.add_trace(trace, row=row, col=1)
    
    # 新增：添加对应模型的基准线历史
    if not baseline_df.empty:
        baseline_trace = go.Scatter(
            x=baseline_df['date'],
            y=baseline_df['cycles'],
            mode='lines+markers',
            name=f"{name} Baseline",
            line=dict(color='red', dash='dot'),
            marker=dict(symbol='triangle-up', color='red'),
            hovertemplate="<b>Date:</b> %{x|%Y-%m-%d}<br><b>Baseline Cycles:</b> %{y}<extra></extra>"
        )
        fig.add_trace(baseline_trace, row=row, col=1)

# 更新布局（完全保留原始设置）
fig.update_layout(
    title_text='Cycles Comparison Across Models',
    title_x=0.5,
    height=900,
    template='plotly_white',
    hovermode='x unified',
    showlegend=True
)

# 更新坐标轴标签（完全保留原始设置）
fig.update_xaxes(title_text="Date", row=3, col=1)
fig.update_yaxes(title_text="Cycles", row=2, col=1)

# 调整子图标题样式（完全保留原始设置）
for annotation in fig['layout']['annotations']:
    annotation['font'] = dict(size=14, color='blue')

# 保存为单个HTML文件（完全保留原始路径）
fig.write_html("/home/test_ci/DLC_Custom_Kernel/DLC_Custom_Kernel/interactive_cycles_chart.html")
