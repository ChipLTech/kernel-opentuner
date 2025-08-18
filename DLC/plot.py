# import pandas as pd
# import plotly.express as px

# # 读取数据
# df = pd.read_csv('/home/CI/autotune/cycles_data_tinyllama.csv', parse_dates=['date'])

# # 创建交互式折线图
# fig = px.line(
#     df,
#     x='date',
#     y='cycles',
#     title='Cycles Over Time',
#     markers=True,  # 显示数据点
#     labels={'date': 'Date', 'cycles': 'Cycles'},
#     template='plotly_white'  # 使用白色主题
# )

# # 自定义悬停信息
# fig.update_traces(
#     hovertemplate="<b>Date:</b> %{x|%Y-%m-%d}<br><b>Cycles:</b> %{y}",
#     line=dict(width=2.5)
# )

# # 更新布局
# fig.update_layout(
#     hovermode='x unified',
#     xaxis_title='Date',
#     yaxis_title='Cycles',
#     title_x=0.5  # 标题居中
# )

# # 显示图形
# # fig.show()

# # 可选：保存为HTML文件
# fig.write_html("/home/test_ci/DLC_Custom_Kernel/DLC_Custom_Kernel/interactive_cycles_chart.html")

import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# 读取三个数据集
df_tinyllama = pd.read_csv('/home/CI/autotune/cycles_data_tinyllama.csv', parse_dates=['date'])
df_gemma = pd.read_csv('/home/CI/autotune/cycles_data_gemma.csv', parse_dates=['date'])
df_llama = pd.read_csv('/home/CI/autotune/cycles_data_llama.csv', parse_dates=['date'])

# 创建子图布局 (3行1列垂直排列)
fig = make_subplots(
    rows=3, 
    cols=1,
    shared_xaxes=True,  # 共享X轴
    vertical_spacing=0.1,  # 子图间距
    subplot_titles=("TinyLlama Cycles", "Gemma Cycles", "Llama Cycles")  # 子图标题
)

# 为每个数据集创建折线图并添加到子图
models = [
    (df_tinyllama, 'TinyLlama', 1),
    (df_gemma, 'Gemma', 2),
    (df_llama, 'Llama', 3)
]

for df, name, row in models:
    trace = go.Scatter(
        x=df['date'],
        y=df['cycles'],
        mode='lines+markers',
        name=name,
        hovertemplate="<b>Date:</b> %{x|%Y-%m-%d}<br><b>Cycles:</b> %{y}<extra></extra>"
    )
    fig.add_trace(trace, row=row, col=1)

# 更新布局
fig.update_layout(
    title_text='Cycles Comparison Across Models',
    title_x=0.5,
    height=900,  # 总高度
    template='plotly_white',
    hovermode='x unified',
    showlegend=False  # 每个图表已有标题，不需要图例
)

# 更新坐标轴标签
fig.update_xaxes(title_text="Date", row=3, col=1)  # 只在最下方子图显示X轴标题
fig.update_yaxes(title_text="Cycles", row=2, col=1)  # 中间子图显示Y轴标题

# 调整子图标题样式
for annotation in fig['layout']['annotations']:
    annotation['font'] = dict(size=14, color='blue')

# 保存为单个HTML文件
fig.write_html("/home/test_ci/DLC_Custom_Kernel/DLC_Custom_Kernel/interactive_cycles_chart.html")