import pandas as pd
import plotly.express as px

# 读取数据
df = pd.read_csv('/home/test_ci/DLC_Custom_Kernel/DLC_Custom_Kernel/cycles_data.csv', parse_dates=['date'])

# 创建交互式折线图
fig = px.line(
    df,
    x='date',
    y='cycles',
    title='Cycles Over Time',
    markers=True,  # 显示数据点
    labels={'date': 'Date', 'cycles': 'Cycles'},
    template='plotly_white'  # 使用白色主题
)

# 自定义悬停信息
fig.update_traces(
    hovertemplate="<b>Date:</b> %{x|%Y-%m-%d}<br><b>Cycles:</b> %{y}",
    line=dict(width=2.5)
)

# 更新布局
fig.update_layout(
    hovermode='x unified',
    xaxis_title='Date',
    yaxis_title='Cycles',
    title_x=0.5  # 标题居中
)

# 显示图形
# fig.show()

# 可选：保存为HTML文件
fig.write_html("/home/test_ci/DLC_Custom_Kernel/DLC_Custom_Kernel/interactive_cycles_chart.html")