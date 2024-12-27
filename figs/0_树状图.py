import plotly.express as px
import pandas as pd
from sympy.benchmarks.bench_meijerint import alpha

# datasets = [
#     "allen_human_neuromorpho",
#     "allman",
#     # "ataman_boulting",
#     "DeKock",
#     "hrvoj-mihic_semendeferi",
#     "jacobs",
#     # "segev",
#     "semendeferi_muotri",
#     "vdheuvel",
#     # "vuksic",
#     # "wittner",
#     "proposed",
# ]
# 数据集简写
datasets = [
    "Allen",
    "Allman",
    # "Ataman",
    "DeKock",
    "Hrvoj",
    "Jacobs",
    # "Segev",
    "Semen",
    "VdHeuvel",
    # "Vuksic",
    # "Wittner",
    "Proposed",
]
counts = [301, 36, 89, 98, 2621, 45, 376, 8676]
# datasets = [f"{dataset}\n({count})" for dataset, count in zip(datasets, counts)]

data = pd.DataFrame({
    'Dataset': datasets,
    'Count': counts
})
# sort
data = data.sort_values(by='Count', ascending=False)
# re idx
data = data.reset_index(drop=True)
print(data)

# 设置清晰度
px.defaults.width = 1920
px.defaults.height = 400
# 使用plotly绘制矩形树图，并设置字体大小
fig = px.treemap(data,
                  path=['Dataset'],  # 树状图的路径，仅此一层数据集
                  values='Count',  # 通过矩形的大小表示数量
                  # color='Count',  # 通过颜色表示数量
                  # color_continuous_scale='Blues',  # 颜色渐变
                  # title="Dataset Counts Treemap"
                 )

fig.update_traces(texttemplate="%{label}<br>(%{value})",  # 设置树状图中每个节点标签的显示格式
                    # textposition='middle center',  # 设置树状图中每个节点标签的位置
                    # hoverinfo='label+value',  # 设置鼠标悬停时显示的信息
                    # marker=dict(line=dict(width=2, color='black'))  # 设置树状图中每个节点的边框
                    )
# 更新文本字体大小
fig.update_traces(
    textfont=dict(
        size=40  # 设置树状图中每个节点标签的字体大小为 14
    )
)


fig.write_image("/data/kfchen/trace_ws/quality_control_test/0_treemap.png")
# fig.saveas("0_树状图.html")
# fig.close()
