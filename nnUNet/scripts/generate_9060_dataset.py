import pandas as pd
import os

# def merge_rows(df):
#     group = df.groupby('Cell ID')
#     for name, group in group:
#         if(group.shape[0] > 1):
#             print(name)
#             print(group)
#             print('-----------------')

def merge_rows(df):
    # 分组并合并
    result_df = df.groupby('Cell ID').apply(merge_if_different)
    # 重置索引，因为 apply 操作可能会引入多级索引
    result_df.reset_index(drop=True, inplace=True)
    return result_df


def merge_if_different(group):
    # 如果只有一行数据，直接返回该组
    if group.shape[0] == 1:
        return group

    # 创建一个空的字典来存放合并后的数据
    merged_data = {}

    # 遍历所有列，根据列的数据类型采取不同的合并策略
    for column in group.columns:
        if pd.api.types.is_numeric_dtype(group[column]):
            # 数值类型：计算平均值
            merged_data[column] = group[column].mean()
        else:
            # 非数值类型：取第一行的值
            merged_data[column] = group[column].iloc[0]

    # 将字典转换为DataFrame
    merged_df = pd.DataFrame([merged_data], columns=group.columns)
    return merged_df





# 假设 'data.csv' 是你的CSV文件，'your_list' 是给定的列表
origin_neuron_info_file = r"/data/kfchen/trace_ws/paper_auto_human_neuron_recon/new_Human_SingleCell_TrackingTable_20240712.csv"
# origin_neuron_info_file = r"/data/kfchen/trace_ws/paper_auto_human_neuron_recon/Human_SingleCell_TrackingTable_20240712.csv"
neuron_info_file = "/data/kfchen/trace_ws/paper_auto_human_neuron_recon/neuron_info_9060.csv"

swc_dir = "/data/kfchen/trace_ws/paper_auto_human_neuron_recon/origin_recon/ptls10"
swc_files = [f for f in os.listdir(swc_dir) if f.endswith('.swc')]
ids = [int(f.split('_')[0]) for f in swc_files]
print(len(ids))


# 读取CSV文件
df = pd.read_csv(origin_neuron_info_file, encoding='gbk', low_memory=False)

def check_condition(value):
    return value in ids

# 过滤数据，只保留第一列值在your_list中的行
filtered_df = df[df.iloc[:, 0].apply(check_condition)]
filtered_df = merge_rows(filtered_df)
# 有多少行
print(filtered_df.shape[0])


if(os.path.exists(neuron_info_file)):
    os.remove(neuron_info_file)
# 将过滤后的数据保存到新的CSV文件
filtered_df.to_csv(neuron_info_file, index=False, encoding='gbk')



