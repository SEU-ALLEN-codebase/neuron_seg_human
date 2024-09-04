import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt

def get_patient_data(df):
    # 提取编号、性别和年龄
    data = []
    seen_id = set()

    for index, row in df.iterrows():
        # 提取编号中的纯数字
        number = row[4]
        if (number == '' or number == '--' or number == np.nan or number is None or type(number) == float):
            continue
        if (number[0] == 'p'):
            number = 'P' + number[1:]
        gender = row[10]
        if (gender == '男'):
            gender = 'M'
        elif (gender == '女'):
            gender = 'F'
        else:
            gender = 'Unknown'

        age = row[9]
        try:
            age = int(age)
        except:
            continue

        if number is not None and number not in seen_id:
            data.append({
                'id': number,
                'gender': gender,
                'age': age
            })
            seen_id.add(number)

    # sort
    data = sorted(data, key=lambda x: x['id'])
    # # 打印结果
    # for item in data:
    #     print(item)
    return data

def get_slice_data(df):
    # 提取number和额外信息
    data = []

    for index, row in df.iterrows():
        number = str(row[1])  # 第二列是number，索引为1
        brain_region = str(row[15])  # 第p列是一些额外信息，假设为第16列，索引为15

        data.append({
            'id': number,
            'brain_region': brain_region
        })

    # 打印结果
    # for item in data:
    #     print(item)
    return data

def merge_data(patient_data, slice_data):
    patient_data_dict = {item['id']: item for item in patient_data}
    merged_data = []
    for item in slice_data:
        number = item['id']
        if number in patient_data_dict:
            combined_item = {**item, **patient_data_dict[number]}
            merged_data.append(combined_item)
        else:
            merged_data.append(item)

    # 打印合并后的数据
    # for item in merged_data:
    #     print(item)

    return merged_data
# 读取Excel文件
file_path = '/data/kfchen/trace_ws/res_vis/Human_SingleCell_TrackingTable_20240719.xlsx'  # 请将此路径替换为实际文件路径
df = pd.read_excel(file_path, sheet_name=2)
patient_data = get_patient_data(df)


# 读取CSV文件
file_path = '/data/kfchen/trace_ws/res_vis/Human_SingleCell_TrackingTable_20240712.csv'  # 请将此路径替换为实际文件路径
df = pd.read_csv(file_path, encoding='latin1')
slice_data = get_slice_data(df)
# print(slice_data)

slice_data = merge_data(patient_data, slice_data)
print(len(slice_data))

# 将 data 转换为 DataFrame
df = pd.DataFrame(slice_data)
csv_file_path = '/data/kfchen/trace_ws/res_vis/data.csv'
df.to_csv(csv_file_path, index=False, encoding='utf-8-sig')

# 将年龄按5岁分组
bins = range(0, df['age'].max() + 5, 5)
labels = [f'{i}-{i+4}' for i in bins[:-1]]
df['age_group'] = pd.cut(df['age'], bins=bins, labels=labels, right=False)

# 创建一个包含四个子图的图形
fig, axs = plt.subplots(2, 2, figsize=(15, 15))

# 绘制 id 的统计饼状图
id_counts = df['id'].value_counts()
axs[0, 0].pie(id_counts, labels=id_counts.index, autopct='%1.1f%%')
axs[0, 0].set_title('ID Distribution')

# 绘制 br 的统计饼状图
br_counts = df['brain_region'].value_counts()
axs[0, 1].pie(br_counts, labels=br_counts.index, autopct='%1.1f%%')
axs[0, 1].set_title('brain_region Distribution')

# 绘制 gender 的统计饼状图
gender_counts = df['gender'].value_counts()
axs[1, 0].pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%')
axs[1, 0].set_title('Gender Distribution')

# 绘制 age 的统计饼状图
age_group_counts = df['age_group'].value_counts().sort_index()
axs[1, 1].pie(age_group_counts, labels=age_group_counts.index, autopct='%1.1f%%')
axs[1, 1].set_title('Age Distribution')

# 调整子图间距
plt.tight_layout()

# 保存图形为 PNG 文件
plt.savefig('/data/kfchen/trace_ws/res_vis/data_distribution.png')








