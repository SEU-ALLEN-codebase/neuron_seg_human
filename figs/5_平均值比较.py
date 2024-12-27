import pandas as pd
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def qq_plot(feasure_list, feasure_name):
    data = feasure_list
    # 绘制直方图和拟合的正态分布曲线
    plt.figure(figsize=(10, 6))
    sns.histplot(data, kde=True, stat="density", bins=30, color="skyblue")
    plt.title('Histogram and KDE of the Data')
    plt.xlabel('Feature Value')
    plt.ylabel('Density')

    # 显示正态分布的拟合曲线
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = stats.norm.pdf(x, np.mean(data), np.std(data))
    plt.plot(x, p, 'k', linewidth=2)

    # 显示图像
    plt.show()

    # 进行Shapiro-Wilk正态性检验
    stat, p_value = stats.shapiro(data)
    print(f"Shapiro-Wilk test statistic: {stat}")
    print(f"Shapiro-Wilk test p-value: {p_value}")

    # 判断正态性
    if p_value > 0.05:
        print("数据符合正态分布")
    else:
        print("数据不符合正态分布")

    # 绘制QQ图
    plt.figure(figsize=(6, 6))
    stats.probplot(data, dist="norm", plot=plt)
    plt.title("QQ Plot")
    plt.show()


manual_l_measure_file = "/data/kfchen/trace_ws/paper_auto_human_neuron_recon/swc_label/1um_swc_lab_l_measure.csv"
l_measure_df = pd.read_csv(manual_l_measure_file)
total_length_list, num_branch_list = l_measure_df["Total Length"].tolist(), l_measure_df["Number of Branches"].tolist()
print("total_length_list mean: ", np.mean(total_length_list))
print("num_branch_list mean: ", np.mean(num_branch_list))
print(len(total_length_list), len(num_branch_list))
# threshold = [
#     [np.mean(total_length_list), np.mean(num_branch_list)],
#     [np.mean(total_length_list) - np.std(total_length_list), np.mean(num_branch_list) - np.std(num_branch_list)]
# ]
# 中位数和四分位数
threshold = [
    [np.median(total_length_list), np.median(num_branch_list)],
    [np.percentile(total_length_list, 25), np.percentile(num_branch_list, 25)]
]
print("threshold: ", threshold)

unlabeled_list_file = "/data/kfchen/trace_ws/paper_trace_result/csv_copy/unlabeled_list.csv"
unlabeled_list = pd.read_csv(unlabeled_list_file)['id'].tolist()
print(len(unlabeled_list), unlabeled_list[:5])
auto_l_measure_file = "/data/kfchen/trace_ws/paper_trace_result/nnunet/proposed_9k/8_estimated_radius_swc_l_measure.csv"
auto_l_measure_df = pd.read_csv(auto_l_measure_file)

label_name = ['Gold', 'Silver', 'Bronze']
label_list = [[], [], []]

for i in range(len(auto_l_measure_df)):
    if auto_l_measure_df.loc[i, 'ID'] not in unlabeled_list:
        continue
    total_length = auto_l_measure_df.loc[i, 'Total Length']
    num_branch = auto_l_measure_df.loc[i, 'Number of Branches']

    if(total_length > threshold[0][0] and num_branch > threshold[0][1]):
        label_list[0].append(auto_l_measure_df.loc[i, 'ID'])
    elif(total_length > threshold[1][0] and num_branch > threshold[1][1]):
        label_list[1].append(auto_l_measure_df.loc[i, 'ID'])
    else:
        label_list[2].append(auto_l_measure_df.loc[i, 'ID'])

print(len(label_list[0]), len(label_list[0]) / len(unlabeled_list),
      len(label_list[1]), len(label_list[1]) / len(unlabeled_list),
      len(label_list[2]), len(label_list[2]) / len(unlabeled_list))

# ID > 6000
auto_l_measure_df = auto_l_measure_df[auto_l_measure_df['ID'] > 6200]
total_length_list = auto_l_measure_df["Total Length"].tolist()
num_branch_list = auto_l_measure_df["Number of Branches"].tolist()

print("total_length_list mean: ", np.mean(total_length_list))
print("num_branch_list mean: ", np.mean(num_branch_list))
# > 6000: 2476, 64
# < 6000 1831, 62

# qq_plot(total_length_list, "Total Length")
# qq_plot(num_branch_list, "Number of Branches")

