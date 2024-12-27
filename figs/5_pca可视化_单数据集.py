import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import seaborn as sns
import os

target_swc_root = "/data/kfchen/trace_ws/quality_control_test"
instersted_dataset_name = [
    "allen_human_neuromorpho",
    "allman",
    # "ataman_boulting",
    "DeKock",
    "hrvoj-mihic_semendeferi",
    "jacobs",
    # "segev",
    "semendeferi_muotri",
    "vdheuvel",
    # "vuksic",
    # "wittner",
    "proposed",
]
instersted_feasures = [
    'N_stem', 'Number of Bifurcatons',
    'Number of Branches', 'Number of Tips', 'Overall Width', 'Overall Height',
    'Overall Depth', 'Total Length',
    'Max Euclidean Distance', 'Max Path Distance',
    'Max Branch Order',
]
radius = 100
lim = (100, 100, 50)
# 假设文件路径在 file_paths 列表中
l_measure_result_files = []
for name in instersted_dataset_name:
    l_measure_result_files.append(os.path.join(target_swc_root, name, "one_point_soma_" + str(radius) + "um_l_measure.csv"))
    # l_measure_result_file = os.path.join(target_swc_root, name,
    #                                      "one_point_soma_box_" + str(lim[0]) + str(lim[1]) + "um_l_measure.csv")
    # l_measure_result_files.append(l_measure_result_file)
# 用于存储所有样本的特征和标签
all_features = []
dataset_labels = []  # 用于标记每个样本所属的数据集
dataset_num = []

# 读取每个 CSV 文件，并将特征数据存储到 all_features 中
for i, file_path in enumerate(l_measure_result_files):
    df = pd.read_csv(file_path)  # 读取 CSV 文件
    # 删掉带有 NaN 的行
    df = df.dropna()
    features = df[instersted_feasures].values  # 仅获取感兴趣的特征

    all_features.append(features)
    dataset_labels.extend([instersted_dataset_name[i]] * features.shape[0])  # 每个样本都标记所属的数据集
    dataset_num.append(features.shape[0])

    # 平均值模式
    # all_features.append(features.mean(axis=0))
    # dataset_labels.append(instersted_dataset_name[i])
    # dataset_num.append(features.shape[0])

# 将所有特征数据合并成一个大矩阵
all_features = np.vstack(all_features)

# 2. 使用 PCA 将特征降到二维
pca = PCA(n_components=2)
reduced_features = pca.fit_transform(all_features)
# reduced_features = reduced_features[:-dataset_num[-1], :]
# dataset_labels = dataset_labels[:-dataset_num[-1]]


# 计算每个数据集的中心点
centroids = []
for dataset_name in instersted_dataset_name:
    # 找出属于当前数据集的所有样本
    dataset_indices = [i for i, label in enumerate(dataset_labels) if label == dataset_name]

    # 获取对应样本的降维特征
    dataset_features = reduced_features[dataset_indices]

    # 计算降维特征的均值（即中心点）
    centroid = np.mean(dataset_features, axis=0)
    centroids.append(centroid)

# 可视化：绘制每个数据集的散点图
plt.figure(figsize=(8, 6))
sns.scatterplot(x=reduced_features[:, 0], y=reduced_features[:, 1], hue=dataset_labels, palette='Set1', s=10, alpha=0.2, legend=None)

# 标出每个数据集的中心点
for i, centroid in enumerate(centroids):
    plt.scatter(centroid[0], centroid[1], color=sns.color_palette('Set1')[i], s=200, marker='X',
                label=f'{instersted_dataset_name[i]} Center')

plt.xlim(-2000, 4000)
plt.ylim(-100, 150)

# 设置标题和标签
plt.title("PCA - Feature Reduction with Datasets and Centers", fontsize=16)
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")

# 显示图例
plt.legend()

# 显示图表
plt.show()
plt.close()