import os
import numpy as np
import pandas as pd
import glob

from pylib.swc_handler import crop_spheric_from_soma
from simple_swc_tool.l_measure_api import l_measure_swc_dir
import shutil
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def shperic_cropping(in_dir, out_dir, radius, remove_axon=True):
    ninswc = 0
    for inswc in glob.glob(os.path.join(in_dir, '*.swc')):
        filename = os.path.split(inswc)[-1]
        outswc = os.path.join(out_dir, filename)
        if os.path.exists(outswc):
            continue
        ninswc += 1
        # if ninswc % 20 == 0:
        #     print(filename)
        df_tree = pd.read_csv(inswc, comment='#', sep=' ', index_col=0 ,
                     names=('id', 'type', 'x', 'y', 'z', 'r', 'pid'))
        if remove_axon:
            df_tree = df_tree[df_tree.type != 2]

        # cropping
        tree_out = crop_spheric_from_soma(df_tree, radius)

        tree_out.to_csv(outswc, sep=' ', index=True, header=False)

dataset = {
    'human': [
        "allen_human_neuromorpho",
        # "allman",
        # "DeKock",
        # "hrvoj-mihic_semendeferi",
        "jacobs",
        # "semendeferi_muotri",
        "vdheuvel",
        "proposed",
    ],
    'mouse':[
        'seu1876',
    ],
}
instersted_feasures = [
    'N_stem', 'Number of Bifurcatons',
    'Number of Branches', 'Number of Tips', 'Overall Width', 'Overall Height',
    'Overall Depth', 'Total Length',
    'Max Euclidean Distance', 'Max Path Distance',
    'Max Branch Order',
]

radius = 100
qc_root = "/data/kfchen/trace_ws/quality_control_test"
# swc_root_dirs = {}
swc_cropped_dirs = {}
l_measure_files = {}
for key in dataset.keys():
    # swc_root_dirs[key] = []
    swc_cropped_dirs[key] = []
    l_measure_files[key] = []
    for dataset_name in dataset[key]:
        current_swc_root = os.path.join(qc_root, key, dataset_name)
        swc_raw_dir = os.path.join(current_swc_root, "raw")
        if(not os.path.exists(swc_raw_dir)):
            old_raw_dir = os.path.join(current_swc_root, "CNG_version")
            if(os.path.exists(old_raw_dir)):
                shutil.move(old_raw_dir, swc_raw_dir)
            else:
                continue

        swc_cropped_dir = os.path.join(current_swc_root, "cropped_" + str(radius) + "um")
        if(not os.path.exists(swc_cropped_dir)):
            old_cropped_dir = os.path.join(current_swc_root, "one_point_soma_" + str(radius) + "um")
            if(os.path.exists(old_cropped_dir)):
                shutil.move(old_cropped_dir, swc_cropped_dir)
            else:
                os.makedirs(swc_cropped_dir, exist_ok=True)
                shperic_cropping(swc_raw_dir, swc_cropped_dir, radius)

        l_measure_result_file = os.path.join(current_swc_root, "l_measure_" + str(radius) + "um.csv")
        if(not os.path.exists(l_measure_result_file)):
            old_l_measure_result_file = os.path.join(current_swc_root, "one_point_soma_" + str(radius) + "um_l_measure.csv")
            if(os.path.exists(old_l_measure_result_file)):
                shutil.move(old_l_measure_result_file, l_measure_result_file)
            else:
                l_measure_swc_dir(swc_cropped_dir, l_measure_result_file)

        swc_cropped_dirs[key].append(swc_cropped_dir)
        l_measure_files[key].append(l_measure_result_file)
        print(f"{key} {dataset_name} {len(os.listdir(swc_cropped_dir))} {l_measure_result_file} done")

all_features = []
dataset_labels = []  # 用于标记每个样本所属的数据集
dataset_num = []

Set3_colors = list(plt.cm.get_cmap('Set3').colors)
candidat_emarker_list = ['o', 's', 'p', 'P', '*', 'h', 'H', 'X', 'D', 'd', 'v', '^', '<', '>', '1', '2', '3', '4', '8']
color_list, marker_list = [], []
for key in l_measure_files.keys():
    key_color = Set3_colors.pop(0)
    key_marker = candidat_emarker_list.pop(0)
    for i, file_path in enumerate(l_measure_files[key]):
        dataset_name = dataset[key][i]
        df = pd.read_csv(file_path)  # 读取 CSV 文件
        # 删掉带有 NaN 的行
        df = df.dropna()
        features = df[instersted_feasures].values  # 仅获取感兴趣的特征

        # all_features.append(features)
        # dataset_labels.extend([instersted_dataset_name[i]] * features.shape[0])  # 每个样本都标记所属的数据集
        # dataset_num.append(features.shape[0])

        # 平均值模式
        all_features.append(features.mean(axis=0))
        dataset_labels.append(dataset_name)
        dataset_num.append(features.shape[0])
        # 颜色
        # 在key_color的基础上，加一点波动
        # current_color = np.array(key_color) + np.random.rand(3) * 0.1
        # current_color = np.clip(current_color, 0, 1)
        color_list.append(key_color)
        marker_list.append(key_marker)

all_features = np.vstack(all_features)
print(all_features.shape, l_measure_files)
pca = PCA(n_components=2)
reduced_features = pca.fit_transform(all_features)

# 绘制 PCA 散点图
plt.figure(figsize=(8, 6))
for i, label in enumerate(dataset_labels):
    plt.scatter(reduced_features[i, 0], reduced_features[i, 1], label=label, s=100, marker=marker_list[i])
plt.legend()
plt.xlabel('PCA1')
plt.ylabel('PCA2')
plt.title('PCA of Neuron Features')
plt.show()
plt.close()



