import os
import pandas as pd
import tifffile
from tqdm import tqdm
from joblib import Parallel, delayed
import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from simple_swc_tool.l_measure_api import l_measure_swc_dir
from simple_swc_tool.opt_topology_analyse import my_opt, opt_analyse_dir_v2
from sklearn.tree import DecisionTreeClassifier

from pylib.swc_handler import parse_swc, trim_swc, crop_spheric_from_soma
N_JOBS = 20
neuron_info_df = pd.read_csv(
    "/data/kfchen/nnUNet/nnUNet_results/Dataset169_hb_10k/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/ptls10/norm_result/Human_SingleCell_TrackingTable_20240712.csv",
    encoding='gbk')


def prepare_extended_swc_info():
    # test_list_file = "/data/kfchen/trace_ws/paper_trace_result/csv_copy/test_list_with_gs.csv"
    l_measure_file = "/data/kfchen/trace_ws/cropped_swc/proposed_1um_l_measure.csv"
    l_measure_df = pd.read_csv(l_measure_file)
    img_dir = "/data/kfchen/trace_ws/paper_trace_result/nnunet/proposed_9k/3_skel_with_soma"

    extended_swc_info_file = "/data/kfchen/trace_ws/cropped_swc/extended_swc_info.csv"

    if(os.path.exists(extended_swc_info_file) == False):
        # img_id_map = {}
        # img_files = [f for f in os.listdir(img_dir) if f.endswith(".tif")]
        # for img_file in img_files:
        #     id = int(img_file.split("_")[0])
        #     img_id_map[id] = img_file
        # # 新加一列
        # l_measure_df["img_x"], l_measure_df["img_y"], l_measure_df["img_z"] = 0, 0, 0
        # # l_measure_df.to_csv(extended_swc_info_file, index=False)
        # # exit()
        # # 遍历每一列
        # # for i in range(len(l_measure_df)):
        # # tqdm
        #
        # def current_task(i, l_measure_df, img_id_map, img_dir):
        #     id = l_measure_df.loc[i, "ID"]
        #     xy_resolution = neuron_info_df.loc[neuron_info_df.iloc[:, 0] == int(id), 'xy拍摄分辨率(*10e-3μm/px)'].values[0]
        #     xy_resolution = float(xy_resolution) / 1000
        #     if id in img_id_map:
        #         img_file = img_id_map[id]
        #         img_file = os.path.join(img_dir, img_file)
        #         img = tifffile.imread(img_file)
        #         img_z, img_y, img_x = img.shape
        #         img_x = int(img_x * xy_resolution)
        #         img_y = int(img_y * xy_resolution)
        #         return i, img_x, img_y, img_z
        #     return i, None, None, None
        #
        # # for i in tqdm(range(len(l_measure_df))):
        # #     id = l_measure_df.loc[i, "ID"]
        # #     if id in img_id_map:
        # #         img_file = img_id_map[id]
        # #         img_file = os.path.join(img_dir, img_file)
        # #         img = tifffile.imread(img_file)
        # #         img_z, img_y, img_x = img.shape
        # #         l_measure_df.loc[i, "img_x"] = img_x
        # #         l_measure_df.loc[i, "img_y"] = img_y
        # #         l_measure_df.loc[i, "img_z"] = img_z
        # results = Parallel(n_jobs=N_JOBS)(
        #     delayed(current_task)(i, l_measure_df, img_id_map, img_dir)
        #     for i in tqdm(range(len(l_measure_df)))
        # )
        # # 将结果写回 DataFrame
        # for i, img_x, img_y, img_z in results:
        #     if img_x is not None:
        #         l_measure_df.loc[i, "img_x"] = img_x
        #         l_measure_df.loc[i, "img_y"] = img_y
            #         l_measure_df.loc[i, "img_z"] = img_z
            # l_measure_df.to_csv(extended_swc_info_file, index=False)
        pass
    l_measure_df = pd.read_csv(extended_swc_info_file)
    test_id = []
    l_measure_df["optj_f1"], l_measure_df["optp_con_prob_f1"], l_measure_df["optg_f1"] = float(0), float(0), float(0)
    opt_file = "/data/kfchen/trace_ws/cropped_swc/proposed_1um_opt_result.csv"
    opt_df = pd.read_csv(opt_file)
    # optj_f1, optp_con_prob_f1, optg_f1
    # 遍历每一行
    for i in range(len(l_measure_df)):
        id = l_measure_df.loc[i, "ID"]
        opt_row = opt_df[opt_df["ID"] == id]
        # print(opt_row)
        if len(opt_row) == 0:
            continue
        # print(id)
        current_opt = opt_row["optj_f1"].values[0], opt_row["optp_con_prob_f1"].values[0], opt_row["optg_f1"].values[0]
        if(current_opt[0] == 0 or current_opt[1] == 0 or current_opt[2] == 0):
            continue
        test_id.append(id)
        l_measure_df.loc[i, "optj_f1"] = opt_row["optj_f1"].values[0]
        l_measure_df.loc[i, "optp_con_prob_f1"] = opt_row["optp_con_prob_f1"].values[0]
        l_measure_df.loc[i, "optg_f1"] = opt_row["optg_f1"].values[0]

    # save
    l_measure_df.to_csv(extended_swc_info_file, index=False)

    l_measure_df = pd.read_csv(extended_swc_info_file)
    l_measure_df['branch_integrity'], l_measure_df['tip_integrity'], l_measure_df['length_integrity'] = float(0), float(0), float(0)
    gt_l_measure_file = "/data/kfchen/trace_ws/cropped_swc/manual_1um_l_measure.csv"
    gt_l_measure_df = pd.read_csv(gt_l_measure_file)
    for i in range(len(l_measure_df)):
        id = l_measure_df.loc[i, "ID"]
        gt_row = gt_l_measure_df[gt_l_measure_df["ID"] == id]
        if len(gt_row) == 0:
            continue
        l_measure_df.loc[i, "branch_integrity"] = l_measure_df.loc[i, "Number of Branches"] / gt_row["Number of Branches"].values[0]
        l_measure_df.loc[i, "tip_integrity"] = l_measure_df.loc[i, "Number of Tips"] / gt_row["Number of Tips"].values[0]
        l_measure_df.loc[i, "length_integrity"] = l_measure_df.loc[i, "Total Length"] / gt_row["Total Length"].values[0]

    l_measure_df.to_csv(extended_swc_info_file, index=False)

    test_id_file = "/data/kfchen/trace_ws/paper_trace_result/csv_copy/test_list_with_gs.csv"
    test_id = pd.read_csv(test_id_file)["id"].tolist()
    l_measure_df = pd.read_csv(extended_swc_info_file)
    l_measure_df["quality"] = 0
    quality_label = [1, 2, 3]
    # quality_label_threshold = [
    #     [0.8 for _ in range(6)],
    #     [0.7 for _ in range(6)],
    #     [0.6 for _ in range(6)],
    # ]
    score_list = [[] for _ in range(6)]
    for i in range(len(l_measure_df)):
        current_id = l_measure_df.loc[i, "ID"]
        if(current_id not in test_id):
            continue
        current_score_list = (l_measure_df.loc[i, "optj_f1"], l_measure_df.loc[i, "optp_con_prob_f1"], l_measure_df.loc[i, "optg_f1"],
                              l_measure_df.loc[i, "branch_integrity"], l_measure_df.loc[i, "tip_integrity"], l_measure_df.loc[i, "length_integrity"])
        if(current_score_list[0:3] == (0, 0, 0) or current_score_list[3:6] == (0, 0, 0)):
            continue
        for j in range(6):
            score_list[j].append(current_score_list[j])
    # drop na
    score_list = [np.array(score) for score in score_list]
    score_list = [score[~np.isnan(score)] for score in score_list]
    print(score_list)
    quality_label_threshold = [
        [np.mean(score_list[j]) * 1.0 + 0.0 * np.std(score_list[j]) for j in range(6)],
        [np.mean(score_list[j]) * 0.9 - 0.0 * np.std(score_list[j]) for j in range(6)],
        [np.mean(score_list[j]) * 0.8 - 0.0 * np.std(score_list[j]) for j in range(6)],
    ]
    print(quality_label_threshold)

    # 遍历每一行
    for i in range(len(l_measure_df)):
        current_id = l_measure_df.loc[i, "ID"]
        if(current_id not in test_id):
            continue
        current_score_list = (l_measure_df.loc[i, "optj_f1"], l_measure_df.loc[i, "optp_con_prob_f1"], l_measure_df.loc[i, "optg_f1"],
                              l_measure_df.loc[i, "branch_integrity"], l_measure_df.loc[i, "tip_integrity"], l_measure_df.loc[i, "length_integrity"])
        if(current_score_list[0:3] == (0, 0, 0) or current_score_list[3:6] == (0, 0, 0)):
            continue
        l_measure_df.loc[i, "quality"] = quality_label[-1] + 1
        for j in range(len(quality_label)):
            if all([current_score >= threshold for current_score, threshold in zip(current_score_list, quality_label_threshold[j])]):
                l_measure_df.loc[i, "quality"] = quality_label[j]
                break
        # print(l_measure_df.loc[i, "quality"])

    # 检查占比
    print(l_measure_df["quality"].value_counts())
    l_measure_df.to_csv(extended_swc_info_file, index=False)

def train_classifier():
    extended_swc_info_file = "/data/kfchen/trace_ws/quality_control_classifier/extended_swc_info.csv"
    l_measure_df = pd.read_csv(extended_swc_info_file)
    l_measure_df = l_measure_df[l_measure_df["quality"] != 0]

    X = l_measure_df.iloc[:, 1:-7].values  # 特征列，去掉最后7列
    y = l_measure_df.iloc[:, -1].values  # 标签列，最后一列

    # 标签进行 one-hot 编码（标签是 1 到 4）
    y = np.array(y - 1)  # 标签从 1 到 4，所以减去 1，使得标签从 0 到 3

    # 数据标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 将数据转为 PyTorch 张量
    X_scaled = torch.tensor(X_scaled, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)  # PyTorch 使用 LongTensor 作为分类标签

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # 创建 DataLoader
    train_data = TensorDataset(X_train, y_train)
    test_data = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

    # 定义 CNN 模型
    class CNN(nn.Module):
        def __init__(self):
            super(CNN, self).__init__()
            self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3, padding=1)
            self.pool = nn.MaxPool1d(2)
            self.fc1 = nn.Linear(64 * (X_train.shape[1] // 2), 128)  # 根据池化层调整大小
            self.fc2 = nn.Linear(128, 4)  # 4 类输出

        def forward(self, x):
            x = self.pool(torch.relu(self.conv1(x)))
            x = x.view(x.size(0), -1)  # 展平为一维
            x = torch.relu(self.fc1(x))
            x = self.fc2(x)
            return x

    # 实例化模型
    model = CNN()

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()  # 适用于多类分类
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练模型
    num_epochs = 1000

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            # 添加一个额外的维度，以适应 Conv1D 输入
            inputs = inputs.unsqueeze(1)  # (batch_size, 1, features)

            optimizer.zero_grad()  # 清零梯度
            outputs = model(inputs)  # 前向传播
            loss = criterion(outputs, labels)  # 计算损失
            loss.backward()  # 反向传播
            optimizer.step()  # 更新权重

            running_loss += loss.item()

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader)}")

    # 评估模型
    model.eval()
    y_pred = []
    y_true = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.unsqueeze(1)  # (batch_size, 1, features)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            y_pred.extend(predicted.numpy())
            y_true.extend(labels.numpy())

    # 计算准确率
    accuracy = accuracy_score(y_true, y_pred)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

def train_classifier_v2():
    extended_swc_info_file = "/data/kfchen/trace_ws/cropped_swc/extended_swc_info.csv"
    l_measure_df = pd.read_csv(extended_swc_info_file)
    l_measure_df = l_measure_df[l_measure_df["quality"] != 0]

    X = l_measure_df.iloc[:, 1:-7].values  # 特征列，去掉最后7列
    y = l_measure_df.iloc[:, -1].values  # 标签列，最后一列
    y = np.array(y - 1)  # 标签从 1 到 4，所以减去 1，使得标签从 0 到 3
    print(y)
    print("shape", X.shape, y.shape)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', learning_rate=0.01,)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")

def train_classifier_v3():
    extended_swc_info_file = "/data/kfchen/trace_ws/cropped_swc/extended_swc_info.csv"
    l_measure_df = pd.read_csv(extended_swc_info_file)
    l_measure_df = l_measure_df[l_measure_df["quality"] != 0]

    X = l_measure_df.iloc[:, 1:-7].values  # 特征列，去掉最后7列
    y = l_measure_df.iloc[:, -1].values  # 标签列，最后一列
    y = np.array(y - 1)  # 标签从 1 到 4，所以减去 1，使得标签从 0 到 3
    print(y)
    print("shape", X.shape, y.shape)

    # MLP
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    classifier = DecisionTreeClassifier(random_state=42)

    # 训练模型
    classifier.fit(X_train, y_train)

    # 在测试集上进行预测
    y_pred = classifier.predict(X_test)

    # 计算并输出准确率
    accuracy = accuracy_score(y_test, y_pred)
    print(f"分类器的准确率: {accuracy:.2f}")



def prepare_unlabeled():
    source_dir = "/data/kfchen/trace_ws/paper_trace_result/nnunet/proposed_9k/8_estimated_radius_swc"
    target_dir = "/data/kfchen/trace_ws/quality_control_classifier/unlabeled/raw"
    unlabeled_list_file = "/data/kfchen/trace_ws/paper_trace_result/csv_copy/unlabeled_list.csv"
    unlabeled_list = pd.read_csv(unlabeled_list_file)["id"].tolist()
    train_list_file = "/data/kfchen/trace_ws/paper_trace_result/csv_copy/train_val_list.csv"
    train_list = pd.read_csv(train_list_file)["id"].tolist()
    test_list_file = "/data/kfchen/trace_ws/paper_trace_result/csv_copy/test_list_with_gs.csv"
    test_list = pd.read_csv(test_list_file)["id"].tolist()
    total_list = train_list + test_list + unlabeled_list
    print((len(total_list)))


    l_measure_df_file = "/data/kfchen/trace_ws/cropped_swc/proposed_1um_l_measure.csv"
    l_measure_df = pd.read_csv(l_measure_df_file)
    l_measure_df = l_measure_df[l_measure_df["ID"].isin(total_list)]
    print((len(l_measure_df)))
    l_measure_df.to_csv("/data/kfchen/trace_ws/cropped_swc/proposed_1um_l_measure_total.csv", index=False)
    exit()

    print((len(unlabeled_list)))

    swc_files = [f for f in os.listdir(source_dir) if f.endswith(".swc")]
    for swc_file in swc_files:
        id = int(swc_file.split("_")[0])
        if id in unlabeled_list:
            source_path = os.path.join(source_dir, swc_file)
            target_path = os.path.join(target_dir, swc_file)
            os.system("cp %s %s" % (source_path, target_path))

feature_name_mapping = {
    'N_node': 'No. of Nodes',
    'Number of Bifurcatons': 'No. of Bifurcations',
    'Number of Branches': 'No. of Branches',
    'Number of Tips': 'No. of Tips',
    'Overall Width': 'Width',
    'Overall Height': 'Height',
    'Overall Depth': 'Depth',
    'Total Length': 'Length',
    'Max Euclidean Distance': 'Max Euclidean',
    'Max Path Distance': 'Max path Dist.',
    'Max Branch Order': 'Max Branch Order',
}
meaningful_feasures = ['N_node', 'Number of Bifurcatons', 'Number of Branches', 'Number of Tips',
                           'Overall Width', 'Overall Height', 'Overall Depth', 'Total Length',
                           'Max Euclidean Distance', 'Max Path Distance', 'Max Branch Order',
                           ]
def plot_corr():
    extended_swc_info_file = "/data/kfchen/trace_ws/cropped_swc/extended_swc_info.csv"
    l_measure_df = pd.read_csv(extended_swc_info_file)
    l_measure_df = l_measure_df[l_measure_df["quality"] != 0]

    # df_columns = l_measure_df.iloc[:, 1:-10]  # 特征列，去掉最后7列
    df_columns = l_measure_df[meaningful_feasures]
    df_rows = l_measure_df.iloc[:, -4:-1]  # 标签列，最后一列
    df_rows["mean_f1"] = df_rows.mean(axis=1)
    # 归一化
    for column in df_columns.columns:
        df_columns[column] = (df_columns[column] - df_columns[column].min()) / (df_columns[column].max() - df_columns[column].min())
    # 计算列与行之间的相关性矩阵
    correlation_matrix = np.corrcoef(df_columns, df_rows, rowvar=False)
    print(correlation_matrix.shape)

    # 重塑数据为一个二维矩阵
    correlation_matrix = correlation_matrix[:df_columns.shape[1], df_columns.shape[1]:].T
    print(correlation_matrix.shape)
    print(correlation_matrix)

    # 设置清晰度 300
    plt.rcParams['savefig.dpi'] = 300

    # 创建热力图
    plt.figure(figsize=(6, 3))
    xticklabels = [feature_name_mapping[feature] for feature in meaningful_feasures]
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", xticklabels=xticklabels,
                yticklabels=['OPT-J', "OPT-P", "OPT-G", "Mean f1"],cbar=True)

    # 设置标题
    # plt.title("Correlation Heatmap", fontsize=16)
    plt.xlabel("l-measure features", fontsize=12)
    plt.ylabel("opt f1", fontsize=12)
    # yticks
    plt.yticks(rotation=0)

    # 显示图像
    plt.tight_layout()
    plt.show()
    plt.close()

def crop_swc_dir(source_swc_dir, target_swc_dir, radius=150):
    swc_files = [f for f in os.listdir(source_swc_dir)]
    def current_task(source_swc_file, target_swc_file, radius, remove_axon=False):
        if(os.path.exists(target_swc_file)):
            return
        df_tree = pd.read_csv(source_swc_file, comment='#', sep=' ', index_col=0,
                              names=('id', 'type', 'x', 'y', 'z', 'r', 'pid'))
        if remove_axon:
            df_tree = df_tree[df_tree.type != 2]

        tree_out = crop_spheric_from_soma(df_tree, radius)
        tree_out.to_csv(target_swc_file, sep=' ', index=True, header=False)

    joblib.Parallel(n_jobs=N_JOBS)(
        joblib.delayed(current_task)(
            os.path.join(source_swc_dir, swc_file),
            os.path.join(target_swc_dir, swc_file),
            radius,
            False
        )
        for swc_file in tqdm(swc_files)
    )

if __name__ == "__main__":
    prepare_unlabeled()
    # crop_swc_dir("/data/kfchen/trace_ws/paper_trace_result/nnunet/proposed_9k/8_estimated_radius_swc",
    #              "/data/kfchen/trace_ws/cropped_swc/proposed_1um")
    # crop_swc_dir("/data/kfchen/trace_ws/paper_auto_human_neuron_recon/swc_label/1um_swc_lab",
    #              "/data/kfchen/trace_ws/cropped_swc/manual_1um")
    cropped_auto_swc_dir = "/data/kfchen/trace_ws/cropped_swc/proposed_1um"
    cropped_manual_swc_dir = "/data/kfchen/trace_ws/cropped_swc/manual_1um"
    # l_measure_swc_dir(cropped_auto_swc_dir, cropped_auto_swc_dir + "_l_measure.csv")
    # l_measure_swc_dir(cropped_manual_swc_dir, cropped_manual_swc_dir + "_l_measure.csv")

    # opt_analyse_dir_v2(cropped_manual_swc_dir, cropped_auto_swc_dir)


    # prepare_extended_swc_info()
    # pass
    # train_classifier()
    # train_classifier_v2()
    # train_classifier_v3()
    # prepare_unlabeled()
    plot_corr()
    # pass