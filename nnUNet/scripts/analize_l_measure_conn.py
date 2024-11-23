import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns
import numpy as np
from scipy.spatial.distance import squareform
from skbio.stats.distance import mantel
from sklearn.manifold import MDS
from scipy.cluster.hierarchy import linkage, dendrogram

from sklearn.metrics import adjusted_rand_score
from scipy.cluster.hierarchy import fcluster


def plot_heatmap(corr_matrix, title):
    # 3. 可视化 - 热力图
    plt.figure(figsize=(10, 8))
    heatmap = sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
    heatmap.set_title(title, fontdict={'fontsize': 18}, pad=16)
    plt.show()
    plt.close()


def compare_heat_map(manual_corr, auto_corr):
    # 设置颜色映射的范围（确保两图一致）
    vmin = min(manual_corr.min().min(), auto_corr.min().min())
    vmax = max(manual_corr.max().max(), auto_corr.max().max())

    # 创建子图
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=True)

    # 绘制手动标注的相关性热力图
    sns.heatmap(manual_corr, ax=axes[0], cmap='coolwarm', annot=False, fmt=".2f",
                cbar_kws={'shrink': 0.8}, vmin=vmin, vmax=vmax, cbar=False)
    axes[0].set_title('Manual Annotation', fontsize=24)
    axes[0].set_xticks([])
    axes[0].tick_params(axis='y', rotation=0, labelsize=16)

    # 绘制自动处理的相关性热力图
    sns.heatmap(auto_corr, ax=axes[1], cmap='coolwarm', annot=False, fmt=".2f",
                cbar_kws={'shrink': 0.8}, vmin=vmin, vmax=vmax)
    axes[1].set_title('Auto Annotation', fontsize=24)
    # axes[1].tick_params(axis='x', rotation=45)
    # 关闭x
    axes[1].set_xticks([])
    axes[1].set_yticks([])

    # 添加总标题
    # fig.suptitle('Comparison of Correlation Matrices', fontsize=16, y=1.02)

    # 显示图表
    plt.show()
    plt.close()

if __name__ == '__main__':
    mutineuron_list_file = "/data/kfchen/trace_ws/paper_trace_result/mutineuron_list.csv"
    mutineuron_list = pd.read_csv(mutineuron_list_file)['id'].tolist()
    train_val_list_file = "/data/kfchen/trace_ws/paper_trace_result/train&val_list.csv"
    train_val_list = pd.read_csv(train_val_list_file)['id'].tolist()


    important_feasures = ['Number of Branches', 'Number of Tips', 'Total Length']
    meaningful_feasures = ['N_node', 'Number of Bifurcatons', 'Number of Branches', 'Number of Tips',
                           'Overall Width', 'Overall Height', 'Overall Depth', 'Total Length',
                           'Max Euclidean Distance', 'Max Path Distance', 'Max Branch Order',
                           'Average Bifurcation Angle Remote']
    other_feasures = ['Soma_surface', 'N_stem', 'Average Diameter', 'Total Surface', 'Total Volume',
                        'Average Contraction', 'Average Fragmentation', 'Average Parent-daughter Ratio',
                        'Average Bifurcation Angle Local', 'Hausdorff Dimension']


    l_measure_result_file = "/data/kfchen/trace_ws/paper_auto_human_neuron_recon/swc_label/1um_swc_lab_l_measure.csv"
    # 读取 CSV 文件
    data = pd.read_csv(l_measure_result_file)
    # 根据样本编号筛选行（例如，样本编号以 'A' 开头）
    filtered_data = data[(~data['ID'].isin(mutineuron_list)) & (data['ID'].isin(train_val_list))]
    print(len(filtered_data))
    features = filtered_data.drop(columns=['ID'])
    manual_corr = features.corr()
    # plot_heatmap(manual_corr, 'Manual_anno')


    l_measure_result_file = "/data/kfchen/trace_ws/paper_trace_result/nnunet/proposed_9k/8_estimated_radius_swc_l_measure.csv"
    # 读取 CSV 文件
    data = pd.read_csv(l_measure_result_file)
    # 根据样本编号筛选行（例如，样本编号以 'A' 开头）
    filtered_data = data[(~data['ID'].isin(mutineuron_list)) & (~data['ID'].isin(train_val_list))]
    # filtered_id = filtered_data['ID'].tolist()
    # ids = [str(int(f)) for f in filtered_id]
    # ids = np.unique(ids)
    # ids = np.sort(ids)
    # df = pd.DataFrame(ids, columns=['id'])
    # df.to_csv(r"/data/kfchen/trace_ws/paper_trace_result/test_list_without_gs.csv", index=False)
    # exit()
    print(len(filtered_data))
    features = filtered_data.drop(columns=['ID'])
    auto_corr = features.corr()
    # plot_heatmap(auto_corr, 'Auto_anno')


    compare_heat_map(manual_corr, auto_corr)


    # diff_corr = manual_corr - auto_corr
    # # 计算平均绝对差值
    # mean_abs_diff = np.mean(np.abs(diff_corr.values.flatten()))
    # print(f'平均绝对差值: {mean_abs_diff}')
    # # 计算均方根误差
    # rmse = np.sqrt(np.mean((diff_corr.values.flatten()) ** 2))
    # print(f'均方根误差: {rmse}')
    #
    #
    # manual_flat = manual_corr.values.flatten()
    # auto_flat = auto_corr.values.flatten()
    # # 计算相关系数
    # corr_between_matrices = np.corrcoef(manual_flat, auto_flat)[0, 1]
    # print(f'两个相关性矩阵之间的相关系数: {corr_between_matrices}')
    #
    #
    # manual_dist = 1 - np.abs(manual_corr)
    # auto_dist = 1 - np.abs(auto_corr)
    # # 进行Mantel检验
    # statistic, p_value, n = mantel(squareform(manual_dist), squareform(auto_dist), method='pearson', permutations=999)
    # print(f'Mantel检验统计量: {statistic}, p值: {p_value}')
    #
    #
    # # 设置随机种子
    # seed = 42
    # # 对人工标注结果进行MDS
    # mds = MDS(n_components=2, dissimilarity='precomputed', random_state=seed)
    # manual_mds = mds.fit_transform(1 - manual_corr)
    # # 对自动处理结果进行MDS
    # auto_mds = mds.fit_transform(1 - auto_corr)
    # # 绘制MDS结果
    # fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    # axes[0].scatter(manual_mds[:, 0], manual_mds[:, 1])
    # axes[0].set_title('人工标注结果的MDS')
    # axes[1].scatter(auto_mds[:, 0], auto_mds[:, 1])
    # axes[1].set_title('自动处理结果的MDS')
    # plt.show()
    # plt.close()
    #
    #
    #
    # # 对人工标注结果聚类
    # manual_linkage = linkage(manual_corr, method='average')
    # plt.figure(figsize=(8, 6))
    # dendrogram(manual_linkage, labels=manual_corr.columns)
    # plt.title('人工标注结果的聚类树')
    # plt.show()
    # # 对自动处理结果聚类
    # auto_linkage = linkage(auto_corr, method='average')
    # plt.figure(figsize=(8, 6))
    # dendrogram(auto_linkage, labels=auto_corr.columns)
    # plt.title('自动处理结果的聚类树')
    # plt.show()
    #
    #
    #
    # manual_labels = fcluster(manual_linkage, t=5, criterion='maxclust')
    # auto_labels = fcluster(auto_linkage, t=5, criterion='maxclust')
    # # 计算ARI
    # ari = adjusted_rand_score(manual_labels, auto_labels)
    # print(f'Adjusted Rand Index (ARI): {ari}')


    '''
    平均绝对差值: 0.13661615688188364
    均方根误差: 0.1979792632812942
    两个相关性矩阵之间的相关系数: 0.8553338278217363
    Mantel检验统计量: 0.8485404328713803, p值: 0.001
    Adjusted Rand Index (ARI): 0.5583849506767607
    
    '''







