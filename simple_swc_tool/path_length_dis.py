import pandas as pd
import networkx as nx
import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np
from sympy.abc import alpha

neuron_info_df = pd.read_csv("/data/kfchen/nnUNet/nnUNet_results/Dataset169_hb_10k/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/ptls10/norm_result/Human_SingleCell_TrackingTable_20240712.csv", encoding='gbk')
def find_resolution(filename):
    filename = os.path.basename(filename)
    # print(filename)
    df = neuron_info_df
    filename = int(filename.split('.')[0].split('_')[0])
    for i in range(len(df)):
        if int(df.iloc[i, 0]) == filename:
            return df.iloc[i, 43]

def load_swc_to_undirected_graph(swc_file_path, upsample):
    """从SWC文件加载数据，构建无向图，并记录每个节点的parent信息"""
    df = pd.read_csv(swc_file_path, delim_whitespace=True, comment='#', header=None,
                     names=['id', 'type', 'x', 'y', 'z', 'radius', 'parent'])
    G = nx.Graph()
    xy_resolution = find_resolution(swc_file_path)
    df['x'] = df['x'] * xy_resolution / 1000
    df['y'] = df['y'] * xy_resolution / 1000

    if upsample:
        df['x'] = df['x'] * 2
        df['y'] = df['y'] * 2
        df['z'] = df['z'] * 2

    for _, row in df.iterrows():
        # 添加节点，同时记录parent信息
        G.add_node(row['id'], pos=(row['x'], row['y'], row['z']), radius=row['radius'], type=row['type'],
                   parent=row['parent'])
        if row['parent'] != -1:
            G.add_edge(row['parent'], row['id'])

    return G

# def calculate_edge_distances(G):
#     distances = []
#     for u, v in G.edges():
#         pos_u = G.nodes[u]['pos']
#         pos_v = G.nodes[v]['pos']
#         # 计算欧几里得距离
#         distance = np.linalg.norm(np.array(pos_u) - np.array(pos_v))
#         distances.append(distance)
#     return distances

def calculate_edge_distances(G):
    """
    计算无向图中所有连接到末梢节点的边的欧几里得距离。

    参数:
    G (networkx.Graph): 无向图，节点具有 'pos' 属性表示其 (x, y, z) 坐标。

    返回:
    list: 所有符合条件的边的欧几里得距离列表。
    """
    # 1. 找到所有的末梢节点（度为1的节点）
    leaf_nodes = {node for node, degree in G.degree() if degree == 1}

    # 2. 使用列表推导式计算符合条件的边的距离
    distances = [
        np.linalg.norm(np.array(G.nodes[u]['pos']) - np.array(G.nodes[v]['pos']))
        for u, v in G.edges()
        if u in leaf_nodes or v in leaf_nodes
    ]

    return distances

def calc_swc_path_length(swc_file, upsample):
    G = load_swc_to_undirected_graph(swc_file, upsample)
    G = G.to_undirected()
    edge_distances = calculate_edge_distances(G)
    return edge_distances

def calc_swc_dir_length(swc_dir, upsample=False):
    edge_distances = []
    leaf_number = []
    for swc_file in os.listdir(swc_dir):
        swc_file = os.path.join(swc_dir, swc_file)
        current_edge = calc_swc_path_length(swc_file, upsample)
        edge_distances += current_edge
        leaf_number.append(len(current_edge))
    return edge_distances, leaf_number

# swc_file = '/data/kfchen/trace_ws/paper_trace_result/nnunet/newcel_0.1/5_swc/2370.swc'
# G = load_swc_to_undirected_graph(swc_file)
# G = G.to_undirected()
# edge_distances = calculate_edge_distances(G)

if __name__ == '__main__':
    swc_dir1 = r'/data/kfchen/trace_ws/paper_trace_result/nnunet/newcel_0.1/5_swc'
    swc_dir2 = r'/data/kfchen/trace_ws/paper_trace_result/nnunet/newcel_0.1/9_swc_no_rescale_no_skel_swc'
    # edge_distances = []
    #
    # for swc_file in os.listdir(swc_dir):
    #     swc_file = os.path.join(swc_dir, swc_file)
    #
    #     edge_distances += calc_swc_path_length(swc_file)

    edge_1, leaf1 = calc_swc_dir_length(swc_dir1)
    edge_2, leaf2 = calc_swc_dir_length(swc_dir2, upsample=True)
    print(sum(leaf1)/len(leaf1), sum(leaf2)/len(leaf2))

    # 查看路径长度的基本统计信息
    # print(f"Total number of edges: {len(edge_distances)}")
    # print(f"Minimum edge distance: {min(edge_distances):.2f}")
    # print(f"Maximum edge distance: {max(edge_distances):.2f}")
    # print(f"Average edge distance: {sum(edge_distances)/len(edge_distances):.2f}")
    print(f"Total number of edges: {len(edge_1)}, {len(edge_2)}")
    print(f"Minimum edge distance: {min(edge_1):.2f}, {min(edge_2):.2f}")
    print(f"Maximum edge distance: {max(edge_1):.2f}, {max(edge_2):.2f}")
    print(f"Average edge distance: {sum(edge_1)/len(edge_1):.2f}, {sum(edge_2)/len(edge_2):.2f}")

    # # 使用 seaborn 绘制路径长度的分布直方图
    # sns.set(style="whitegrid")
    # plt.figure(figsize=(10, 6))
    # # sns.histplot(edge_distances, bins=100, kde=False, color='skyblue', edgecolor='black')
    # sns.histplot(edge_2, bins=100, kde=True, color='red', edgecolor='black', label='Origin', alpha=0)
    # sns.histplot(edge_1, bins=100, kde=True, color='skyblue', edgecolor='black', label='Skeleton', alpha=0)
    #
    # plt.title('Distribution of Path Lengths Between Neighbors in the Graph')
    # plt.xlabel('Path Length (Euclidean Distance)')
    # plt.ylabel('Frequency')
    # plt.legend()
    # plt.show()
    # plt.close()

    # 定义类名
    class_names = ['Origin', 'Skeleton']

    # 准备数据为 DataFrame 格式，便于 seaborn 处理
    data = pd.DataFrame({
        'Class': ['Origin'] * len(leaf2) + ['Skeleton'] * len(leaf1),
        'Distance': leaf2 + leaf1
    })

    plt.figure(figsize=(3, 3))

    # leaf
    sns.boxplot(x='Class', y='Distance', data=data, palette='Set2', linewidth=1, width=0.5)
    plt.tick_params(axis='both', which='major')  # 调整刻度标签大小
    plt.legend().set_visible(False)
    # plt.set_ylabel('Number of Tips')
    plt.ylabel('Number of Tips',  fontsize=15)
    plt.xlabel('')
    plt.xticks(fontsize=15)

    plt.tight_layout()  # 调整布局
    plt.show()
    # plt.savefig(box_file)
    plt.close()