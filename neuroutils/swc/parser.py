import os
import platform
import subprocess
from neuroutils.config.settings import V3D_X_PATH, V3D_3_PATH
import pandas as pd
import tempfile
import shutil
from neuroutils.swc.io import load_swc, save_swc
from contextlib import contextmanager


def rescale_swc(swc_points, xy_resolution, z_resolution):
    """
    Rescale the SWC points based on the given xy and z resolutions.

    Parameters:
    swc_points (list of lists): The SWC points to be rescaled.
    xy_resolution (float): The resolution for the x and y coordinates.
    z_resolution (float): The resolution for the z coordinate.

    Returns:
    list of lists: The rescaled SWC points.
    """
    swc_points['x'] = swc_points['x'] * xy_resolution
    swc_points['y'] = swc_points['y'] * xy_resolution
    swc_points['z'] = swc_points['z'] * z_resolution
    return swc_points

def resample_swc_file(swc_in, swc_out, step=1, correction=True):
    '''
    Resample the SWC file using V3D_X. (COPY FROM YUFENG'S CODE, https://github.com/SEU-ALLEN-codebase/pylib/blob/b60eea8f2ab30885a5830b7abdb3184d11e4feac/swc_handler.py#L474)
    Parameters:
    swc_in (str): The input SWC file path.
    swc_out (str): The output SWC file path.
    step (int): The resampling step. Default is 1 (1um).
    correction (bool): Whether to correct the SWC file. Default is True.

    '''
    cmd_str = f'xvfb-run -a -s "-screen 0 640x480x16" {V3D_X_PATH} -x resample_swc -f resample_swc -i {swc_in} -o {swc_out} -p {step}'
    # print(swc_in, swc_out)
    p = subprocess.check_output(cmd_str, shell=True)
    if correction:
        # The built-in resample_swc has a bug: the first node is commented out, and there are two additional columns
        subprocess.run(f"sed -i 's/pid1/pid\\n1/g' {swc_out}; sed -i 's/ -1 -1//g' {swc_out}", shell=True)
    return True

# def resample_swc(swc_file, save_file, resample_step=2):
#     """
#     Resample the SWC file using V3D_X.
#     Parameters:
#     swc_file (str): The input SWC file path.
#     save_file (str): The output SWC file path.
#     resample_step (int): The resampling step. Default is 1 (1um).
#     """
#
#     def check_resample(resample_swc_path):
#         with open(resample_swc_path, "r") as f:
#             lines = f.readlines()
#             data = []
#             if (lines[0][0] == '#' and lines[1][0] == '2'):
#                 lines[0] = lines[0][lines[0].find(",pid") + 4:]
#             for i in lines:
#                 temp_line = i.split()
#                 line = "%d %d %f %f %f %f %d\n" % (
#                     int(temp_line[0]), int(temp_line[1]), float(temp_line[2]),
#                     float(temp_line[3]),
#                     float(temp_line[4]), float(temp_line[5]), int(temp_line[6])
#                 )
#                 data.append(line)  # 记录每一行
#         # write
#         with open(resample_swc_path, "w") as f:
#             for i in data:
#                 f.writelines(i)
#
#     assert platform.system() == "Linux", "V3D_X only supports Linux system"
#     cmd = f'xvfb-run -a -s "-screen 0 640x480x16" {V3D_X_PATH} -x resample_swc -f resample_swc -i {swc_file} -o {save_file} -p {resample_step}'
#     cmd = cmd.replace('(', '\(').replace(')', '\)')
#     subprocess.run(cmd, stdout=subprocess.DEVNULL, shell=True)
#
#     # check_resample(save_file)
#
#     return save_file

def sort_swc_file(swc_in, swc_out):
    cmd_str = f'xvfb-run -a -s "-screen 0 640x480x16" {V3D_X_PATH} -x sort_neuron_swc -f sort_swc -i {swc_in} -o {swc_out}'
    p = subprocess.check_output(cmd_str, shell=True)

    # retype
    df = pd.read_csv(swc_out, sep=' ', names=('#id', 'type', 'x', 'y', 'z', 'r', 'p'), comment='#', index_col=False)
    df['type'] = 3
    df.loc[0, 'type'] = 1
    df.to_csv(swc_out, sep=' ', index=False)

    return True

def find_leaf_branches(swc):
    """
    找到所有的叶子分支
    叶子分支定义：从一个分叉点到叶子节点之间的路径，且路径中不包含其他分支点

    参数:
        swc: DataFrame, 包含SWC文件数据的DataFrame

    返回:
        list: 包含所有叶子分支的列表，每个分支是一个节点ID的列表
    """
    # 创建节点到子节点的映射
    child_map = {}
    for _, row in swc.iterrows():
        parent = row['parent']
        if parent != -1:  # 跳过根节点
            if parent not in child_map:
                child_map[parent] = []
            child_map[parent].append(row['n'])

    # 识别所有分支点（有多个子节点的节点）
    branch_points = set()
    for parent, children in child_map.items():
        if len(children) > 1:
            branch_points.add(parent)

    # 识别所有叶子节点（没有子节点的节点）
    all_nodes = set(swc['n'])
    leaf_nodes = all_nodes - set(child_map.keys())
    # print(leaf_nodes)

    # 找到所有叶子分支
    leaf_branches = []

    for leaf in leaf_nodes:
        current_branch = []
        current_node = leaf
        parent_node = swc.loc[swc['n'] == current_node, 'parent'].values[0]

        # 向上追溯直到遇到分支点或根节点
        # while parent_node != -1 and parent_node not in branch_points:
        #     current_branch.insert(0, parent_node)
        #     current_node = parent_node
        #     parent_node = swc.loc[swc['n'] == current_node, 'parent'].values[0]
        while(True):
            current_branch.insert(0, current_node)
            if current_node in branch_points or parent_node == -1:
                # print(f"Branch point {current_node} found")
                break
            current_node = parent_node
            parent_node = swc.loc[swc['n'] == current_node, 'parent'].values[0]

        leaf_branches.append(current_branch)

    # print("leaf_branches", leaf_branches)
    return leaf_branches

def get_branch_length(swc, branch):
    """
    计算给定分支的长度
    参数:
        swc: DataFrame, 包含SWC文件数据的DataFrame
        branch: list, 分支节点ID的列表
    返回:
        float: 分支的长度
    """
    length = 0.0
    # print(branch)
    for i in range(len(branch) - 1):
        node1 = swc.loc[swc['n'] == branch[i]]
        node2 = swc.loc[swc['n'] == branch[i + 1]]
        if node1.empty or node2.empty:
            print(f"Node {branch[i]} or {branch[i + 1]} not found in SWC data.")
            print(branch)
            exit()
        length += ((node1['x'].values[0] - node2['x'].values[0]) ** 2 +
                    (node1['y'].values[0] - node2['y'].values[0]) ** 2 +
                    (node1['z'].values[0] - node2['z'].values[0]) ** 2) ** 0.5
    return length

def prune_small_branches(swc, min_length=5):
    """
    修剪小于指定长度的分支
    参数:
        swc: DataFrame, 包含SWC文件数据的DataFrame
        min_length: float, 最小长度，单位为微米
    返回:
        DataFrame: 修剪后的SWC数据
    """
    # 找到所有叶子分支
    origin_branch_num = len(swc)
    leaf_branches = find_leaf_branches(swc)

    # 计算每个分支的长度并修剪小于最小长度的分支
    for branch in leaf_branches:
        length = get_branch_length(swc, branch)
        if length < min_length:
            for node in branch[1:]:  # 从第二个节点开始修剪, 因为第一个节点是分支点
                swc = swc[swc['n'] != node]

    processed_branch_num = len(swc)
    if(origin_branch_num - processed_branch_num) > 0:
        # print(f"Pruned {origin_branch_num - processed_branch_num} branches shorter than {min_length}um")
        # 百分之多少的点被修剪掉了
        print(f"Pruned {(origin_branch_num - processed_branch_num) / origin_branch_num * 100:.2f}% nodes")
    return swc

def prune_swc(swc_file, save_file, min_length=5):
    """
    修剪SWC文件中小于指定长度的分支
    参数:
        swc_file: str, 输入SWC文件路径
        save_file: str, 输出SWC文件路径
        min_length: float, 最小长度，单位为微米
    """
    # print(f"Pruning branches shorter than {min_length}um in {swc_file} and saving to {save_file}")
    # 读取SWC文件
    swc = load_swc(swc_file)

    # 修剪小分支
    swc = prune_small_branches(swc, min_length)

    # 保存修剪后的SWC文件
    save_swc(swc, save_file)

# swc 标准化
def standardize_swc(swc_file, save_file,
                    eswc2swc_flag=False,
                    rescale_flag=False, xy_resolution=1, z_resolution=1,
                    resample_swc_flag=True, resample_step=1,
                    sort_swc_flag=False):
    '''
    Standardize the SWC file by rescaling, resampling, and sorting.
    Parameters:
    swc_file (str): The input SWC file path.
    save_file (str): The output SWC file path.
    rescale_flag (bool): Whether to rescale the SWC file. Default is False.
    xy_resolution (float): The resolution for the x and y coordinates. Default is 1.
    z_resolution (float): The resolution for the z coordinate. Default is 1.
    resample_swc_flag (bool): Whether to resample the SWC file. Default is True.
    resample_step (int): The resampling step. Default is 1 (1um).
    sort_swc_flag (bool): Whether to sort the SWC file. Default is False. ***Bug exists.***
    '''
    if(eswc2swc_flag):
        # 如果是eswc文件，先转换为swc文件
        swc_file = swc_file.replace('.eswc', '.swc')
        eswc2swc(swc_file, swc_file)

    with temp_swc_files(3) as temp_files:
        current_file, temp_intermediate, temp_final = temp_files

        # 将原始文件复制到工作临时文件
        shutil.copyfile(swc_file, current_file)

        # 第一步：重新缩放处理
        if rescale_flag:
            swc_data = load_swc(current_file)
            swc_data = rescale_swc(swc_data, xy_resolution, z_resolution)
            save_swc(swc_data, temp_intermediate)
            current_file, temp_intermediate = temp_intermediate, current_file

        # 第二步：重采样处理
        if resample_swc_flag:
            resample_swc_file(current_file, temp_intermediate, step=resample_step)
            current_file, temp_intermediate = temp_intermediate, current_file
        #
        # # 第三步：排序处理
        if sort_swc_flag:
            sort_swc_file(current_file, temp_final)
            current_file = temp_final

        # 保存最终结果
        os.makedirs(os.path.dirname(save_file), exist_ok=True)
        shutil.copyfile(current_file, save_file)

    # remove
    if(eswc2swc_flag and os.path.exists(swc_file)):
        os.remove(swc_file)
    return save_file

def eswc2swc(eswc_file, swc_file):
    if(os.path.exists(swc_file)):
        os.remove(swc_file)
    result_lines = []
    with open(eswc_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if(line.startswith("#")):
                result_lines.append(line)
            else:
                line = line.strip().split()
                line = line[:7]
                result_lines.append(" ".join(line) + "\n")

    with open(swc_file, 'w') as f:
        f.writelines(result_lines)

# 上下文管理器用于创建和自动清理临时SWC文件
@contextmanager
def temp_swc_files(count=1):
    """
    上下文管理器，用于创建和自动清理临时SWC文件
    """
    temp_files = []
    try:
        for _ in range(count):
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".swc")
            temp_files.append(temp_file)
            temp_file.close()  # 关闭文件句柄，但保留文件

        if count == 1:
            yield temp_files[0].name
        else:
            yield [f.name for f in temp_files]
    finally:
        # 清理所有临时文件
        for temp_file in temp_files:
            try:
                if os.path.exists(temp_file.name):
                    os.remove(temp_file.name)
            except OSError:
                pass  # 忽略删除失败的情况