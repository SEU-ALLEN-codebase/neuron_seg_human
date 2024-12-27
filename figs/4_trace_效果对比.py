from simple_swc_tool.big_neuron_tracers import Advantra_trace_file, CWlab_method_v1, MOST_trace_file, Mst_tracing_file, APP1_trace_file, APP2_trace_file, NeuroGPSTree_trace_file, neuTube_trace_file
import os
import pandas as pd
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from skimage import io
import numpy as np
from scipy import ndimage
from skimage import io
from scipy.ndimage import binary_dilation
import tifffile
from simple_swc_tool.swc_io import read_swc, write_swc
from simple_swc_tool.sort_swc import sort_swc as sort_swc2
import skimage
from skimage.measure import regionprops
import matplotlib.pyplot as plt

v3d_path = r"/home/kfchen/Vaa3D-x.1.1.4_Ubuntu/Vaa3D-x"
v3d_v3_path = r"/home/kfchen/Vaa3D_CentOS_64bit_v3.601/bin/vaa3d"
neuron_info_df = pd.read_csv("/data/kfchen/nnUNet/nnUNet_results/Dataset169_hb_10k/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/ptls10/norm_result/Human_SingleCell_TrackingTable_20240712.csv", encoding='gbk')

def plot_box_of_swc_list_l_measure(l_measure_files, labels, box_file):
    def get_common_rows_from_dfs(dfs):
        # 获取所有 DataFrame 第一列的共同项
        # 假设 df 列名为 'col1'
        common_items = set(dfs[0].iloc[:, 0])  # 假设所有 DataFrame 第一列都是一样的列名
        for df in dfs[1:]:
            common_items &= set(df.iloc[:, 0])  # 交集操作，找出共同的元素

        # 将共有项作为索引过滤每个 DataFrame
        common_df_list = []
        for df in dfs:
            filtered_df = df[df.iloc[:, 0].isin(common_items)]  # 根据第一列的共有项筛选
            # 找到有多少行
            print(len(filtered_df))
            common_df_list.append(filtered_df)

        # 返回包含共同项的所有 DataFrame
        return common_df_list
    feature_names = ['Number of Tips', 'N_stem']
    feature_name_maps = {'Number of Branches': 'Number of Branches', 'Total Length': 'Total Length (μm) ↑',
                         'Max Path Distance': 'Max Path Distance (μm)', 'N_stem': 'Number of Stems',
                         'Number of Tips': 'Number of Tips', 'Max Branch Order': 'Max Branch Order'}

    num_features = len(feature_names)
    cols = 2
    rows = (num_features + cols - 1) // cols
    # 图像清晰度
    plt.rcParams['savefig.dpi'] = 800
    fig, axes = plt.subplots(rows, cols, figsize=(cols*6, 4 * rows))  # 调整figsize和dpi提高清晰度
    axes = axes.flatten()
    # plt.rcParams.update({'font.size': 20})  # 更新字体大小
    # 设置字体 Arial
    # plt.rcParams['font.family'] = 'Arial'
    tab20c_colors = plt.get_cmap('tab20c').colors
    set3_colors = plt.get_cmap('Set3').colors

    colors = [None for _ in range(len(labels))]
    colors[4], colors[8] = tab20c_colors[5], tab20c_colors[6]
    colors[5], colors[9] = tab20c_colors[9], tab20c_colors[10]
    colors[6], colors[10] = tab20c_colors[1], tab20c_colors[2]
    colors[7], colors[11] = tab20c_colors[13], tab20c_colors[14]
    colors[0], colors[1], colors[2] = set3_colors[1], set3_colors[2], set3_colors[3]


    dfs = [pd.read_csv(f) for f in l_measure_files]
    dfs = get_common_rows_from_dfs(dfs)
    for i, df in enumerate(dfs):
        df['Type'] = labels[i]

    df = pd.concat(dfs, axis=0)
    average_values = df.groupby('Type')[feature_names].mean()
    print("各类各特征的平均值：")
    print(average_values)
    df_long = pd.melt(df, id_vars=['Type'], value_vars=feature_names, var_name='Feature', value_name='Value')

    # 绘图
    for idx, feature in enumerate(feature_names):
        ax = axes[idx]
        # if feature == 'Number of Tips':
        #     ax.set_ylim(0, 150)
        # elif feature == 'Total Length':
        #     ax.set_ylim(-50, 5000)

        # 获取当前特征的数据子集
        feature_data = df_long[df_long['Feature'] == feature]

        # 计算positions, 每个Feature会有多个箱体
        # positions的数量要等于每个Feature和Type的组合数量
        positions = [i + 1 for i in range(len(feature_data['Type'].unique()))]
        positions[6:] = [i + 1 for i in positions[6:]]
        positions[9] = positions[9] + 1
        # print(positions)
        hue_order = feature_data['Type'].unique()  # 获取所有的hue分类，通常是['Type 1', 'Type 2']

        # # 绘制箱线图
        # sns.boxplot(x='Feature', y='Value', hue='Type', data=feature_data, ax=ax,
        #             palette=colors, dodge=False, fliersize=0, native_scale=True, positions=positions)
        current_data = []
        for i, hue in enumerate(hue_order):
            current_data.append(feature_data[feature_data['Type'] == hue]['Value'].to_numpy())
            ax.boxplot(current_data[i], positions=[positions[i]], widths=0.5, patch_artist=True,
                       showfliers=True, boxprops=dict(facecolor=colors[i], color='black'),
                       medianprops=dict(color='black'), flierprops=dict(marker='o', color='black', markersize=3))

        # p_values = []
        # for i in range(len(hue_order) - 1):
        #     group1 = current_data[i]
        #     group2 = current_data[4]
        #     t_stat, p_value = stats.Z(group1, group2)
        #     p_values.append((hue_order[i], hue_order[4], p_value))
        #
        # # 打印p值
        # print(f"P-values for feature {feature}:")
        # for pair_0, pair_1, p_value in p_values:
        #     print(f"p({pair_0} vs {pair_1}): {p_value}")
        #
        # y_offset = 0.1
        # # 标记p值：可以选择显示p值和显著性标志
        # for idx, (group1, group2, p_value) in enumerate(p_values):
        #     x1 = positions[hue_order.tolist().index(group1)]
        #     x2 = positions[hue_order.tolist().index(group2)]
        #     y_max = max(max(current_data[hue_order.tolist().index(group1)]),
        #                 max(current_data[hue_order.tolist().index(group2)]))
        #     ax.plot([x1, x2], [y_max * (1.05 + idx * y_offset), y_max * (1.05 + idx * y_offset)], color='black', lw=1)  # 画线连接两个箱体
        #     ax.text((x1 + x2) / 2, y_max * (1.05 + idx * y_offset), f"p = {p_value:.3f}", ha='center', va='bottom', fontsize=12)

        # ax.set_title(feature_name_maps[feature], fontsize=15)
        ax.set_title("")

        ax.set_xticks(positions)  # 设置 x 轴刻度位置
        ax.set_xticklabels(labels)  # 每个位置显示相同的 feature 名称
        # 设置tick的字体大小和right
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, fontsize=8, ha='right')


        # ax.get_xaxis().set_ticks(labels)
        ax.legend().set_visible(False)
        ax.set_ylabel(feature_name_maps[feature], fontsize=15)
        ax.tick_params(axis='both', which='major', labelsize=15)


    # 隐藏不需要的子图
    for ax in axes[num_features:]:
        ax.axis('off')

    plt.tight_layout(pad=1.0)  # 调整布局
    # plt.show()
    plt.savefig(box_file)
    plt.close()

def plot_box_of_swc_list_opt(opt_files, labels, box_file):
    def get_common_rows_from_dfs(dfs):
        # 获取所有 DataFrame 第一列的共同项
        # 假设 df 列名为 'col1'
        common_items = set(dfs[0].iloc[:, 0])  # 假设所有 DataFrame 第一列都是一样的列名
        for df in dfs[1:]:
            common_items &= set(df.iloc[:, 0])  # 交集操作，找出共同的元素

        # 将共有项作为索引过滤每个 DataFrame
        common_df_list = []
        for df in dfs:
            filtered_df = df[df.iloc[:, 0].isin(common_items)]  # 根据第一列的共有项筛选
            # 找到有多少行
            print(len(filtered_df))
            common_df_list.append(filtered_df)

        # 返回包含共同项的所有 DataFrame
        return common_df_list

    feature_names = ['optj_f1', 'optp_con_prob_f1', 'optg_f1', 'mean_f1']
    feature_name_maps = {'optj_f1': 'OPT-J F1 ↑',
                         'optp_con_prob_f1': 'OPT-P F1 ↑',
                         'optg_f1': 'OPT-G F1 ↑',
                            'mean_f1': 'Mean F1 ↑'}

    num_features = len(feature_names)
    cols = 2
    rows = (num_features + cols - 1) // cols
    # 图像清晰度
    plt.rcParams['savefig.dpi'] = 800
    fig, axes = plt.subplots(rows, cols, figsize=(cols*5, 4 * rows))  # 调整figsize和dpi提高清晰度
    axes = axes.flatten()
    # plt.rcParams.update({'font.size': 20})  # 更新字体大小
    # 设置字体 Arial
    # plt.rcParams['font.family'] = 'Arial'
    tab20c_colors = plt.get_cmap('tab20c').colors
    set3_colors = plt.get_cmap('Set3').colors

    colors = [None for _ in range(len(labels))]

    colors[4], colors[8] = tab20c_colors[5], tab20c_colors[6]
    colors[5], colors[9] = tab20c_colors[9], tab20c_colors[10]
    colors[6], colors[10] = tab20c_colors[1], tab20c_colors[2]
    colors[7], colors[11] = tab20c_colors[13], tab20c_colors[14]
    colors[0], colors[1], colors[2], colors[3] = set3_colors[1], set3_colors[3], set3_colors[6], set3_colors[7]


    dfs = [pd.read_csv(f) for f in opt_files]
    dfs = get_common_rows_from_dfs(dfs)
    # get mean_f1

    for i, df in enumerate(dfs):
        df['Type'] = labels[i]
        df['mean_f1'] = (df['optj_f1'] + df['optp_con_prob_f1'] + df['optg_f1']) / 3

    df = pd.concat(dfs, axis=0)
    average_values = df.groupby('Type')[feature_names].mean()
    print("各类各特征的平均值：")
    print(average_values)
    df_long = pd.melt(df, id_vars=['Type'], value_vars=feature_names, var_name='Feature', value_name='Value')

    # 绘图
    for idx, feature in enumerate(feature_names):
        ax = axes[idx]
        # if feature == 'Number of Tips':
        #     ax.set_ylim(0, 150)
        # elif feature == 'Total Length':
        #     ax.set_ylim(-50, 5000)

        # 获取当前特征的数据子集
        feature_data = df_long[df_long['Feature'] == feature]

        # 计算positions, 每个Feature会有多个箱体
        # positions的数量要等于每个Feature和Type的组合数量
        positions = [i + 1 for i in range(len(feature_data['Type'].unique()))]
        positions[8:] = [i + 1 for i in positions[8:]]
        # print(positions)
        hue_order = feature_data['Type'].unique()  # 获取所有的hue分类，通常是['Type 1', 'Type 2']

        # # 绘制箱线图
        # sns.boxplot(x='Feature', y='Value', hue='Type', data=feature_data, ax=ax,
        #             palette=colors, dodge=False, fliersize=0, native_scale=True, positions=positions)
        current_data = []
        for i, hue in enumerate(hue_order):
            current_data.append(feature_data[feature_data['Type'] == hue]['Value'].to_numpy())
            ax.boxplot(current_data[i], positions=[positions[i]], widths=0.5, patch_artist=True,
                       showfliers=True, boxprops=dict(facecolor=colors[i], color='black'),
                       medianprops=dict(color='black'), flierprops=dict(marker='o', color='black', markersize=3))

        # p_values = []
        # for i in range(len(hue_order) - 1):
        #     group1 = current_data[i]
        #     group2 = current_data[4]
        #     t_stat, p_value = stats.Z(group1, group2)
        #     p_values.append((hue_order[i], hue_order[4], p_value))
        #
        # # 打印p值
        # print(f"P-values for feature {feature}:")
        # for pair_0, pair_1, p_value in p_values:
        #     print(f"p({pair_0} vs {pair_1}): {p_value}")
        #
        # y_offset = 0.1
        # # 标记p值：可以选择显示p值和显著性标志
        # for idx, (group1, group2, p_value) in enumerate(p_values):
        #     x1 = positions[hue_order.tolist().index(group1)]
        #     x2 = positions[hue_order.tolist().index(group2)]
        #     y_max = max(max(current_data[hue_order.tolist().index(group1)]),
        #                 max(current_data[hue_order.tolist().index(group2)]))
        #     ax.plot([x1, x2], [y_max * (1.05 + idx * y_offset), y_max * (1.05 + idx * y_offset)], color='black', lw=1)  # 画线连接两个箱体
        #     ax.text((x1 + x2) / 2, y_max * (1.05 + idx * y_offset), f"p = {p_value:.3f}", ha='center', va='bottom', fontsize=12)

        # ax.set_title(feature_name_maps[feature], fontsize=15)
        ax.set_title("")

        ax.set_xticks(positions)  # 设置 x 轴刻度位置
        ax.set_xticklabels(labels)  # 每个位置显示相同的 feature 名称
        # 设置tick的字体大小和right
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, fontsize=8, ha='right')


        # ax.get_xaxis().set_ticks(labels)
        ax.legend().set_visible(False)
        ax.set_ylabel(feature_name_maps[feature], fontsize=15)
        ax.tick_params(axis='both', which='major', labelsize=15)


    # 隐藏不需要的子图
    for ax in axes[num_features:]:
        ax.axis('off')

    plt.tight_layout(pad=1.0)  # 调整布局
    # plt.show()
    plt.savefig(box_file)
    plt.close()

def rescale_xy_resolution(swc_file, unified_swc_file):
    id = int(os.path.basename(swc_file).split('_')[0].split('.')[0])
    xy_resolution = float(neuron_info_df.loc[neuron_info_df.iloc[:, 0] == id, 'xy拍摄分辨率(*10e-3μm/px)'].values[0])

    swc = np.loadtxt(swc_file)
    if len(swc.shape) == 1:
        swc = np.expand_dims(swc, axis=0)
    swc = np.array([line for line in swc if line[0] != '#'])
    # print(swc[:3])
    swc[:, 2] = swc[:, 2] * xy_resolution / 1000
    swc[:, 3] = swc[:, 3] * xy_resolution / 1000
    np.savetxt(unified_swc_file, swc, fmt='%d %d %f %f %f %f %d')

def trace_app2_with_soma(img_file, swc_file, v3d_path):
    def process_path(path):
        return path.replace('\\', '/')

    # somamarker_file = input_files[1]
    somamarker_file = 'NULL'
    ini_swc_path = img_file.replace('.tif', '.tif_ini.swc')
    # if (os.path.exists(swc_file) or (not os.path.exists(img_file)) or (not os.path.exists(somamarker_file))):
    #     return
    '''
        **** Usage of APP2 ****
        vaa3d -x plugin_name -f app2 -i <inimg_file> -o <outswc_file> -p [<inmarker_file> [<channel> [<bkg_thresh> 
        [<b_256cube> [<b_RadiusFrom2D> [<is_gsdt> [<is_gap> [<length_thresh> [is_resample][is_brightfield][is_high_intensity]]]]]]]]]
        inimg_file          Should be 8/16/32bit image
        inmarker_file       If no input marker file, please set this para to NULL and it will detect soma automatically.
                            When the file is set, then the first marker is used as root/soma.
        channel             Data channel for tracing. Start from 0 (default 0).
        bkg_thresh          Default 10 (is specified as AUTO then auto-thresolding)
        b_256cube           If trace in a auto-downsampled volume (1 for yes, and 0 for no. Default 1.)
        b_RadiusFrom2D      If estimate the radius of each reconstruction node from 2D plane only (1 for yes as many 
        times the data is anisotropic, and 0 for no. Default 1 which which uses 2D estimation.)
        is_gsdt             If use gray-scale distance transform (1 for yes and 0 for no. Default 0.)
                       If allow gap (1 for yes and 0 for no. Default 0.)
        length_thresh       Default 5
        is_resample         If allow resample (1 for yes and 0 for no. Default 1.)
        is_brightfield      If the signals are dark instead of bright (1 for yes and 0 for no. Default 0.)
        is_high_intensity   If the image has high intensity background (1 for yes and 0 for no. Default 0.)
        outswc_file         If not be specified, will be named automatically based on the input image file name.
    '''

    resample = 1
    gsdt = 1
    b_RadiusFrom2D = 1

    # temp_img_file = os.path.join(os.path.dirname(swc_file), str(os.path.basename(swc_file).split('_')[0]) + "_temp.tif")
    # temp_swc_file = os.path.join(os.path.dirname(swc_file), str(os.path.basename(swc_file).split('_')[0]) + "_temp.swc")
    # temp_marker_file = os.path.join(os.path.dirname(swc_file), str(os.path.basename(swc_file).split('_')[0]) + "_temp.marker")
    # shutil.copyfile(img_file, temp_img_file)

    try:
        if (sys.platform == "linux"):
            cmd = f'xvfb-run -a -s "-screen 0 640x480x16" "{v3d_path}" -x vn2 -f app2 -i "{img_file}" -o "{swc_file}" -p "{somamarker_file}" 0 10 1 {b_RadiusFrom2D} {gsdt} 1 {resample} 0 0'
            cmd = process_path(cmd)
            subprocess.run(cmd, stdout=subprocess.DEVNULL, shell=True)
        else:
            pass
    except:
        pass

    # os.remove(temp_img_file)
    # os.remove(temp_marker_file)
    # # rename
    # if(os.path.exists(temp_swc_file)):
    #     os.rename(temp_swc_file, swc_file)

    if (os.path.exists(ini_swc_path)):
        os.remove(ini_swc_path)


def connect_to_soma_file(swc_file, soma_region_file, conn_swc_file):
    def compute_centroid(mask):
        # 计算三维 mask 的重心
        labeled_mask = skimage.measure.label(mask)
        props = regionprops(labeled_mask)

        if len(props) > 0:
            # 获取第一个区域的重心坐标
            centroid = props[0].centroid
            return centroid
    def prune_fiber_in_soma(point_l, soma_region):
        edge_p_list = []
        x_limit, y_limit, z_limit = soma_region.shape[2], soma_region.shape[1], soma_region.shape[0]

        for p in point_l.p:
            if (p.n == 0 or p.n == 1): continue
            x, y, z = p.x, p.y, p.z
            y = soma_region.shape[1] - y

            x = min(int(x), x_limit - 1)
            y = min(int(y), y_limit - 1)
            z = min(int(z), z_limit - 1)

            if (soma_region[int(z), int(y), int(x)]):
                edge_p_list.append(p)

        for p in edge_p_list:
            if (len(p.s) == 0):
                temp_p = point_l.p[p.n]
                while (True):
                    if (temp_p.n == 1): break
                    if (temp_p.pruned == True): break
                    if (not len(temp_p.s) == 1): break
                    point_l.p[temp_p.n].pruned = True
                    temp_p = point_l.p[temp_p.p]
        for p in point_l.p:
            for s in p.s:
                if (point_l.p[s].pruned == True):
                    p.s.remove(s)

        return point_l

    soma_region = tifffile.imread(soma_region_file).astype("uint8")
    # soma_region = get_main_soma_region_in_msoma_from_gsdt(soma_region,,
    soma_region = binary_dilation(soma_region, iterations=4).astype("uint8")
    x_limit, y_limit, z_limit = soma_region.shape[2], soma_region.shape[1], soma_region.shape[0]
    new_soma_coord = compute_centroid(np.flip(soma_region, axis=1))

    point_l = read_swc(swc_file)
    # point_l.p[1].x, point_l.p[1].y, point_l.p[1].z = new_soma_coord[2], new_soma_coord[1], new_soma_coord[0]

    labeled_img, num_objects = ndimage.label(soma_region)
    if (len(point_l.p) <= 1):
        return
    for obj_id in range(1, num_objects + 1):
        obj_img = np.where(labeled_img == obj_id, 1, 0)
        x, y, z = point_l.p[1].x, point_l.p[1].y, point_l.p[1].z
        if (obj_img[int(z), int(y), int(x)]):
            soma_region = obj_img
            del obj_img
            break
        del obj_img
    # tifffile.imwrite(os.path.join(conn_folder, file_name+"1.tif"), soma_region.astype("uint8")*255, compression='zlib')
    labeled_img, num_objects = ndimage.label(soma_region)
    if (num_objects > 1):
        # soma_region = dusting(soma_region)
        write_swc(conn_swc_file, point_l)
        del soma_region, point_l
        return

    point_l = prune_fiber_in_soma(point_l, soma_region)

    # strict strategy
    edge_p_list = []
    edge_p_id_list = []
    for p in point_l.p:
        if (p.n == 0 or p.n == 1): continue
        x, y, z = p.x, p.y, p.z
        y = soma_region.shape[1] - y

        x = min(int(x), x_limit - 1)
        y = min(int(y), y_limit - 1)
        z = min(int(z), z_limit - 1)

        if (soma_region[int(z), int(y), int(x)]):
            edge_p_list.append(p)
            edge_p_id_list.append(p.n)

    for p in edge_p_list:
        temp_p = point_l.p[p.p]
        while (True):
            if (temp_p.n == 1): break
            if (temp_p.pruned == True): break
            if(temp_p.n not in edge_p_id_list): break
            point_l.p[temp_p.n].pruned = True
            temp_p = point_l.p[temp_p.p]

    for p in edge_p_list:
        if (point_l.p[p.n].pruned == False):
            if (not len(point_l.p[p.n].s)):
                point_l.p[p.n].pruned = True
            else:
                point_l.p[p.n].p = 1
                point_l.p[1].s.append(p.n)
        else:
            for s in point_l.p[p.n].s:
                point_l.p[s].p = 1
                point_l.p[1].s.append(s)

    # Conservative strategy
    # for s in point_l.p[1].s:
    #     temp_p = point_l.p[s]
    #     x, y, z = temp_p.x, temp_p.y, temp_p.z
    #     y = soma_region.shape[1] - y
    #     x = min(int(x), x_limit - 1)
    #     y = min(int(y), y_limit - 1)
    #     z = min(int(z), z_limit - 1)
    #
    #     if(not soma_region[int(z), int(y), int(x)]):
    #         continue
    #
    #     for s2 in point_l.p[s].s:
    #         point_l.p[s2].p = 1
    #         point_l.p[1].s.append(s2)
    #
    #     point_l.p[1].s.remove(s)
    #     point_l.p[s].pruned = True
    # print(conn_path)
    if (os.path.exists(conn_swc_file)):
        os.remove(conn_swc_file)
    write_swc(conn_swc_file, point_l)
    # print(len(point_l.p))
    del soma_region, point_l


def trace_img_dir(img_dir, swc_dir_root, trace_methods):
    tif_files = [f for f in os.listdir(img_dir) if f.endswith('.tif')]
    def current_task(tif_file):
        img_file = os.path.join(img_dir, tif_file)
        for method_name, method in trace_methods.items():
            swc_file = os.path.join(swc_dir_root, method_name, tif_file.replace('.tif', '.swc'))
            os.makedirs(os.path.dirname(swc_file), exist_ok=True)
            if (not os.path.exists(swc_file)):
                if("APP2" in method_name or "CWlab" in method_name or "Advantra" in method_name):
                    method(img_file, swc_file, v3d_path)
                else:
                    method(img_file, swc_file, v3d_v3_path)

    # for tif_file in tif_files:
    #     current_task(tif_file)
    pbar = tqdm(total=len(tif_files))
    with ThreadPoolExecutor(max_workers=10) as executor:
        for _ in executor.map(current_task, tif_files):
            pbar.update(1)

    pbar.close()

def reconnect_soma_to_swc(swc_dir, soma_region_dir, conn_swc_dir):
    os.makedirs(conn_swc_dir, exist_ok=True)
    swc_files = [f for f in os.listdir(swc_dir) if f.endswith('.swc')]
    def current_task(swc_file):
        soma_region_file = os.path.join(soma_region_dir, swc_file.replace('.swc', '.tif'))
        conn_swc_file = os.path.join(conn_swc_dir, swc_file)
        if(os.path.exists(os.path.join(swc_dir, swc_file)) and not os.path.exists(conn_swc_file)):
            connect_to_soma_file(os.path.join(swc_dir, swc_file), soma_region_file, conn_swc_file)

    pbar = tqdm(total=len(swc_files))
    with ThreadPoolExecutor(max_workers=10) as executor:
        for _ in executor.map(current_task, swc_files):
            pbar.update(1)


def sort_swc_dir(swc_dir, soma_region_dir, sorted_swc_dir):
    os.makedirs(sorted_swc_dir, exist_ok=True)
    swc_files = [f for f in os.listdir(swc_dir) if f.endswith('.swc')]

    def compute_centroid(mask):
        # 计算三维 mask 的重心
        labeled_mask = skimage.measure.label(mask)
        props = regionprops(labeled_mask)

        if len(props) > 0:
            # 获取第一个区域的重心坐标
            centroid = props[0].centroid
            return centroid

    def current_task(swc_file):
        soma_region_file = os.path.join(soma_region_dir, swc_file.replace('.swc', '.tif'))
        sorted_swc_file = os.path.join(sorted_swc_dir, swc_file)
        if(os.path.exists(os.path.join(swc_dir, swc_file)) and not os.path.exists(sorted_swc_file)):
            soma = tifffile.imread(soma_region_file)
            soma = np.flip(soma, axis=1)
            soma_coord = compute_centroid(soma)
            soma_coord = (soma_coord[2], soma_coord[1], soma_coord[0])
            sort_swc2(os.path.join(swc_dir, swc_file), sorted_swc_file, soma_coord, merge_nodes=False)


    pbar = tqdm(total=len(swc_files))
    with ThreadPoolExecutor(max_workers=10) as executor:
        for _ in executor.map(current_task, swc_files):
            pbar.update(1)

if __name__ == '__main__':

    resized_seg_dir = "/data/kfchen/trace_ws/topology_test/512_seg"
    skel_tracing_source_dir = "/data/kfchen/trace_ws/paper_trace_result/nnunet/newcel_0.1/3_skel_with_soma"
    soma_region_dir = "/data/kfchen/trace_ws/paper_trace_result/nnunet/newcel_0.1/2_soma_region"
    skel_seg_dir = "/data/kfchen/trace_ws/topology_test/512_skel_seg"
    if(not os.path.exists(skel_seg_dir)):
        os.makedirs(skel_seg_dir)
        for seg_file in os.listdir(skel_tracing_source_dir):
            seg = io.imread(os.path.join(skel_tracing_source_dir, seg_file))
            seg = (seg - seg.min()) / (seg.max() - seg.min()) * 255
            seg = seg.astype('uint8')
            io.imsave(os.path.join(skel_seg_dir, seg_file), seg)


    swc_dir_root = "/data/kfchen/trace_ws/topology_test/ori_trace_result"
    trace_methods = {
        "Advantra": Advantra_trace_file,
        'APP1': APP1_trace_file,
        "MOST": MOST_trace_file,
        'neuTube': neuTube_trace_file,


        'APP2': APP2_trace_file,
        "CWlab": CWlab_method_v1,
        'MST': Mst_tracing_file,
        'NeuroGPSTree': NeuroGPSTree_trace_file,
    }
    insterested_trace_method = {
        'APP2': APP2_trace_file,
        "CWlab": CWlab_method_v1,
        'MST': Mst_tracing_file,
        # 'neuTube': neuTube_trace_file,
        'NeuroGPSTree': NeuroGPSTree_trace_file,
    }

    # trace_img_dir(skel_seg_dir, os.path.join(swc_dir_root, 'from_skel'), insterested_trace_method)
    # trace_img_dir(resized_seg_dir, os.path.join(swc_dir_root, 'from_seg'), trace_methods)



    # for method in insterested_trace_method.keys():
    #     sort_swc_dir(os.path.join(swc_dir_root, 'from_skel', method), soma_region_dir, os.path.join(swc_dir_root, 'from_skel_sorted', method))
    #     # sort_swc_dir(os.path.join(swc_dir_root, 'from_seg', method), soma_region_dir, os.path.join(swc_dir_root, 'from_seg_sorted', method))
    # #
    # #
    # for method in insterested_trace_method.keys():
    #     reconnect_soma_to_swc(os.path.join(swc_dir_root, 'from_skel_sorted', method), soma_region_dir, os.path.join(swc_dir_root, 'from_skel_re_connect', method))
    #     # reconnect_soma_to_swc(os.path.join(swc_dir_root, 'from_seg_sorted', method), soma_region_dir, os.path.join(swc_dir_root, 'from_seg_re_connect', method))
    #
    #
    rescaled_swc_dir_root = "/data/kfchen/trace_ws/topology_test/rescaled_trace_result"
    # for method in insterested_trace_method.keys():
    #     os.makedirs(os.path.join(rescaled_swc_dir_root, 'from_skel_re_connect', method), exist_ok=True)
    #     for swc_file in os.listdir(os.path.join(swc_dir_root, 'from_skel_re_connect', method)):
    #         source_swc_file = os.path.join(swc_dir_root, 'from_skel_re_connect', method, swc_file)
    #         target_swc_file = os.path.join(rescaled_swc_dir_root, 'from_skel_re_connect', method, swc_file)
    #         if(os.path.exists(source_swc_file) and not os.path.exists(target_swc_file)):
    #             rescale_xy_resolution(source_swc_file, target_swc_file)
    # for method in trace_methods.keys():
    #     os.makedirs(os.path.join(rescaled_swc_dir_root, 'from_seg', method), exist_ok=True)
    #     for swc_file in os.listdir(os.path.join(swc_dir_root, 'from_seg', method)):
    #         source_swc_file = os.path.join(swc_dir_root, 'from_seg', method, swc_file)
    #         target_swc_file = os.path.join(rescaled_swc_dir_root, 'from_seg', method, swc_file)
    #         if(os.path.exists(source_swc_file) and not os.path.exists(target_swc_file)):
    #             rescale_xy_resolution(source_swc_file, target_swc_file)

    swc_result_dirs = []
    label_list = []
    l_measure_csv_files = []
    for method in trace_methods.keys():
        swc_result_dirs.append(os.path.join(rescaled_swc_dir_root, 'from_seg', method))
        label_list.append(method)
        l_measure_csv_files.append(os.path.join(rescaled_swc_dir_root, 'from_seg', method + '_l_measure.csv'))

    for method in insterested_trace_method.keys():
        current_swc_dir = os.path.join(rescaled_swc_dir_root, 'from_skel_re_connect', method)
        if(os.path.exists(current_swc_dir)):
            swc_result_dirs.append(current_swc_dir)
            label_list.append(method + '*')
            l_measure_csv_files.append(os.path.join(rescaled_swc_dir_root, 'from_skel_re_connect', method + '_l_measure.csv'))


    label_list.append("Manual")
    l_measure_csv_files.append("/data/kfchen/trace_ws/paper_auto_human_neuron_recon/swc_label/1um_swc_lab_l_measure.csv")
    # # l_measure
    from simple_swc_tool.l_measure_api import l_measure_swc_dir
    for swc_result_dir, l_measure_csv_file in zip(swc_result_dirs, l_measure_csv_files):
        if(not os.path.exists(l_measure_csv_file)):
            l_measure_swc_dir(swc_result_dir, l_measure_csv_file)

    plot_box_of_swc_list_l_measure(l_measure_csv_files, label_list, os.path.join(rescaled_swc_dir_root, 'l_measure_box.png'))

    opt_result_files = []
    from simple_swc_tool.opt_topology_analyse import opt_analyse_dir_v2
    for swc_result_dir in swc_result_dirs:
        # print("???")
        # my_opt("/data/kfchen/trace_ws/paper_auto_human_neuron_recon/swc_label/1um_swc_lab", swc_result_dir, swc_result_dir + '_opt_result.csv')
        opt_result_file = swc_result_dir + '_opt_result.csv'
        if(not os.path.exists(opt_result_file)):
            opt_analyse_dir_v2("/data/kfchen/trace_ws/paper_auto_human_neuron_recon/swc_label/1um_swc_lab", swc_result_dir)
        opt_result_files.append(swc_result_dir + '_opt_result.csv')

    plot_box_of_swc_list_opt(opt_result_files, label_list[:-1], os.path.join(rescaled_swc_dir_root, 'opt_box.png'))