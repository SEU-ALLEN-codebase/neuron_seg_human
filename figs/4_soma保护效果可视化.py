import os
import sys
import subprocess
import shutil
from skimage import io
from skimage.transform import resize
import numpy as np
import pandas as pd
from simple_swc_tool.l_measure_api import l_measure_swc_dir
import matplotlib.pyplot as plt



def trace_app2_with_soma(img_file, swc_file):
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

def rescale_xy_resolution(swc_file, unified_swc_file):
    id = int(os.path.basename(swc_file).split('_')[0].split('.')[0])
    xy_resolution = float(neuron_info_df.loc[neuron_info_df.iloc[:, 0] == id, 'xy拍摄分辨率(*10e-3μm/px)'].values[0])

    swc = np.loadtxt(swc_file)
    # print(swc[:3])
    swc = np.array([line for line in swc if line[0] != '#'])
    # print(swc[:3])
    swc[:, 2] = swc[:, 2] * xy_resolution / 1000
    swc[:, 3] = swc[:, 3] * xy_resolution / 1000
    np.savetxt(unified_swc_file, swc, fmt='%d %d %f %f %f %f %d')

def plot_box_of_swc_list(l_measure_files, labels, box_file):
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
    cols = 3
    rows = (num_features + cols - 1) // cols
    # 图像清晰度
    plt.rcParams['savefig.dpi'] = 800
    fig, axes = plt.subplots(rows, cols, figsize=(cols*3.5, 3 * rows))  # 调整figsize和dpi提高清晰度
    axes = axes.flatten()
    # plt.rcParams.update({'font.size': 20})  # 更新字体大小
    # 设置字体 Arial
    # plt.rcParams['font.family'] = 'Arial'
    colors = plt.get_cmap('Set3').colors
    colors = [colors[4], colors[3], colors[0]]

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
        # if feature == 'Number of Branches':
        #     ax.set_ylim(-1.5, 150)
        # elif feature == 'Total Length':
        #     ax.set_ylim(-50, 5000)

        # 获取当前特征的数据子集
        feature_data = df_long[df_long['Feature'] == feature]

        # 计算positions, 每个Feature会有多个箱体
        # positions的数量要等于每个Feature和Type的组合数量
        positions = [0, 0.6, 1.5]
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

        # ax.set_ylabel('')
        ax.get_xaxis().set_visible(False)
        # ax.get_xaxis().set_ticks([])
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

if __name__ == '__main__':
    # seg_dir = "/data/kfchen/trace_ws/paper_trace_result/nnunet/newcel_0.1/0_seg"
    # resized_seg_dir = "/data/kfchen/trace_ws/topology_test/512_seg"
    # tracing_source_dir = "/data/kfchen/trace_ws/paper_trace_result/nnunet/newcel_0.1/3_skel_with_soma"
    # out_swc_dir = r"/data/kfchen/trace_ws/topology_test/no_skel_swc"
    # rescaled_swc_dir = r"/data/kfchen/trace_ws/topology_test/no_skel_swc_rescaled"
    # if(not os.path.exists(out_swc_dir)):
    #     os.makedirs(out_swc_dir)
    # if(not os.path.exists(rescaled_swc_dir)):
    #     os.makedirs(rescaled_swc_dir)
    #
    # seg_files = [f for f in os.listdir(seg_dir) if f.endswith('.tif')]
    # for seg_file in seg_files:
    #     swc_file = os.path.join(out_swc_dir, seg_file.replace('.tif', '.swc'))
    #     if(not os.path.exists(swc_file)):
    #         seg = io.imread(os.path.join(seg_dir, seg_file))
    #         tracing_source = io.imread(os.path.join(tracing_source_dir, seg_file))
    #
    #         seg = resize(seg, tracing_source.shape, order=0, preserve_range=True)
    #         seg = np.flip(seg, axis=1)
    #         seg = (seg - seg.min()) / (seg.max() - seg.min()) * 255
    #         resized_seg_file = os.path.join(resized_seg_dir, seg_file)
    #         io.imsave(resized_seg_file, seg.astype("uint8"))
    #
    #         trace_app2_with_soma(resized_seg_file, swc_file)
    #
    #
    #     rescaled_swc_file = os.path.join(rescaled_swc_dir, seg_file.replace('.tif', '.swc'))
    #     if(not os.path.exists(rescaled_swc_file)):
    #         rescale_xy_resolution(swc_file, rescaled_swc_file)
    #
    # result_csv = os.path.join("/data/kfchen/trace_ws/topology_test/no_skel_rescaled_l_measure_result.csv")
    # if(not os.path.exists(result_csv)):
    #     l_measure_swc_dir(rescaled_swc_dir, result_csv, v3d_path)
    #
    # plot_file = "/data/kfchen/trace_ws/topology_test/l_measure_result_box.png"
    # plot_box_of_swc_list(
    #     [result_csv, "/data/kfchen/trace_ws/paper_trace_result/nnunet/newcel_0.1/8_estimated_radius_swc_l_measure.csv", "/data/kfchen/trace_ws/paper_auto_human_neuron_recon/swc_label/1um_swc_lab_l_measure.csv"],
    #     ['No Skel', 'With Skel', 'Manual'],
    #     plot_file
    # )

    # from_seg_swc_dir = "/data/kfchen/trace_ws/topology_test/ori_trace_result/from_seg/APP2"
    # reconn_seg_swc_dir = "/data/kfchen/trace_ws/paper_trace_result/nnunet/newcel_0.1/8_estimated_radius_swc"
    # flag1, flag2 = 0, 0
    #
    # rescaled_swc_dir = r"/data/kfchen/trace_ws/topology_test/no_skel_swc_rescaled/APP2"
    # for swc_file in os.listdir(from_seg_swc_dir):
    #     rescaled_swc_file = os.path.join(rescaled_swc_dir, swc_file)
    #     if(not os.path.exists(rescaled_swc_file)):
    #         rescale_xy_resolution(os.path.join(from_seg_swc_dir, swc_file), rescaled_swc_file)
    #
    # manual_swc_dir = "/data/kfchen/trace_ws/paper_auto_human_neuron_recon/swc_label/1um_swc_lab"
    #
    # simple_soma_dist_list, reconn_soma_dist_list = [], []
    # for swc_file in os.listdir(rescaled_swc_dir):
    #     simple_swc_file = os.path.join(rescaled_swc_dir, swc_file)
    #     reconn_swc_file = os.path.join(reconn_seg_swc_dir, swc_file)
    #     manual_swc_file = os.path.join(manual_swc_dir, swc_file)
    #
    #     true_soma = np.loadtxt(manual_swc_file)[0, 2:5]
    #     simple_soma = np.loadtxt(simple_swc_file)[0, 2:5]
    #     reconn_soma = np.loadtxt(reconn_swc_file)[0, 2:5]
    #
    #     # print(f"{true_soma}: {simple_soma}: {reconn_soma}")
    #     print(f"dist1: {np.linalg.norm(true_soma - simple_soma)}, dist2: {np.linalg.norm(true_soma - reconn_soma)}")
    #     simple_soma_dist_list.append(np.linalg.norm(true_soma - simple_soma))
    #     reconn_soma_dist_list.append(np.linalg.norm(true_soma - reconn_soma))
    #     if(np.linalg.norm(true_soma - simple_soma) > np.linalg.norm(true_soma - reconn_soma)):
    #         flag1 += 1
    #     else:
    #         flag2 += 1
    #
    # print(f"mean dist1: {np.mean(simple_soma_dist_list)}, mean dist2: {np.mean(reconn_soma_dist_list)}")
    # print(f"flag1: {flag1}, flag2: {flag2}")
    #
    # # plot box
    #
    # fig = plt.figure(figsize=(6, 6))
    # ax = fig.add_subplot(111)
    # ax.boxplot([simple_soma_dist_list, reconn_soma_dist_list], labels=['Simple', 'Reconn'])
    # # y轴的范围
    # ax.set_ylim(0, 100)
    # plt.ylabel('Distance to True Soma (μm)')
    # plt.show()
    # plt.close()




