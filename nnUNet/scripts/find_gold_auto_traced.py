import cv2
import pandas as pd
import numpy as np
import ast

import tifffile
from scipy.stats import chisquare, ks_2samp
import os
from nnUNet.scripts.mip import get_mip_swc, get_mip
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def decode_sholl_hist(sholl_x, sholl_y, calc_threshold=0.75):
    bins_str = sholl_x
    samples_str = sholl_y
    bins = ast.literal_eval(bins_str)
    samples = ast.literal_eval(samples_str)
    max_bin = max(bins)

    # sholl_x = [x for x in sholl_x if float(x) > max_x * calc_threshold]
    # sholl_y = sholl_y[len(sholl_y) - len(sholl_x):]
    bins = [bin for bin in bins if bin >= max_bin * calc_threshold]
    samples = samples[len(samples) - len(bins):]
    # print(len(bins), len(samples))
    avg_sample = 0
    for i in range(len(bins)):
        avg_sample += bins[i] * samples[i]
    return avg_sample / len(bins)

def add_opt_metrics(opt_result_file, gold_standard_metrics_result):
    opt_result = pd.read_csv(opt_result_file)
    # sort
    opt_result = opt_result.sort_values(by='ID')
    ids, opt_j_f1s, opt_p_f1s, opt_g_f1s = opt_result['ID'], opt_result['optj_f1'], opt_result['optp_con_prob_f1'], \
    opt_result['optg_f1']
    avg_opt_f1s = (opt_j_f1s + opt_p_f1s + opt_g_f1s) / 3

    for id, opt_avg_f1 in zip(ids, avg_opt_f1s):
        gold_standard_metrics_result['id'].append(id)
        gold_standard_metrics_result['topological_similarity'].append(opt_avg_f1)

    return ids, gold_standard_metrics_result

def add_l_measure_metrics(ids, gt_l_measure_file, auto_l_measure_file, gold_standard_metrics_result):
    gt_l_measure = pd.read_csv(gt_l_measure_file)
    auto_l_measure = pd.read_csv(auto_l_measure_file)

    gt_l_measure = gt_l_measure.sort_values(by='ID')
    auto_l_measure = auto_l_measure.sort_values(by='ID')

    # print(gt_l_measure)
    for id in ids:
        # gt_total_length = gt_l_measure[gt_l_measure['ID'] == id]['Total Length'].values[0]
        # auto_total_length = auto_l_measure[auto_l_measure['ID'] == id]['Total Length'].values[0]
        # r1 = auto_total_length / gt_total_length

        gt_number_of_branches = gt_l_measure[gt_l_measure['ID'] == id]['Number of Branches'].values[0]
        auto_number_of_branches = auto_l_measure[auto_l_measure['ID'] == id]['Number of Branches'].values[0]
        r2 = auto_number_of_branches / gt_number_of_branches

        gt_number_of_tips = gt_l_measure[gt_l_measure['ID'] == id]['Number of Tips'].values[0]
        auto_number_of_tips = auto_l_measure[auto_l_measure['ID'] == id]['Number of Tips'].values[0]
        r3 = auto_number_of_tips / gt_number_of_tips

        # low level branches形象语言 / major chains计算语言
        # gt_low_level_branches =
        # auto_low_level_branches =

        gold_standard_metrics_result['branch_similarity'].append((r2 + r3) / 2)

    return gold_standard_metrics_result

def find_line(id, df):
    id = id.split('.')[0]
    for i in range(len(df)):
        if str(int(df['ID'][i].split('_')[0])) == id:
            return i


def add_sholl_metrics(ids, gt_sholl_file, auto_sholl_file, gold_standard_metrics_result):
    gt_sholl = pd.read_csv(gt_sholl_file, encoding='gbk').sort_values(by='ID')
    auto_sholl = pd.read_csv(auto_sholl_file, encoding='gbk').sort_values(by='ID')
    # print(gt_sholl)
    # print(auto_sholl)

    for id in ids:
        gt_sholl_x = gt_sholl[gt_sholl['ID'] == id]['Full_sholl_x'].values[0]
        gt_sholl_y = gt_sholl[gt_sholl['ID'] == id]['Full_sholl_y'].values[0]
        auto_sholl_x = auto_sholl[auto_sholl['ID'] == id]['Full_sholl_x'].values[0]
        auto_sholl_y = auto_sholl[auto_sholl['ID'] == id]['Full_sholl_y'].values[0]
        # gt_sholl_x = gt_sholl['Full_sholl_x'][find_line(id, gt_sholl)]
        # gt_sholl_y = gt_sholl['Full_sholl_y'][find_line(id, gt_sholl)]
        # auto_sholl_x = auto_sholl['Full_sholl_x'][find_line(id, auto_sholl)]
        # auto_sholl_y = auto_sholl['Full_sholl_y'][find_line(id, auto_sholl)]

        # r1 = decode_sholl_hist(auto_sholl_x, auto_sholl_y) / decode_sholl_hist(gt_sholl_x, gt_sholl_y)
        # gold_standard_metrics_result['distent_terminal_similarity'].append(r1)
        gold_standard_metrics_result['low_level_branch_similarity'].append(compare_histograms(gt_sholl_x, gt_sholl_y, auto_sholl_x, auto_sholl_y, compare_range=[0, 0.25]))
        gold_standard_metrics_result['middle_level_branch_similarity'].append(compare_histograms(gt_sholl_x, gt_sholl_y, auto_sholl_x, auto_sholl_y, compare_range=[0.25, 0.75]))

    return gold_standard_metrics_result



def compare_histograms(bins_str1, samples_str1, bins_str2, samples_str2, compare_range=[0, 1], smooth=1e-10):
    # 将字符串转换为列表
    bins1 = ast.literal_eval(bins_str1)
    samples1 = ast.literal_eval(samples_str1)
    bins2 = ast.literal_eval(bins_str2)
    samples2 = ast.literal_eval(samples_str2)

    # 确定全局的最小和最大bins
    global_min_bin = min(min(bins1), min(bins2))
    global_max_bin = max(max(bins1), max(bins2))

    # 创建全局bins数组
    global_bins = np.arange(global_min_bin, global_max_bin+bins1[1] - bins1[0], step=bins1[1] - bins1[0])

    # 创建对应的直方图数组，初始化为0
    histogram1 = np.zeros(len(global_bins))
    histogram2 = np.zeros(len(global_bins))
    # histogram1 = np.ones(len(global_bins)) * smooth
    # histogram2 = np.ones(len(global_bins)) * smooth

    # 填充存在的bins对应的samples
    def fill_histogram(global_bins, bins, samples, histogram):
        # 找到子集bins在全局bins中的索引位置
        indices = np.searchsorted(global_bins, bins[:])
        for i in range(len(samples)):
            histogram[indices[i]] = samples[i]

    fill_histogram(global_bins, bins1, samples1, histogram1)
    fill_histogram(global_bins, bins2, samples2, histogram2)

    # 可以选择在特定范围内比较
    if compare_range != [0, 1]:
        compare_range = [global_max_bin * compare_range[0], global_max_bin * compare_range[1]]
        start_index = np.searchsorted(global_bins, compare_range[0], 'left')
        end_index = np.searchsorted(global_bins, compare_range[1], 'right')
        histogram1 = histogram1[start_index:end_index]
        histogram2 = histogram2[start_index:end_index]

    # 归一化直方图以确保总和相同
    sum1 = np.sum(histogram1)
    sum2 = np.sum(histogram2)
    histogram1 = histogram1 / sum1
    histogram2 = histogram2 / sum2

    # 计算卡方检验
    # chi2, p_value = chisquare(histogram2, f_exp=histogram1)
    # 执行KS测试
    statistic, p_value = ks_2samp(histogram1, histogram2)
    # print("-----------------")
    # print(bins1, bins2, samples1, samples2)
    # print(histogram1, histogram2)
    # print(p_value)

    # p越接近1，两个分布越相似, s越接近0，两个分布越相似
    # if(p_value==1):
    #     print(histogram1, histogram2, statistic, p_value)
    return 1-statistic

def plot_samples(sample_list, img_dir, gt_swc_dir, auto_swc_dir, info_file, plot_number=5, out_dir=None):
    if(out_dir is None):
        out_dir = os.path.join(img_dir, 'sample_mips')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    info_df = pd.read_csv(info_file, encoding='gbk')
    # random select samples
    sample_list = np.random.choice(sample_list, plot_number, replace=False)
    # print(sample_list)
    plt.figure(figsize=(plot_number * 5, 15))

    # for idx, sample in enumerate(sample_list):
    #     img_file = os.path.join(img_dir, sample.replace('.swc', '.tif'))
    #     swc_file1 = os.path.join(gt_swc_dir, sample)
    #     swc_file2 = os.path.join(auto_swc_dir, sample)
    #
    #     img = tifffile.imread(img_file)
    #     # print(img_file)
    #     img_mip = get_mip(img)
    #     # img_mip = np.stack(img_mip, axis=-1)
    #     # tifffile.imwrite(os.path.join(out_dir, f"{sample}_mip.png"), img_mip)
    #
    #     swc_mip1 = get_mip_swc(swc_file1, img, ignore_background=True)
    #     swc_mip2 = get_mip_swc(swc_file2, img, ignore_background=True)
    #     img_mip = cv2.cvtColor(img_mip, cv2.COLOR_GRAY2RGB)
    #
    #     ID = int(sample.split(".")[0])
    #     brain_region = info_df[info_df['Cell ID'] == ID]['脑区'].values[0]
    #
    #     file_label = f"No.0{sample.split('.')[0]}({brain_region})"
    #     print(file_label)
    #
    #     plt.subplot(3, plot_number, idx + 1)
    #     print(np.max(img_mip), np.min(img_mip))
    #     plt.imshow(img_mip)
    #     # plt.title(file_label)
    #     plt.axis('off')
    #
    #     # # Plot GT SWC MIP
    #     # plt.subplot(3, plot_number, idx + 1 + plot_number)
    #     # plt.imshow(swc_mip1)
    #     # plt.axis('off')
    #     #
    #     # # Plot Auto SWC MIP
    #     # plt.subplot(3, plot_number, idx + 1 + 2 * plot_number)
    #     # plt.imshow(swc_mip2)
    #     # plt.axis('off')

    # plt.tight_layout()
    # plt.show()

    for idx, sample in enumerate(sample_list):
        img_file = os.path.join(img_dir, sample.replace('.swc', '.tif'))
        swc_file1 = os.path.join(gt_swc_dir, sample)
        swc_file2 = os.path.join(auto_swc_dir, sample)

        ID = int(sample.split(".")[0])
        brain_region = info_df[info_df['Cell ID'] == ID]['脑区'].values[0]
        label_info = f"No.0{sample.split('.')[0]} ({brain_region})"
        xy_resolution = info_df[info_df['Cell ID'] == ID]['xy拍摄分辨率(*10e-3μm/px)'].values[0]

        img = tifffile.imread(img_file)
        img_mip = get_mip(img)
        img_mip = cv2.cvtColor(img_mip, cv2.COLOR_GRAY2RGB)
        # print(img.shape)
        x_limit = int(img.shape[2] * xy_resolution / 1000)
        y_limit = int(img.shape[1] * xy_resolution / 1000)

        swc_mip1 = get_mip_swc(swc_file1, img[:,:y_limit, :x_limit], ignore_background=True)
        swc_mip1 = np.flip(swc_mip1, axis=0)
        swc_mip2 = get_mip_swc(swc_file2, img[:,:y_limit, :x_limit], ignore_background=True)

        # resize
        default_size = 256
        img_mip = cv2.resize(img_mip, (default_size, default_size))
        swc_mip1 = cv2.resize(swc_mip1, (default_size, default_size))
        swc_mip2 = cv2.resize(swc_mip2, (default_size, default_size))



        # Plotting each sample in a grid
        ax1 = plt.subplot(3, plot_number, idx + 1)
        plt.imshow(img_mip)
        # plt.title(f"{sample} - {brain_region}")
        plt.axis('off')
        # label info
        # plt.text(0, 30, label_info, fontsize=30, color='white', backgroundcolor='black')
        plt.text(img_mip.shape[1]/2, 20, label_info, fontsize=30, color='white', ha='center', va='top', backgroundcolor='black')

        ax2 = plt.subplot(3, plot_number, idx + 1 + plot_number)
        plt.imshow(swc_mip1)
        # plt.title("GT Annotation")
        plt.axis('off')
        ax2.add_patch(Rectangle((0, 0), swc_mip1.shape[1] - 1, swc_mip1.shape[0] - 1, edgecolor='black', facecolor='none',
                                linewidth=1))
        plt.text(10, 20, "Gold standard", fontsize=20, color='black')

        ax3 = plt.subplot(3, plot_number, idx + 1 + 2 * plot_number)
        plt.imshow(swc_mip2)
        # plt.title("Automated Annotation")
        plt.axis('off')
        ax3.add_patch(Rectangle((0, 0), swc_mip2.shape[1] - 1, swc_mip2.shape[0] - 1, edgecolor='black', facecolor='none',
                                linewidth=1))
        plt.text(10, 20, "Proposed", fontsize=20, color='black')

        # Optionally save each MIP image
        # plt.savefig(os.path.join(out_dir, f"{sample}_preview.png"))

    plt.tight_layout()
    # plt.show()


    # Optionally save the figure
    if out_dir:
        file_path = os.path.join(out_dir, 'sample_mips_preview.png')
        if(os.path.exists(file_path)):
            os.remove(file_path)
        plt.savefig(file_path)









if __name__ == '__main__':
    img_dir = r"/data/kfchen/trace_ws/to_gu/img"
    gt_swc_dir = r"/data/kfchen/nnUNet/nnUNet_results/Dataset169_hb_10k/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/ptls10/norm_result/pruned_unified_GS"
    auto_swc_dir = r"/data/kfchen/nnUNet/nnUNet_results/Dataset169_hb_10k/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/ptls10/norm_result/pruned_unified_Auto"

    opt_result_file = '/data/kfchen/nnUNet/nnUNet_results/Dataset169_hb_10k/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/ptls10/norm_result/opt_result.csv'
    gt_l_measure_file = "/data/kfchen/nnUNet/nnUNet_results/Dataset169_hb_10k/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/ptls10/norm_result/pruned_gt_swc.csv"
    auto_l_measure_file = "/data/kfchen/nnUNet/nnUNet_results/Dataset169_hb_10k/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/ptls10/norm_result/pruned_pred_swc.csv"
    gt_sholl_file = "/data/kfchen/nnUNet/nnUNet_results/Dataset169_hb_10k/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/ptls10/norm_result/gt_sholl_analysis_results_full.csv"
    auto_sholl_file = "/data/kfchen/nnUNet/nnUNet_results/Dataset169_hb_10k/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/ptls10/norm_result/auto_sholl_analysis_results_full.csv"
    info_file = "/data/kfchen/nnUNet/nnUNet_results/Dataset169_hb_10k/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/ptls10/norm_result/Human_SingleCell_TrackingTable_20240712.csv"

    # opt_result_file = r"/data/kfchen/New Folder/to_Kaifeng/opt_result_A2_A3.csv"
    # gt_sholl_file = r"/data/kfchen/New Folder/to_Kaifeng/final_doubleChecked_annotation.csv"
    # auto_sholl_file = r"/data/kfchen/New Folder/to_Kaifeng/final_oneChecked_annotation.csv"
    # opt_result_file = r"/data/kfchen/New Folder/to_Kaifeng/opt_result_A1_A3.csv"
    # gt_sholl_file = r"/data/kfchen/New Folder/to_Kaifeng/final_doubleChecked_annotation.csv"
    # auto_sholl_file = r"/data/kfchen/New Folder/to_Kaifeng/final_original_annotation.csv"

    gold_standard_metrics_result = {
        "id": [],
        "topological_similarity": [],
        "branch_similarity": [],
        "distent_terminal_similarity": [],
        "low_level_branch_similarity": [],
        "middle_level_branch_similarity": [],

        'score': [],
    }


    ids, gold_standard_metrics_result = add_opt_metrics(opt_result_file, gold_standard_metrics_result)
    # gold_standard_metrics_result = add_l_measure_metrics(ids, gt_l_measure_file, auto_l_measure_file, gold_standard_metrics_result)
    add_sholl_metrics(ids, gt_sholl_file, auto_sholl_file, gold_standard_metrics_result)

    # gold_threshold = [0.9853, 0.9900, 0.9404] # C2_C3
    # silver_threshold = [0.8698, 0.9592, 0.7202] # C1_C3
    # gold_threshold = [0.9, 0.9, 0.9]
    # silver_threshold = [0.85, 0.85, 0.85]
    # bronze_threshold = [0.8, 0.8, 0.8]
    # manual_c_std=[0.05239526404138365, 0.12183239475430767, 0.08393750920696284]
    # manual_c_mean=[0.8698618361824826, 0.7491597829608033, 0.7285503014738852]
    # gold_threshold = [manual_c_mean[0], manual_c_mean[1], manual_c_mean[2]]
    # # silver_threshold = [manual_c_mean[0] - manual_c_std[0], manual_c_mean[1] - manual_c_std[1], manual_c_mean[2] - manual_c_std[2]]
    # # bronze_threshold = [manual_c_mean[0] - 2 * manual_c_std[0], manual_c_mean[1] - 2 * manual_c_std[1], manual_c_mean[2] - 2 * manual_c_std[2]]
    # silver_threshold = 0.8 * np.array(gold_threshold)
    # bronze_threshold = 0.6 * np.array(gold_threshold)
    gold_threshold, silver_threshold, bronze_threshold, iron = 0.8402641335501222, 0.7825239735390571, 0.724783813527992, 0.6670436535169269

    gold_samples = []
    silver_samples = []
    bronze_samples = []
    iron_samples = []

    for i in range(len(ids)):
        gold_standard_metrics_result['score'].append(
            (gold_standard_metrics_result['topological_similarity'][i] * 1.0 +
            gold_standard_metrics_result['low_level_branch_similarity'][i] * 1.0 +
            gold_standard_metrics_result['middle_level_branch_similarity'][i] * 1.0) / 3
        )

    for i in range(len(ids)):
        if(gold_standard_metrics_result['score'][i] >= gold_threshold):
            gold_samples.append(ids[i])
        elif(gold_standard_metrics_result['score'][i] >= silver_threshold):
            silver_samples.append(ids[i])
        elif(gold_standard_metrics_result['score'][i] >= bronze_threshold):
            bronze_samples.append(ids[i])
        elif(gold_standard_metrics_result['score'][i] >= iron):
            iron_samples.append(ids[i])

        # flag1 = int(gold_standard_metrics_result['topological_similarity'][i] >= gold_threshold[0])
        # flag2 = int(gold_standard_metrics_result['low_level_branch_similarity'][i] >= gold_threshold[1])
        # flag3 = int(gold_standard_metrics_result['middle_level_branch_similarity'][i] >= gold_threshold[2])
        #
        # # if flag1 + flag2 + flag3 == 3:
        # #     gold_samples.append(ids[i])
        # # elif flag1 + flag2 + flag3 == 2:
        # #     silver_samples.append(ids[i])
        # # elif flag1 + flag2 + flag3 == 1:
        # #     bronze_samples.append(ids[i])
        # if(gold_standard_metrics_result['topological_similarity'][i] >= gold_threshold[0]
        #         and gold_standard_metrics_result['low_level_branch_similarity'][i] >= gold_threshold[1]
        #         and gold_standard_metrics_result['middle_level_branch_similarity'][i] >= gold_threshold[2]):
        #     gold_samples.append(ids[i])
        # elif(gold_standard_metrics_result['topological_similarity'][i] >= silver_threshold[0]
        #      and gold_standard_metrics_result['low_level_branch_similarity'][i] >= silver_threshold[1]
        #      and gold_standard_metrics_result['middle_level_branch_similarity'][i] >= silver_threshold[2]):
        #     silver_samples.append(ids[i])
        # elif(gold_standard_metrics_result['topological_similarity'][i] >= bronze_threshold[0]
        #      and gold_standard_metrics_result['low_level_branch_similarity'][i] >= bronze_threshold[1]
        #      and gold_standard_metrics_result['middle_level_branch_similarity'][i] >= bronze_threshold[2]):
        #     bronze_samples.append(ids[i])


    # print avg
    # print(np.mean(gold_standard_metrics_result['topological_similarity']), np.mean(gold_standard_metrics_result['low_level_branch_similarity']), np.mean(gold_standard_metrics_result['middle_level_branch_similarity']))
    print(f"gold samples: {len(gold_samples)}, silver samples: {len(silver_samples)}, bronze samples: {len(bronze_samples)}, iron samples: {len(iron_samples)}",
          f"gold rate: {len(gold_samples) / len(ids)}, silver rate: {len(silver_samples) / len(ids)}, bronze rate: {len(bronze_samples) / len(ids)}, iron rate: {len(iron_samples) / len(ids)}")

    # s1, s2, s3 = gold_standard_metrics_result['topological_similarity'], gold_standard_metrics_result['low_level_branch_similarity'], gold_standard_metrics_result['middle_level_branch_similarity']
    # print(f"mean - std: {np.mean(s1) - np.std(s1), np.mean(s2) - np.std(s2), np.mean(s3) - np.std(s3)}")
    # print(f"mean: {np.mean(s1), np.mean(s2), np.mean(s3)}")
    # print(f"mean + std: {np.mean(s1) + np.std(s1), np.mean(s2) + np.std(s2), np.mean(s3) + np.std(s3)}")
    # print(f"std: {np.std(s1), np.std(s2), np.std(s3)}")
    # # 中位数
    # print(f"median: {np.median(s1), np.median(s2), np.median(s3)}")
    score = gold_standard_metrics_result['score']
    print(f"mean - std: {np.mean(score) - np.std(score)}")
    print(f"mean: {np.mean(score)}")
    print(f"mean + std: {np.mean(score) + np.std(score)}")
    print(score)

    ggs = ["3397.swc", "3275.swc", "3030.swc", "3101.swc", "3112.swc", "2447.swc"]
    plot_samples(ggs, img_dir, gt_swc_dir, auto_swc_dir, info_file, plot_number=6, out_dir=None)

