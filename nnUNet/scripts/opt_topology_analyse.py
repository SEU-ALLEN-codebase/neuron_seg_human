import os.path
import time
import glob
from simple_swc_tool.Topology_scoring import metrics_delin as md
import tifffile
import pandas as pd
import numpy as np
import concurrent.futures
from tqdm import tqdm

def main_opt(G_gt_path, out_dir):
    result_list, failed_files = [], []

    # num = os.path.splitext(os.path.split(G_gt_path)[-1])[0]
    # out_swc_name = num + ".swc"
    out_swc_name = os.path.split(G_gt_path)[-1]
    G_pred_path = os.path.join(out_dir, out_swc_name)
    if (not os.path.exists(G_pred_path)):
        return result_list, failed_files
    result_list.append(os.path.split(G_gt_path)[-1]) # ["ID"]

    G_gt = md.load_graph_swc(G_gt_path)
    G_pred = md.load_graph_swc(G_pred_path)

    try:
        # --------------------------------------------------opt_j
        f1, precision, recall, \
            tp, pp, ap, \
            matches_g, matches_hg, \
            g_gt_snap, g_pred_snap = md.opt_j(G_gt,
                                              G_pred,
                                              th_existing=1,  # 在捕获过程中，只有当该边的所有端点都不在th_existing范围内时，才会将一个附加节点插入到该边中
                                              th_snap=25,  # 如果一个点到最近的边的距离小于th_snap，那么它就被折入图中
                                              alpha=100)  # 鼓励匹配具有相似顺序的两个节点
        result_list.append(precision) # ["optj_precision"]
        result_list.append(recall) # ["optj_recall"]
        result_list.append(f1) # ["optj_f1"]

        # --------------------------------------------------opt_p
        n_conn_precis, n_conn_recall, \
            n_inter_precis, n_inter_recall, \
            con_prob_precis, con_prob_recall, con_prob_f1 = md.opt_p(G_gt, G_pred)

        result_list.append(con_prob_precis) # ["optp_con_prob_precis"]
        result_list.append(con_prob_recall) # ["optp_con_prob_recall"]
        result_list.append(con_prob_f1) # ["optp_con_prob_f1"]

        # --------------------------------------------------opt_g
        f1, spurious, missings, \
            n_preds_sum, n_gts_sum, \
            n_spurious_marbless_sum, \
            n_empty_holess_sum = md.opt_g(G_gt, G_pred,
                                          spacing=10,
                                          dist_limit=300,
                                          dist_matching=25,
                                          N=50,  # to speed up this script
                                          verbose=False)
        result_list.append(spurious) # ["optg_spurious"]
        result_list.append(missings) # ["optg_missings"]
        result_list.append(f1) # ["optg_f1"]

        # print(f"{file_num} files done!\n")
    except:
        failed_files.append(G_gt_path)
        return [], failed_files

    # print(result_list)
    return result_list, failed_files

if __name__ == '__main__':
    num_false_set = [1, 5, 10, 15, 20, 25, 30, 35, 40]

    # gt_dir = r"/data/kfchen/nnUNet/nnUNet_results/Dataset169_hb_10k/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/ptls10/norm_result/pruned_unified_GS"
    # out_dir = r"/data/kfchen/nnUNet/nnUNet_results/Dataset169_hb_10k/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/ptls10/norm_result/pruned_unified_Auto"
    gt_dir = r"/data/kfchen/New Folder/to_Kaifeng/final_doubleChecked_annotation"
    out_dir = r"/data/kfchen/New Folder/to_Kaifeng/final_oneChecked_annotation"
    img_dir = r"/data/kfchen/trace_ws/to_gu/img"
    save_path = r"/data/kfchen/New Folder/to_Kaifeng/result.txt"
    csv_file = '/data/kfchen/New Folder/to_Kaifeng/opt_result.csv'


    flip_flag = False
    if(flip_flag):
        csv_file = "/data/kfchen/nnUNet/nnUNet_results/Dataset169_hb_10k/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/ptls10/norm_result/Human_SingleCell_TrackingTable_20240712.csv"
        df = pd.read_csv(csv_file, encoding='gbk')
        from nnUNet.scripts.resolution_unifier import find_resolution
        fliped_out_dir = out_dir.replace("Auto", "Auto_fliped")
        if(not os.path.exists(fliped_out_dir)):
            os.makedirs(fliped_out_dir)
        swc_files = glob.glob(os.path.join(out_dir, '*.swc'))
        for swc_file in swc_files:
            fliped_swc_file = swc_file.replace("Auto", "Auto_fliped")
            if(os.path.exists(fliped_swc_file)):
                continue
            img_file = os.path.join(img_dir, os.path.split(swc_file)[-1].replace(".swc", ".tif"))
            img = tifffile.imread(img_file)
            xy_resolution = find_resolution(df, os.path.split(swc_file)[-1])
            result_lines = []
            with open(swc_file, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    if line.startswith("#"):
                        result_lines.append(line)
                    else:
                        line = line.strip().split()
                        y = img.shape[1] * xy_resolution / 1000 - float(line[3])
                        result_lines.append(f"{line[0]} {line[1]} {line[2]} {y} {line[4]} {line[5]} {line[6]}\n")
            with open(fliped_swc_file, 'w') as f:
                f.writelines(result_lines)
        out_dir = fliped_out_dir
    rename_flag = True
    if(rename_flag):
        swc_files = glob.glob(os.path.join(out_dir, '*.swc'))
        for swc_file in swc_files:
            new_swc_file = str(int(swc_file.split("/")[-1].split("_")[0])) + ".swc"
            # print(new_swc_file)
            os.rename(swc_file, os.path.join(out_dir, new_swc_file))

        # swc_files = glob.glob(os.path.join(gt_dir, '*.swc'))
        # for swc_file in swc_files:
        #     new_swc_file = str(int(swc_file.split("/")[-1].split("_")[0])) + ".swc"
        #     # print(new_swc_file)
        #     os.rename(swc_file, os.path.join(gt_dir, new_swc_file))

    result_dict = {
        "ID": [],

        "optj_precision": [],
        "optj_recall": [],
        "optj_f1": [],

        "optp_con_prob_precis": [],
        "optp_con_prob_recall": [],
        "optp_con_prob_f1": [],

        "optg_spurious": [],
        "optg_missings": [],
        "optg_f1": []
    }
    result_list = []
    failed_files = []
    swc_files = glob.glob(os.path.join(gt_dir, '*.swc'))

    # for G_gt_path in swc_files:  # 读所有的swc文件
    #     result_list, failed_files = main_opt(G_gt_path, out_dir, result_list, failed_files)

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        progress = tqdm(total=len(swc_files), desc='Processing SWC Files', unit='file')
        future_to_file = {executor.submit(main_opt, G_gt_path, out_dir): G_gt_path for
                          G_gt_path in swc_files}
        for future in concurrent.futures.as_completed(future_to_file):
            progress.update(1)  # 更新进度条
            try:
                res_list, fail_files = future.result()
                if(not res_list == []):result_list.append(res_list)
                if(not fail_files == []):failed_files.append(fail_files)
            except Exception as e:
                print(f"Error processing file {future_to_file[future]}: {e}")
        progress.close()

    # 整理result_list
    for result in result_list:
        # print(result)
        result_dict["ID"].append(result[0])
        result_dict["optj_precision"].append(result[1])
        result_dict["optj_recall"].append(result[2])
        result_dict["optj_f1"].append(result[3])
        result_dict["optp_con_prob_precis"].append(result[4])
        result_dict["optp_con_prob_recall"].append(result[5])
        result_dict["optp_con_prob_f1"].append(result[6])
        result_dict["optg_spurious"].append(result[7])
        result_dict["optg_missings"].append(result[8])
        result_dict["optg_f1"].append(result[9])


    print("mean optj_precision: ", np.mean(result_dict["optj_precision"]),
          "mean optj_recall: ", np.mean(result_dict["optj_recall"]),
          "mean optj_f1: ", np.mean(result_dict["optj_f1"]))
    print("mean optp_con_prob_precis: ", np.mean(result_dict["optp_con_prob_precis"]),
          "mean optp_con_prob_recall: ", np.mean(result_dict["optp_con_prob_recall"]),
          "mean optp_con_prob_f1: ", np.mean(result_dict["optp_con_prob_f1"]))
    print("mean optg_spurious: ", np.mean(result_dict["optg_spurious"]),
          "mean optg_missings: ", np.mean(result_dict["optg_missings"]),
          "mean optg_f1: ", np.mean(result_dict["optg_f1"]))

    print("failed_files: ", failed_files)


    with open(csv_file, 'w') as f:
        f.write("ID,optj_precision,optj_recall,optj_f1,optp_con_prob_precis,optp_con_prob_recall,optp_con_prob_f1,optg_spurious,optg_missings,optg_f1\n")
        for i in range(len(result_dict["ID"])):
            f.write("{},{},{},{},{},{},{},{},{}, {}\n".format(
                result_dict["ID"][i],
                result_dict["optj_precision"][i],
                result_dict["optj_recall"][i],
                result_dict["optj_f1"][i],
                result_dict["optp_con_prob_precis"][i],
                result_dict["optp_con_prob_recall"][i],
                result_dict["optp_con_prob_f1"][i],
                result_dict["optg_spurious"][i],
                result_dict["optg_missings"][i],
                result_dict["optg_f1"][i]
            ))