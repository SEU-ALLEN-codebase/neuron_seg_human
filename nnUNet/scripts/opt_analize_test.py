import os.path
import time
import glob

# from nnUNet.scripts.val_gamma_via_hist import common_files
from simple_swc_tool.Topology_scoring import metrics_delin as md
import tifffile
import pandas as pd
import numpy as np
import concurrent.futures
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from concurrent.futures import ThreadPoolExecutor
import cProfile
import multiprocessing


def opt_analyse_file_v2(gt_swc_file, pred_swc_file, result_file):
    G_gt = md.load_graph_swc(gt_swc_file)
    G_pred = md.load_graph_swc(pred_swc_file)

    result_list = []

    id = os.path.basename(gt_swc_file).split(".")[0].split('_')[0]
    id = int(id)
    result_list.append(id)  # ["ID"]

    f1, precision, recall, \
        tp, pp, ap, \
        matches_g, matches_hg, \
        g_gt_snap, g_pred_snap = md.opt_j(G_gt,
                                          G_pred,
                                          th_existing=1,  # 在捕获过程中，只有当该边的所有端点都不在th_existing范围内时，才会将一个附加节点插入到该边中
                                          th_snap=25,  # 如果一个点到最近的边的距离小于th_snap，那么它就被折入图中
                                          alpha=100)  # 鼓励匹配具有相似顺序的两个节点
    result_list.extend([precision, recall, f1])

    # --------------------------------------------------opt_p
    n_conn_precis, n_conn_recall, \
        n_inter_precis, n_inter_recall, \
        con_prob_precis, con_prob_recall, con_prob_f1 = md.opt_p(G_gt, G_pred)
    result_list.extend([con_prob_precis, con_prob_recall, con_prob_f1])

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
    result_list.extend([spurious, missings, f1])
    print(result_list)

    if(len(result_list) == 10):
        np.savetxt(result_file, result_list, fmt='%s', delimiter=',')

def current_task(swc_file, gt_swc_dir, pred_swc_dir, opt_result_dir):
    gt_swc_file = os.path.join(gt_swc_dir, swc_file)
    pred_swc_file = os.path.join(pred_swc_dir, swc_file)
    result_file = os.path.join(opt_result_dir, swc_file.replace('.swc', '.npy'))
    # if((not os.path.exists(result_file)) and os.path.exists(gt_swc_file) and os.path.exists(pred_swc_file)):
    opt_analyse_file_v2(gt_swc_file, pred_swc_file, result_file)

def opt_analyse_dir_v2(gt_swc_dir, pred_swc_dir):
    opt_result_dir = pred_swc_dir + "_opt_result"
    os.makedirs(opt_result_dir, exist_ok=True)

    gt_swc_files = [f for f in os.listdir(gt_swc_dir) if f.endswith('.swc')]
    pred_swc_files = [f for f in os.listdir(pred_swc_dir) if f.endswith('.swc')]
    common_files = set(gt_swc_files) & set(pred_swc_files)
    common_files = sorted(list(common_files))

    # 多进程
    with multiprocessing.Pool() as pool:
        list(
            tqdm(pool.starmap(current_task, [(file, gt_swc_dir, pred_swc_dir, opt_result_dir) for file in common_files]),
                 total=len(common_files)))
    # 多线程
    # with ThreadPoolExecutor(max_workers=10) as executor:
    #     list(
    #         tqdm(executor.map(current_task, common_files, [gt_swc_dir]*len(common_files), [pred_swc_dir]*len(common_files), [opt_result_dir]*len(common_files)),
    #              total=len(common_files)))



if __name__ == '__main__':
    gt_dir = r"/data/kfchen/trace_ws/paper_auto_human_neuron_recon/swc_label/1um_swc_lab"
    out_dirs = [
        "/data/kfchen/trace_ws/topology_test/rescaled_trace_result/from_seg/Advantra",
    ]

    for out_dir in out_dirs:
        print('processing:', out_dir)
        # out_dir = r"/data/kfchen/trace_ws/paper_trace_result/nnunet/newcel_0.1/8_estimated_radius_swc"
        # img_dir = r"/data/kfchen/trace_ws/to_gu/img"
        # save_path = r"/data/kfchen/trace_ws/paper_trace_result/nnunet/newcel_0.1/result.txt"
        csv_file = out_dir + "_opt_result.csv"
        if(not os.path.exists(csv_file)):
            opt_analyse_dir_v2(gt_dir, out_dir)









