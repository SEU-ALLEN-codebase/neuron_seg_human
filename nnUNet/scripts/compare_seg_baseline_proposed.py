import os
import random
import tifffile
from nnUNet.nnunetv2.dataset_conversion.generate_nnunet_dataset import augment_gamma
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from matplotlib.patches import Rectangle
from nnUNet.scripts.mip import get_mip_swc

def find_swc(swc_dir, tif_file):
    file_name = tif_file[:-4]
    swc_files = os.listdir(swc_dir)
    for swc_file in swc_files:
        if int(file_name.split("_")[0]) == int(swc_file.split("_")[0].split(".")[0]):
            return os.path.join(swc_dir, swc_file)


def plot_compare_result(file_list, img_dir, seg_dirs, seg_labels, recon_dirs, recon_labels, info_file, index):
    plt.figure(figsize=(5 * 5, 7 * 5))

    for idx, img_file in enumerate(file_list):
        img_path = os.path.join(img_dir, img_file)
        seg_paths = [os.path.join(seg_dir, img_file) for seg_dir in seg_dirs]

        img = tifffile.imread(img_path)
        img = augment_gamma(img)
        img_mip = np.max(img, axis=0)

        segs = [tifffile.imread(seg_path) for seg_path in seg_paths]
        segs = [(seg - np.min(seg)) / (np.max(seg) - np.min(seg)) * 255 for seg in segs]
        seg_mips = [np.max(seg, axis=0) for seg in segs]

        recon_paths = [find_swc(recon_dir, img_file) for recon_dir in recon_dirs]
        recon_mips = [get_mip_swc(recon_path, img, ignore_background=True) for recon_path in recon_paths]

        info_df = pd.read_csv(info_file, encoding='gbk')
        ID = int(img_file.split(".")[0])
        brain_region = info_df[info_df['Cell ID'] == ID]['脑区'].values[0]
        label_info = f"No.0{img_file.split('.')[0]} ({brain_region})"

        default_size = 256
        img_mip = cv2.resize(img_mip, (default_size, default_size))
        seg_mips = [cv2.resize(seg_mip, (default_size, default_size)) for seg_mip in seg_mips]

        img_mip = cv2.cvtColor(img_mip.astype('uint8'), cv2.COLOR_GRAY2RGB)
        seg_mips = [cv2.cvtColor(seg_mip.astype('uint8'), cv2.COLOR_GRAY2RGB) for seg_mip in seg_mips]

        recon_mips = [cv2.resize(recon_mip, (default_size, default_size)) for recon_mip in recon_mips]
        recon_mips = [cv2.rectangle(recon_mip, (0, 0), (recon_mip.shape[1], recon_mip.shape[0]), (0, 0, 0), 3) for
                      recon_mip in recon_mips] # 添加边框

        labels = [label_info, seg_labels[0], recon_labels[0], seg_labels[1], recon_labels[1], seg_labels[2], recon_labels[2]]
        plot_mips = [img_mip, seg_mips[0], recon_mips[0], seg_mips[1], recon_mips[1], seg_mips[2], recon_mips[2]]

        for i, plot_mip in enumerate(plot_mips):
            # 横向
            # ax = plt.subplot(6, 5, idx * 5 + i + 1)
            # 纵向
            ax = plt.subplot(7, 5, i * 5 + idx + 1)
            ax.imshow(plot_mip)
            ax.axis('off')
            if(i == 0):
                ax.text(plot_mip.shape[1] / 2, 20, labels[i], fontsize=30, color='white', ha='center', va='top',
                        backgroundcolor='black')

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.01, hspace=0.01)
    plt.savefig(r"/data/kfchen/trace_ws/paper_auto_human_neuron_recon/test_seg_220/compare_result/" + str(index) + ".png")
    plt.close()

def plot_compare_result_0(file_list, img_dir, seg_dirs, seg_labels, info_file, index):
    plt.figure(figsize=(5 * 5, 4 * 5))

    for idx, img_file in enumerate(file_list):
        img_path = os.path.join(img_dir, img_file)
        seg_paths = [os.path.join(seg_dir, img_file) for seg_dir in seg_dirs]

        img = tifffile.imread(img_path)
        img = augment_gamma(img)
        img_mip = np.max(img, axis=0)

        segs = [tifffile.imread(seg_path) for seg_path in seg_paths]
        segs = [(seg - np.min(seg)) / (np.max(seg) - np.min(seg)) * 255 for seg in segs]
        seg_mips = [np.max(seg, axis=0) for seg in segs]

        info_df = pd.read_csv(info_file, encoding='gbk')
        ID = int(img_file.split(".")[0])
        brain_region = info_df[info_df['Cell ID'] == ID]['脑区'].values[0]
        label_info = f"No.0{img_file.split('.')[0]} ({brain_region})"

        default_size = 256
        img_mip = cv2.resize(img_mip, (default_size, default_size))
        seg_mips = [cv2.resize(seg_mip, (default_size, default_size)) for seg_mip in seg_mips]

        img_mip = cv2.cvtColor(img_mip.astype('uint8'), cv2.COLOR_GRAY2RGB)
        seg_mips = [cv2.cvtColor(seg_mip.astype('uint8'), cv2.COLOR_GRAY2RGB) for seg_mip in seg_mips]

        ax1 = plt.subplot(4, 5, idx + 1)
        ax1.imshow(img_mip)
        ax1.axis('off')
        ax1.text(img_mip.shape[1] / 2, 20, label_info, fontsize=30, color='white', ha='center', va='top',
                 backgroundcolor='black')

        for i, seg_mip in enumerate(seg_mips):
            ax = plt.subplot(4, 5, 5 + idx + 1 + i * 5)
            ax.imshow(seg_mip)
            ax.axis('off')
            # ax.text(seg_mip.shape[1] / 2, 20, seg_labels[i], fontsize=30, color='white', ha='center', va='top',
            #         backgroundcolor='black')

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.01, hspace=0.01)
    plt.savefig(r"/data/kfchen/trace_ws/paper_auto_human_neuron_recon/test_seg_220/compare_result/" + str(index) + ".png")
    plt.close()


if __name__ == "__main__":
    img_dir = r"/data/kfchen/trace_ws/to_gu/img"
    info_file = "/data/kfchen/nnUNet/nnUNet_results/Dataset169_hb_10k/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/ptls10/norm_result/Human_SingleCell_TrackingTable_20240712.csv"
    seg_dirs = [
        r"/data/kfchen/trace_ws/paper_auto_human_neuron_recon/test_seg_220/label",
        r"/data/kfchen/trace_ws/paper_auto_human_neuron_recon/test_seg_220/source500",
        r"/data/kfchen/trace_ws/paper_auto_human_neuron_recon/test_seg_220/ptls10",
    ]
    seg_labels = [
        "Label",
        "Baseline",
        "Proposed",
    ]
    recon_dirs = [
        r'/data/kfchen/trace_ws/to_gu/new_sort_lab/2_flip_after_sort',
        r"/data/kfchen/trace_ws/paper_auto_human_neuron_recon/origin_recon/source500",
        r"/data/kfchen/trace_ws/paper_auto_human_neuron_recon/origin_recon/ptls10",
    ]
    recon_labels = [
        "Manual Reconstruction",
        "Baseline Reconstruction",
        "Proposed Reconstruction",
    ]

    # tif
    img_files = os.listdir(seg_dirs[-1])
    img_files = [img_file for img_file in img_files if img_file.endswith(".tif")]
    # img_files = random.sample(img_files, 5)
    img_files = ["3515.tif", "3083.tif", "5364.tif", "2735.tif", "2497.tif"]
    # plot_compare_result_0(img_files, img_dir, seg_dirs, seg_labels, info_file, "chosen")
    plot_compare_result(img_files, img_dir, seg_dirs, seg_labels, recon_dirs, recon_labels, info_file, "chosen")

    # for index in range(int(len(img_files)/5)):
    #     plot_compare_result(img_files[index*5:index*5+5], img_dir, seg_dirs, seg_labels, info_file, index)



