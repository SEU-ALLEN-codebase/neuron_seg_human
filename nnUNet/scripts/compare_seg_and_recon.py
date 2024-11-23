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
from pylib.file_io import load_image

neuron_info_file =
def find_resolution(filename):
    # print(filename)
    df = neuron_info_df
    filename = int(filename.split('.')[0].split('_')[0])
    for i in range(len(df)):
        if int(df.iloc[i, 0]) == filename:
            return df.iloc[i, 43]


def find_v3draw_file(file_name, v3draw_dir="/PBshare/SEU-ALLEN/Projects/Human_Neurons/all_human_cells/all_human_cells_v3draw_8bit"):
    file_name = file_name.split(".")[0].split("_")[0]
    for root, dirs, files in os.walk(v3draw_dir):
        if("IHC" in root):
            continue
        for file in files:
            if file.endswith('.v3draw') and "_i" not in file and "_p" not in file:
                if(int(file_name)) == int(file.split("_")[0]):
                    return os.path.join(root, file)
    return None

def plot_compare_result(file_list, img_dir, seg_dirs, seg_labels, recon_dirs, recon_labels, info_file, index):
    plt.figure(figsize=(5 * 5, 5 * 5))
    info_df = pd.read_csv(info_file, encoding='gbk')

    for idx, img_file in enumerate(file_list):
        ID = int(img_file.split(".")[0].split("_")[0])
        brain_region = info_df[info_df['Cell ID'] == ID]['脑区'].values[0]
        label_info = f"No.{img_file.split('_')[0]} ({brain_region})"


        img_path = find_v3draw_file(img_file)
        img = load_image(img_path)[0]
        img = augment_gamma(img)
        img_mip = np.max(img, axis=0)


        seg_paths = [os.path.join(seg_dir, img_file) for seg_dir in seg_dirs]
        segs = [tifffile.imread(seg_path) for seg_path in seg_paths]
        segs = [(seg - np.min(seg)) / (np.max(seg) - np.min(seg)) * 255 for seg in segs]
        seg_mips = [np.max(seg, axis=0) for seg in segs]

        recon_paths = [os.path.join(recon_dir, img_file.replace(".tif", ".swc")) for recon_dir in recon_dirs]
        recon_mips = [get_mip_swc(recon_path, img, ignore_background=True) for recon_path in recon_paths]


        default_size = 256
        img_mip = cv2.resize(img_mip, (default_size, default_size))
        img_mip = cv2.cvtColor(img_mip.astype('uint8'), cv2.COLOR_GRAY2RGB)
        seg_mips = [cv2.resize(seg_mip, (default_size, default_size)) for seg_mip in seg_mips]
        seg_mips = [cv2.cvtColor(seg_mip.astype('uint8'), cv2.COLOR_GRAY2RGB) for seg_mip in seg_mips]
        recon_mips = [cv2.resize(recon_mip, (default_size, default_size)) for recon_mip in recon_mips]
        recon_mips = [cv2.rectangle(recon_mip, (0, 0), (recon_mip.shape[1], recon_mip.shape[0]), (0, 0, 0), 3) for
                      recon_mip in recon_mips] # 添加边框
        # recon_mips = [cv2.cvtColor(recon_mip.astype('uint8'), cv2.COLOR_GRAY2RGB) for recon_mip in recon_mips]

        labels = [label_info, seg_labels[0], recon_labels[0], seg_labels[1], recon_labels[1]]
        plot_mips = [img_mip, seg_mips[0], recon_mips[0], seg_mips[1], recon_mips[1]]

        for i, plot_mip in enumerate(plot_mips):
            # 横向
            ax = plt.subplot(5, 5, idx * 5 + i + 1)
            # 纵向
            # ax = plt.subplot(5, 5, i * 5 + idx + 1)
            ax.imshow(plot_mip)
            ax.axis('off')
            if(i == 0):
                ax.text(plot_mip.shape[1] / 2, 20, labels[i], fontsize=30, color='white', ha='center', va='top',
                        backgroundcolor='black')


    plt.tight_layout()
    plt.subplots_adjust(wspace=0.01, hspace=0.01)
    # plt.savefig(r"/data/kfchen/trace_ws/paper_auto_human_neuron_recon/seg/" + str(index) + ".png")
    plt.show()
    plt.close()


if __name__ == "__main__":
    # img_dir = r"/data/kfchen/trace_ws/to_gu/img"
    v3draw_dir = "/PBshare/SEU-ALLEN/Projects/Human_Neurons/all_human_cells/all_human_cells_v3draw_8bit"
    info_file = "/data/kfchen/nnUNet/nnUNet_results/Dataset169_hb_10k/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/ptls10/norm_result/Human_SingleCell_TrackingTable_20240712.csv"
    seg_dirs = [
        r"/data/kfchen/trace_ws/paper_auto_human_neuron_recon/seg/source500",
        r"/data/kfchen/trace_ws/paper_auto_human_neuron_recon/seg/ptls10",
    ]
    seg_labels = [
        "Baseline Segmentation",
        "Proposed Segmentation",
    ]
    recon_dirs = [
        r"/data/kfchen/trace_ws/paper_auto_human_neuron_recon/origin_recon/source500",
        r"/data/kfchen/trace_ws/paper_auto_human_neuron_recon/origin_recon/ptls10",
    ]
    recon_labels = [
        "Baseline Reconstruction",
        "Proposed Reconstruction",
    ]

    # tif
    img_files = os.listdir(seg_dirs[-1])
    img_files = [img_file for img_file in img_files if img_file.endswith(".tif")]
    img_files = random.sample(img_files, 5)
    img_files = [
        "08719_P053_T01_(1)_S029_-_TL.R_R0613_OMZ_20230915_LD.tif",
        "06052_P031_T02_(3)_S030_-_RTL_R0613_YS_20230522_YS.tif",
        "00935_P008_T01-S010_SPL_R0368_LJ-20220607_YXQ.tif",
        "00675_P005_T01-S010_MFG_R0368_LJ-20220525_XJ.tif",
        "04181_P029_T01_-S033_FL_R0460_RJ-20230303_RJ.tif"
    ]
    plot_compare_result(img_files, v3draw_dir, seg_dirs, seg_labels, recon_dirs, recon_labels, info_file, "chosen")

    # for index in range(int(len(img_files)/5)):
    #     plot_compare_result(img_files[index*5:index*5+5], img_dir, seg_dirs, seg_labels, info_file, index)



