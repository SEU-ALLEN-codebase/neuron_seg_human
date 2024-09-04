import os
import random
import tifffile
from nnUNet.nnunetv2.dataset_conversion.generate_nnunet_dataset import augment_gamma
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from matplotlib.patches import Rectangle

def find_soma(img_file, soma_dir):
    id = img_file.split(".")[0]
    soma_files = os.listdir(soma_dir)
    for soma_file in soma_files:
        if id in soma_file:
            if(int(soma_file.split("_")[0]) == int(id)):
                return soma_file

img_dir = r"/data/kfchen/trace_ws/to_gu/img"
soma_dir = r"/data/kfchen/trace_ws/14k_seg_result/soma"
seg_dir = r"/data/kfchen/trace_ws/14k_seg_result/tif"
info_file = "/data/kfchen/nnUNet/nnUNet_results/Dataset169_hb_10k/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/ptls10/norm_result/Human_SingleCell_TrackingTable_20240712.csv"

# tif
img_files = os.listdir(img_dir)
img_files = [img_file for img_file in img_files if img_file.endswith(".tif")]
# random choose 1000 images
img_files = random.sample(img_files, 5)
plt.figure(figsize=(5*5, 3*5))

for idx, img_file in enumerate(img_files):
    img_path = os.path.join(img_dir, img_file)
    soma_file = find_soma(img_file, soma_dir)
    soma_path = os.path.join(soma_dir, soma_file)
    seg_path = os.path.join(seg_dir, soma_file)
    img = tifffile.imread(img_path)
    img = augment_gamma(img)
    img_mip = np.max(img, axis=0)

    soma = tifffile.imread(soma_path)
    seg = tifffile.imread(seg_path)


    seg_mip = np.max(seg, axis=0)
    soma_mip = np.max(soma, axis=0)
    info_df = pd.read_csv(info_file, encoding='gbk')
    ID = int(img_file.split(".")[0])
    brain_region = info_df[info_df['Cell ID'] == ID]['脑区'].values[0]
    label_info = f"No.0{img_file.split('.')[0]} ({brain_region})"


    default_size = 256
    img_mip = cv2.resize(img_mip, (default_size, default_size))
    seg_mip = cv2.resize(seg_mip, (default_size, default_size))
    soma_mip = cv2.resize(soma_mip, (default_size, default_size))

    soma_mask = soma_mip > 0
    # print(img_mip.shape)
    img_mip = cv2.cvtColor(img_mip.astype('uint8'), cv2.COLOR_GRAY2RGB)
    seg_mip = cv2.cvtColor(seg_mip.astype('uint8'), cv2.COLOR_GRAY2RGB)
    soma_mip = cv2.cvtColor(soma_mip.astype('uint8'), cv2.COLOR_GRAY2RGB)
    soma_mask_color = np.zeros_like(soma_mip)

    soma_mask_color[soma_mask == True] = [255, 128, 0]


    ax1 = plt.subplot(3, 5, idx + 1)
    ax1.imshow(img_mip)
    ax1.imshow(np.ma.masked_where(soma_mask_color==[255, 128, 0], soma_mask_color), alpha=0.5)
    ax1.axis('off')
    ax1.text(img_mip.shape[1] / 2, 20, label_info, fontsize=30, color='white', ha='center', va='top',
             backgroundcolor='black')

    ax2 = plt.subplot(3, 5, idx + 1 + 5)
    ax2.imshow(seg_mip)
    ax2.imshow(np.ma.masked_where(soma_mask_color==[255, 128, 0], soma_mask_color), alpha=0.5)
    # plt.title("GT Annotation")
    ax2.axis('off')
    # ax2.add_patch(Rectangle((0, 0), seg_mip.shape[1] - 1, seg_mip.shape[0] - 1, edgecolor='black', facecolor='none',
    #                 linewidth=1))
    ax2.text(10, 20, "Neuron Segment", fontsize=20, color='white')

    ax3 = plt.subplot(3, 5, idx + 1 + 2 * 5)
    ax3.imshow(soma_mip, cmap='gray')
    # plt.title("Automated Annotation")
    ax3.axis('off')
    # ax3.add_patch(Rectangle((0, 0), soma_mip.shape[1] - 1, soma_mip.shape[0] - 1, edgecolor='black', facecolor='none',
    #                         linewidth=1))
    ax3.text(10, 20, "Soma", fontsize=20, color='white')

plt.tight_layout()
plt.savefig(r"/data/kfchen/trace_ws/14k_seg_result/compare_soma.png")

