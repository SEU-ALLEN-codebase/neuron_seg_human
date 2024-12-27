import os

import matplotlib.pyplot as plt
import numpy as np
from skimage import io
import cv2
from skimage.transform import resize
import pandas as pd
import torchio as tio
from monai.transforms import Resize, Compose, Pad, CenterSpatialCrop
plt.rcParams['figure.dpi'] = 800
name_mapping_df = pd.read_csv(os.path.join("/data/kfchen/nnUNet/nnUNet_raw/Dataset180_deflu_gamma", "name_mapping.csv"))
neuron_info_df = pd.read_csv("/data/kfchen/nnUNet/nnUNet_results/Dataset169_hb_10k/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/ptls10/norm_result/Human_SingleCell_TrackingTable_20240712.csv", encoding='gbk')


def get_img_with_mask(img, mask, alpha = 0.5):
    # print(type(img))
    image_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    mask_rgb = np.zeros_like(image_rgb, dtype=np.uint8)
    mask_rgb[mask == 255] = [0, 255, 0]  # 将 mask 区域设为绿色，可以换成其他颜色
    mask_rgb = mask_rgb.astype(np.float32) / 255.0  # 转换为 0-1 范围

    mask_stack = np.stack([mask, mask, mask], axis=2)
    # 将 soma mask 和 segment 进行加权叠加
    output = image_rgb * (mask_stack<1) + ((1 - alpha) * image_rgb + alpha * mask_rgb * 255)*(mask_stack>=1)
    output = np.clip(output, 0, 255).astype(np.uint8)
    return output

def croporpad(img, soma, target_fig_size):
    if(img.shape[0] > target_fig_size[0]):
        img = img[img.shape[0]//2 - target_fig_size[0]//2:img.shape[0]//2 + target_fig_size[0]//2, :]
        soma = soma[soma.shape[0]//2 - target_fig_size[0]//2:soma.shape[0]//2 + target_fig_size[0]//2, :]
    else:
        padding = (target_fig_size[0] - img.shape[0]) // 2
        img = np.pad(img, ((padding, target_fig_size[0] - img.shape[0] - padding), (0, 0)), mode='constant', constant_values=0)
        soma = np.pad(soma, ((padding, target_fig_size[0] - soma.shape[0] - padding), (0, 0)), mode='constant', constant_values=0)


    if(img.shape[1] > target_fig_size[1]):
        img = img[:, img.shape[1]//2 - target_fig_size[1]//2:img.shape[1]//2 + target_fig_size[1]//2]
        soma = soma[:, soma.shape[1]//2 - target_fig_size[1]//2:soma.shape[1]//2 + target_fig_size[1]//2]
    else:
        padding = (target_fig_size[1] - img.shape[1]) // 2
        img = np.pad(img, (padding, target_fig_size[1] - img.shape[1] - padding), mode='constant', constant_values=0)
        soma = np.pad(soma, (padding, target_fig_size[1] - soma.shape[1] - padding), mode='constant', constant_values=0)


    # print(img.shape, soma.shape)

    return np.array(img), np.array(soma)

seg_dir = "/data/kfchen/trace_ws/paper_trace_result/nnunet/newcel_0.1/0_seg"
soma_dir = "/data/kfchen/trace_ws/paper_trace_result/nnunet/newcel_0.1/2_soma_region"
mip_dir = "/data/kfchen/trace_ws/paper_trace_result/nnunet/newcel_0.1/soma_mip"
img_dir = "/data/kfchen/trace_ws/to_gu/img"

seg_files = [f for f in os.listdir(seg_dir) if f.endswith(".tif")]

for seg_file in seg_files:
    id = int(seg_file.split("_")[0].split(".")[0])
    xy_resolution = neuron_info_df.loc[neuron_info_df.iloc[:, 0] == id, 'xy拍摄分辨率(*10e-3μm/px)'].values[0]

    soma = io.imread(os.path.join(soma_dir, seg_file))
    soma = np.flip(soma, axis=1)
    img = io.imread(os.path.join(img_dir, seg_file))
    seg = io.imread(os.path.join(seg_dir, seg_file))

    origin_shape = (soma.shape[0], soma.shape[1] * xy_resolution / 1000, soma.shape[2] * xy_resolution / 1000)

    soma = resize(soma, origin_shape, order=0, anti_aliasing=False, preserve_range=True).astype(np.uint8)
    seg = resize(seg, origin_shape, order=0, anti_aliasing=False, preserve_range=True).astype(np.uint8)
    img = resize(img, origin_shape, order=2, anti_aliasing=False, preserve_range=True).astype(np.uint8)


    background_list, mask_list = [], []
    target_fig_size = (200, 200)

    for axis in range(3):
        current_bkg, current_mask = np.max(img, axis=axis), np.max(soma, axis=axis)
        current_bkg, current_mask = croporpad(current_bkg, current_mask, target_fig_size)
        background_list.append(current_bkg)
        mask_list.append(current_mask)

        current_bkg, current_mask = np.max(seg, axis=axis), np.max(soma, axis=axis)
        current_bkg, current_mask = croporpad(current_bkg, current_mask, target_fig_size)
        background_list.append(current_bkg)
        mask_list.append(current_mask)

    # mask_list = [np.max(soma) for i in range(len(background_list))]

    output = [get_img_with_mask(currnt_img, currnt_mask) for currnt_img, currnt_mask in zip(background_list, mask_list)]



    fig, ax = plt.subplots(3, 2, figsize=(8, 12))
    ax = ax.flatten()
    for i in range(len(output)):
        ax[i].imshow(output[i], cmap="gray")
        ax[i].axis("off")

    plt.tight_layout(w_pad=1, h_pad=1)
    plt.savefig(os.path.join(mip_dir, seg_file.replace(".tif", ".png")))
    plt.close()

