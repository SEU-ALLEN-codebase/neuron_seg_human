import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2

brain_region_map = {
    "superior frontal gyrus": ["SFG.R", "SFG.L", "SFG", "S(M)FG.R", "S(M)FG.R", "M(I)FG.L"],
    "middle frontal gyrus": ["MFG.R", "MFG", "MFG.L"],
    "inferior frontal gyrus": ["IFG.R", "(X)FG", "IFG"],

    "superior temporal gyrus": ["STG", "STG.R", "S(M)TG.R", "S(M)TG.L", "STG-AP", "S(M,I)TG"],
    "middle temporal gyrus": ["MTG.R", "MTG.L", "MTG"],

    "parietal lobe": ["PL.L", "PL.L_OL.L", "PL"],
    "inferior parietal lobe": ["IPL-near-AG", "IPL.L"],

    "temporal pole": ["TP", "TP.L", "TP.R"],

    'others': ['CB_tonsil.L', 'FP.R', 'FP.L', 'BN.L', 'FT.L', 'CC.L'],
    'occipital lobe': ['OL.L', 'OL.R'],
    'temporal Lobe': ['TL.L', 'TL.R'],
    'frontal lobe': ['FL.L', 'FL.R', 'FL_TL.L'],
    'posterior lateral ventricle': ['pLV.L'],
}

cell_ids = {
    '1574': 'superior frontal gyrus',
    '1617': 'middle frontal gyrus',
    '2647': 'inferior frontal gyrus',

    '534': 'superior temporal gyrus',
    '6076': 'middle temporal gyrus',

    '709': 'parietal lobe',
    '68': 'inferior parietal lobe',

    '830': 'temporal pole',

    '1312': 'occipital lobe',
    '2407': 'temporal Lobe',
    '2412': 'frontal lobe',
}

def augment_gamma(data_sample, gamma_range=(0.5, 2), epsilon=1e-7, per_channel=False,
                  retain_stats=False, p=1):
    """Function directly copied from batchgenerators"""
    if(np.random.random() > p):
        return data_sample
    # gamma = np.random.uniform(gamma_range[0], 1)
    gamma = 0.5
    if not per_channel:
        if retain_stats:
            mn = data_sample.mean()
            sd = data_sample.std()
        minm = data_sample.min()
        rnge = data_sample.max() - minm
        data_sample = np.power(((data_sample - minm) / float(rnge + epsilon)), gamma) * rnge + minm
        if retain_stats:
            data_sample = data_sample - data_sample.mean()
            data_sample = data_sample / (data_sample.std() + 1e-8) * sd
            data_sample = data_sample + mn
    return data_sample

def enhance_dark_areas(image, clip_limit=2.0, tile_grid_size=(8, 8), gamma=1.5):
    """
    Enhance the dark areas of an image using CLAHE and Gamma correction.

    Parameters:
        image_path (str): Path to the input image.
        clip_limit (float): Threshold for contrast limiting in CLAHE.
        tile_grid_size (tuple): Size of the grid for the CLAHE algorithm.
        gamma (float): Gamma value for gamma correction.

    Returns:
        enhanced_image (numpy.ndarray): The enhanced image.
    """

    # 转换到LAB色彩空间
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    # 分离L通道
    l_channel, a_channel, b_channel = cv2.split(lab)

    # 应用CLAHE到L通道
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_channel = clahe.apply(l_channel)

    # 将处理后的L通道和原始的A和B通道合并
    lab = cv2.merge((l_channel, a_channel, b_channel))

    # 转换回BGR色彩空间
    enhanced_image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # Gamma校正
    gamma_correction = np.array(255 * (enhanced_image / 255) ** gamma, dtype='uint8')

    gamma_correction = cv2.convertScaleAbs(gamma_correction, alpha=2, beta=0)

    return gamma_correction


def reduce_contrast(image, alpha=0.5):
    """
    Reduce the contrast of an image.

    Parameters:
        image_path (str): Path to the input image.
        alpha (float): Factor by which to reduce the contrast (0.0 to 1.0).
                       0.0 will result in a completely gray image, and 1.0 will
                       leave the image unchanged.

    Returns:
        contrast_reduced_image (numpy.ndarray): The image with reduced contrast.
    """

    # 创建一个全为128的灰色图像（中间灰度值）
    gray_image = np.full_like(image, 128)

    # 线性插值降低对比度
    contrast_reduced_image = cv2.addWeighted(image, alpha, gray_image, 1 - alpha, 0)

    # print(np.min(contrast_reduced_image), np.max(contrast_reduced_image))
    contrast_reduced_image = (contrast_reduced_image - np.min(contrast_reduced_image)) / (
            np.max(contrast_reduced_image) - np.min(contrast_reduced_image)) * 255
    contrast_reduced_image = contrast_reduced_image.astype(np.uint8)

    # print(np.min(contrast_reduced_image), np.max(contrast_reduced_image))

    return contrast_reduced_image


def find_neuron_img(cell_id, root_dir):
    # walk
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if cell_id in file and '.v3draw' in file and 'p' not in file and 'i' not in file:
                if(int(cell_id) == int(file.split('_')[0])):
                    return os.path.join(root, file)
    return None

def find_neuron_annotation(cell_id, root_dir):
    # walk
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if cell_id in file and '.swc' in file:
                if (int(cell_id) == int(file.split('_')[0])):
                    return os.path.join(root, file)
    return None

from nnUNet.scripts.mip import get_mip_swc, get_mip
from pylib.file_io import load_image

def get_mip_swc_path(cell_id, img_root_dir, swc_root_dir):
    img_path = find_neuron_img(cell_id, img_root_dir)
    swc_path = find_neuron_annotation(cell_id, swc_root_dir)
    # print(img_path, swc_path)
    if img_path is None or swc_path is None:
        return None
    img = load_image(img_path)[0]
    return get_mip(img), get_mip_swc(swc_path, img, ignore_background=True)

if __name__ == '__main__':
    img_root_dir = '/PBshare/SEU-ALLEN/Projects/Human_Neurons/all_human_cells/all_human_cells_v3draw_8bit'
    swc_root_dir = '/PBshare/SEU-ALLEN/Projects/Human_Neurons/Different_versions_human_dendrite_reconstructed/humanNeuron_auto/humanNeuron_10847_auto_reconstructed/10847_auto_v1.4/swc'
    mip_dir = '/data/kfchen/nnUNet/nnUNet_results/Dataset169_hb_10k/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/ptls10/example_mips'

    for cell_id in cell_ids.keys():
        mip_img, mip_swc = get_mip_swc_path(cell_id, img_root_dir, swc_root_dir)
        if mip_img is None or mip_swc is None:
            continue
        # 三通道
        mip_img = np.stack([mip_img, mip_img, mip_img], axis=-1)
        # mip_img = enhance_dark_areas(mip_img)
        mip_img = augment_gamma(mip_img)
        mip_img = reduce_contrast(mip_img)

        concat_img = np.concatenate([mip_img,  mip_swc], axis=1)
        # print(mip_img.shape, mip_swc.shape, concat_img.shape)
        save_path = os.path.join(mip_dir, f'{cell_id}.png')
        plt.imsave(save_path, concat_img, cmap='gray')
        print(f'save to {save_path}')


