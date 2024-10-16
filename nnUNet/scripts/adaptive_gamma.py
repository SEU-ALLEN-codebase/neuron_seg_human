import os

import matplotlib.pyplot as plt

from simple_swc_tool.soma_detection import simple_get_soma
import tifffile
import numpy as np
import cv2
from concurrent.futures import ThreadPoolExecutor
from skimage.measure import block_reduce
import pandas as pd
from skimage import img_as_float
from scipy.ndimage import gaussian_filter
from skimage import exposure
from skimage.filters import laplace


def find_resolution(df, filename):
    # print(filename)
    filename = str(int(filename.split('.')[0].split('_')[0]))
    row = df[df.iloc[:, 0] == filename]
    return row.iloc[0, 43]

# def adaptive_augment_gamma(data, center, spacing=(1, 1, 1)):
#     min_data = np.min(data)
#     data_range = np.max(data) - min_data
#     data = (data - min_data) / data_range
#
#     max_dis = np.sqrt((data.shape[0] * spacing[0]) ** 2 + (data.shape[1] * spacing[1]) ** 2 + (data.shape[2] * spacing[2]) ** 2)
#
#     for z in range(data.shape[0]):
#         for y in range(data.shape[1]):
#             for x in range(data.shape[2]):
#                 dis_to_center = ((center[0] - z) * spacing[0]) ** 2 + ((center[1] - y) * spacing[1]) ** 2 + ((center[2] - x) * spacing[2]) ** 2
#                 dis_to_center = np.sqrt(dis_to_center) / max_dis
#                 power_val = 1 - dis_to_center
#                 data[z, y, x] = data[z, y, x] ** power_val
#
#     data = data * data_range + min_data
#     return data

def adaptive_augment_gamma(data, center, spacing=(1, 1, 1)):
    # Normalize data
    min_data = np.min(data)
    max_data = np.max(data)
    data_normalized = (data - min_data) / (max_data - min_data)

    # Prepare grid coordinates
    z, y, x = np.ogrid[:data.shape[0], :data.shape[1], :data.shape[2]]
    distances = np.sqrt(((z - center[0]) * spacing[0]) ** 2 +
                        ((y - center[1]) * spacing[1]) ** 2 +
                        ((x - center[2]) * spacing[2]) ** 2)

    # Normalize distances
    max_dis = np.max(distances)
    distances_normalized = distances / max_dis

    # Compute gamma power values
    b = 0.25
    a = 1-b
    power_a = 4
    power_values = a*(1-distances_normalized) ** power_a + b
    # power_values = (1 - distances_normalized * 0.75)

    # max_power = 0.5
    # power_values = np.clip(power_values, 0, max_power)

    min_power = 0.5
    power_values = np.clip(power_values, min_power, 1)

    # Apply gamma correction
    data_corrected = data_normalized ** power_values

    data = data_corrected

    # Scale data back to original range
    data = (data - np.min(data)) / (np.max(data) - np.min(data)) * 255

    return data.astype("uint8")



def de_fluorophore_gamma_augment(img):
    def norm2(img):
        img = (img - np.min(img)) / (np.max(img) - np.min(img))
        return img

    def sharpen_3d_image(image):
        # 将图像转换为浮点类型，以避免数据类型问题
        image = img_as_float(image)

        # 应用高斯滤波进行平滑处理
        blurred = gaussian_filter(image, sigma=1)

        # 计算拉普拉斯滤波器的响应
        laplacian = laplace(blurred)
        laplacian = norm2(laplacian)
        sharpened_image = laplacian
        sharpened_image = exposure.adjust_gamma(sharpened_image, gamma=0.5)

        return sharpened_image

    img = img_as_float(img)
    img = (img - np.min(img)) / (np.max(img) - np.min(img))

    sharpen_img = sharpen_3d_image(img)

    equal_img = exposure.equalize_adapthist(img, clip_limit=0.02, nbins=256)
    fluorophore = gaussian_filter(equal_img, sigma=5)
    de_flu_img = (equal_img - fluorophore)
    de_flu_img = norm2(de_flu_img)
    de_flu_img = exposure.adjust_gamma(de_flu_img, gamma=0.5)

    add_img = sharpen_img + de_flu_img
    add_img = norm2(add_img)
    add_img = exposure.adjust_gamma(add_img, gamma=0.5)

    return add_img

def de_fluorophore_test():
    img_dir = "/data/kfchen/nnUNet/nnUNet_raw/Dataset177_14k_de_fluorophore_gamma/imagesTr"
    mask_dir = "/data/kfchen/nnUNet/nnUNet_raw/Dataset177_14k_de_fluorophore_gamma/labelsTr"
    mip_dir = "/data/kfchen/trace_ws/de_f_test"

    img_files = [f for f in os.listdir(img_dir) if f.endswith(".tif")]
    img_files.sort()
    img_files = img_files[:10]

    for img_file in img_files:
        img = tifffile.imread(os.path.join(img_dir, img_file))
        mask = tifffile.imread(os.path.join(mask_dir, img_file.replace("_0000.tif", ".tif")))

        img_mip = np.max(img, axis=0)
        mask_mip = np.max(mask, axis=0)

        plt.figure()
        plt.subplot(121)
        plt.imshow(img_mip, cmap='gray')
        plt.subplot(122)
        plt.imshow(mask_mip, cmap='gray')

        plt.savefig(os.path.join(mip_dir, img_file.replace(".tif", ".png"))
                    , dpi=300)

def down_sample(img, factor=2):
    img = block_reduce(img, block_size=(factor, factor, factor), func=np.max)
    return img

def visulize_aug(img_file, img_dir, seg_dir, mip_dir, spacing):
    img = tifffile.imread(os.path.join(img_dir, img_file))
    seg = tifffile.imread(os.path.join(seg_dir, img_file))
    mip_file = os.path.join(mip_dir, img_file.replace(".tif", ".png"))

    img = ((img - np.min(img)) / (np.max(img) - np.min(img)) * 255).astype('uint8')

    soma_pos = simple_get_soma(seg, os.path.join(seg_dir, img_file)) # zyx

    img = down_sample(img)
    soma_pos = np.array(soma_pos) / 2

    aug_img = adaptive_augment_gamma(img, soma_pos, spacing)

    img_mip = np.max(img, axis=0)
    aug_img_mip = np.max(aug_img, axis=0)
    # cv2 concat
    img_seg = np.concatenate((img_mip, aug_img_mip), axis=1).astype('uint8')
    tifffile.imwrite(mip_file, img_seg)
    print(f"{img_file} done")


def visulize_soma(seg_file, seg_dir, images_dir, mip_dir):
    seg = tifffile.imread(os.path.join(seg_dir, seg_file))
    soma_pos = simple_get_soma(seg, os.path.join(seg_dir, seg_file)) # zyx
    print(seg_file, soma_pos)

    img = tifffile.imread(os.path.join(images_dir, seg_file))

    img_mip = np.max(img, axis=0)
    seg_mip = np.max(seg, axis=0)

    img_mip = cv2.cvtColor(img_mip, cv2.COLOR_GRAY2BGR)
    seg_mip = cv2.cvtColor(seg_mip, cv2.COLOR_GRAY2BGR)

    cv2.circle(img_mip, (round(soma_pos[2]), round(soma_pos[1])), 10, (0, 0, 255), -1)
    cv2.circle(seg_mip, (round(soma_pos[2]), round(soma_pos[1])), 10, (0, 0, 255), -1)

    # cv2 concat
    img_seg_mip = np.concatenate((img_mip, seg_mip), axis=1)

    # save
    cv2.imwrite(os.path.join(mip_dir, seg_file.replace(".tif", ".png")), img_seg_mip)

if __name__ == "__main__":
    de_fluorophore_test()
    exit()


    images_dir = "/PBshare/SEU-ALLEN/Users/KaifengChen/human_brain/img/raw"
    seg_dir = "/PBshare/SEU-ALLEN/Users/KaifengChen/human_brain/img/mask"
    raw_info_path = "/data/kfchen/nnUNet/nnUNet_results/Dataset169_hb_10k/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/ptls10/norm_result/Human_SingleCell_TrackingTable_20240712.csv"
    # mip_dir = "/PBshare/SEU-ALLEN/Users/KaifengChen/human_brain/img/mip"
    mip_dir = "/PBshare/SEU-ALLEN/Users/KaifengChen/human_brain/img/mip_aug"

    seg_files = [f for f in os.listdir(seg_dir) if f.endswith(".tif")]
    seg_files.sort()

    # for seg_file in seg_files:
    #     visulize_soma(seg_file, seg_dir, images_dir, mip_dir)
    # 多线程

    # with ThreadPoolExecutor(max_workers=16) as executor:
    #     executor.map(visulize_soma, seg_files, [seg_dir]*len(seg_files), [images_dir]*len(seg_files), [mip_dir]*len(seg_files))

    img_files = [f for f in os.listdir(images_dir) if f.endswith(".tif")]
    img_files.sort()
    info_df = pd.read_csv(raw_info_path, header=None, encoding='gbk')
    spacing_list = []
    for img_file in img_files:
        spacing = find_resolution(info_df, img_file)
        spacing = (1, float(spacing)/1000, float(spacing)/1000)
        spacing_list.append(spacing)
    print("get spacing list done")
    with ThreadPoolExecutor(max_workers=16) as executor:
        executor.map(visulize_aug, img_files, [images_dir]*len(img_files), [seg_dir]*len(img_files), [mip_dir]*len(img_files), spacing_list)
