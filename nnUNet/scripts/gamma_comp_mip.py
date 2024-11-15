import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from scipy.spatial import distance
import seaborn as sns
from scipy.ndimage import distance_transform_edt
from skimage import exposure
from tqdm import tqdm

from scipy.stats import gaussian_kde

def quick_soma_from_seg(seg):
    # 进行距离变换
    distance_transformed = distance_transform_edt(seg)

    # 找到最大值及其所在的体素坐标
    # max_distance = np.max(distance_transformed)
    max_coord = np.unravel_index(np.argmax(distance_transformed), distance_transformed.shape)
    return max_coord

def calc_dt_from_soma(img, soma_coord):
    # 创建一个与图像相同大小的空白掩码，并在起始点设为1
    mask = np.zeros_like(img, dtype=bool)
    mask[soma_coord] = True

    # 计算从起始点到图像中每个点的欧几里得距离
    distance_map = distance_transform_edt(~mask)

    # 计算不同距离范围内的平均体素强度
    max_distance = int(np.ceil(distance_map.max()))
    mean_intensity = []

    for r in range(1, max_distance + 1):
        # 找到距离在 r - 1 和 r 之间的体素
        mask_r = (distance_map >= r - 1) & (distance_map < r)

        # 如果该距离范围内有体素，计算平均强度
        if np.any(mask_r):
            mean_intensity.append(np.mean(img[mask_r]))
        else:
            mean_intensity.append(0)

    # 将距离归一化到100个bins中
    normalized_bins = np.linspace(0, max_distance, 101)
    binned_intensity = []

    for i in range(1, len(normalized_bins)):
        bin_mask = (distance_map >= normalized_bins[i - 1]) & (distance_map < normalized_bins[i])
        if np.any(bin_mask):
            binned_intensity.append(np.mean(img[bin_mask]))
        else:
            binned_intensity.append(0)

    # print(len(binned_intensity))
    return binned_intensity

def get_gamma_img(img, gamma=0.7):
    img = exposure.adjust_gamma(img, gamma=gamma)
    img = exposure.equalize_adapthist(img, clip_limit=0.02, nbins=256)
    img = (img - img.min()) / (img.max() - img.min())
    return img

def get_equalize_img(img):
    img = exposure.equalize_adapthist(img, clip_limit=0.02, nbins=256)
    img = (img - img.min()) / (img.max() - img.min())
    return img

# 指定两个文件夹路径
folder1 = r'/data/kfchen/nnUNet/nnUNet_raw/Dataset175_14k_hb_neuron_aug_pure/imagesTr'
folder2 = r'/data/kfchen/nnUNet/nnUNet_raw/Dataset180_deflu_gamma/imagesTr'
label_dir = r"/data/kfchen/nnUNet/nnUNet_raw/Dataset180_deflu_gamma/labelsTr"
save_dir = r'/data/kfchen/nnUNet/nnUNet_raw/Dataset180_deflu_gamma'
mip_dir = r"/data/kfchen/nnUNet/nnUNet_raw/Dataset180_deflu_gamma/full_com_gamma_0.7/mip"

# 获取两个文件夹中的所有文件名
files1 = set(f for f in os.listdir(folder1) if f.endswith('.tif'))
files2 = set(f for f in os.listdir(folder2) if f.endswith('.tif'))

# 找到同名的图像对
common_files = files1.intersection(files2)
# sort
common_files = sorted(list(common_files))
# common_files = common_files[:3]

# 初始化累计直方图和频率信息
hist1_sum = np.zeros(128)
hist2_sum = np.zeros(128)

# 初始化用于存储平均强度结果
mean_intensities1 = []
mean_intensities2 = []
mi_hist_sum1 = np.zeros(100)
mi_hist_sum2 = np.zeros(100)

# 进度条
pbar = tqdm(total=len(common_files))
# 遍历同名的图像对并绘制MIP对照图和直方图
for file_name in common_files:
    # 进度条
    pbar.update(1)

    img_list = [
        io.imread(os.path.join(folder1, file_name)),
        io.imread(os.path.join(folder2, file_name))
    ]

    img_list = [f.astype("float32") for f in img_list]
    img_list = [(f - f.min()) / (f.max() - f.min()) for f in img_list]

    img_list.append(get_gamma_img(img_list[0], gamma=0.7))
    img_list.append(get_equalize_img(img_list[0]))

    img_list = [(f * 255).astype(np.uint8) for f in img_list]

    mip_list = [np.max(f, axis=0) for f in img_list]
    name_list = ["Raw", "Adaptive_Gamma", "Global_Gamma", "Equalization"]

    plt.figure(figsize=(16, 4))

    # 循环显示并保存每个MIP图像
    for i, mip in enumerate(mip_list):
        plt.subplot(1, len(mip_list), i + 1)
        plt.imshow(mip, cmap='gray')
        plt.title(name_list[i])
        plt.axis('off')

    plt.tight_layout()
    save_path = os.path.join(mip_dir, file_name.replace(".tif", ".jpg"))
    # 保存每张 MIP 图像
    plt.savefig(save_path)
    plt.close()

pbar.close()

# mip_dir = r"/data/kfchen/nnUNet/nnUNet_raw/Dataset180_deflu_gamma/full_com_gamma_0.5/mip"
# pbar = tqdm(total=len(common_files))
# for file_name in common_files:
#     # 进度条
#     pbar.update(1)
#
#     img_list = [
#         io.imread(os.path.join(folder1, file_name)),
#         io.imread(os.path.join(folder2, file_name))
#     ]
#
#     img_list = [f.astype("float32") for f in img_list]
#     img_list = [(f - f.min()) / (f.max() - f.min()) for f in img_list]
#
#     img_list.append(get_gamma_img(img_list[0], gamma=0.5))
#     img_list.append(get_equalize_img(img_list[0]))
#
#     img_list = [(f * 255).astype(np.uint8) for f in img_list]
#
#     mip_list = [np.max(f, axis=0) for f in img_list]
#     name_list = ["Raw", "Adaptive_Gamma", "Global_Gamma", "Equalization"]
#
#     plt.figure(figsize=(16, 4))
#
#     # 循环显示并保存每个MIP图像
#     for i, mip in enumerate(mip_list):
#         plt.subplot(1, len(mip_list), i + 1)
#         plt.imshow(mip, cmap='gray')
#         plt.title(name_list[i])
#         plt.axis('off')
#
#     plt.tight_layout()
#     save_path = os.path.join(mip_dir, file_name.replace(".tif", ".jpg"))
#     # 保存每张 MIP 图像
#     plt.savefig(save_path)
#     plt.close()
#
# pbar.close()
