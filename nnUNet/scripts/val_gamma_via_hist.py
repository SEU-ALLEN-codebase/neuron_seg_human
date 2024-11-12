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

def get_gamma_img(img):
    return exposure.adjust_gamma(img, gamma=0.7)

def get_equalize_img(img):
    return exposure.equalize_hist(img)

# 指定两个文件夹路径
folder1 = r'/data/kfchen/nnUNet/nnUNet_raw/Dataset169_hb_10k/imagesTr'
folder2 = r'/data/kfchen/nnUNet/nnUNet_raw/Dataset180_deflu_gamma/imagesTr'
label_dir = r"/data/kfchen/nnUNet/nnUNet_raw/Dataset180_deflu_gamma/labelsTr"
save_dir = r'/data/kfchen/nnUNet/nnUNet_raw/Dataset180_deflu_gamma'

if(not os.path.exists(os.path.join(save_dir, 'hist1_sum.npy'))):
    # 获取两个文件夹中的所有文件名
    files1 = set(f for f in os.listdir(folder1) if f.endswith('.tif'))
    files2 = set(f for f in os.listdir(folder2) if f.endswith('.tif'))

    # 找到同名的图像对
    common_files = files1.intersection(files2)
    # sort
    common_files = sorted(list(common_files))
    # common_files = common_files[:1]

    # 初始化累计直方图和频率信息
    hist1_sum = np.zeros(128)
    hist2_sum = np.zeros(128)
    cumulative_low_freq1 = 0
    cumulative_low_freq2 = 0
    cumulative_high_freq1 = 0
    cumulative_high_freq2 = 0

    # 初始化用于存储平均强度结果
    mean_intensities1 = []
    mean_intensities2 = []
    mi_hist_sum1 = np.zeros(100)
    mi_hist_sum2 = np.zeros(100)


    # 遍历同名的图像对并绘制MIP对照图和直方图
    for file_name in common_files:
        # print(file_name)
        # 读取图像
        image1 = io.imread(os.path.join(folder1, file_name))
        image2 = io.imread(os.path.join(folder2, file_name))

        image1 = image1.astype(np.float32)
        image2 = image2.astype(np.float32)

        image1 = (image1 - image1.min()) / (image1.max() - image1.min()) * 255
        image2 = (image2 - image2.min()) / (image2.max() - image2.min()) * 255

        # 计算全图的直方图
        hist1, _ = np.histogram(image1.ravel(), bins=128, range=(0, 256), density=True)
        hist2, _ = np.histogram(image2.ravel(), bins=128, range=(0, 256), density=True)

        # 累加直方图
        hist1_sum += hist1
        hist2_sum += hist2

        lab_file = io.imread(os.path.join(label_dir, file_name.replace("_0000.tif", ".tif")))
        soma_coord = quick_soma_from_seg(lab_file)
        # print(soma_coord)
        mean_intensity1 = calc_dt_from_soma(image1, soma_coord)
        mean_intensity2 = calc_dt_from_soma(image2, soma_coord)
        mi_hist_sum1 += mean_intensity1
        mi_hist_sum2 += mean_intensity2


        mean_intensities1.append(mean_intensity1)
        mean_intensities2.append(mean_intensity2)

        #
        # # 计算低灰度区域的累计频率（例如灰度值0-50）
        # low_intensity_threshold1 = int(255*0.2)
        # low_intensity_threshold2 = int(255*0.9)
        # cumulative_low_freq1 += np.sum(hist1[low_intensity_threshold1:low_intensity_threshold2])
        # cumulative_low_freq2 += np.sum(hist2[low_intensity_threshold1:low_intensity_threshold2])
        #
        # # 计算高灰度区域的累计频率（例如灰度值200-255）
        # high_intensity_threshold1 = int(0.9 * 255)
        # high_intensity_threshold2 = int(1.0 * 255)
        # cumulative_high_freq1 += np.sum(hist1[high_intensity_threshold1:high_intensity_threshold2])
        # cumulative_high_freq2 += np.sum(hist2[high_intensity_threshold1:high_intensity_threshold2])


    # save data
    np.save(os.path.join(save_dir, 'hist1_sum.npy'), hist1_sum)
    np.save(os.path.join(save_dir, 'hist2_sum.npy'), hist2_sum)
    np.save(os.path.join(save_dir, 'mi_hist_sum1.npy'), mi_hist_sum1)
    np.save(os.path.join(save_dir, 'mi_hist_sum2.npy'), mi_hist_sum2)
else:
    print("data 1 and 2 already exists")

if(not os.path.exists(os.path.join(save_dir, 'hist3_sum.npy'))):
    # 获取两个文件夹中的所有文件名
    files1 = set(f for f in os.listdir(folder1) if f.endswith('.tif'))
    files2 = set(f for f in os.listdir(folder2) if f.endswith('.tif'))

    # 找到同名的图像对
    common_files = files1.intersection(files2)
    # sort
    common_files = sorted(list(common_files))
    # common_files = common_files[:10]

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

        # print(file_name)
        # 读取图像
        image0 = io.imread(os.path.join(folder1, file_name))
        image0 = image0.astype(np.float32)
        image0 = (image0 - image0.min()) / (image0.max() - image0.min())

        image1 = get_gamma_img(image0)
        image2 = get_equalize_img(image0)

        image1 = (image1 - image1.min()) / (image1.max() - image1.min()) * 255
        image2 = (image2 - image2.min()) / (image2.max() - image2.min()) * 255

        # 计算全图的直方图
        hist1, _ = np.histogram(image1.ravel(), bins=128, range=(0, 256), density=True)
        hist2, _ = np.histogram(image2.ravel(), bins=128, range=(0, 256), density=True)

        # 累加直方图
        hist1_sum += hist1
        hist2_sum += hist2

        lab_file = io.imread(os.path.join(label_dir, file_name.replace("_0000.tif", ".tif")))
        soma_coord = quick_soma_from_seg(lab_file)
        # print(soma_coord)
        mean_intensity1 = calc_dt_from_soma(image1, soma_coord)
        mean_intensity2 = calc_dt_from_soma(image2, soma_coord)
        mi_hist_sum1 += mean_intensity1
        mi_hist_sum2 += mean_intensity2

        mean_intensities1.append(mean_intensity1)
        mean_intensities2.append(mean_intensity2)

    # 进度条
    pbar.close()
    # save data
    np.save(os.path.join(save_dir, 'hist3_sum.npy'), hist1_sum)
    np.save(os.path.join(save_dir, 'hist4_sum.npy'), hist2_sum)
    np.save(os.path.join(save_dir, 'mi_hist_sum3.npy'), mi_hist_sum1)
    np.save(os.path.join(save_dir, 'mi_hist_sum4.npy'), mi_hist_sum2)
else:
    print("data 3 and 4 already exists")



print("find data")
# load data
hist1_sum = np.load(os.path.join(save_dir, 'hist1_sum.npy'))
hist2_sum = np.load(os.path.join(save_dir, 'hist2_sum.npy'))
hist3_sum = np.load(os.path.join(save_dir, 'hist3_sum.npy'))
hist4_sum = np.load(os.path.join(save_dir, 'hist4_sum.npy'))
mi_hist_sum1 = np.load(os.path.join(save_dir, 'mi_hist_sum1.npy'))
mi_hist_sum2 = np.load(os.path.join(save_dir, 'mi_hist_sum2.npy'))
mi_hist_sum3 = np.load(os.path.join(save_dir, 'mi_hist_sum3.npy'))
mi_hist_sum4 = np.load(os.path.join(save_dir, 'mi_hist_sum4.npy'))

hist_sum_list = [hist1_sum, hist2_sum, hist3_sum, hist4_sum]
mi_hist_sum_list = [mi_hist_sum1, mi_hist_sum2, mi_hist_sum3, mi_hist_sum4]
color_list = ['black', 'red', 'blue', 'green']
label_list = ['origin', 'adaptive_gamma', 'global_gamma', 'equalize']

# 绘制两个文件夹的全图直方图和累计频率信息
bins = np.linspace(0, 1, 128)
sns.set(style="whitegrid")

# 绘制0-33%和66-100%最大强度的直方图
low_range = int(0.33 * 128)
high_range = int(0.33 * 128)

plt.figure(figsize=(9, 3))
plt.subplot(1, 3, 1)


# sns.histplot(x=bins, weights=hist1_sum, color='black', label='Folder 1', alpha=0, kde=True, bins=40)
# sns.histplot(x=bins, weights=hist2_sum, color='red', label='Folder 2', alpha=0, kde=True, bins=40)
for i in range(4):
    print(len(hist_sum_list[i]))
for i in range(4):
    sns.histplot(x=bins, weights=hist_sum_list[i], color=color_list[i], label=label_list[i], alpha=0, kde=True, bins=40)
plt.title('')
plt.xlabel('Normalized Intensity')
plt.ylabel('Normalized Frequency')
plt.ylim(0, 80)
plt.xlim(0, 0.33)
# plt.legend()


plt.subplot(1, 3, 2)
# sns.histplot(x=bins, weights=hist1_sum, color='black', label='Folder 1', alpha=0, kde=True, bins=40)
# sns.histplot(x=bins, weights=hist2_sum, color='red', label='Folder 2', alpha=0, kde=True, bins=40)
for i in range(4):
    sns.histplot(x=bins, weights=hist_sum_list[i], color=color_list[i], label=label_list[i], alpha=0, kde=True, bins=40)
plt.title('')
plt.xlabel('Normalized Intensity')
plt.ylabel('Normalized Frequency')
plt.ylim(0, 15)
plt.xlim(0.33, 1)

# plt.tight_layout()
# plt.show()

# plot mean intensity
plt.subplot(1, 3, 3)
# sns.lineplot(x=np.linspace(0, 1, 100), y=mi_hist_sum1/np.max(mi_hist_sum1), color='black', label='origin')
# sns.lineplot(x=np.linspace(0, 1, 100), y=mi_hist_sum2/np.max(mi_hist_sum2), color='red', label='gamma')
for i in range(4):
    sns.lineplot(x=np.linspace(0, 1, 100), y=mi_hist_sum_list[i]/np.max(mi_hist_sum_list[i]), color=color_list[i], label=label_list[i])
plt.title('')
plt.xlabel('Normolized Dist from Soma')
plt.ylabel('Normolized Mean Intensity')
# plt.legend()
plt.tight_layout()
plt.show()


# 显示累计频率信息
# print(f'Low-Intensity Cumulative (0-50):\nFolder 1: {cumulative_low_freq1:.4f}\nFolder 2: {cumulative_low_freq2:.4f}')
# print(f'High-Intensity Cumulative (200-255):\nFolder 1: {cumulative_high_freq1:.4f}\nFolder 2: {cumulative_high_freq2:.4f}')


