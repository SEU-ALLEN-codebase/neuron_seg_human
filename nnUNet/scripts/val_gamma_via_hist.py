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
from skimage import io
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

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
    img = exposure.adjust_gamma(img, gamma=0.7)
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
save_dir = r'/data/kfchen/nnUNet/nnUNet_raw/Dataset180_deflu_gamma/full_com_gamma_0.7'
# mip_dir = r"/data/kfchen/nnUNet/nnUNet_raw/Dataset180_deflu_gamma/full_com_gamma_0.7/mip"

# 用于更新进度条的回调函数
def update_pbar(pbar, value):
    pbar.update(value)


# 处理每个文件的函数
def process_file(file_name, label_dir, folder1, folder2):
    lab_file = io.imread(os.path.join(label_dir, file_name.replace("_0000.tif", ".tif")))
    soma_coord = quick_soma_from_seg(lab_file)

    img_list = [
        io.imread(os.path.join(folder1, file_name)),
        io.imread(os.path.join(folder2, file_name))
    ]

    img_list = [f.astype("float32") for f in img_list]
    img_list = [(f - f.min()) / (f.max() - f.min()) for f in img_list]

    img_list.append(get_gamma_img(img_list[0]))
    img_list.append(get_equalize_img(img_list[0]))

    img_list = [(f * 255).astype(np.uint8) for f in img_list]

    # 更新直方图和平均强度值
    local_hist_sum = [np.zeros(128) for _ in range(4)]
    local_mi_hist_sum = [np.zeros(100) for _ in range(4)]

    for i, img in enumerate(img_list):
        current_hist, _ = np.histogram(img.ravel(), bins=128, range=(0, 256), density=True)
        local_hist_sum[i] += current_hist

        current_mean_intensity = calc_dt_from_soma(img, soma_coord)
        local_mi_hist_sum[i] += current_mean_intensity

    return local_hist_sum, local_mi_hist_sum


# 主执行函数
def process_files_in_parallel(common_files, label_dir, folder1, folder2):
    pbar = tqdm(total=len(common_files))

    # 线程池执行任务
    with ThreadPoolExecutor() as executor:
        futures = {}
        for file_name in common_files:
            futures[executor.submit(process_file, file_name, label_dir, folder1, folder2)] = file_name

        # 等待所有线程完成
        for future in as_completed(futures):
            local_hist_sum, local_mi_hist_sum = future.result()

            # 更新全局变量
            for i in range(4):
                hist_sum_list[i] += local_hist_sum[i]
                mi_hist_sum_list[i] += local_mi_hist_sum[i]

            # 更新进度条
            update_pbar(pbar, 1)

    pbar.close()


if(not os.path.exists(os.path.join(save_dir, 'hist_sum1.npy'))):
    # 获取两个文件夹中的所有文件名
    files1 = set(f for f in os.listdir(folder1) if f.endswith('.tif'))
    files2 = set(f for f in os.listdir(folder2) if f.endswith('.tif'))

    # 找到同名的图像对
    common_files = files1.intersection(files2)
    # sort
    common_files = sorted(list(common_files))
    # common_files = common_files[:20]
    hist_sum_list = [np.zeros(128) for i in range(4)]
    mi_hist_sum_list = [np.zeros(100) for i in range(4)]
    #
    # pbar = tqdm(total=len(common_files))
    # # 遍历同名的图像对并绘制MIP对照图和直方图
    # for file_name in common_files:
    #     pbar.update(1)
    #     lab_file = io.imread(os.path.join(label_dir, file_name.replace("_0000.tif", ".tif")))
    #     soma_coord = quick_soma_from_seg(lab_file)
    #
    #     img_list = [
    #         io.imread(os.path.join(folder1, file_name)),
    #         io.imread(os.path.join(folder2, file_name))
    #     ]
    #
    #     img_list = [f.astype("float32") for f in img_list]
    #     img_list = [(f - f.min()) / (f.max() - f.min()) for f in img_list]
    #
    #     img_list.append(get_gamma_img(img_list[0]))
    #     img_list.append(get_equalize_img(img_list[0]))
    #
    #     img_list = [(f * 255).astype(np.uint8) for f in img_list]
    #
    #     for i, img in enumerate(img_list):
    #         current_hist, _ = np.histogram(img.ravel(), bins=128, range=(0, 256), density=True)
    #         hist_sum_list[i] += current_hist
    #
    #         current_mean_intensity = calc_dt_from_soma(img, soma_coord)
    #         mi_hist_sum_list[i] += current_mean_intensity
    # pbar.close()
    # save data

    pbar = tqdm(total=len(common_files))

    # 线程池执行任务
    with ThreadPoolExecutor() as executor:
        futures = {}
        for file_name in common_files:
            futures[executor.submit(process_file, file_name, label_dir, folder1, folder2)] = file_name

        # 等待所有线程完成
        for future in as_completed(futures):
            local_hist_sum, local_mi_hist_sum = future.result()

            # 更新全局变量
            for i in range(4):
                hist_sum_list[i] += local_hist_sum[i]
                mi_hist_sum_list[i] += local_mi_hist_sum[i]

            # 更新进度条
            update_pbar(pbar, 1)

    pbar.close()

    for i in range(4):
        np.save(os.path.join(save_dir, 'hist_sum'  + str(i) + '.npy'), hist_sum_list[i])
        np.save(os.path.join(save_dir, 'mi_hist_sum' + str(i) + '.npy'), mi_hist_sum_list[i])
else:
    print("data 1 and 2 already exists")


print("find data")

hist_sum_list, mi_hist_sum_list = [], []
for i in range(4):
    hist_sum_list.append(np.load(os.path.join(save_dir, 'hist_sum'  + str(i) + '.npy')))
    mi_hist_sum_list.append(np.load(os.path.join(save_dir, 'mi_hist_sum' + str(i) + '.npy')))

color_list = ['black', 'red', 'blue', 'green']
label_list = ['Raw', 'Adaptive_gamma', 'Global_gamma', 'Equalization']

# 绘制两个文件夹的全图直方图和累计频率信息
bins = np.linspace(0, 1, 128)
sns.set(style="whitegrid")

# 绘制0-33%和66-100%最大强度的直方图
low_range = int(0.33 * 128)
high_range = int(0.33 * 128)
plt.figure(figsize=(9, 3))
plt.subplot(1, 2, 1)


# sns.histplot(x=bins, weights=hist1_sum, color='black', label='Folder 1', alpha=0, kde=True, bins=40)
# sns.histplot(x=bins, weights=hist2_sum, color='red', label='Folder 2', alpha=0, kde=True, bins=40)
for i in range(4):
    print(len(hist_sum_list[i]))

middle_range = (0.2,  0.8) # 550
for i in range(4):
    # print middle range
    print(f'{label_list[i]}: {np.sum(hist_sum_list[i][int(middle_range[0]*128):int(middle_range[1]*128)]):.4f}')
# 0.01426
for i in range(4):
    sns.histplot(x=bins, weights=hist_sum_list[i]/1.28, color=color_list[i], label=label_list[i], alpha=0.2, kde=True, bins=40)
plt.title('')
plt.xlabel('Normalized Intensity')
plt.ylabel('Normalized Frequency')
plt.ylim(0, 90)
# plt.xlim(0, 0.33)
# plt.legend()


# plt.subplot(1, 3, 2)
# # sns.histplot(x=bins, weights=hist1_sum, color='black', label='Folder 1', alpha=0, kde=True, bins=40)
# # sns.histplot(x=bins, weights=hist2_sum, color='red', label='Folder 2', alpha=0, kde=True, bins=40)
# for i in range(4):
#     sns.histplot(x=bins, weights=hist_sum_list[i], color=color_list[i], label=label_list[i], alpha=0, kde=True, bins=40)
# plt.title('')
# plt.xlabel('Normalized Intensity')
# plt.ylabel('Normalized Frequency')
# plt.ylim(0, 15)
# plt.xlim(0.33, 1)

# plt.tight_layout()
# plt.show()

# plot mean intensity
plt.subplot(1, 2, 2)
# sns.lineplot(x=np.linspace(0, 1, 100), y=mi_hist_sum1/np.max(mi_hist_sum1), color='black', label='origin')
# sns.lineplot(x=np.linspace(0, 1, 100), y=mi_hist_sum2/np.max(mi_hist_sum2), color='red', label='gamma')
for i in range(4):
    sns.lineplot(x=np.linspace(0, 1, 100), y=mi_hist_sum_list[i]/np.max(mi_hist_sum_list[i]), color=color_list[i], label=label_list[i])
plt.title('')
plt.xlabel('Normolized Dist from Soma')
plt.ylabel('Normolized Mean Intensity')
plt.legend()
# plt.legend().set_visible(False)

fig = plt.gcf()  # 获取当前图形对象
# fig.legend(label_list, loc='upper center', bbox_to_anchor=(0.5, 0.2), ncol=2)
fig.tight_layout()
fig.show()


# 显示累计频率信息
# print(f'Low-Intensity Cumulative (0-50):\nFolder 1: {cumulative_low_freq1:.4f}\nFolder 2: {cumulative_low_freq2:.4f}')
# print(f'High-Intensity Cumulative (200-255):\nFolder 1: {cumulative_high_freq1:.4f}\nFolder 2: {cumulative_high_freq2:.4f}')


