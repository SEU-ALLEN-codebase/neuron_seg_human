import os
import numpy as np
import matplotlib.pyplot as plt
import tifffile
from skimage import io
from scipy.spatial import distance
import seaborn as sns
from skimage import exposure


def get_gamma_img(img):
    img = exposure.adjust_gamma(img, gamma=0.5)
    img = exposure.equalize_adapthist(img, clip_limit=0.02, nbins=256)
    img = (img - img.min()) / (img.max() - img.min())
    return img

def get_equalize_img(img):
    img = exposure.equalize_adapthist(img, clip_limit=0.02, nbins=256)
    img = (img - img.min()) / (img.max() - img.min())
    return img

def count_unique_voxel_intensities(image_path):
    # 读取三维图像
    image_3d = io.imread(image_path)

    # 找到图像中所有不同的体素强度
    unique_intensities = np.unique(image_3d)

    # 输出不同体素强度的数量
    print(f'The number of unique voxel intensities is: {len(unique_intensities)}')
    return unique_intensities

# 指定两个文件夹路径
folder1 = r'/data/kfchen/nnUNet/nnUNet_raw/Dataset169_hb_10k/imagesTr'
folder2 = r'/data/kfchen/nnUNet/nnUNet_raw/Dataset180_deflu_gamma/imagesTr'

# 获取两个文件夹中的所有文件名
files1 = set(f for f in os.listdir(folder1) if f.endswith('.tif'))
files2 = set(f for f in os.listdir(folder2) if f.endswith('.tif'))

# 找到同名的图像对
common_files = files1.intersection(files2)
common_files = sorted(list(common_files))
common_files = common_files[:1]

# 初始化累计直方图和频率信息
hist_sum_list = [np.zeros(128) for _ in range(4)]
example_names = ['Original', 'Adaptive_Gamma', 'Global_Gamma', 'Equalization']
colors = ['black', 'red', 'blue', 'green']

# 遍历同名的图像对并绘制MIP对照图和直方图
for file_name in common_files:
    img_list = [
        io.imread(os.path.join(folder1, file_name)),
        io.imread(os.path.join(folder2, file_name))
    ]

    img_list = [f.astype("float32") for f in img_list]
    img_list = [(f - f.min()) / (f.max() - f.min()) for f in img_list]

    img_list.append(get_gamma_img(img_list[0]))
    img_list.append(get_equalize_img(img_list[0]))

    img_list = [(f * 255).astype(np.uint8) for f in img_list]

    mip_list = [np.max(f, axis=0) for f in img_list]
    for i, mip in enumerate(mip_list):
        save_path = r"/data/kfchen/nnUNet/nnUNet_raw/Dataset180_deflu_gamma/comp_mip/"
        save_path = save_path + str(i) + '.jpg'
        print(mip.shape)
        tifffile.imsave(save_path, mip)
        # io.imsave(save_path, mip)


    for i, img in enumerate(img_list):
        # 计算全图的直方图
        hist, _ = np.histogram(img.ravel(), bins=128, range=(0, 256), density=True)
        hist_sum_list[i] += hist

# 绘制两个文件夹的全图直方图和累计频率信息
bins = np.arange(128)
sns.set(style="whitegrid")

# 绘制0-33%和66-100%最大强度的直方图
low_range = int(128* 0.33)
high_range = int(128 * 0.66)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
for i in range(4):
    sns.histplot(x=bins, weights=hist_sum_list[i], color=colors[i], label=example_names[i], alpha=0, kde=True)
# sns.histplot(x=bins[:low_range], weights=hist1_sum[:low_range], color='blue', label='Folder 1', alpha=0, kde=True)
# sns.histplot(x=bins[:low_range], weights=hist2_sum[:low_range], color='red', label='Folder 2', alpha=0, kde=True).title('Histogram Comparison (0-33% Intensity)')
plt.xlabel('Pixel Intensity')
plt.ylabel('Normalized Frequency')
plt.legend()

plt.subplot(1, 2, 2)
for i in range(4):
    sns.histplot(x=bins, weights=hist_sum_list[i], color=colors[i], label=example_names[i], alpha=0, kde=True)
# sns.barplot(x=bins[high_range:], y=hist1_sum[high_range:], color='blue', label='Folder 1', alpha=0.6)
# sns.barplot(x=bins[high_range:], y=hist2_sum[high_range:], color='red', label='Folder 2', alpha=0.6)
plt.title('Histogram Comparison (66-100% Intensity)')
plt.xlabel('Pixel Intensity')
plt.ylabel('Normalized Frequency')
plt.legend()

plt.tight_layout()
plt.show()

