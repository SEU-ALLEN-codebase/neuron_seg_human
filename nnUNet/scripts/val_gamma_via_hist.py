import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from scipy.spatial import distance
import seaborn as sns

# 指定两个文件夹路径
folder1 = r'/data/kfchen/nnUNet/nnUNet_raw/Dataset169_hb_10k/imagesTr'
folder2 = r'/data/kfchen/nnUNet/nnUNet_raw/Dataset180_deflu_gamma/imagesTr'


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
cumulative_low_freq1 = 0
cumulative_low_freq2 = 0
cumulative_high_freq1 = 0
cumulative_high_freq2 = 0

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


# 绘制两个文件夹的全图直方图和累计频率信息
bins = np.arange(128)
sns.set(style="whitegrid")

plt.figure(figsize=(12, 6))
sns.barplot(x=bins, y=hist1_sum, color='blue', label='Folder 1', alpha=0.6)
sns.barplot(x=bins, y=hist2_sum, color='red', label='Folder 2', alpha=0.6)
plt.title('Overall Histogram Comparison Between Folders')
plt.xlabel('Pixel Intensity')
plt.ylabel('Normalized Frequency')
plt.legend()
plt.show()

# 绘制0-33%和66-100%最大强度的直方图
low_range = int(128* 0.33)
high_range = int(128 * 0.66)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.barplot(x=bins[:low_range], y=hist1_sum[:low_range], color='blue', label='Folder 1', alpha=0.6)
sns.barplot(x=bins[:low_range], y=hist2_sum[:low_range], color='red', label='Folder 2', alpha=0.6)
plt.title('Histogram Comparison (0-33% Intensity)')
plt.xlabel('Pixel Intensity')
plt.ylabel('Normalized Frequency')
plt.legend()

plt.subplot(1, 2, 2)
sns.barplot(x=bins[high_range:], y=hist1_sum[high_range:], color='blue', label='Folder 1', alpha=0.6)
sns.barplot(x=bins[high_range:], y=hist2_sum[high_range:], color='red', label='Folder 2', alpha=0.6)
plt.title('Histogram Comparison (66-100% Intensity)')
plt.xlabel('Pixel Intensity')
plt.ylabel('Normalized Frequency')
plt.legend()

plt.tight_layout()
plt.show()

# 显示累计频率信息
print(f'Low-Intensity Cumulative (0-50):\nFolder 1: {cumulative_low_freq1:.4f}\nFolder 2: {cumulative_low_freq2:.4f}')
print(f'High-Intensity Cumulative (200-255):\nFolder 1: {cumulative_high_freq1:.4f}\nFolder 2: {cumulative_high_freq2:.4f}')


