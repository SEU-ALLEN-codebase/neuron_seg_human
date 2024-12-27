from traceback import print_exc

from pylib.file_io import load_image
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import Normalize
import tifffile
from matplotlib.colors import ListedColormap
import cv2
from scipy.ndimage import zoom

from pylib.math_utils import calc_included_angles_from_coords

img_file = r"D:\tracing_ws\mni_annotation.v3draw"
image = load_image(img_file)[0]
tifffile.imsave(r"D:\tracing_ws\mni_annotation.tif", image)
image = np.array(image)
# 选择z轴的前半部分
image = image[:image.shape[0] // 2]

# 上采样
image = zoom(image, (2, 2, 2), order=0)


interested_categories = [89, 52, 73, 60, 61, 79, 54, 85, 54, 73]
categories_names = ["superior frontal gyrus", "middle frontal gyrus", "inferior frontal gyrus",
                    "parietal lobe", "inferior parietal lobe",
                    "superior temporal gyrus", "middle temporal gyrus",
                    'occipital lobe',
                    'temporal lobe',
                    'frontal lobe',
                    ]
brain_region_map = {
    'frontal lobe': ['FL.L', 'FL.R', 'FL_TL.L'],
    "superior frontal gyrus": ["SFG.R", "SFG.L", "SFG", "S(M)FG.R", "M(I)FG.L"],
    "middle frontal gyrus": ["MFG.R", "MFG", "MFG.L"],
    "inferior frontal gyrus": ["IFG.R", "(X)FG", "IFG"],
    #
    'temporal lobe': ['TL.L', 'TL.R'],
    "superior temporal gyrus": ["STG", "STG.R", "S(M)TG.R", "S(M)TG.L", "STG-AP", "S(M,I)TG"],
    "middle temporal gyrus": ["MTG.R", "MTG.L", "MTG"],

    "parietal lobe": ["PL.L", "PL.L_OL.L", "PL"],
    "inferior parietal lobe": ["IPL-near-AG", "IPL.L"],

    'occipital lobe': ['OL.L', 'OL.R'],


    # 'posterior lateral ventricle': ['pLV.L'],
    'others': ['CB_tonsil.L', 'FP.R', 'FP.L', 'BN.L', 'FT.L', 'CC.L', "TP", "TP.L", "TP.R"],
}


full_interested_categories = len(interested_categories)
rows = cols = int(np.ceil(np.sqrt(len(interested_categories))))

# 创建足够的子图，每个类别一个子图
fig, axes = plt.subplots(1, 1, figsize=(10, 10))

print(full_interested_categories)
colors = plt.cm.get_cmap('Set3').colors[:11]
# print(len(colors))

colored_mip = np.max(image, axis=0)
colored_mip = (colored_mip - np.min(colored_mip)) / (np.max(colored_mip) - np.min(colored_mip)) * 0.9
colored_mip = np.stack([colored_mip, colored_mip, colored_mip], axis=-1)
axes.imshow(colored_mip)
axes.axis('off')  # 不显示坐标轴
plt.tight_layout()
plt.show()
plt.close()






fig, axes = plt.subplots(1, 1, figsize=(10, 10))
colored_mip = np.ones_like(colored_mip)

# 把黑色的部分变成白色
# print(colored_mip.shape)
colored_mip.astype(np.uint8)
for row in range(colored_mip.shape[0]):
    for col in range(colored_mip.shape[1]):
        if colored_mip[row, col, :].sum() == 0:
            colored_mip[row, col, :] = 1

# colored_mip = cv2.fastNlMeansDenoisingColored(colored_mip, None, 10, 10, 7, 21)
i = -1
for category in interested_categories[:8]:
    i = i + 1
    # 为当前类别创建掩码
    mask = image == category

    # 计算MIP
    mip_image = np.max(mask, axis=0)  # 沿Z轴计算最大值
    mip_image = cv2.medianBlur(mip_image.astype(np.uint8), 15)
    color = colors[i]  # 取颜色的RGB分量
    alpha = 0.7
    for j in range(3):
        colored_mip[:, :, j] = colored_mip[:, :, j] * (1 - alpha * mip_image) + alpha * color[j] * mip_image
# 在子图中显示彩色MIP
axes.imshow(colored_mip)
axes.axis('off')  # 不显示坐标轴
plt.tight_layout()
plt.show()
plt.close()







fig, axes = plt.subplots(1, 1, figsize=(10, 10))
colored_mip = np.max(image, axis=0)
colored_mip = (colored_mip - np.min(colored_mip)) / (np.max(colored_mip) - np.min(colored_mip)) * 0.9
colored_mip = np.stack([colored_mip, colored_mip, colored_mip], axis=-1)
colored_mip = np.ones_like(colored_mip)

# 把黑色的部分变成白色
# print(colored_mip.shape)
colored_mip.astype(np.uint8)
for row in range(colored_mip.shape[0]):
    for col in range(colored_mip.shape[1]):
        if colored_mip[row, col, :].sum() == 0:
            colored_mip[row, col, :] = 1

# TL: 把像素点为79的部分修改为54
image[image == 79] = 54
# FL: 把像素点为98, 52的部分修改为73
image[image == 89] = 73
image[image == 52] = 73
for category in interested_categories[8:]:
    i = i + 1
    mask = image == category

    mip_image = np.max(mask, axis=0)  # 沿Z轴计算最大值
    mip_image = mip_image.astype(np.uint8) * 255
    mip_image = cv2.medianBlur(mip_image.astype(np.uint8), 15)
    edges = cv2.Canny(mip_image, 10, 200)  # 阈值可根据实际需求调整
    # edges = mip_image

    # 膨胀
    kernel = np.ones((2, 2), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    edges = edges / 255

    alpha = 2
    color = colors[i]  # 取颜色的RGB分量
    for j in range(3):
        colored_mip[:, :, j] = colored_mip[:, :, j] * (1 - alpha * edges) + alpha * color[j] * edges
    print(i)

# 在子图中显示彩色MIP
axes.imshow(colored_mip)
axes.axis('off')  # 不显示坐标轴

plt.tight_layout()
plt.show()
plt.close()


