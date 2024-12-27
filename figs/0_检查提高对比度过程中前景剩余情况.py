import tifffile
import numpy as np
import matplotlib.pyplot as plt

img_path = "/data/kfchen/trace_ws/to_gu/img/2376.tif"
mask_path = "/PBshare/SEU-ALLEN/Users/KaifengChen/human_brain/img/mask/2376.tif"

img = tifffile.imread(img_path)
mask = tifffile.imread(mask_path)
print(img.shape, mask.shape)
mask = np.where(mask > 0, 1, 0)
img, mask = img.astype(np.float32), mask.astype(np.float32)
img = (img - img.min()) / (img.max() - img.min())
# img = np.power(img, 0.7)
origin_img = img.copy()

def current_task(img, mask, threshold):
    img = np.where(img > threshold, 1, 0)
    rt = np.sum(mask * img) / np.sum(mask)
    return rt
# img = origin_img
# img = (img - img.min()) / (img.max() - img.min())
# img = np.where(img > 0.6, 1, 0)
# rt = np.sum(mask * img) / np.sum(mask)
# print(rt)
#
# img = origin_img
# img = (img - img.min()) / (img.max() - img.min())
# img = np.where(img > 0.65, 1, 0)
# rt = np.sum(mask * img) / np.sum(mask)
# print(rt)
#
# img = origin_img
# img = (img - img.min()) / (img.max() - img.min())
# img = np.where(img > 0.7, 1, 0)
# rt = np.sum(mask * img) / np.sum(mask)
# print(rt)

for i in range(20):
    img = origin_img
    img = (img - img.min()) / (img.max() - img.min())
    img = np.where(img > i * 0.05, 1, 0)
    rt = np.sum(mask * img) / np.sum(mask)
    print(rt, i * 0.05)

img_mip, mask_mip = img.max(axis=0), mask.max(axis=0)
save_file = "/data/kfchen/trace_ws/img_noise_test/2376_.png"
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(img_mip, cmap="gray")
ax[1].imshow(mask_mip, cmap="gray")

plt.savefig(save_file)
plt.close()

