import matplotlib.pyplot as plt
import numpy as np
import os
import tifffile

def process_file(process_file_pair):
    img_file, seg_file = process_file_pair
    img = tifffile.imread(os.path.join(img_dir, img_file))
    seg = tifffile.imread(os.path.join(seg_dir, seg_file))

    mip_list = [
        np.max(img, axis=0),
        np.max(img, axis=1),
        np.max(img, axis=2),

        np.max(seg, axis=0),
        np.max(seg, axis=1),
        np.max(seg, axis=2),
    ]

    plt.figure(figsize=(15, 15))
    plt.title(img_file)
    for j in range(6):
        plt.subplot(2, 3, j + 1)
        plt.imshow(mip_list[j], cmap='gray')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(mip_dir, img_file.replace('.tif', '.png')))
    plt.close()

img_dir = "/data/kfchen/nnUNet/nnUNet_raw/Dataset170_14k_hb_neuron/imagesTs"
seg_dir = "/data/kfchen/nnUNet/nnUNet_results/Dataset169_hb_10k/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/ptls10/14k_seg_result"
mip_dir = "/data/kfchen/trace_ws/paper_auto_human_neuron_recon/mip1016"
if not os.path.exists(mip_dir):
    os.makedirs(mip_dir)

img_files = os.listdir(img_dir)
img_files = [f for f in img_files if f.endswith('.tif')]
img_ids = [int(f.split('_')[1]) for f in img_files]
# 大于6000
img_ids = [i for i in img_ids if i >= 10000]

seg_files = os.listdir(seg_dir)
seg_files = [f for f in seg_files if f.endswith('.tif')]
seg_ids = [int(f.split('_')[1].split('.')[0]) for f in seg_files]

img_ids = list(set(img_ids) & set(seg_ids))
new_img_files = []
new_seg_files = []

for i in img_ids:
    new_img_file = [f for f in img_files if int(f.split('_')[1]) == i][0]
    new_seg_file = [f for f in seg_files if int(f.split('_')[1].split('.')[0]) == i][0]
    new_img_files.append(new_img_file)
    new_seg_files.append(new_seg_file)

# img_files.sort()
# seg_files.sort()

img_files = new_img_files
seg_files = new_seg_files
print(len(img_files), len(seg_files))
print(img_files[:20])
print(seg_files[:20])

# for i in range(len(img_files)):
#     process_file(img_files[i], seg_files[i])

from multiprocessing import Pool
from functools import partial

with Pool(10) as p:
    p.map(partial(process_file), zip(img_files, seg_files))

