import matplotlib.pyplot as plt
import tifffile
import numpy as np
import os
import pandas as pd
from scipy.ndimage import zoom


def find_resolution(df, filename):
    # print(filename)
    filename = int(filename.split('.')[0])
    for i in range(len(df)):
        if int(df.iloc[i, 0]) == filename:
            return df.iloc[i, 43]
    return None

img_dir = r"/data/kfchen/nnUNet/nnUNet_raw/Dataset170_14k_hb_neuron/imagesTs"
mip_dir = r"/data/kfchen/nnUNet/nnUNet_raw/Dataset170_14k_hb_neuron/mip"
# neuron_info_file = r"/data/kfchen/nnUNet/nnUNet_results/Dataset169_hb_10k/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/ptls10/norm_result/Human_SingleCell_TrackingTable_20240712.csv"
# df = pd.read_csv(neuron_info_file, encoding='gbk')
img_files = os.listdir(img_dir)
img_files = [f for f in img_files if f.endswith('.tif')]
img_files = sorted(img_files, key=lambda x: int(x.split('_')[1]))
img_files = np.random.choice(img_files, 500, replace=False)

for img_file in img_files:
    img = tifffile.imread(os.path.join(img_dir, img_file))
    mip_file = os.path.join(mip_dir, img_file.replace('.tif', '.png'))
    if(os.path.exists(mip_file)):
        continue
    # xy_resolution = find_resolution(df, img_file)
    xy_resolution = 500

    print(img.shape, xy_resolution)
    new_img_shape = (img.shape[0], round(img.shape[1] * xy_resolution / 1000), round(img.shape[2] * xy_resolution / 1000))
    # zoom
    img = zoom(img, (1, new_img_shape[1] / img.shape[1], new_img_shape[2] / img.shape[2]), order=1)
    print(img.shape)

    z_mip = np.max(img, axis=0)
    y_mip = np.max(img, axis=1)
    x_mip = np.max(img, axis=2)



    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(z_mip, cmap='gray')
    plt.title('Z MIP')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(y_mip, cmap='gray')
    plt.title('Y MIP')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(x_mip, cmap='gray')
    plt.title('X MIP')
    plt.axis('off')

    plt.savefig(mip_file)
    plt.close()