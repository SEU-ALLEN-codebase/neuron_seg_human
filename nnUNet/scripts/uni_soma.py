import tifffile
import os
import numpy as np
from skimage.transform import resize

img_dir = "/PBshare/SEU-ALLEN/Users/KaifengChen/human_brain/img/mask"
soma_dir = "/data/kfchen/nnUNet/nnUNet_results/Dataset169_hb_10k/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/source500/validation_traced/soma"
uni_soma_dir = "/data/kfchen/nnUNet/nnUNet_results/Dataset169_hb_10k/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/source500/validation_traced/uni_soma"

soma_files = [f for f in os.listdir(soma_dir) if f.endswith('.tif')]

for f in soma_files:
    soma = tifffile.imread(os.path.join(soma_dir, f))
    # soma = (soma - np.min(soma)) / (np.max(soma) - np.min(soma)) * 255
    img = tifffile.imread(os.path.join(img_dir, f))

    img_shape = img.shape
    soma_shape = soma.shape

    # resize soma
    soma = resize(soma, img_shape, order=0, anti_aliasing=False)
    soma = (soma > 0).astype("uint8") * 255

    tifffile.imwrite(os.path.join(uni_soma_dir, f), soma)