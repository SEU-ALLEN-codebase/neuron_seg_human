import os
import numpy as np
import tifffile
from pylib.file_io import load_image
from nnUNet.scripts.mip import get_mip_swc
import cv2
import joblib
N_JOBS = 20

swc_dir = "/PBshare/SEU-ALLEN/Users/KaifengChen/human_brain/10847_auto_v1.5/swc/connswc_with_radius"
img_dir = "/PBshare/SEU-ALLEN/Projects/Human_Neurons/all_human_cells/all_human_cells_v3draw_8bit"
save_dir = "/PBshare/SEU-ALLEN/Users/KaifengChen/human_brain/10847_auto_v1.5/mip"

swc_files = [os.path.join(swc_dir, f) for f in os.listdir(swc_dir) if f.endswith('.swc')]
if('v3draw' in img_dir):
    img_root = img_dir
    # walk
    img_files = []
    for root, dirs, files in os.walk(img_root):
        if("human_brain_data_v3draw" not in root):
            continue
        for f in files:
            if f.endswith('.v3draw'):
                img_files.append(os.path.join(root, f))
else:
    img_files = [f for f in os.listdir(img_dir) if f.endswith('.tif')]

img_dict = {int(os.path.basename(f).split('_')[0]): f for f in img_files}
swc_dict = {int(os.path.basename(f).split('_')[0]): f for f in swc_files}
shared_ids = set(img_dict.keys()).intersection(swc_dict.keys())
todo_pairs = [(img_dict[i], swc_dict[i]) for i in shared_ids]

print("Number of shared ids: ", len(shared_ids))

def current_task(img_file, swc_file):
    save_file = os.path.join(save_dir, os.path.basename(swc_file).replace('.swc', '.png'))
    if(os.path.exists(save_file)):
        return

    try:
        if('v3draw' in img_file):
            img = load_image(img_file)[0]
        elif('tif' in img_file):
            img = tifffile.imread(img_file)

        img = (img - img.min()) / (img.max() - img.min()) * 255
        mip_swc = get_mip_swc(swc_file, img)
        cv2.imwrite(save_file, mip_swc)
    except Exception as e:
        print(e, img_file, swc_file)

joblib.Parallel(n_jobs=10)(joblib.delayed(current_task)(img_file, swc_file) for img_file, swc_file in tqdm(todo_pairs))

