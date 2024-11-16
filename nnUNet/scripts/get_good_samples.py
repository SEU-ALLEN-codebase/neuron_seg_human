import tifffile
import numpy as np
import os
import pandas as pd
import shutil

from simple_swc_tool.Topology_scoring.metrics_delin import mkdir

tif_dir = "/data/kfchen/nnUNet/nnUNet_raw/Dataset170_14k_hb_neuron/imagesTs"
new_tif_dir = "/data/kfchen/nnUNet/nnUNet_raw/Dataset170_14k_hb_neuron/imagesTs_good_14k"
if(not os.path.exists(new_tif_dir)):
    mkdir(new_tif_dir)
tif_files = [f for f in os.listdir(tif_dir) if f.endswith('.tif')]
# tif_files = tif_files[:10]

good_sample_list_file = "/data/kfchen/nnUNet/nnUNet_raw/Dataset170_14k_hb_neuron/good_samples_14k.csv"
df = pd.read_csv(good_sample_list_file)
good_samples = df['Cell_id'].values
print(len(good_samples))
good_samples = [int(s) for s in good_samples]

for f in tif_files:
    sample_id = int(f.split('_')[1])
    if sample_id not in good_samples:
        # print(f)
        continue
    else:
        shutil.copy(os.path.join(tif_dir, f), os.path.join(new_tif_dir, f))
        shutil.copy(os.path.join(tif_dir, f.replace("_0000.tif", ".json")), os.path.join(new_tif_dir, f.replace("_0000.tif", ".json")))
