import pandas as pd
import os

def find_resolution(df, filename):
    # print(filename)
    filename = int(filename.split('.')[0])
    for i in range(len(df)):
        if int(df.iloc[i, 0]) == filename:
            return df.iloc[i, 43]
    return None

csv_file = "/data/kfchen/nnUNet/nnUNet_results/Dataset169_hb_10k/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/ptls10/norm_result/Human_SingleCell_TrackingTable_20240712.csv"
df = pd.read_csv(csv_file, encoding='gbk')
swc_dir = "/data/kfchen/nnUNet/nnUNet_results/Dataset169_hb_10k/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/ptls10/norm_result/2_sort"
unified_swc_dir = "/data/kfchen/nnUNet/nnUNet_results/Dataset169_hb_10k/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/ptls10/norm_result/unified_GS"
if(not os.path.exists(unified_swc_dir)):
    os.makedirs(unified_swc_dir)

swc_files = os.listdir(swc_dir)
# swc_files = swc_files[:10]
for swc_file in swc_files:
    # print(find_resolution(df, swc_file))
    xy_resolution = find_resolution(df, swc_file)
    unified_swc_file = os.path.join(unified_swc_dir, swc_file)
    with open(os.path.join(swc_dir, swc_file), 'r') as f:
        lines = f.readlines()
    with open(unified_swc_file, 'w') as f:
        result_lines = []
        for line in lines:
            if line.startswith("#"):
                result_lines.append(line)
            else:
                line = line.strip().split()
                line[2] = str(float(line[2]) * xy_resolution / 1000)
                line[3] = str(float(line[3]) * xy_resolution / 1000)
                result_lines.append(" ".join(line) + "\n")
        f.writelines(result_lines)
