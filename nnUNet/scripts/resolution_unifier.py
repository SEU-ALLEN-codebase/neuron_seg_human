import pandas as pd
import os
from multiprocessing import Pool


"""
for Dataset169_hb_10k test data
"""
def find_resolution(df, filename):
    # print(filename)
    filename = int(filename.split('.')[0])
    for i in range(len(df)):
        if int(df.iloc[i, 0]) == filename:
            return df.iloc[i, 43]
    return None

csv_file = "/data/kfchen/nnUNet/nnUNet_results/Dataset169_hb_10k/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/ptls10/norm_result/Human_SingleCell_TrackingTable_20240712.csv"
df = pd.read_csv(csv_file, encoding='gbk')
swc_dir = "/data/kfchen/trace_ws/paper_auto_human_neuron_recon/test_seg_220/recon/source500"
unified_swc_dir = "/data/kfchen/trace_ws/paper_auto_human_neuron_recon/test_seg_220/unified_recon_1um/source500"
if(not os.path.exists(unified_swc_dir)):
    os.makedirs(unified_swc_dir)

swc_files = [f for f in os.listdir(swc_dir) if f.endswith('.swc')]
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

#
#
# """
# for 9k auto recon single neuron
# """
# def find_resolution(df, filename):
#     # print(filename)
#     filename = int(filename.split('.')[0].split('_')[0])
#     for i in range(len(df)):
#         if int(df.iloc[i, 0]) == filename:
#             return df.iloc[i, 43]
#     return None
#
# def unify_swc(swc_file, unified_swc_file, df):
#     xy_resolution = find_resolution(df, os.path.basename(swc_file))
#     with open(swc_file, 'r') as f:
#         lines = f.readlines()
#     with open(unified_swc_file, 'w') as f:
#         result_lines = []
#         for line in lines:
#             if line.startswith("#"):
#                 result_lines.append(line)
#             else:
#                 line = line.strip().split()
#                 line[2] = str(float(line[2]) * xy_resolution / 1000)
#                 line[3] = str(float(line[3]) * xy_resolution / 1000)
#                 result_lines.append(" ".join(line) + "\n")
#         f.writelines(result_lines)
#
# if __name__ == '__main__':
#     csv_file = "/data/kfchen/nnUNet/nnUNet_results/Dataset169_hb_10k/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/ptls10/norm_result/Human_SingleCell_TrackingTable_20240712.csv"
#     df = pd.read_csv(csv_file, encoding='gbk')
#     swc_dir = "/data/kfchen/trace_ws/paper_auto_human_neuron_recon/origin_recon/source500"
#     unified_swc_dir = "/data/kfchen/trace_ws/paper_auto_human_neuron_recon/unified_recon_1um/source500"
#     if(not os.path.exists(unified_swc_dir)):
#         os.makedirs(unified_swc_dir)
#
#     swc_files = [f for f in os.listdir(swc_dir) if f.endswith('.swc')]
#
#     swc_paths = [os.path.join(swc_dir, f) for f in swc_files]
#     unified_swc_paths = [os.path.join(unified_swc_dir, f) for f in swc_files]
#     # for swc_file, unified_swc_file in zip(swc_paths, unified_swc_paths):
#     #     unify_swc(swc_file, unified_swc_file, df)
#
#     workers = 16
#     pool = Pool(workers)
#     pool.starmap(unify_swc, zip(swc_paths, unified_swc_paths, [df]*len(swc_paths)))
#     pool.close()
#     pool.join()
#     print("done")