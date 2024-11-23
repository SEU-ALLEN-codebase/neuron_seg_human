import pandas as pd
import numpy as np
import os

def get_origin_name(name, name_mapping_df):
    return name_mapping_df[name_mapping_df['nnunet_name'] == name]['ID'].values[0]

# tif_dir = "/data/kfchen/nnUNet/nnUNet_raw/Dataset180_deflu_gamma/imagesTr"
# name_mapping_file = "/data/kfchen/nnUNet/nnUNet_raw/Dataset180_deflu_gamma/name_mapping.csv"
# csv_path = "/data/kfchen/trace_ws/paper_trace_result/train&val_list.csv"
#
# name_mapping_df = pd.read_csv(name_mapping_file)
#
# tif_files = [f for f in os.listdir(tif_dir) if f.endswith('.tif')]
# tif_files = [f.replace("_0000.tif", "") for f in tif_files]
# ids = [str(int(get_origin_name(f, name_mapping_df))) for f in tif_files]
#
# # ids = [str(int(f.split('_')[0].split('.')[0])) for f in tif_files]
# ids = np.unique(ids)
# # sort
# ids = np.sort(ids)
# df = pd.DataFrame(ids, columns=['id'])
# df.to_csv(csv_path, index=False)

#  检查是否有交集
# csv1 = "/data/kfchen/trace_ws/paper_trace_result/train&val_list.csv" # 1100个
# csv2 = "/data/kfchen/trace_ws/paper_trace_result/test_list_with_gs.csv" # 242个
#
# df1 = pd.read_csv(csv1)
# df2 = pd.read_csv(csv2)
#
# ids1 = df1['id'].values
# ids2 = df2['id'].values
# print(len(ids1), len(ids2))
#
# print(np.intersect1d(ids1, ids2))
# # 无交集

# csv1 = "/data/kfchen/trace_ws/paper_trace_result/test_list_with_gs.csv"
# df1 = pd.read_csv(csv1)
# ids1 = df1['id'].values
#
# swc_dir = r"/data/kfchen/trace_ws/paper_trace_result/nnunet/proposed_9k/8_estimated_radius_swc"
# swc_files = [f for f in os.listdir(swc_dir) if f.endswith('.swc')]
# ids2 = [str(int(f.split('.')[0].split('_')[0])) for f in swc_files]
# ids2 = np.unique(ids2)
# print(len(np.intersect1d(ids1, ids2)))
# # 完全包含

