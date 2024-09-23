import pandas as pd
import os

def get_full_v3draw_files(v3draw_dir, suffix=".v3draw"):
    v3draw_files = []
    for root, dirs, files in os.walk(v3draw_dir):
        if("IHC" in root):
            continue
        for file in files:
            if file.endswith(suffix) and "_i" not in file and "_p" not in file:
                v3draw_files.append(file)

    v3draw_files.sort()
    return v3draw_files

def from_nnunet_name_to_full_name(name, name_mapping_df):
    # from nnunet name to full name
    name = name.split(".")[0]
    # print(name_mapping_df["nnunet_name"])
    full_name = name_mapping_df[name_mapping_df["nnunet_name"] == name].values[0][1]
    full_name = str(full_name)
    return full_name

def from_nnunet_name_to_seu_name(name, name_mapping_df, v3draw_files):
    # from nnunet name to full name
    name = name.split(".")[0]
    # print(name_mapping_df["nnunet_name"])
    full_name = name_mapping_df[name_mapping_df["nnunet_name"] == name].values[0][1]
    full_name = str(full_name)

    # from full name to seu name
    for v3draw_file in v3draw_files:
        if full_name in v3draw_file:
            if(int(v3draw_file.split("_")[0]) == int(full_name)):
                return v3draw_file
    return None


name_mapping_csv = "/data/kfchen/trace_ws/paper_auto_human_neuron_recon/test_seg_220/name_mapping.csv"
name_mapping_df = pd.read_csv(name_mapping_csv)
# img_info_file = "/data/kfchen/trace_ws/paper_auto_human_neuron_recon/Human_SingleCell_TrackingTable_20240712.csv"
# img_info_df = pd.read_csv(img_info_file, encoding='gbk')
v3draw_dir = "/PBshare/SEU-ALLEN/Projects/Human_Neurons/all_human_cells/all_human_cells_v3draw_8bit"
v3draw_files = get_full_v3draw_files(v3draw_dir)

seg_dir = r"/data/kfchen/trace_ws/paper_auto_human_neuron_recon/test_seg_220/ptls10"
seg_files = [f for f in os.listdir(seg_dir) if f.endswith(".tif")]
# print(seg_files)

for seg_file in seg_files:
    # v3draw_file = from_nnunet_name_to_seu_name(seg_file, name_mapping_df, v3draw_files)
    full_name = from_nnunet_name_to_full_name(seg_file, name_mapping_df)
    # print(full_name)
    os.rename(os.path.join(seg_dir, seg_file), os.path.join(seg_dir, full_name + ".tif"))
    # print("v3draw file: ", v3draw_file[:-7] + '.tif')
    # print("swc file: ", seg_file)
    # print("")
    # rename
    # os.rename(os.path.join(seg_dir, seg_file), os.path.join(seg_dir, v3draw_file[:-7] + ".tif"))