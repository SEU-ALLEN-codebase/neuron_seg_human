import os
import shutil
# #
# val_dir = "/data/kfchen/nnUNet/nnUNet_raw/Dataset169_hb_10k/imagesTs"
# val_files = [f for f in os.listdir(val_dir) if f.endswith('.tif')]
#
# source_tif_dir = "/data/kfchen/nnUNet/nnUNet_raw/Dataset173_14k_hb_neuron_aug_4power/imagesTr"
# target_tif_dir = "/data/kfchen/nnUNet/nnUNet_raw/Dataset173_14k_hb_neuron_aug_4power/imagesTs"
#
# for val_file in val_files:
#     source_tif = os.path.join(source_tif_dir, val_file)
#     target_tif = os.path.join(target_tif_dir, val_file)
#     source_json_file = os.path.join(source_tif_dir, val_file.replace("_0000.tif", ".json"))
#     target_json_file = os.path.join(target_tif_dir, val_file.replace("_0000.tif", ".json"))
#     # shutil.copy(source_tif, target_tif)
#     # shutil.copy(source_json_file, target_json_file)
#     # 剪切
#     shutil.move(source_tif, target_tif)
#     shutil.move(source_json_file, target_json_file)
#
#     # print(f"Copy {source_tif} to {target_tif}")

# v3d_swc_dir = "/data/kfchen/nnUNet/nnUNet_raw/Dataset172_14k_hb_neuron_aug/my_test/v3dswc"
# v3d_swc_files = [f for f in os.listdir(v3d_swc_dir) if f.endswith('.swc')]
#
# for v3d_swc_file in v3d_swc_files:
#     # rename
#     new_v3d_swc_file = v3d_swc_file.replace(".tif.swc", ".swc")
#     os.rename(os.path.join(v3d_swc_dir, v3d_swc_file), os.path.join(v3d_swc_dir, new_v3d_swc_file))



# img_dir1 = "/data/kfchen/nnUNet/nnUNet_raw/Dataset172_14k_hb_neuron_aug/imagesTr"
# img_dir2 = "/data/kfchen/nnUNet/nnUNet_raw/Dataset173_14k_hb_neuron_aug_4power/imagesTr"
#
# img_files1 = [f for f in os.listdir(img_dir1) if f.endswith('.tif')]
# img_files2 = [f for f in os.listdir(img_dir2) if f.endswith('.tif')]
# # sort
# img_files1.sort()
# img_files2.sort()
#
# # shared_files = set(img_files1) & set(img_files2)
# non_shared_files = set(img_files2) - set(img_files1)
#
# print(non_shared_files)
# print(len(non_shared_files))



img_dir = "/data/kfchen/trace_ws/paper_auto_human_neuron_recon/test_seg_220/adaptive_gamma_source500_2"
img_files = [f for f in os.listdir(img_dir) if f.endswith('.tif')]

for img_file in img_files:
    if(img_file.endswith(".tif.tif")):
        os.rename(os.path.join(img_dir, img_file), os.path.join(img_dir, img_file.replace(".tif.tif", ".tif")))