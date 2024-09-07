import os
import shutil

# swc_dir1 = r"/data/kfchen/trace_ws/14k_seg_result_ptls10/v3dswc"
target_dir1 = r"/data/kfchen/trace_ws/paper_auto_human_neuron_recon/origin_recon/ptls10"
img_dir1 = r"/data/kfchen/trace_ws/paper_auto_human_neuron_recon/seg/ptls10"
# swc_dir2 = r"/data/kfchen/trace_ws/14k_seg_result_source500/v3dswc"
target_dir2 = r"/data/kfchen/trace_ws/paper_auto_human_neuron_recon/origin_recon/source500"
img_dir2 = r"/data/kfchen/trace_ws/paper_auto_human_neuron_recon/seg/source500"
mutineuron_sample_marker_dir = r"/data/kfchen/trace_ws/14k_seg_result_source500/newest_muti_soma_markers"


# swcs = [f for f in os.listdir(swc_dir) if f.endswith(".swc")]
#
# mutineuron_sample_markers = [f for f in os.listdir(mutineuron_sample_marker_dir) if f.endswith(".marker")]
# ids = [int(f.split("_")[0]) for f in mutineuron_sample_markers]
# max_id = 10000
#
# for swc in swcs:
#     swc_id = int(swc.split("_")[0])
#     if swc_id in ids:
#         continue
#     if(swc_id > max_id):
#         continue
#     shutil.copy(os.path.join(swc_dir, swc), os.path.join(target_dir, swc))
#
# final_swcs = [f for f in os.listdir(target_dir) if f.endswith(".swc")]
# print(f"final swc count: {len(final_swcs)}")





# swcs_1 = [f for f in os.listdir(target_dir1) if f.endswith(".swc")]
# swcs_2 = [f for f in os.listdir(target_dir2) if f.endswith(".swc")]
#
# shared_swcs = set(swcs_1) & set(swcs_2)
# print(f"swc1 count: {len(swcs_1)}")
# print(f"swc2 count: {len(swcs_2)}")
# print(f"shared swc count: {len(shared_swcs)}")
#
# # remove non-shared swcs
# for swc in swcs_1:
#     if swc not in shared_swcs:
#         os.remove(os.path.join(target_dir1, swc))
# for swc in swcs_2:
#     if swc not in shared_swcs:
#         os.remove(os.path.join(target_dir2, swc))



swcs_1 = [f for f in os.listdir(target_dir1) if f.endswith(".swc")]
imgs_1 = [f for f in os.listdir(img_dir1) if f.endswith(".tif")]
shared_ids = set([int(f.split("_")[0]) for f in swcs_1]) & set([int(f.split("_")[0]) for f in imgs_1])
for img_1 in imgs_1:
    img_id = int(img_1.split("_")[0])
    if img_id not in shared_ids:
        os.remove(os.path.join(img_dir1, img_1))
