from nnUNet.scripts.mip import get_mip_swc, get_mip
import os
import tifffile
import numpy as np
import cv2



swc_dir1 = '/data/kfchen/trace_ws/to_gu/lab/2_flip_after_sort'
swc_dir2 = '/data/kfchen/nnUNet/nnUNet_results/Dataset169_hb_10k/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/ptls10/validation_traced/v3dswc_copy'
img_dir = r'/data/kfchen/trace_ws/to_gu/img'
mip_dir = '/data/kfchen/nnUNet/nnUNet_results/Dataset169_hb_10k/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/ptls10/concat_mip'
if(not os.path.exists(mip_dir)):
    os.makedirs(mip_dir)

swc_dir_result = "/data/kfchen/nnUNet/nnUNet_results/Dataset169_hb_10k/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/ptls10/norm_result/pruned_unified_Auto"
img_dir2 = "/data/kfchen/trace_ws/trace_consensus_test/img"
swc_files = [f for f in os.listdir(swc_dir_result) if f.endswith('.swc')]
for swc_file in swc_files:
    img_file = os.path.join(img_dir, os.path.basename(swc_file).replace('.swc', '.tif'))
    copy_img_file = os.path.join(img_dir2, os.path.basename(swc_file).replace('.swc', '.tif'))
    # copy img file
    os.system('cp %s %s' % (img_file, copy_img_file))


exit(0)

swc_list1 = [f for f in os.listdir(swc_dir1) if f.endswith('.swc')]
swc_list2 = [f for f in os.listdir(swc_dir2) if f.endswith('.swc')]
shared_list = [f for f in swc_list1 if f in swc_list2]
print(len(shared_list))

shared_list = [f for f in shared_list if f.startswith('2836')]

for swc_name in shared_list:
    img_file = os.path.join(img_dir, os.path.basename(swc_name).replace('.swc', '.tif'))
    mip_file = os.path.join(mip_dir, os.path.basename(swc_name).replace('.swc', '.png'))
    swc_file1 = os.path.join(swc_dir1, os.path.basename(swc_name))
    swc_file2 = os.path.join(swc_dir2, os.path.basename(swc_name))

    img = tifffile.imread(img_file)
    img_mip = get_mip(img)
    # 转成三通道
    img_mip = np.stack([img_mip, img_mip, img_mip], axis=2)

    swc_mip1 = get_mip_swc(swc_file1, img)
    swc_mip2 = get_mip_swc(swc_file2, img)

    # print(img_mip.shape, swc_mip1.shape, swc_mip2.shape)

    # concat_mip = np.concatenate([img_mip, swc_mip1, swc_mip2], axis=1)
    # tifffile.imsave(mip_file, concat_mip)
    interested_window = [img_mip.shape[0] // 2 - 128, img_mip.shape[0] // 2 + 128 - 64,
                         img_mip.shape[1] // 2 - 256 + 64, img_mip.shape[1] // 2]

    # img_mip = img_mip[interested_window[0]:interested_window[1], interested_window[2]:interested_window[3], :]
    # swc_mip1 = swc_mip1[interested_window[0]:interested_window[1], interested_window[2]:interested_window[3], :]
    # swc_mip2 = swc_mip2[interested_window[0]:interested_window[1], interested_window[2]:interested_window[3], :]

    # 长方形边框
    cv2.rectangle(img_mip, (interested_window[2], interested_window[0]), (interested_window[3], interested_window[1]), (0, 255, 255), 2)
    cv2.rectangle(swc_mip1, (interested_window[2], interested_window[0]), (interested_window[3], interested_window[1]), (0, 255, 255), 2)
    cv2.rectangle(swc_mip2, (interested_window[2], interested_window[0]), (interested_window[3], interested_window[1]), (0, 255, 255), 2)

    # img_mip = np.all(img_mip == [255, 0, 0], axis=-1)
    mask = np.all(swc_mip1 == [255, 0, 0], axis=-1)
    swc_mip1[mask] = [0, 0, 255]
    mask = np.all(swc_mip2 == [255, 0, 0], axis=-1)
    swc_mip2[mask] = [0, 0, 255]

    cv2.imwrite(mip_file.replace('.png', '_img.png'), img_mip)
    cv2.imwrite(mip_file.replace('.png', '_swc1.png'), swc_mip1)
    cv2.imwrite(mip_file.replace('.png', '_swc2.png'), swc_mip2)
    # tifffile.imsave(mip_file.replace('.png', '_img.png'), img_mip)
    # tifffile.imsave(mip_file.replace('.png', '_swc1.png'), swc_mip1)
    # tifffile.imsave(mip_file.replace('.png', '_swc2.png'), swc_mip2)


    # concat_mip = np.concatenate([img_mip[interested_window[0]:interested_window[1], interested_window[2]:interested_window[3], :],
    #                              swc_mip1[interested_window[0]:interested_window[1], interested_window[2]:interested_window[3], :],
    #                              swc_mip2[interested_window[0]:interested_window[1], interested_window[2]:interested_window[3], :]], axis=1)
    # tifffile.imsave(mip_file.replace('.png', '_crop.png'), concat_mip)

