import tifffile
import numpy as np
import scipy.ndimage
from scipy.ndimage import label, find_objects
from skimage.morphology import ball
from skimage.morphology import binary_opening
import cupy as cp
from cupyx.scipy.ndimage import binary_opening
import cupyx

def get_min_diameter_3d(binary_image):
    labeled_array, num_features = scipy.ndimage.label(binary_image)
    largest_cc = np.argmax(np.bincount(labeled_array.flat)[1:]) + 1
    slice_x, slice_y, slice_z = find_objects(labeled_array == largest_cc)[0]
    diameter_x = slice_x.stop - slice_x.start
    diameter_y = slice_y.stop - slice_y.start
    diameter_z = slice_z.stop - slice_z.start

    return min(diameter_x, diameter_y, diameter_z)

def opening_get_soma_region(soma_region):
    mip_list = []

    soma_region_copy = soma_region.copy()
    radius = get_min_diameter_3d(soma_region)

    # on cpu
    max_rate = 10
    for i in range(max_rate):
        print(i)
        spherical_selem = ball(radius * (max_rate - i) / 10 / 2)
        # soma_region_res = binary_opening(soma_region, spherical_selem).astype("uint8")
        soma_region_res = scipy.ndimage.binary_opening(soma_region, spherical_selem).astype("uint8")
        if (soma_region_res.sum() == 0):
            continue
        soma_region = soma_region_res
        mip_list.append(np.max(soma_region_res, axis=0))

    # soma_region = binary_erosion(soma_region, spherical_selem).astype("uint8")
    del spherical_selem, radius, soma_region_res
    if (soma_region.sum() == 0):
        soma_region = soma_region_copy
    del soma_region_copy

    return soma_region, mip_list

def opening_get_soma_region_gpu(soma_region):
    soma_region_copy = soma_region.copy()
    radius = get_min_diameter_3d(soma_region)

    # on gpu
    # try:
    max_rate = 10
    soma_region_gpu = cp.array(soma_region)
    mip_list = []

    for i in range(max_rate):
        mip_list.append(cp.asnumpy(cp.max(soma_region_gpu, axis=0)))
        spherical_selem = ball(radius * (max_rate - i) / 10 / 2)
        spherical_selem_gpu = cp.array(spherical_selem)

        # 在 GPU 上执行 binary_opening
        # soma_region_res_gpu = binary_opening(soma_region_gpu, spherical_selem_gpu)
        soma_region_res_gpu = cupyx.scipy.ndimage.binary_opening(soma_region_gpu, spherical_selem_gpu)

        if soma_region_res_gpu.sum() == 0:
            continue

        mip_list.append(cp.asnumpy(cp.max(soma_region_res_gpu, axis=0)))
        soma_region_gpu = soma_region_res_gpu

    soma_region = cp.asnumpy(soma_region_gpu)
    del spherical_selem, radius, soma_region_res_gpu, soma_region_gpu
    # except:
    #     pass
    if (soma_region.sum() == 0):
        soma_region = soma_region_copy
    del soma_region_copy

    return soma_region, mip_list

tif_file = r"/data/kfchen/trace_ws/14k_seg_result/tif/02368_P020_T01-S020_ROL_R0613_RJ-20221021_LD.tif"
image = tifffile.imread(tif_file)
soma_region = image > 0
# soma_region, mip_list = opening_get_soma_region(soma_region)
soma_region, mip_list = opening_get_soma_region_gpu(soma_region)

concat_mip = np.concatenate(mip_list, axis=1)
concat_mip = concat_mip.astype(np.uint8) * 255
tifffile.imsave(r"/data/kfchen/nnUNet/nnUNet_results/Dataset169_hb_10k/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/ptls10/2364_mip.png", concat_mip)

