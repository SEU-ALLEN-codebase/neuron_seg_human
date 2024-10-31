import os
import shutil
import subprocess
import time
import uuid
from functools import partial
from multiprocessing import Pool

import cc3d
import matplotlib.pyplot as plt
import numpy as np
import scipy
import skimage.morphology
import tifffile
from scipy.ndimage import binary_dilation
from scipy.ndimage import label, find_objects
from skimage.measure import regionprops, label
from skimage.morphology import ball
from skimage.morphology import skeletonize_3d
from tqdm import tqdm

# os.environ["CUDA_PATH"] = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.2"


import cupy as cp
from cupyx.scipy.ndimage import binary_opening
import cupyx

import SimpleITK as sitk
# from gcut.python.neuron_segmentation import NeuronSegmentation
import sys
from simple_swc_tool.swc_io import read_swc, write_swc

from scipy import ndimage
# from nnunetv2.training.loss.fastanison import anisodiff3
import pandas as pd
from pylib.file_io import load_image

v3d_path = r"/home/kfchen/Vaa3D-x.1.1.4_Ubuntu/Vaa3D-x"

class SomaRegionFinder():
    def process_path(self, pstr):
        return pstr.replace('(', '\(').replace(')', '\)')

    def dusting(self, img):
        if (img.sum() == 0):
            return img
        labeled_image = cc3d.connected_components(img, connectivity=6)
        largest_label = np.argmax(np.bincount(labeled_image.flat)[1:]) + 1
        largest_component_binary = ((labeled_image == largest_label)).astype("uint8")
        return largest_component_binary

    def crop_nonzero(self, image):
        non_zero_coords = np.argwhere(image)
        min_coords = non_zero_coords.min(axis=0)
        max_coords = non_zero_coords.max(axis=0) + 1

        cropped_image = image[min_coords[0]:max_coords[0],
                        min_coords[1]:max_coords[1],
                        min_coords[2]:max_coords[2]]

        return cropped_image, image.shape, min_coords

    def restore_original_size(self, cropped_image, original_shape, min_coords):
        restored_image = np.zeros(original_shape, dtype=cropped_image.dtype)
        restored_image[min_coords[0]:min_coords[0] + cropped_image.shape[0],
        min_coords[1]:min_coords[1] + cropped_image.shape[1],
        min_coords[2]:min_coords[2] + cropped_image.shape[2]] = cropped_image

        return restored_image

    def get_min_diameter_3d(self, binary_image):
        labeled_array, num_features = scipy.ndimage.label(binary_image)
        largest_cc = np.argmax(np.bincount(labeled_array.flat)[1:]) + 1
        slice_x, slice_y, slice_z = find_objects(labeled_array == largest_cc)[0]
        diameter_x = slice_x.stop - slice_x.start
        diameter_y = slice_y.stop - slice_y.start
        diameter_z = slice_z.stop - slice_z.start

        return min(diameter_x, diameter_y, diameter_z)

    def opening_get_soma_region_gpu(self, soma_region):
        soma_region_copy = soma_region.copy()
        radius = self.get_min_diameter_3d(soma_region)

        # on gpu
        # try:
        max_rate = 10
        soma_region_gpu = cp.array(soma_region)

        for i in range(max_rate):
            spherical_selem = ball(radius * (max_rate - i) / 10 / 2)
            spherical_selem_gpu = cp.array(spherical_selem)

            # 在 GPU 上执行 binary_opening
            # soma_region_res_gpu = binary_opening(soma_region_gpu, spherical_selem_gpu)
            soma_region_res_gpu = cupyx.scipy.ndimage.binary_opening(soma_region_gpu, spherical_selem_gpu)

            if soma_region_res_gpu.sum() == 0:
                continue

            soma_region_gpu = soma_region_res_gpu

        soma_region = cp.asnumpy(soma_region_gpu)
        del spherical_selem, radius, soma_region_res_gpu, soma_region_gpu
        # except:
        #     pass
        if (soma_region.sum() == 0):
            soma_region = soma_region_copy
        del soma_region_copy

        return soma_region

    def get_soma_region(self, img_path):
        # print(img_path, marker_path)
        in_tmp = img_path
        out_tmp = in_tmp.replace('.tif', '_gsdt.tif')

        if (sys.platform == "linux"):
            cmd_str = f'xvfb-run -a -s "-screen 0 640x480x16" {v3d_path} -x gsdt -f gsdt -i {in_tmp} -o {out_tmp} -p 0 1 0 1.5'
            cmd_str = self.process_path(cmd_str)
            # print(cmd_str)
            subprocess.run(cmd_str, stdout=subprocess.DEVNULL, shell=True)
        else:
            cmd = f'{v3d_path} /x gsdt /f gsdt /i {in_tmp} /o {out_tmp} /p 0 1 0 1.5'
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

        pred = tifffile.imread(img_path).astype("uint8")
        pred[pred <= 255 / 2] = 0
        pred[pred > 255 / 2] = 1

        gsdt = tifffile.imread(out_tmp).astype("uint8")
        gsdt = np.flip(gsdt, axis=1)
        if (os.path.exists(out_tmp)): os.remove(out_tmp)
        del out_tmp, in_tmp

        max_gsdt = np.max(gsdt)
        gsdt[gsdt <= max_gsdt / 2] = 0
        gsdt[gsdt > max_gsdt / 2] = 1

        gsdt = binary_dilation(gsdt, iterations=5).astype("uint8")
        soma_region = np.logical_and(pred, gsdt).astype("uint8")
        soma_region = self.dusting(soma_region)
        del pred, gsdt, max_gsdt

        soma_region, original_shape, min_coords = self.crop_nonzero(soma_region)
        # soma_region = opening_get_soma_region(soma_region)
        soma_region = self.opening_get_soma_region_gpu(soma_region)
        soma_region = self.dusting(soma_region)
        # restore original size
        soma_region = self.restore_original_size(soma_region, original_shape, min_coords)

        return soma_region

    def get_soma_regions_file(self, file_name, tif_folder, soma_folder):
        if ("gsdt" in file_name):
            os.remove(file_name)
            return
        tif_path = os.path.join(tif_folder, file_name)
        soma_region_path = os.path.join(soma_folder, os.path.splitext(file_name)[0] + '.tif')
        # muti_soma_marker_path = os.path.join(muti_soma_marker_folder, os.path.splitext(file_name)[0] + '.marker')
        # muti_soma_marker_path = find_muti_soma_marker_file(file_name, muti_soma_marker_folder)

        if (os.path.exists(soma_region_path)):
            return
        try:
            soma_region = self.get_soma_region(tif_path)
        except:
            return
        if (soma_region is None):
            return
        # binary
        soma_region = np.where(soma_region > 0, 1, 0).astype("uint8")
        tifffile.imwrite(soma_region_path, soma_region * 255, compression='zlib')

        del soma_region