import os
import shutil
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
from simple_swc_tool.get_soma_region_from_seg import SomaRegionFinder

MAX_PROCESSES = 16

import os


class FileProcessingStep:
    def __init__(self, step_name, process_function, file_name, input_dirs, output_dir=None, supplementary_files=None):
        self.step_name = step_name
        self.process_function = process_function  # Function that performs the processing
        self.file_name = file_name  # Common file name for inputs and outputs
        self.input_dirs = input_dirs  # List of directories containing input files
        self.output_dir = output_dir  # Directory to store the output files
        self.supplementary_files = supplementary_files if supplementary_files else []  # List of supplementary files

    def execute(self):
        # Create output directory if it doesn't exist
        if(self.output_dir is not None):
            os.makedirs(self.output_dir, exist_ok=True)

        # Collect input files with the specified file_name from input directories
        input_files = [os.path.join(input_dir, self.file_name) for input_dir in self.input_dirs]
        output_file = os.path.join(self.output_dir, self.file_name)

        self.process_function(input_files, output_file, self.supplementary_files)


class FileProcessingPipeline:
    def __init__(self, root_dir, file_name):
        if not root_dir:
            raise ValueError("Root directory must be specified.")
        self.root_dir = root_dir
        self.steps = []
        self.file_name = file_name

    def add_step(self, step_name, process_function, input_steps, supplementary_files=None):
        # Define the output directory for the step
        output_dir = os.path.join(self.root_dir, step_name)
        input_dirs = [os.path.join(self.root_dir, input_step) for input_step in input_steps]
        step = FileProcessingStep(step_name, process_function, self.file_name, input_dirs, output_dir, supplementary_files)
        self.steps.append(step)

    def run(self):
        if not self.steps:
            print("No steps to run in the pipeline.")
            return

        for step in self.steps:
            step.execute()


def copy_file_from_seg_result(origin_seg_file, seg_dir, name_mapping_file):
    def get_full_name(file_name, df):
        full_name = df[df['nnunet_name'] == file_name]['full_name']
        if (full_name.empty):
            return None
        else:
            return str(full_name.values[0])

    if((not origin_seg_file.endswith('.tif')) or (not origin_seg_file.endswith('.nii.gz'))):
        return
    # copy the seg file to the output directory
    file_name = os.path.basename(origin_seg_file)
    if(file_name.endswith('.nii.gz')):
        file_name = file_name.replace('.nii.gz', '.tif')
    shutil.copy(origin_seg_file, seg_dir)

    df = pd.read_csv(name_mapping_file)
    full_name = get_full_name(file_name.split('.')[0], df)
    if full_name is None:
        print(f"Error: {file_name} not found in the name mapping file.")
        return
    # rename the seg file
    new_file_path = os.path.join(seg_dir, full_name)
    if(file_name.endswith('.nii.gz')):
        new_file_name = new_file_path.replace('.tif', '.nii.gz')
    os.rename(os.path.join(seg_dir, file_name), new_file_path)

    # to uint8
    if(file_name.endswith('.tif')):
        seg = tifffile.imread(new_file_path)
        seg = (np.where(seg > 0, 1, 0) * 255).astype("uint8")
    elif(file_name.endswith('.nii.gz')):
        seg = sitk.ReadImage(new_file_path)
        seg = sitk.GetArrayFromImage(seg)
        seg = (np.where(seg > 0, 1, 0) * 255).astype("uint8")
        new_file_path = new_file_path.replace('.nii.gz', '.tif')
    tifffile.imsave(new_file_path, seg, compression='zlib')

def prepare_seg_files(origin_seg_dir, seg_dir, name_mapping_file):
    origin_seg_files = [os.path.join(origin_seg_dir, f) for f in os.listdir(origin_seg_dir)]
    # for origin_seg_file in origin_seg_files:
    #     copy_file_from_seg_result(origin_seg_file, seg_dir, name_mapping_file)

    # 多线程
    pool = Pool(MAX_PROCESSES)
    func = partial(copy_file_from_seg_result, seg_dir=seg_dir, name_mapping_file=name_mapping_file)
    pool.map(func, origin_seg_files)
    pool.close()
    pool.join()





class AutoTracePipeline(FileProcessingPipeline):
    def __init__(self, root_dir, file_name):
        super().__init__(root_dir, file_name)

        self.add_step("1_skel_seg", self.skel_seg, ["0_seg"])
        self.add_step('2_soma_region', self.get_soma_region, ['0_seg'])
        self.add_step("3_skel_with_soma", self.skel_with_soma, ["1_skel_seg", "2_soma_region"])
        self.add_step("4_soma_marker", self.get_soma_marker, ["2_soma_region"])



    def skel_seg(self, input_files, output_file):
        seg_file = input_files[0]
        skel_file = output_file

        if ((not os.path.exists(seg_file)) or os.path.exists(skel_file)):
            return

        data = tifffile.imread(seg_file).astype("uint8")
        skel = skeletonize_3d(data).astype("uint8")
        # skel = binary_dilation(skel, iterations=1).astype("uint8")
        tifffile.imwrite(skel_file, skel * 255, compression='zlib')

    def get_soma_region(self, input_files, output_file):
        def clear_gsdt_file(input_dir):
            for f in os.listdir(input_dir):
                if 'gsdt' in f:
                    os.remove(os.path.join(input_dir, f))

        seg_file = input_files[0]
        soma_region_file = output_file

        if ((not os.path.exists(seg_file)) or os.path.exists(soma_region_file)):
            return

        try:
            soma_region = SomaRegionFinder.get_soma_region(seg_file)
        except:
            clear_gsdt_file(os.path.dirname(seg_file))
            return
        if(soma_region is None):
            clear_gsdt_file(os.path.dirname(seg_file))
            return
        soma_region = np.where(soma_region > 0, 1, 0).astype("uint8")
        tifffile.imwrite(soma_region_file, soma_region * 255, compression='zlib')
        clear_gsdt_file(os.path.dirname(seg_file))

    def skel_with_soma(self, input_files, output_file):
        skel_file = input_files[0]
        soma_region_file = input_files[1]
        skel_with_soma_file = output_file

        if ((not os.path.exists(skel_file)) or (not os.path.exists(soma_region_file)) or os.path.exists(skel_with_soma_file)):
            return

        skel = tifffile.imread(skel_file).astype("uint8")
        soma = tifffile.imread(soma_region_file).astype("uint8")
        skelwithsoma = (np.logical_or(skel, soma))
        # normalize to [0, 255]
        skelwithsoma = (skelwithsoma * 255).astype("uint8")
        tifffile.imwrite(skel_with_soma_file, skelwithsoma, compression='zlib')

    def get_soma_marker(self, input_files, output_file):
        def compute_centroid(mask):
            # 计算三维 mask 的重心
            labeled_mask = skimage.measure.label(mask)
            props = regionprops(labeled_mask)

            if len(props) > 0:
                # 获取第一个区域的重心坐标
                centroid = props[0].centroid
                return centroid
            else:
                return None
        soma_path = input_files[0]
        somamarker_path = output_file

        if ((not os.path.exists(soma_path)) or os.path.exists(somamarker_path)):
            return

        soma_region = tifffile.imread(soma_path).astype("uint8")
        centroid = compute_centroid(soma_region)
        soma_x, soma_y, soma_z, soma_r = centroid[2], centroid[1], centroid[0], 1
        soma_y = soma_region.shape[1] - soma_y
        # print(soma_x, soma_y, soma_z, soma_r)
        marker_str = f"{soma_x}, {soma_y}, {soma_z}, {soma_r}, 1, , , 255,0,0"
        with open(somamarker_path, 'w') as f:
            f.write(marker_str)

    def trace_


if __name__ == "__main__":
    origin_seg_dir = r"/data/kfchen/trace_ws/paper_trace_result/nnunet/baseline/origin_seg"
    raw_dataset_dir = r"/data/kfchen/nnUNet/nnUNet_raw/Dataset180_deflu_gamma"

    # work_dir = os.path.join(r"/data/kfchen/trace_ws", origin_seg_dir.split('/')[-1])
    work_dir = "/data/kfchen/trace_ws/paper_trace_result/nnunet/baseline"
    seg_dir = os.path.join(work_dir, "0_seg")
    name_mapping_file = os.path.join(work_dir, "name_mapping.csv")
    prepare_seg_files(origin_seg_dir, seg_dir, name_mapping_file)

    pipeline_list = []
    file_names = [f for f in os.listdir(seg_dir) if f.endswith('.tif')]
    for file_name in file_names:
        pipeline = AutoTracePipeline(work_dir)
        pipeline_list.append(pipeline)

    #多线程
    pool = Pool(MAX_PROCESSES)
    pool.map(FileProcessingPipeline.run, pipeline_list)
    pool.close()
    pool.join()






