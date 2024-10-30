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

MAX_PROCESSES = 16

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
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # Collect input files with the specified file_name from input directories
        input_files = []
        for input_dir in self.input_dirs:
            input_file_path = os.path.join(input_dir, self.file_name)
            if os.path.exists(input_file_path):
                input_files.append(input_file_path)

        # Run the process function on the collected input files and supplementary files
        if input_files:
            self.process_function(self.file_name, input_files, self.output_dir, self.supplementary_files)

        print(f"Step '{self.step_name}' completed. Output saved to '{self.output_dir}'.")

class FileProcessingPipeline:
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.steps = []

    def add_step(self, step_name, process_function, file_name, input_dirs, supplementary_files=None):
        # Define the output directory for the step
        output_dir = os.path.join(self.root_dir, step_name)
        step = FileProcessingStep(step_name, process_function, file_name, input_dirs, output_dir, supplementary_files)
        self.steps.append(step)

    def run(self):
        for step in self.steps:
            step.execute()


def copy_file_from_seg_result(origin_seg_file, seg_dir, name_mapping_file):
    def get_full_name(file_name, df):
        full_name = df[df['nnunet_name'] == file_name]['full_name']
        if (full_name.empty):
            return None
        else:
            return str(full_name.values[0])

    # copy the seg file to the output directory
    file_name = os.path.basename(origin_seg_file)
    shutil.copy(origin_seg_file, seg_dir)

    df = pd.read_csv(name_mapping_file)
    full_name = get_full_name(file_name.split('.')[0], df)
    if full_name is None:
        print(f"Error: {file_name} not found in the name mapping file.")
        return
    # rename the seg file
    new_file_name = os.path.join(seg_dir, full_name)
    os.rename(os.path.join(seg_dir, file_name), new_file_name)

    # to uint8
    seg = tifffile.imread(new_file_name)
    seg = (np.where(seg > 0, 1, 0) * 255).astype("uint8")
    tifffile.imsave(new_file_name, seg, compression='zlib')

def prepare_seg_files(origin_seg_dir, seg_dir, name_mapping_file):
    origin_seg_files = [os.path.join(origin_seg_dir, f) for f in os.listdir(origin_seg_dir) if f.endswith('.tif')]
    # for origin_seg_file in origin_seg_files:
    #     copy_file_from_seg_result(origin_seg_file, seg_dir, work_dir + '/name_mapping.csv')

    # 多线程
    pool = Pool(MAX_PROCESSES)
    func = partial(copy_file_from_seg_result, seg_dir=seg_dir, name_mapping_file=name_mapping_file)
    pool.map(func, origin_seg_files)
    pool.close()
    pool.join()

def skel_seg_file(file_name, input_files, output_dir):
    seg_file = input_files[0]
    skel_file = os.path.join(output_dir, file_name)

    if(os.path.exists(skel_file)):
        return

    data = tifffile.imread(seg_file).astype("uint8")
    skel = skeletonize_3d(data).astype("uint8")
    # skel = binary_dilation(skel, iterations=1).astype("uint8")
    tifffile.imwrite(skel_file, skel * 255, compression='zlib')

class AutoTracePipeline(FileProcessingPipeline):
    def __init__(self, root_dir):
        super().__init__(root_dir)

        self.add_step("skel_seg", skel_seg_file, "seg.tif", [seg_dir])


if __name__ == "__main__":
    origin_seg_dir = r"/data/kfchen/nnUNet/nnUNet_results/Dataset180_deflu_gamma/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/ptls10/14k_result"
    raw_dataset_dir = r"/data/kfchen/nnUNet/nnUNet_raw/Dataset170_14k_hb_neuron"

    work_dir = os.path.join(r"/data/kfchen/trace_ws", origin_seg_dir.split('/')[-1])
    seg_dir = os.path.join(work_dir, "seg")
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






