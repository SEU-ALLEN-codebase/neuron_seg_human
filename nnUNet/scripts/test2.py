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

from nnUNet.scripts.resolution_unifier import swc_file
from simple_swc_tool.swc_io import read_swc, write_swc

from scipy import ndimage
# from nnunetv2.training.loss.fastanison import anisodiff3
import pandas as pd
from pylib.file_io import load_image
from simple_swc_tool.get_soma_region_from_seg import SomaRegionFinder, SmartSomaRegionFinder

import concurrent.futures
from tqdm import tqdm

v3d_path = r"/home/kfchen/Vaa3D-x.1.1.4_Ubuntu/Vaa3D-x"
MAX_PROCESSES = 16

import os
import inspect


class FileProcessingStep:
    def __init__(self, step_name, process_function, file_name, input_dirs, output_dirs=None, supplementary_files=None):
        self.step_name = step_name
        self.process_function = process_function  # Function that performs the processing
        self.file_name = file_name  # Common file name for inputs and outputs
        self.input_dirs = input_dirs  # List of directories containing input files
        self.output_dirs = output_dirs  # Directory to store the output files
        self.supplementary_files = supplementary_files if supplementary_files else []  # List of supplementary files

    def count_function_parameters(self, function):
        signature = inspect.signature(function)
        return len(signature.parameters)

    def execute(self):
        # Create output directory if it doesn't exist
        for output_dir in self.output_dirs:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)

        # Collect input files with the specified file_name from input directories
        input_files = [os.path.join(input_dir, self.file_name) for input_dir in self.input_dirs]
        for i in range(len(input_files)):
            if(self.input_dirs[i].endswith('swc')):
                input_files[i] = input_files[i].replace('.tif', '.swc')
            elif(self.input_dirs[i].endswith('marker')):
                input_files[i] = input_files[i].replace('.tif', '.marker')

        output_files = [os.path.join(output_dir, self.file_name) for output_dir in self.output_dirs]
        for i in range(len(output_files)):
            if(self.output_dirs[i].endswith('swc')):
                output_files[i] = output_files[i].replace('.tif', '.swc')
            elif(self.output_dirs[i].endswith('marker')):
                output_files[i] = output_files[i].replace('.tif', '.marker')

        if(self.count_function_parameters(self.process_function) == 2):
            self.process_function(input_files, output_files)
        elif(self.count_function_parameters(self.process_function) == 3):
            self.process_function(input_files, output_files, self.supplementary_files)
        else:
            raise ValueError("The number of parameters of the process function must be 2 or 3.")


class FileProcessingPipeline:
    def __init__(self, root_dir, file_name, supplementary_files=None):
        if not root_dir:
            raise ValueError("Root directory must be specified.")
        self.root_dir = root_dir
        self.steps = []
        self.file_name = file_name
        self.supplementary_files = supplementary_files

    def add_step(self, step_name, process_function, input_steps, output_steps, supplementary_files=None):
        # Define the output directory for the step
        # output_dirs = os.path.join(self.root_dir, step_name)
        input_dirs = [os.path.join(self.root_dir, input_step) for input_step in input_steps]
        output_dirs = [os.path.join(self.root_dir, output_step) for output_step in output_steps]
        step = FileProcessingStep(step_name, process_function, self.file_name, input_dirs, output_dirs,
                                  supplementary_files)
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
    # print(origin_seg_file)
    if((not origin_seg_file.endswith('.tif')) and (not origin_seg_file.endswith('.nii.gz'))):
        return

    if(not os.path.exists(seg_dir)):
        os.makedirs(seg_dir, exist_ok=True)

    # copy the seg file to the output directory
    file_name = os.path.basename(origin_seg_file)
    if (file_name.endswith('.nii.gz')):
        seg = sitk.ReadImage(origin_seg_file)
        seg = sitk.GetArrayFromImage(seg)
        seg = (np.where(seg > 0, 1, 0) * 255).astype("uint8")
        file_name = file_name.replace('_pred0.nii.gz', '.tif')
        tifffile.imwrite(os.path.join(seg_dir, file_name), seg, compression='zlib')
    else:
        seg = tifffile.imread(origin_seg_file)
        seg = (np.where(seg > 0, 1, 0) * 255).astype("uint8")
        tifffile.imwrite(os.path.join(seg_dir, file_name), seg, compression='zlib')

    df = pd.read_csv(name_mapping_file)
    full_name = get_full_name(file_name.split('.')[0], df)
    if full_name is None:
        print(f"Error: {file_name} not found in the name mapping file.")
        return
    # rename the seg file
    new_file_path = os.path.join(seg_dir, full_name)
    os.rename(os.path.join(seg_dir, file_name), new_file_path)
    # print(new_file_path, "ok")


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
    def __init__(self, root_dir, file_name, supplementary_files=None):
        super().__init__(root_dir, file_name, supplementary_files)

        self.name_mapping_file = supplementary_files[0]
        self.neuron_info_file = supplementary_files[1]
        self.name_mapping_df = pd.read_csv(self.name_mapping_file)
        self.neuron_info_df = pd.read_csv(self.neuron_info_file, encoding='gbk')
        self.add_step("1_skel_seg", self.skel_seg, ["0_seg"], ["1_skel_seg"])
        self.add_step('2_soma_region', self.get_soma_region, ['0_seg'], ['2_soma_region', "4_soma_marker"])
        self.add_step("3_skel_with_soma", self.skel_with_soma, ["1_skel_seg", "2_soma_region"], ["3_skel_with_soma"])
        # self.add_step("4_soma_marker", self.get_soma_marker, ["2_soma_region"])

        self.add_step("5_swc", self.trace_app2_with_soma, ["3_skel_with_soma", "4_soma_marker"], ["5_swc"])

        self.add_step("6_reconnect_soma", self.connect_to_soma_file, ["5_swc", "2_soma_region"], ["6_connect_soma_swc"])

        self.add_step("7_rescale_xy_resolution", self.rescale_xy_resolution, ["6_connect_soma_swc"], ["7_scaled_1um_swc"])

    def find_resolution(self):
        # print(filename)
        filename = self.file_name
        df = self.neuron_info_df
        filename = int(filename.split('.')[0])
        for i in range(len(df)):
            if int(df.iloc[i, 0]) == filename:
                return df.iloc[i, 43]
        return None
    def get_origin_img_size(self):
        df = self.name_mapping_df
        # print(self.name_mapping_file)
        full_name = self.file_name.replace('.swc', '').replace('.tif', '').replace('image_', '')
        # print(full_name)
        img_size = df[df['ID'] == int(full_name)]['img_size'].values[0]
        img_size = img_size.split(',')
        x_limit, y_limit, z_limit = img_size[2], img_size[1], img_size[0]
        x_limit, y_limit, z_limit = "".join(filter(str.isdigit, x_limit)), \
            "".join(filter(str.isdigit, y_limit)), \
            "".join(filter(str.isdigit, z_limit))
        origin_size = (int(z_limit), int(y_limit), int(x_limit))
        return origin_size

    def skel_seg(self, input_files, output_files):
        if(not input_files):
            return
        seg_file = input_files[0]
        skel_file = output_files[0]

        if ((not os.path.exists(seg_file)) or os.path.exists(skel_file)):
            return

        data = tifffile.imread(seg_file).astype("uint8")
        data = np.flip(data, axis=1)
        origin_size = self.get_origin_img_size()
        data = skimage.transform.resize(data, origin_size, order=0, anti_aliasing=False)

        skel = skeletonize_3d(data).astype("uint8")
        # skel = binary_dilation(skel, iterations=1).astype("uint8")
        tifffile.imwrite(skel_file, skel * 255, compression='zlib')

    def get_soma_region(self, input_files, output_files):
        def clear_gsdt_file(seg_file):
            input_dir = os.path.dirname(seg_file)
            for f in os.listdir(input_dir):
                if 'gsdt' in f and os.path.basename(seg_file).split('.')[0] in f:
                    os.remove(os.path.join(input_dir, f))

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

        if(not input_files):
            return
        seg_file = input_files[0]
        soma_region_file = output_files[0]
        soma_marker_file = output_files[1]

        if ((not os.path.exists(seg_file)) or os.path.exists(soma_region_file)):
            return


        ''' SomaRegionFinder '''
        # # try:
        # somaregionfinder = SomaRegionFinder()
        # soma_region = somaregionfinder.get_soma_region(seg_file)
        # # except:
        # #     clear_gsdt_file(seg_file)
        # #     return
        # if(soma_region is None):
        #     clear_gsdt_file(seg_file)
        #     return
        # soma_region = np.where(soma_region > 0, 1, 0).astype("uint8")
        # tifffile.imwrite(soma_region_file, soma_region * 255, compression='zlib')
        # clear_gsdt_file(seg_file)

        ''' SmartSomaRegionFinder '''
        df = self.name_mapping_df
        somaregionfinder = SmartSomaRegionFinder()
        soma_region = somaregionfinder.test_soma_detectison([seg_file, df, soma_region_file])

        centroid = compute_centroid(soma_region)
        soma_x, soma_y, soma_z, soma_r = centroid[2], centroid[1], centroid[0], 1
        # soma_y = soma_region.shape[1] - soma_y - 1
        origin_size = self.get_origin_img_size()
        new_size = soma_region.shape
        x_ratio, y_ratio, z_ratio = origin_size[0] / new_size[0], origin_size[1] / new_size[1], origin_size[2] / \
                                    new_size[2]
        # print(x_ratio, y_ratio, z_ratio)


        marker_str = f"{float(soma_x) * x_ratio}, {float(soma_y) * y_ratio}, {float(soma_z) * z_ratio}, {soma_r}, 1, , , 255,0,0"
        with open(soma_marker_file, 'w') as f:
            f.write(marker_str)

        soma_region = np.where(soma_region > 0, 1, 0).astype("uint8")
        soma_region = np.flip(soma_region, axis=1)
        origin_size = self.get_origin_img_size()
        soma_region = skimage.transform.resize(soma_region, origin_size, order=0, anti_aliasing=False)

        tifffile.imwrite(soma_region_file, soma_region * 255, compression='zlib')


    def skel_with_soma(self, input_files, output_files):
        if(not input_files):
            return
        skel_file = input_files[0]
        soma_region_file = input_files[1]
        skel_with_soma_file = output_files[0]

        if ((not os.path.exists(skel_file)) or (not os.path.exists(soma_region_file)) or os.path.exists(skel_with_soma_file)):
            return

        skel = tifffile.imread(skel_file).astype("uint8")
        soma = tifffile.imread(soma_region_file).astype("uint8")
        skelwithsoma = (np.logical_or(skel, soma))
        # normalize to [0, 255]
        skelwithsoma = (skelwithsoma * 255).astype("uint8")
        tifffile.imwrite(skel_with_soma_file, skelwithsoma, compression='zlib')

    def get_soma_marker(self, input_files, output_files):
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
        if(not input_files):
            return
        soma_path = input_files[0]
        somamarker_path = output_files[0]

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

    def trace_app2_with_soma(self, input_files, output_files):
        def process_path(path):
            return path.replace('\\', '/')
        if(not input_files):
            return
        img_file = input_files[0]
        somamarker_file = input_files[1]
        swc_file = output_files[0]
        ini_swc_path = img_file.replace('.tif', '.tif_ini.swc')
        if (os.path.exists(swc_file) or (not os.path.exists(img_file)) or (not os.path.exists(somamarker_file))):
            return
        '''
            **** Usage of APP2 ****
            vaa3d -x plugin_name -f app2 -i <inimg_file> -o <outswc_file> -p [<inmarker_file> [<channel> [<bkg_thresh> 
            [<b_256cube> [<b_RadiusFrom2D> [<is_gsdt> [<is_gap> [<length_thresh> [is_resample][is_brightfield][is_high_intensity]]]]]]]]]
            inimg_file          Should be 8/16/32bit image
            inmarker_file       If no input marker file, please set this para to NULL and it will detect soma automatically.
                                When the file is set, then the first marker is used as root/soma.
            channel             Data channel for tracing. Start from 0 (default 0).
            bkg_thresh          Default 10 (is specified as AUTO then auto-thresolding)
            b_256cube           If trace in a auto-downsampled volume (1 for yes, and 0 for no. Default 1.)
            b_RadiusFrom2D      If estimate the radius of each reconstruction node from 2D plane only (1 for yes as many 
            times the data is anisotropic, and 0 for no. Default 1 which which uses 2D estimation.)
            is_gsdt             If use gray-scale distance transform (1 for yes and 0 for no. Default 0.)
                           If allow gap (1 for yes and 0 for no. Default 0.)
            length_thresh       Default 5
            is_resample         If allow resample (1 for yes and 0 for no. Default 1.)
            is_brightfield      If the signals are dark instead of bright (1 for yes and 0 for no. Default 0.)
            is_high_intensity   If the image has high intensity background (1 for yes and 0 for no. Default 0.)
            outswc_file         If not be specified, will be named automatically based on the input image file name.
        '''

        resample = 1
        gsdt = 1
        b_RadiusFrom2D = 1

        if (not os.path.exists(somamarker_file)):
            somamarker_path = "NULL"

        try:
            if (sys.platform == "linux"):
                cmd = f'xvfb-run -a -s "-screen 0 640x480x16" {v3d_path} -x vn2 -f app2 -i {img_file} -o {swc_file} -p {somamarker_file} 0 10 1 {b_RadiusFrom2D} {gsdt} 1 {resample} 0 0'
                cmd = process_path(cmd)
                subprocess.run(cmd, stdout=subprocess.DEVNULL, shell=True)
            else:
                pass
        except:
            pass

        if (os.path.exists(ini_swc_path)):
            os.remove(ini_swc_path)

    def prune_fiber_in_soma(self, point_l, soma_region):
        edge_p_list = []
        x_limit, y_limit, z_limit = soma_region.shape[2], soma_region.shape[1], soma_region.shape[0]

        for p in point_l.p:
            if (p.n == 0 or p.n == 1): continue
            x, y, z = p.x, p.y, p.z
            y = soma_region.shape[1] - y

            x = min(int(x), x_limit - 1)
            y = min(int(y), y_limit - 1)
            z = min(int(z), z_limit - 1)

            if (soma_region[int(z), int(y), int(x)]):
                edge_p_list.append(p)

        for p in edge_p_list:
            if (len(p.s) == 0):
                temp_p = point_l.p[p.n]
                while (True):
                    if (temp_p.n == 1): break
                    if (temp_p.pruned == True): break
                    if (not len(temp_p.s) == 1): break
                    point_l.p[temp_p.n].pruned = True
                    temp_p = point_l.p[temp_p.p]
        for p in point_l.p:
            for s in p.s:
                if (point_l.p[s].pruned == True):
                    p.s.remove(s)

        return point_l

    def connect_to_soma_file(self, input_files, output_files):
        if (not input_files):
            return
        swc_file = input_files[0]
        soma_refion_file = input_files[1]
        conn_swc_file = output_files[0]

        if ((not os.path.exists(swc_file)) or (not os.path.exists(soma_refion_file)) or os.path.exists(conn_swc_file)):
            return

        soma_region = tifffile.imread(soma_refion_file).astype("uint8")
        # soma_region = get_main_soma_region_in_msoma_from_gsdt(soma_region,,
        soma_region = binary_dilation(soma_region, iterations=4).astype("uint8")
        x_limit, y_limit, z_limit = soma_region.shape[2], soma_region.shape[1], soma_region.shape[0]

        point_l = read_swc(swc_file)

        labeled_img, num_objects = ndimage.label(soma_region)
        if (len(point_l.p) <= 1):
            return
        for obj_id in range(1, num_objects + 1):
            obj_img = np.where(labeled_img == obj_id, 1, 0)
            x, y, z = point_l.p[1].x, point_l.p[1].y, point_l.p[1].z
            if (obj_img[int(z), int(y), int(x)]):
                soma_region = obj_img
                del obj_img
                break
            del obj_img
        # tifffile.imwrite(os.path.join(conn_folder, file_name+"1.tif"), soma_region.astype("uint8")*255, compression='zlib')
        labeled_img, num_objects = ndimage.label(soma_region)
        if (num_objects > 1):
            # soma_region = dusting(soma_region)
            write_swc(conn_swc_file, point_l)
            del soma_region, point_l
            return

        point_l = self.prune_fiber_in_soma(point_l, soma_region)

        # strict strategy
        edge_p_list = []
        for p in point_l.p:
            if (p.n == 0 or p.n == 1): continue
            x, y, z = p.x, p.y, p.z
            y = soma_region.shape[1] - y

            x = min(int(x), x_limit - 1)
            y = min(int(y), y_limit - 1)
            z = min(int(z), z_limit - 1)

            if (soma_region[int(z), int(y), int(x)]):
                edge_p_list.append(p)

        for p in edge_p_list:
            temp_p = point_l.p[p.p]
            while (True):
                if (temp_p.n == 1): break
                if (temp_p.pruned == True): break
                point_l.p[temp_p.n].pruned = True
                temp_p = point_l.p[temp_p.p]

        for p in edge_p_list:
            if (point_l.p[p.n].pruned == False):
                if (not len(point_l.p[p.n].s)):
                    point_l.p[p.n].pruned = True
                else:
                    point_l.p[p.n].p = 1
                    point_l.p[1].s.append(p.n)
            else:
                for s in point_l.p[p.n].s:
                    point_l.p[s].p = 1
                    point_l.p[1].s.append(s)

        # Conservative strategy
        # for s in point_l.p[1].s:
        #     temp_p = point_l.p[s]
        #     x, y, z = temp_p.x, temp_p.y, temp_p.z
        #     y = soma_region.shape[1] - y
        #     x = min(int(x), x_limit - 1)
        #     y = min(int(y), y_limit - 1)
        #     z = min(int(z), z_limit - 1)
        #
        #     if(not soma_region[int(z), int(y), int(x)]):
        #         continue
        #
        #     for s2 in point_l.p[s].s:
        #         point_l.p[s2].p = 1
        #         point_l.p[1].s.append(s2)
        #
        #     point_l.p[1].s.remove(s)
        #     point_l.p[s].pruned = True
        # print(conn_path)
        if (os.path.exists(conn_swc_file)):
            os.remove(conn_swc_file)
        write_swc(conn_swc_file, point_l)
        # print(len(point_l.p))
        del soma_region, point_l

    def rescale_xy_resolution(self, input_files, output_files):
        if(not input_files):
            return

        swc_file = input_files[0]
        unified_swc_file = output_files[0]

        if(not os.path.exists(swc_file) or os.path.exists(unified_swc_file)):
            return

        xy_resolution = self.find_resolution()
        # print(xy_resolution)
        with open(swc_file, 'r') as f:
            lines = f.readlines()
        # print(len(lines))
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

def process_pipeline(file_name):
    pipeline = AutoTracePipeline(work_dir, file_name, [name_mapping_file, neuron_info_file])
    pipeline.run()  # 如果需要执行任务，可以取消注释这行
    return file_name

if __name__ == "__main__":
    net_work_list = ["nnunet"]
    loss_list = ['baseline', 'cldice', 'skelrec', 'newcel_0.1']
    work_dir_list = [f"/data/kfchen/trace_ws/paper_trace_result/nnunet/{loss}/" for loss in loss_list]

    for work_dir in work_dir_list:
        print(f"Processing {work_dir}")
        # work_dir = "/data/kfchen/trace_ws/paper_trace_result/nnunet/newcel_0.1/"
        origin_seg_dir = os.path.join(work_dir, "origin_seg")# "origin_seg"


        raw_dataset_dir = r"/data/kfchen/nnUNet/nnUNet_raw/Dataset180_deflu_gamma"
        seg_dir = os.path.join(work_dir, "0_seg")
        name_mapping_file = os.path.join(raw_dataset_dir, "name_mapping.csv")
        neuron_info_file = "/data/kfchen/nnUNet/nnUNet_results/Dataset169_hb_10k/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/ptls10/norm_result/Human_SingleCell_TrackingTable_20240712.csv"

        if(not os.path.exists(seg_dir)):
            prepare_seg_files(origin_seg_dir, seg_dir, name_mapping_file)

        print("Data preparation is done.")
        # pipeline_list = []
        file_names = [f for f in os.listdir(seg_dir) if f.endswith('.tif')]
        # file_names = file_names[:10]
        # for file_name in file_names:
        #     pipeline = AutoTracePipeline(work_dir, file_name, [name_mapping_file, neuron_info_file])
        #     pipeline_list.append(pipeline)
        #     # pipeline.run()

        with concurrent.futures.ThreadPoolExecutor() as executor:
            # 使用tqdm添加进度条
            results = list(tqdm(executor.map(process_pipeline, file_names), total=len(file_names)))









