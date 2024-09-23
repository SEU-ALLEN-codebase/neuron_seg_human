import glob
import os
import subprocess
import time

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tqdm import tqdm

from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from simple_swc_tool.swc_io import read_swc

from nnUNet.scripts.mip import get_mip_swc, get_mip
from nnUNet.nnunetv2.dataset_conversion.generate_nnunet_dataset import augment_gamma
import tifffile
import numpy as np

def calc_tips_to_soma(swc_file):
    file1_tip_to_soma_dist_list = []
    point_l1 = read_swc(swc_file)

    for p1 in point_l1.p:
        if (p1.n == 0 or p1.n == 1):
            continue
        if (len(p1.s) == 0):  # tip
            file1_tip_to_soma_dist_list.append(point_l1.calc_p_to_soma(p1.n))

    return file1_tip_to_soma_dist_list



def compare_tip_to_soma(traced_dir1 = r"/data/kfchen/trace_ws/result500_new_resized_test_noptls/connswc",
                        traced_dir2 = r"/data/kfchen/trace_ws/result500_new_resized_test_ptls/connswc"):
    dir1_files = glob.glob(os.path.join(traced_dir1, '*swc'))
    dir2_files = glob.glob(os.path.join(traced_dir2, '*swc'))
    dir1_files.sort()
    dir2_files.sort()

    dir1_ids = [int(os.path.split(f)[-1].split('_')[0].split('.')[0]) for f in dir1_files]
    dir2_ids = [int(os.path.split(f)[-1].split('_')[0].split('.')[0]) for f in dir2_files]
    shared_ids = list(set(dir1_ids) & set(dir2_ids))

    dir1_mean_tip_to_soma_dist_list = []
    dir2_mean_tip_to_soma_dist_list = []
    better_list = []

    for idx in shared_ids:
        dir1_swc_file = [f for f, id in zip(dir1_files, dir1_ids) if id == idx][0]
        dir2_swc_file = [f for f, id in zip(dir2_files, dir2_ids) if id == idx][0]

        point_l1 = read_swc(dir1_swc_file)
        point_l2 = read_swc(dir2_swc_file)

        file1_tip_to_soma_dist_list = []
        file2_tip_to_soma_dist_list = []

        for p1 in point_l1.p:
            if(p1.n == 0 or p1.n == 1):
                continue
            if(len(p1.s) == 0): # tip
                file1_tip_to_soma_dist_list.append(point_l1.calc_p_to_soma(p1.n))

        for p2 in point_l2.p:
            if (p2.n == 0 or p2.n == 1):
                continue
            if(len(p2.s) == 0): # tip
                file2_tip_to_soma_dist_list.append(point_l2.calc_p_to_soma(p2.n))

        # print(len(file1_tip_to_soma_dist_list), len(file2_tip_to_soma_dist_list))
        mean1 = sum(file1_tip_to_soma_dist_list) / len(file1_tip_to_soma_dist_list)
        mean2 = sum(file2_tip_to_soma_dist_list) / len(file2_tip_to_soma_dist_list)

        dir1_mean_tip_to_soma_dist_list.append(mean1)
        dir2_mean_tip_to_soma_dist_list.append(mean2)

        print(mean1, mean2)
        if(mean1 < mean2):
            better_list.append(1)
        else:
            better_list.append(0)

    print(f"mean dir1_mean_tip_to_soma_dist_list: {sum(dir1_mean_tip_to_soma_dist_list) / len(dir1_mean_tip_to_soma_dist_list)}")
    print(f"mean dir2_mean_tip_to_soma_dist_list: {sum(dir2_mean_tip_to_soma_dist_list) / len(dir2_mean_tip_to_soma_dist_list)}")
    print("better rate: ", sum(better_list) / len(better_list))
    print(len(better_list))