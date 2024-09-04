import os

import cv2

from pylib import file_io
import numpy as np
from simple_swc_tool.soma_detection import simple_get_soma
import platform
import subprocess
import scipy.ndimage
import tifffile
import skimage
import cc3d
from skimage.morphology import binary_dilation
from scipy.ndimage import binary_fill_holes, label, binary_dilation
from simple_swc_tool.swc_io import read_swc, write_swc
import math
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import SimpleITK as sitk
from nnUNet.scripts.mip import get_mip_swc

def flip_swc(img_path, swc_path, out_path=""):
    if (not out_path):
        out_path = swc_path[0:-4] + "_flip.swc"
    if (os.path.exists(out_path)):
        """debug mode off"""
        return out_path
        os.remove(out_path)

    img = file_io.load_image(img_path)
    y_limit = img.shape[1]

    with open(swc_path, 'r') as f:
        lines = f.readlines()
    res_lines = []
    for line in lines:
        if (line[0] == '#'): continue
        temp_line = line.split()
        temp_line[2] = str(temp_line[2])
        temp_line[3] = str(y_limit - float(temp_line[3]))
        temp_line[4] = str(temp_line[4])
        temp_line[5] = str(temp_line[5])
        res_line = "%s %s %s %s %s %s %s\n" % (
            temp_line[0], temp_line[1], temp_line[2],
            temp_line[3],
            temp_line[4], temp_line[5], temp_line[6]
        )
        res_lines.append(res_line)

    file_handle = open(out_path, mode="a")
    file_handle.writelines(res_lines)
    file_handle.close()

    return out_path

def flip_and_resize_swc(img_path, swc_path, scale_factors, pad_width, out_path=""):
    if (not out_path):
        out_path = swc_path[0:-4] + "_flip.swc"
    if (os.path.exists(out_path)):
        """debug mode off"""
        # return out_path
        os.remove(out_path)

    img = file_io.load_image(img_path)
    y_limit = img.shape[1]

    with open(swc_path, 'r') as f:
        lines = f.readlines()
    res_lines = []
    for line in lines:
        if (line[0] == '#'): continue
        temp_line = line.split()
        temp_line[2] = str((float(temp_line[2]) + pad_width[1]) * scale_factors[2])
        temp_line[3] = str(y_limit - (float(temp_line[3]) + pad_width[0]) * scale_factors[1])
        temp_line[4] = str(float(temp_line[4]) * scale_factors[0])
        temp_line[5] = str(float(temp_line[5]) * min(scale_factors))
        res_line = "%s %s %s %s %s %s %s\n" % (
            temp_line[0], temp_line[1], temp_line[2],
            temp_line[3],
            temp_line[4], temp_line[5], temp_line[6]
        )
        res_lines.append(res_line)

    file_handle = open(out_path, mode="a")
    file_handle.writelines(res_lines)
    file_handle.close()

    return out_path

def binarize_img(image, threshold):
    binary_image = np.zeros_like(image)
    binary_image[image > threshold] = 255
    return binary_image

def simple_soma_region(img_path, out_path = ""):
    if (not out_path):
        out_path = img_path[:-4] + "_soma.tif"
    if (os.path.exists(out_path)):
        """debug mode off"""
        # return out_path
        os.remove(out_path)
    image = file_io.load_image(img_path).astype("uint8")
    # image = edge_detection_and_fill(image)
    image = binarize_img(image, 255*0.5)

    kern = 3
    kernel = np.ones((kern, kern, round(kern / 2)), "uint8")
    closing_image = skimage.morphology.closing(image, kernel)
    labeled_image = cc3d.connected_components(closing_image, connectivity=6)
    largest_label = np.argmax(np.bincount(labeled_image.flat)[1:]) + 1
    largest_component_binary = ((labeled_image == largest_label)*255).astype("uint8")
    labeled_image, num_labels = label(largest_component_binary)
    boundaries = binary_dilation(labeled_image > 0) & ~labeled_image
    filled_image = binary_fill_holes(boundaries).astype("uint8")*255

    file_io.save_image(out_path, filled_image)
    return out_path

def sort_swc(img_path, swc_path, soma_num=-1, out_path="", v3d_path=r"/home/kfchen/Vaa3D-x.1.1.4_Ubuntu/Vaa3D-x"):
    def get_soma_num(img_path, swc_path):

        # img = file_io.load_image(img_path).astype(np.uint16)
        img = tifffile.imread(img_path)
        soma_num_res = -1
        soma_num_dis = -1
        temp_path1 = swc_path + "1.marker"

        center = simple_get_soma(img, img_path)
        x_center, y_center, z_center = center[2], img.shape[1] - center[1], center[0]

        # img = tifffile.imread(img_path)
        # mip = get_mip_swc(swc_path, img)
        # # center = simple_get_soma(img, img_path)
        # # x_center, y_center, z_center = center[2], center[1], center[0]
        # cv2.circle(mip, (int(x_center), int(y_center)), radius=3, color=(255, 0, 0))
        # tifffile.imwrite(img_path + "mip_origin.png", mip)

        # print(f"x_center, y_center, z_center, r_center", x_center, y_center, z_center, r_center)
        if (os.path.exists(temp_path1)): os.remove(temp_path1)
        print(x_center, y_center, z_center, img.shape)
        # x_center, z_center = z_center, x_center
        # y_center = img.shape[1] - y_center
        soma_x, soma_y, soma_z = 0, 0, 0

        with open(swc_path, 'r') as f:
            lines = f.readlines()
        for line in lines:
            if (line[0] == '#'): continue
            temp_line = line.split()
            # print(temp_line)
            if (not float(temp_line[6]) == -1): continue
            temp_dis = (x_center - float(temp_line[2])) ** 2 + \
                       (y_center - float(temp_line[3])) ** 2
            # print(f"{temp_line[0]} temp_dis {temp_dis}, x_center {temp_line[2]}, y_center {temp_line[3]}")
            if (soma_num_res == -1 or soma_num_dis > temp_dis):
                soma_num_res, soma_x, soma_y, soma_z = temp_line[0], temp_line[2], temp_line[3], temp_line[4]
                soma_num_dis = temp_dis

        # print(soma_num_res, soma_x, soma_y, soma_z)
        return soma_num_res, soma_x, soma_y, soma_z

    def check_0(swc_path, out_path):
        with open(out_path, 'r') as f:
            lines = f.readlines()
        out_path2 = swc_path[:-4] + "_sort2.swc"
        if (os.path.exists(out_path2)): os.remove(out_path2)
        res_lines = []
        for line in lines:
            if (line[0] == '#'): continue
            temp_line = line.split()
            if (float(temp_line[2]) == 0 and float(temp_line[3]) == 0 and float(temp_line[4]) == 0): continue
            res_lines.append(line)

        file_handle = open(out_path2, mode="a")
        file_handle.writelines(res_lines)
        file_handle.close()
        if (os.path.exists(out_path)): os.remove(out_path)

        return out_path2

    def check_soma(swc_path, out_path, soma_x, soma_y, soma_z):
        with open(out_path, 'r') as f:
            lines = f.readlines()
        # print(out_path)
        temp_soma_num = -1
        for line in lines:
            if (line[0] == '#'): continue
            temp_line = line.split()
            if (abs(float(temp_line[2]) - float(soma_x) < 0.1) and
                    abs(float(temp_line[3]) - float(soma_y)) < 0.1 and
                    abs(float(temp_line[4]) - float(soma_z)) < 0.1):
                # print("!!!!!")
                temp_soma_num = int(temp_line[0])
                break
        # print(f"soma {out_path, soma_x, soma_y, soma_z, temp_soma_num}")
        if (temp_soma_num == 1): return out_path, True
        if (temp_soma_num == -1):
            print("fail to find soma")
            exit(0)

        out_path2 = out_path[:-4] + "_sort3.swc"
        if (os.path.exists(out_path2)): os.remove(out_path2)
        if (platform.system() == "Windows"):
            subprocess.run(
                f'{v3d_path} /x sort /f sort_swc /i {out_path} /o {out_path2} /p 0 {temp_soma_num}',
                stdout=subprocess.DEVNULL)  # 全路径
        else:
            # cmd_str = f'xvfb-run -a -s "-screen 0 640x480x16" {v3d_path} -x gsdt -f gsdt -i {in_tmp} -o {out_tmp} -p 0 1 0 1.5'
            cmd_str = f'xvfb-run -a -s "-screen 0 640x480x16" {v3d_path} -x sort -f sort_swc -i {out_path} -o {out_path2} -p 0 {temp_soma_num}'
            cmd_str = cmd_str.replace('(', '\(').replace(')', '\)')
            # print(cmd_str)
            subprocess.run(cmd_str, stdout=subprocess.DEVNULL, shell=True)
        if (os.path.exists(out_path)): os.remove(out_path)
        return out_path2, False

    if (not out_path):
        out_path = swc_path[:-4] + "_sort.swc"
    origin_out_path = out_path
    # img = tifffile.imread(img_path)
    # mip = get_mip_swc(swc_path, img)
    # tifffile.imwrite(img_path + "mip_origin.png", mip)
    if (os.path.exists(out_path)):
        """debug mode off"""
        return out_path

    # print("????")

    if (soma_num == -1):
        soma_num, soma_x, soma_y, soma_z = get_soma_num(img_path, swc_path)

    # print("...")
    if (platform.system() == "Windows"):
        subprocess.run(
            f'{v3d_path} /x sort /f sort_swc /i {swc_path} /o {out_path} /p 0 {soma_num}',
            stdout=subprocess.DEVNULL)  # 全路径
    else:
        cmd_str = f'xvfb-run -a -s "-screen 0 640x480x16" {v3d_path} -x sort -f sort_swc -i {swc_path} -o {out_path} -p 0 {soma_num}'
        cmd_str = cmd_str.replace('(', '\(').replace(')', '\)')
        # print(cmd_str)
        subprocess.run(cmd_str, stdout=subprocess.DEVNULL, shell=True)

    # print("????")
    out_path = check_0(swc_path, out_path)
    check_times = 0
    while(True):
        check_times = check_times + 1
        out_path, check_res = check_soma(swc_path, out_path, soma_x, soma_y, soma_z)
        if(check_res) == True:
            folder_path, file_name = os.path.split(swc_path)
            file_name = file_name[:-4]
            soma_marker_path = os.path.join(folder_path, file_name + ".marker")
            line = "%s, %s, %s, 0.000, 1, , , 255,0,0\n" % (soma_x, soma_y, soma_z)
            file_handle = open(soma_marker_path, mode="a")
            file_handle.writelines(line)
            file_handle.close()
            break
        if(check_times > 3):exit(100)
    if("sort" in out_path):
        # rename
        os.rename(out_path, origin_out_path)

    img = tifffile.imread(img_path)
    mip = get_mip_swc(origin_out_path, img)
    center = simple_get_soma(img, img_path)
    x_center, y_center, z_center = center[2], img.shape[1] - center[1], center[0]
    cv2.circle(mip, (int(x_center), int(y_center)), radius=3, color=(255,0,0))
    tifffile.imwrite(img_path + "mip_origin.png", mip)

    return origin_out_path

from simple_swc_tool.sort_swc import sort_swc as sort_swc2
def my_sort_swc(img_path, swc_path, out_path):
    if(os.path.exists(out_path)):
        return out_path
    img = tifffile.imread(img_path)
    center = simple_get_soma(img, img_path)
    x_center, y_center, z_center = center[2], img.shape[1] - center[1], center[0]
    sort_swc2(swc_path, out_path, (x_center, y_center, z_center))



def calc_radius(img_path, swc_path, tho=-1, radius2d=1, out_path="", v3d_path=r"/home/kfchen/Vaa3D-x.1.1.4_Ubuntu/Vaa3D-x"):
    if (not out_path):
        out_path = swc_path[0:-4] + "_radius.swc"
    if (os.path.exists(out_path)):
        """debug mode off"""
        # return out_path
        os.remove(out_path)
    if (tho == -1):
        # img = PBD().load(img_path)[0].astype("uint8")
        img = file_io.load_image(img_path)
        img_mean, img_std = np.mean(img), np.std(img)
        tho = int(img_mean + 1 * img_std)
        img_b = binarize_img(img.copy(), tho)
        # file_io.save_image(img_path+"b.tif", img_b, False)
        file_io.save_image(img_path + "b.tif", img_b, True)

    if (platform.system() == "Windows"):
        subprocess.run(
            f'{v3d_path} /x neuron_radius /f neuron_radius /i {img_path + "b.tif"} {swc_path} /o {out_path} /p {tho} {radius2d}',
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)  # 全路径
    else:
        cmd_str = f'xvfb-run -a -s "-screen 0 640x480x16" {v3d_path} -x neuron_radius -f neuron_radius -i {img_path + "b.tif"} {swc_path} -o {out_path} -p {tho} {radius2d}'
        cmd_str = cmd_str.replace('(', '\(').replace(')', '\)')
        print(cmd_str)
        subprocess.run(cmd_str, stdout=subprocess.DEVNULL, shell=True)

    if (os.path.exists(img_path + "b.tif")): os.remove(img_path + "b.tif")
    return out_path

def kill_point_in_soma(swc_path, soma_region_path, out_path=""):
    if (not out_path):
        out_path = swc_path[:-4] + "_soma_killed.swc"
    if (os.path.exists(out_path)):
        """debug mode off"""
        # return out_path
        os.remove(out_path)
    soma_region = file_io.load_image(soma_region_path)
    kernel = np.ones((3, 3, 1), "uint8")
    soma_region = skimage.morphology.dilation(soma_region, kernel)
    point_l = read_swc(swc_path)

    for p in point_l.p:
        x, y, z = round(p.x), round(p.y), round(p.z)
        x = max(0, min(x, soma_region.shape[2]-1))
        y = max(0, min(y, soma_region.shape[1]-1))
        z = max(0, min(z, soma_region.shape[0]-1))
        if(soma_region[z,y,x]):
            point_l.p[p.n].r = 1
        else:
            point_l.p[p.n].r = math.sqrt(point_l.p[p.n].r)
    write_swc(out_path, point_l)
    return out_path

def swc2img(img_path, swc_path, out_path="", v3d_path=r"/home/kfchen/Vaa3D-x.1.1.4_Ubuntu/Vaa3D-x"):
    # "Usage v3d -x swc_to_maskimage_sphere_unit -f swc_to_maskimage -i <input.swc> [-p <sz0> <sz1> <sz2>] [-o <output_image.raw>]\n"
    # "Usage v3d -x swc_to_maskimage_sphere_unit -f swc_filter -i <input.tif> <input.swc> [-o <output_image.raw>]\n"
    if (not out_path):
        out_path = swc_path[:-4] + "_ano.tif"
    if (os.path.exists(out_path)):
        """debug mode off"""
        # return out_path
        os.remove(out_path)

    img = file_io.load_image(img_path)

    # tree = parse_swc(swc_path)
    # lab = swc_to_image(tree, imgshape=img.shape)
    # file_io.save_image(out_path, lab)

    if (platform.system() == "Windows"):
        subprocess.run(
            f'{v3d_path} /x swc_to_maskimage_sphere_unit /f swc_to_maskimage /i {swc_path} '
            f'/p {img.shape[2]} {img.shape[1]} {img.shape[0]} /o {out_path}',
            stdout=subprocess.DEVNULL)  # 全路径
    else:
        cmd_str = f'xvfb-run -a -s "-screen 0 640x480x16" {v3d_path} -x swc_to_maskimage_sphere_unit -f swc_to_maskimage -i {swc_path} ' \
                  f'-p {img.shape[2]} {img.shape[1]} {img.shape[0]} -o {out_path}'
        cmd_str = cmd_str.replace('(', '\(').replace(')', '\)')
        # print(cmd_str)
        subprocess.run(cmd_str, stdout=subprocess.DEVNULL, shell=True)
    return out_path

def or_img(img_path1, img_path2, out_path=""):
    if (not out_path):
        out_path = img_path1[:-4] + "_or.tif"
    if (os.path.exists(out_path)):
        """debug mode off"""
        # return out_path
        os.remove(out_path)
    img1 = file_io.load_image(img_path1).astype("uint8")
    # file_io.save_image(out_path+"_test1.tif", img1)
    img2 = file_io.load_image(img_path2).astype("uint8")
    # file_io.save_image(out_path + "_test2.tif", img2)
    # img_or = cv2.bitwise_or(img1, img2)
    img_or = np.logical_or(img1, img2).astype("uint8")
    file_io.save_image(out_path, img_or)

    return out_path

def dilate_img(img_path, kern = 3, out_path=""):
    # cv2.dilate(img, kernel, iteration)
    if (not out_path):
        out_path = img_path[:-4] + "_dilate.tif"
    if (os.path.exists(out_path)):
        """debug mode off"""
        # return out_path
        os.remove(out_path)

    img = file_io.load_image(img_path)
    # kernel = np.ones((kern, kern, kern), "uint8")
    # img_d = cv2.dilate(img, kernel=kernel, iterations=iter)
    # kernel = skimage.morphology.cube(kern)
    kernel = np.ones((kern, kern, round(kern / 2)), "uint8")
    img_d = skimage.morphology.dilation(img, kernel)
    file_io.save_image(out_path, img_d)

    return out_path

def and_img(origin_img_path, ano_path, out_path=""):
    if (not out_path):
        out_path = origin_img_path[:-4] + "_mask.tif"
    if (os.path.exists(out_path)):
        """debug mode off"""
        # return out_path
        os.remove(out_path)
    img = file_io.load_image(origin_img_path)
    img_mean, img_std = np.mean(img), np.std(img)
    tho = int(img_mean + 1 * img_std)
    img_b = binarize_img(img.copy(), tho)
    img_a = file_io.load_image(ano_path)

    # print(img.shape, img_a.shape, img_b.shape)

    # img_mask = cv2.bitwise_and(img_b, img_a)
    img_mask = np.logical_and(img_b, img_a).astype("uint8")
    file_io.save_image(out_path, img_mask)

    return out_path

def dust_img(img_path, kern=3, out_path = ""):
    if (not out_path):
        out_path = img_path[:-4] + "_dust.tif"
    if (os.path.exists(out_path)):
        """debug mode off"""
        # return out_path
        os.remove(out_path)

    img = file_io.load_image(img_path)

    kernel = np.ones((kern, kern, kern), "uint8")
    closing_image = skimage.morphology.closing(img, kernel)

    labeled_image = cc3d.connected_components(closing_image, connectivity=26)
    largest_label = np.argmax(np.bincount(labeled_image.flat)[1:]) + 1
    largest_component_binary = ((labeled_image == largest_label) * 255).astype("uint8")

    file_io.save_image(out_path, largest_component_binary)

    return out_path

def get_spacing(img_path, raw_info_path="/data/kfchen/trace_ws/lab_info.xlsx"):

    raw_info = pd.read_excel(raw_info_path)
    preffix = img_path.split('/')[-1].split('.')[0]
    spacing = raw_info[raw_info['number'] == float(preffix)][['resolution']].values[0]
    return (1, float(spacing[0] / 1000), float(spacing[0] / 1000))

def gen_seg_mask(origin_swc_path, img_path, target_img_dir, target_swc_dir, target_lab_dir, patch_size=(64, 256, 256)):
    file_name, extension = os.path.splitext(os.path.basename(origin_swc_path))
    target_img_path = os.path.join(target_img_dir, file_name + ".tif")
    target_swc_path = os.path.join(target_swc_dir, file_name + ".swc")
    target_lab_path = os.path.join(target_lab_dir, file_name + ".tif")

    # if(os.path.exists(target_lab_path)):
    #     return
    if(not os.path.exists(target_img_path)):
        img = tifffile.imread(img_path)

        # spacing = get_spacing(img_path)
        # pad_width = (int((1-spacing[1]) * img.shape[1]), int((1-spacing[2]) * img.shape[2]))
        # pad_params = ((0, 0),) + (pad_width,) * 2
        # img = np.pad(img, pad_width=pad_params, mode='constant', constant_values=0)
        # print(img.shape)
        #
        # scale_factors = [n / o for n, o in zip(patch_size, img.shape)]
        # img = scipy.ndimage.zoom(img, scale_factors, order=3)
        # print(f"...", img.shape)
        img = (img - np.min(img)) / (np.max(img) - np.min(img)) * 255
        img = img.astype('uint8')
        # print(np.max(img), np.min(img))
        # tifffile.imwrite(str(target_img_path), img)
        sitk.WriteImage(sitk.GetImageFromArray(img), str(target_img_path), useCompression=False)

    flip_dir = os.path.join(target_lab_dir, "1_flip")
    sort_dir = os.path.join(target_lab_dir, "2_sort")
    soma_region_dir = os.path.join(target_lab_dir, "3_soma_region")
    radius_dir = os.path.join(target_lab_dir, "4_radius")
    killed_soma_swc_dir = os.path.join(target_lab_dir, "4_killed_soma_swc")
    ano_dir = os.path.join(target_lab_dir, "6_swc2img")
    ano3_dir = os.path.join(target_lab_dir, "7_or_img")
    dilate_dir = os.path.join(target_lab_dir, "8_dilate")
    mask_dir = os.path.join(target_lab_dir, "9_and_img")
    ano2_dir = os.path.join(target_lab_dir, "10_swc2img")
    mask2_dir = os.path.join(target_lab_dir, "11_or_img")
    dust_dir = os.path.join(target_lab_dir, "12_dust")

    flip_dir2 = os.path.join(target_lab_dir, "2_flip_after_sort")

    if (not os.path.exists(flip_dir)): os.makedirs(flip_dir)
    if (not os.path.exists(sort_dir)): os.makedirs(sort_dir)
    if (not os.path.exists(soma_region_dir)): os.makedirs(soma_region_dir)
    if (not os.path.exists(radius_dir)): os.makedirs(radius_dir)
    if (not os.path.exists(killed_soma_swc_dir)): os.makedirs(killed_soma_swc_dir)
    if (not os.path.exists(ano_dir)): os.makedirs(ano_dir)
    if (not os.path.exists(ano3_dir)): os.makedirs(ano3_dir)
    if (not os.path.exists(dilate_dir)): os.makedirs(dilate_dir)
    if (not os.path.exists(mask_dir)): os.makedirs(mask_dir)
    if (not os.path.exists(ano2_dir)): os.makedirs(ano2_dir)
    if (not os.path.exists(mask2_dir)): os.makedirs(mask2_dir)
    if (not os.path.exists(dust_dir)): os.makedirs(dust_dir)
    if (not os.path.exists(flip_dir2)): os.makedirs(flip_dir2)

    flip_path = flip_swc(target_img_path, origin_swc_path, os.path.join(flip_dir, file_name + ".swc"))
    sort_path = my_sort_swc(target_img_path, flip_path, out_path=os.path.join(sort_dir, file_name + ".swc"))
    flip_path2 = flip_swc(target_img_path, sort_path, os.path.join(flip_dir2, file_name + ".swc"))
    # soma_region_path = simple_soma_region(target_img_path, os.path.join(soma_region_dir, file_name + ".tif"))
    # radius_path = calc_radius(target_img_path, sort_path, out_path=os.path.join(radius_dir, file_name + ".swc"))
    # killed_soma_swc_path = kill_point_in_soma(radius_path, soma_region_path, os.path.join(killed_soma_swc_dir, file_name + ".swc"))
    # ano_path = swc2img(target_img_path, killed_soma_swc_path, os.path.join(ano_dir, file_name + ".tif"))
    # ano_path3 = or_img(ano_path, soma_region_path, os.path.join(ano3_dir, file_name + ".tif"))
    # dilate_path = dilate_img(ano_path3, out_path=os.path.join(dilate_dir, file_name + ".tif"))
    # mask_path = and_img(target_img_path, dilate_path, os.path.join(mask_dir, file_name + ".tif"))
    # ano_path2 = swc2img(target_img_path, sort_path, os.path.join(ano2_dir, file_name + ".tif"))
    # mask_path2 = or_img(mask_path, ano_path2, os.path.join(mask2_dir, file_name + ".tif"))
    # dust_path = dust_img(mask_path2, out_path=os.path.join(dust_dir, file_name + ".tif"))





    # if(os.path.exists(flip_path)):os.remove(flip_path)
    # if (os.path.exists(sort_path)): os.remove(sort_path)
    # if(os.path.exists(soma_region_path)): os.remove(soma_region_path)
    # if (os.path.exists(radius_path)): os.remove(radius_path)
    # if (os.path.exists(killed_soma_swc_path)): os.remove(killed_soma_swc_path)
    # if (os.path.exists(ano_path)): os.remove(ano_path)
    # if (os.path.exists(ano_path3)): os.remove(ano_path3)
    # if (os.path.exists(dilate_path)): os.remove(dilate_path)
    # if (os.path.exists(mask_path)): os.remove(mask_path)
    # if (os.path.exists(ano_path2)): os.remove(ano_path2)
    # if (os.path.exists(mask_path2)): os.remove(mask_path2)

def process_file(tif_file, tif_source_dir, swc_source_dir, target_img_dir, target_swc_dir, target_lab_dir):
    # try:
    tif_file_path = os.path.join(tif_source_dir, tif_file)
    swc_file_path = os.path.join(swc_source_dir, tif_file[:-4] + ".swc")
    gen_seg_mask(swc_file_path, tif_file_path, target_img_dir, target_swc_dir, target_lab_dir)
    # except:
    #     print(f"Error: {tif_file}")


def gen_mip(target_img_dir, target_lab_dir, target_mip_dir):
    tif_files = os.listdir(target_img_dir)
    tif_files = [f for f in tif_files if f.endswith(".tif")]
    tif_files.sort()
    for tif_file in tif_files:
        tif_file_path = os.path.join(target_img_dir, tif_file)

        # get mip
        img = tifffile.imread(tif_file_path)
        img_mip = np.max(img, axis=0)
        # to 0-255
        img_mip = (img_mip - np.min(img_mip)) / (np.max(img_mip) - np.min(img_mip)) * 255
        lab = tifffile.imread(os.path.join(target_lab_dir, tif_file[:-4] + ".tif"))
        lab_mip = np.max(lab, axis=0)
        lab_mip = (lab_mip - np.min(lab_mip)) / (np.max(lab_mip) - np.min(lab_mip)) * 255

        # print(f"{img.shape}, {lab.shape}")

        mip_combined = np.zeros((img_mip.shape[0], img_mip.shape[1] * 2), dtype="uint8")
        mip_combined[:, :img_mip.shape[1]] = img_mip
        mip_combined[:, img_mip.shape[1]:] = lab_mip
        # save png
        plt.imsave(os.path.join(target_mip_dir, tif_file[:-4] + ".png"), mip_combined, cmap="gray")

if __name__ == "__main__":
    tif_source_dir = "/PBshare/SEU-ALLEN/Users/KaifengChen/human_brain/img/raw"
    # swc_source_dir = "/PBshare/SEU-ALLEN/Users/KaifengChen/human_brain/label/origin_swc"
    swc_source_dir = r"/data/kfchen/trace_ws/to_gu/origin_swc"

    target_dir = r"/data/kfchen/trace_ws/to_gu"
    target_img_dir = target_dir + "/img"
    target_swc_dir = target_dir + "/swc"
    target_lab_dir = target_dir + "/new_lab"
    target_mip_dir = target_dir + "/mip"
    if (not os.path.exists(target_img_dir)): os.makedirs(target_img_dir)
    if (not os.path.exists(target_swc_dir)): os.makedirs(target_swc_dir)
    if (not os.path.exists(target_lab_dir)): os.makedirs(target_lab_dir)
    if (not os.path.exists(target_mip_dir)): os.makedirs(target_mip_dir)

    # origin_swc_path = r"/data/kfchen/trace_ws/my_test/2364.swc"
    # img_path = r"/data/kfchen/trace_ws/my_test/2364.tif"
    # gen_seg_mask(origin_swc_path, img_path, target_img_dir, target_swc_dir, target_lab_dir)

    tif_files = os.listdir(tif_source_dir)
    tif_files = [f for f in tif_files if f.endswith(".tif")]
    # 随机挑选20个
    # tif_files = np.random.choice(tif_files, 50, replace=False)

    with ThreadPoolExecutor(max_workers=12) as executor:
        # 使用tqdm显示进度条
        for _ in tqdm(executor.map(process_file, tif_files, [tif_source_dir] * len(tif_files),
                                   [swc_source_dir] * len(tif_files),
                                   [target_img_dir] * len(tif_files), [target_swc_dir] * len(tif_files),
                                   [target_lab_dir] * len(tif_files)),
                      total=len(tif_files), desc="Processing"):
            pass


    gen_mip(target_img_dir, os.path.join(target_lab_dir, "2_flip_after_sort"), target_mip_dir)

    # del swc dir
    # if (os.path.exists(target_swc_dir)): os.removedirs(target_swc_dir)