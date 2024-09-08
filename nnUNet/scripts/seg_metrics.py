import os
import time

from PIL import Image
import numpy as np
import cc3d
import tifffile

from scipy import ndimage
from scipy.ndimage import binary_dilation, binary_erosion
from skimage.morphology import skeletonize_3d

import concurrent.futures
from tqdm import tqdm

import nibabel as nib
import matplotlib.pyplot as plt

def close_operation_3d(image, structure=None):
    """
    对三维图像进行闭操作。

    :param image: 输入的三维numpy数组
    :param structure: 结构元素，用于定义邻域。如果为None，则使用默认的结构元素。
    :return: 经过闭操作的图像
    """

    # 先膨胀后腐蚀
    dilated = binary_dilation(image, structure=structure)
    closed_image = binary_erosion(dilated, structure=structure)

    return closed_image


def find_largest_connected_components(img_data, target_cc_num):
    # 使用cc3d计算连通块
    labels_out, N = cc3d.connected_components(img_data, connectivity=26, return_N=True)

    # 统计每个标签的体素数
    label_sizes = np.bincount(labels_out.ravel())
    label_sizes[0] = 0  # 忽略背景的体素

    # 找到体积最大的n个连通块的标签
    largest_labels = np.argsort(label_sizes)[-target_cc_num:]

    # 创建一个数组用来存储最大的n个连通块的数据
    largest_components = np.zeros_like(labels_out)

    # 提取每个连通块并放入largest_components数组
    for label in largest_labels:
        largest_components[labels_out == label] = label

    return largest_components


def get_single_c_relative_coverage(seg_data, target_cc_num):
    close_data = close_operation_3d(seg_data)
    max_cc_seg_data = find_largest_connected_components(close_data, target_cc_num)

    skel_seg = skeletonize_3d(seg_data > 0)
    skel_max_cc = skeletonize_3d(max_cc_seg_data > 0)

    if(np.sum(skel_seg > 0) > 0):
        return np.sum(skel_max_cc > 0) / np.sum(skel_seg > 0)
    else:
        return 0

# 计算相对前景比例
def get_relative_foreground_ratio(seg_data, gt_data):
# 计算前景比例
    seg_foreground_ratio = np.sum(seg_data > 0) / np.sum(seg_data >= 0)
    gt_foreground_ratio = np.sum(gt_data > 0) / np.sum(gt_data >= 0)

    # 计算相对前景比例
    relative_foreground_ratio = seg_foreground_ratio / gt_foreground_ratio
    return relative_foreground_ratio


def get_single_overlap(seg_data, gt_data):
    # 计算交集和并集
    intersection = np.sum((seg_data > 0) & (gt_data > 0))
    union = np.sum((seg_data > 0) | (gt_data > 0))

    # 计算Jaccard指数（Overlap）
    if union == 0:
        jaccard_index = 1.0  # 避免除以零的情况
    else:
        jaccard_index = intersection / union
    return jaccard_index

def get_single_c_overlap(seg_data, gt_data, target_cc_num):
    seg_data = close_operation_3d(seg_data)
    seg_data = find_largest_connected_components(seg_data, target_cc_num)
    # 计算交集和并集
    intersection = np.sum((seg_data > 0) & (gt_data > 0))
    union = np.sum((seg_data > 0) | (gt_data > 0))

    # 计算Jaccard指数（Overlap）
    if union == 0:
        jaccard_index = 1.0  # 避免除以零的情况
    else:
        jaccard_index = intersection / union
    return jaccard_index

def dice_coefficient(seg, gt):
    # print(seg.shape, gt.shape)
    intersection = np.sum(seg[gt > 0])
    seg_sum = np.sum(seg)
    gt_sum = np.sum(gt)
    dice = 2 * intersection / (seg_sum + gt_sum)
    return dice


def get_single_c_dice(seg, gt, target_cc_num):
    seg = close_operation_3d(seg)
    seg = find_largest_connected_components(seg, target_cc_num)
    return dice_coefficient(seg > 0, gt > 0)


def get_fingures_single_result(seg_folder, gt_folder, seg_file, gt_file, target_cc_num):
    # Load the segmentation result and ground truth image
    seg_path = os.path.join(seg_folder, seg_file)
    gt_path = os.path.join(gt_folder, gt_file)

    if(seg_path[-4:] == ".tif"):
        seg_img = tifffile.imread(seg_path)
        gt_img = tifffile.imread(gt_path)
    elif(seg_path[-7:] == ".nii.gz"):
        seg_img = nib.load(seg_path)
        gt_img = nib.load(gt_path)
        seg_img = seg_img.get_fdata()
        gt_img = gt_img.get_fdata()

    # Convert images to NumPy arrays
    seg_data = np.array(seg_img).astype(np.uint8)
    gt_data = np.array(gt_img).astype(np.uint8)
    # print(seg_data.shape, gt_data.shape)

    # print(seg_file)
    if(seg_file[:5] == 'image'):
        seg_id = seg_file.split('_')[1].replace('.tif', '')
    else:
        seg_id = seg_file.split('_')[0]

    return (seg_id,
            dice_coefficient(seg_data > 0, gt_data > 0),
            get_single_c_dice(seg_data > 0, gt_data > 0, target_cc_num),
            get_single_overlap(seg_data, gt_data),
            get_single_c_overlap(seg_data, gt_data, target_cc_num),
            get_single_c_relative_coverage(seg_data, target_cc_num),
            get_relative_foreground_ratio(seg_data, gt_data),
            get_single_broken_points(seg_data, gt_data),
            get_skel_accuracy(seg_data, gt_data),
    )

def get_single_broken_points(seg_data, gt_data):
    # 计算骨架
    skel_gt = skeletonize_3d(gt_data > 0)
    seg_data = binary_dilation(seg_data)

    tp_skel = (seg_data > 0) & (skel_gt > 0)
    fn_skel = skel_gt - tp_skel

    _, cc_num = cc3d.connected_components(fn_skel, connectivity=26, return_N=True)
    return cc_num


def get_skel_accuracy(seg_data, gt_data):
    skel_gt = skeletonize_3d(gt_data > 0)
    seg_data = binary_dilation(seg_data)

    tp_skel = (seg_data > 0) & (skel_gt > 0)
    return np.sum(tp_skel > 0) / np.sum(skel_gt > 0)


def get_normalized_foreground_breaking_points(seg_data, gt_data, target_ratio):
    while(np.sum(seg_data > 0) > np.sum(gt_data > 0) * target_ratio):
        seg_data = binary_erosion(seg_data)
    return get_single_broken_points(seg_data, gt_data)

def process_file_pair(seg_folder, gt_folder, file_pair, target_cc_num):
    seg_file, gt_file = file_pair
    return get_fingures_single_result(seg_folder, gt_folder, seg_file, gt_file, target_cc_num)


def compute_metrics_for_all_pairs(seg_folder, gt_folder, target_cc_num, prefix=""):
    seg_files = [f for f in os.listdir(seg_folder) if f.endswith(prefix)]
    gt_files = [f for f in os.listdir(gt_folder) if f.endswith(prefix)]

    gt_files_copy = gt_files.copy()
    for gt_file in gt_files_copy:
        flag = False
        for seg_file in seg_files:
            if prefix == '.tif' and seg_file == gt_file:
                flag = True
                break
            if(prefix == '.nii.gz' and seg_file[-10:] == gt_file[-10:]):
                flag = True
                break
        if(flag == False):
            gt_files.remove(gt_file)
    # sort
    gt_files = sorted(gt_files)
    seg_files = sorted(seg_files)
    print(len(seg_files), len(gt_files))
    if(not len(seg_files) == len(gt_files)):
        return

    file_pairs = [(seg_file, gt_file) for seg_file, gt_file in zip(seg_files, gt_files)]

    # debug
    for i in range(len(file_pairs)):
        # print(file_pairs[i])
        if(file_pairs[i][0][:5] != file_pairs[i][1][:5]):
            print("error")
    # file_pairs = file_pairs[:10]

    results = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_file_pair, seg_folder, gt_folder, pair, target_cc_num) for pair in file_pairs]
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing Files"):
            results.append(future.result())

    return results

# 把nii.gz转成tif
def nii2tif(nii_path, tif_path):
    img = nib.load(nii_path)
    img_data = img.get_fdata()
    img_data = (img_data - np.min(img_data)) / (np.max(img_data) - np.min(img_data)) * 255
    img_data = np.array(img_data).astype(np.uint8)
    tifffile.imsave(tif_path, img_data)

def move_img(source_folder, target_folder, move_ratio=0.2):
    img_files = os.listdir(source_folder)
    for img_file in img_files:
        if(np.random.rand() < move_ratio):
            os.rename(os.path.join(source_folder, img_file), os.path.join(target_folder, img_file))

def generate_mip_and_compare(seg_folder1, seg_folder2, mip_folder):
    if(not os.path.exists(mip_folder)):
        os.makedirs(mip_folder)
    files = os.listdir(seg_folder1)
    for file in files:
        seg_path1 = os.path.join(seg_folder1, file)
        seg_path2 = os.path.join(seg_folder2, file)
        seg_img1 = tifffile.imread(seg_path1)
        seg_img2 = tifffile.imread(seg_path2)
        seg_img1 = (seg_img1 > 0).astype(np.uint8)
        seg_img2 = (seg_img2 > 0).astype(np.uint8)
        mip1 = np.max(seg_img1, axis=0) * 255
        mip2 = np.max(seg_img2, axis=0) * 255

        cat_img = np.concatenate([mip1, mip2], axis=1)
        cat_img = Image.fromarray(cat_img)
        # save png
        cat_img.save(os.path.join(mip_folder, file[:-4] + "_mip.png"))

# 比较前景比例
def compare_foreground_ratio(seg_folder, gt_folder):
    seg_files = os.listdir(seg_folder)
    gt_files = os.listdir(gt_folder)
    seg_files = sorted(seg_files)
    gt_files = sorted(gt_files)

    seg_front_ratios = []
    gt_front_ratios = []

    for i in range(len(seg_files)):
        seg_path = os.path.join(seg_folder, seg_files[i])
        gt_path = os.path.join(gt_folder, gt_files[i])
        seg_img = tifffile.imread(seg_path)
        gt_img = tifffile.imread(gt_path)
        seg_foreground_ratio = np.sum(seg_img > 0) / np.sum(seg_img >= 0)
        gt_foreground_ratio = np.sum(gt_img > 0) / np.sum(gt_img >= 0)
        seg_front_ratios.append(seg_foreground_ratio)
        gt_front_ratios.append(gt_foreground_ratio)

    print("Mean Foreground Ratio of Segmentation:", np.mean(seg_front_ratios))
    print("Mean Foreground Ratio of Ground Truth:", np.mean(gt_front_ratios))

def normalized_foreground(seg_folder1, seg_folder2, result_folder): # ptls baseline
    seg_files1 = os.listdir(seg_folder1)
    num = 1
    for seg_file1 in seg_files1:
        if(seg_file1[-4:] != ".tif"):
            continue
        print(f"Processing {num}/{len(seg_files1)}")
        num += 1
        seg_path1 = os.path.join(seg_folder1, seg_file1)
        seg_path2 = os.path.join(seg_folder2, seg_file1)
        seg_img1 = tifffile.imread(seg_path1)
        seg_img2 = tifffile.imread(seg_path2)
        seg_img1 = (seg_img1 > 0).astype(np.uint8)
        seg_img2 = (seg_img2 > 0).astype(np.uint8)
        print(np.sum(seg_img1 > 0), np.sum(seg_img2 > 0))
        fg1, fg2 = np.sum(seg_img1 > 0), np.sum(seg_img2 > 0)
        print(fg1, fg2, fg1>fg2)
        while (fg1 * 0.77 > fg2):
            seg_img2 = binary_dilation(seg_img2)
            fg2 = np.sum(seg_img2 > 0)
        print(np.sum(seg_img1 > 0), np.sum(seg_img2 > 0))
        print("-----------")
        tifffile.imwrite(os.path.join(result_folder, seg_file1), seg_img2)

def plot_box(result_csv1, result_csv2, label=["Baseline", "Proposed"]):
    metric_names = ["Broken Points", "Skeleton Accuracy"]
    metrics1 = []
    metrics2 = []
    with open(result_csv1, "r") as f:
        lines = f.readlines()
        for line in lines[1:]:
            items = line.split(',')
            metrics1.append([float(items[7]), float(items[8])])
    with open(result_csv2, "r") as f:
        lines = f.readlines()
        for line in lines[1:]:
            items = line.split(',')
            metrics2.append([float(items[7]), float(items[8])])




# main
if __name__ == '__main__':
    # nii_files = os.listdir("/data/kfchen/nnUNet/nnUNet_results/Dataset202_HepaticVessel01/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/close_result/noptls500/validation")
    # for nii_file in nii_files:
    #     if(nii_file[-7:] == ".nii.gz"):
    #         nii_path = os.path.join("/data/kfchen/nnUNet/nnUNet_results/Dataset202_HepaticVessel01/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/close_result/noptls500/validation", nii_file)
    #         tif_path = os.path.join("/data/kfchen/nnUNet/nnUNet_results/Dataset202_HepaticVessel01/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/close_result/noptls500/validation_tif", nii_file[:-7] + ".tif")
    #         nii2tif(nii_path, tif_path)

    # source_folder = r"/data/kfchen/nnUNet/nnUNet_raw/Dataset401_mb1891/imageTr"
    # target_folder = r"/data/kfchen/nnUNet/nnUNet_raw/Dataset401_mb1891/imageTs"
    # move_img(source_folder, target_folder, move_ratio=0.2)

    # lab_folder = r"/data/kfchen/nnUNet/nnUNet_raw/Dataset401_mb1891/labelsTr"
    # lab_files = os.listdir(lab_folder)
    # for lab_file in lab_files:
    #     if(lab_file[-4:] == ".tif"):
    #         lab_path = os.path.join(lab_folder, lab_file)
    #         lab_img = tifffile.imread(lab_path)
    #         lab_img = (lab_img > 0).astype(np.uint8)
    #         tifffile.imsave(lab_path, lab_img)
    #
    # print("ok")

    # normalized_foreground("/data/kfchen/nnUNet/nnUNet_results/Dataset169_hb_10k/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/ptls100_finle??/validation",
    #                       "/data/kfchen/nnUNet/nnUNet_results/Dataset169_hb_10k/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/source500/validation",
    #                       "/data/kfchen/nnUNet/nnUNet_results/Dataset169_hb_10k/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/source500/foreground_norm")
    # print("done")
    # # time.sleep(100000)

    dataset_list = {
        'hb_gt': '/data/kfchen/nnUNet/nnUNet_raw/Dataset169_hb_10k/labelsTr',
        'hb_seg_baseline': '/data/kfchen/nnUNet/nnUNet_results/Dataset169_hb_10k/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/source500/validation',
        'hb_seg_ptls': '/data/kfchen/nnUNet/nnUNet_results/Dataset169_hb_10k/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/ptls10/validation',

        'HepaticVessel_gt': "/data/kfchen/nnUNet/nnUNet_raw/Dataset202_HepaticVessel01/labelsTr",
        'HepaticVessel_seg_baseline': "/data/kfchen/nnUNet/nnUNet_results/Dataset202_HepaticVessel01/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/close_result/noptls500/validation",
        'HepaticVessel_seg_ptls': "/data/kfchen/nnUNet/nnUNet_results/Dataset202_HepaticVessel01/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/close_result/ptls500/validation",

        'cas_gt': "/home/kfchen/nnUNet_raw/Dataset301_CAS/labelsTr",
        'cas_seg_baseline': "/home/kfchen/nnUNet_results/Dataset301_CAS/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/source/validation",
        'cas_seg_ptls': "/home/kfchen/nnUNet_results/Dataset301_CAS/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/ptls/validation",
    }
    seg_folder = dataset_list['hb_seg_ptls']
    gt_folder = dataset_list['hb_gt']

    seg_folder = "/data/kfchen/trace_ws/paper_auto_human_neuron_recon/test_seg_220/source500"
    gt_folder = "/data/kfchen/trace_ws/paper_auto_human_neuron_recon/test_seg_220/label"

    # generate_mip_and_compare(dataset_list['hb_seg_baseline'], dataset_list['hb_seg_ptls'],
    #                          "/data/kfchen/nnUNet/nnUNet_results/Dataset167_human_brain_10000_noptls/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/ptls_full_weight200_250+250/mip")



    all_metrics = compute_metrics_for_all_pairs(seg_folder, gt_folder, target_cc_num=1, prefix=".tif")

    for i in range(len(all_metrics)):
        print(all_metrics[i])

    mean_dice = np.mean([metrics[1] for metrics in all_metrics])
    mean_c_dice = np.mean([metrics[2] for metrics in all_metrics])
    mean_overlap = np.mean([metrics[3] for metrics in all_metrics])
    mean_c_overlap = np.mean([metrics[4] for metrics in all_metrics])
    mean_c_relative_coverage = np.mean([metrics[5] for metrics in all_metrics])
    mean_relative_foreground_ratio = np.mean([metrics[6] for metrics in all_metrics])
    mean_broken_points = np.mean([metrics[7] for metrics in all_metrics])
    mean_skel_acc = np.mean([metrics[8] for metrics in all_metrics])

    # 4位小数输出
    print("Mean Dice Coefficient:", round(mean_dice, 4))
    print("Mean Overlap:", round(mean_overlap, 4))
    print("Mean C-Dice Coefficient:", round(mean_c_dice, 4))
    print("Mean C-Overlap:", round(mean_c_overlap, 4))
    print("Mean C-Relative Coverage:", round(mean_c_relative_coverage, 4))
    print("Mean Relative Foreground Ratio:", round(mean_relative_foreground_ratio, 4))
    print("Mean Broken Points:", round(mean_broken_points, 4))
    print("Mean Skeleton Accuracy:", round(mean_skel_acc, 4))

    print(round(mean_dice, 4), round(mean_overlap, 4), round(mean_c_dice, 4), round(mean_c_overlap, 4),
          round(mean_c_relative_coverage, 4), round(mean_relative_foreground_ratio, 4), round(mean_broken_points, 4),
            round(mean_skel_acc, 4))

    result_csv = "/data/kfchen/trace_ws/paper_auto_human_neuron_recon/test_seg_220/source500.csv"
    if(os.path.exists(result_csv)):
        os.remove(result_csv)
    # sort
    all_metrics = sorted(all_metrics, key=lambda x: x[0])
    with open(result_csv, "w") as f:
        f.write("ID,Dice,Overlap,C-Dice,C-Overlap,C-Relative Coverage,Relative Foreground Ratio,Broken Points,Skeleton Accuracy\n")
        for metrics in all_metrics:
            f.write(f"{metrics[0]},{metrics[1]},{metrics[3]},{metrics[2]},{metrics[4]},{metrics[5]},{metrics[6]},{metrics[7]},{metrics[8]}\n")
        # f.write(f"Mean,{round(mean_dice, 4)},{round(mean_overlap, 4)},{round(mean_c_dice, 4)},{round(mean_c_overlap, 4)},{round(mean_c_relative_coverage, 4)},{round(mean_relative_foreground_ratio, 4)},{round(mean_broken_points, 4)},{round(mean_skel_acc, 4)}\n")