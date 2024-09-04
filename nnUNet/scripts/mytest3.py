import os
import numpy as np
import tifffile
from simple_swc_tool.sort_swc import sort_swc as sort_swc2

import concurrent.futures
from tqdm import tqdm

def process_file(swc_file, gt_swc_dir, swc_dir, result_swc_dir):
    if (swc_file[-4:] != ".swc"):
        return
    id = str(int(swc_file.split("_")[0]))
    gt_swc_file = id + ".swc"

    x_center, y_center, z_center = 0, 0, 0

    with open(os.path.join(gt_swc_dir, gt_swc_file), "r") as f:
        gt_lines = f.readlines()
        for line in gt_lines:
            if (line[0] == "#"):
                continue
            line = line.split()
            if (line[-1] == -1):
                x_center, y_center, z_center = float(line[2]), float(line[3]), float(line[4])
                break

    swc_path = os.path.join(swc_dir, swc_file)
    out_path = os.path.join(result_swc_dir, swc_file)

    sort_swc2(swc_path, out_path, (x_center, y_center, z_center))

if __name__ == '__main__':
    # swc_dir = "/data/kfchen/trace_ws/to_gu/new_sort_lab/2_sort"
    # fliped_dir = "/data/kfchen/trace_ws/to_gu/new_sort_lab/2_flip_after_sort"
    # img_dir = "/data/kfchen/trace_ws/to_gu/img"
    #
    # swc_files = os.listdir(swc_dir)
    #
    # for swc_file in swc_files:
    #     if(swc_file[-4:] != ".swc"):
    #         continue
    #
    #     swc_path = os.path.join(swc_dir, swc_file)
    #     fliped_path = os.path.join(fliped_dir, swc_file)
    #     img_path = os.path.join(img_dir, swc_file[:-4] + ".tif")
    #
    #     img = tifffile.imread(img_path)
    #     y_limit = img.shape[1]
    #
    #     with open(swc_path, "r") as f:
    #         lines = f.readlines()
    #         for line in lines:
    #             if(line[0] == "#"):
    #                 continue
    #             line = line.split()
    #             y = y_limit - float(line[3])
    #             line[3] = str(y)
    #             with open(fliped_path, "a") as f:
    #                 f.write(" ".join(line) + "\n")

    gt_swc_dir = "/data/kfchen/trace_ws/to_gu/new_sort_lab/2_flip_after_sort"
    swc_dir = "/data/kfchen/New Folder/to_Kaifeng/oneChecked_annotationn"
    result_swc_dir = "/data/kfchen/New Folder/to_Kaifeng/final_oneChecked_annotation"
    if(not os.path.exists(result_swc_dir)):
        os.makedirs(result_swc_dir)

    swc_files = os.listdir(swc_dir)
    # for swc_file in swc_files:
    #     process(swc_file, gt_swc_dir, swc_dir, result_swc_dir)

    # 使用 ThreadPoolExecutor 来创建线程池
    # max_workers 参数可以根据你的系统和任务类型调整
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        # 创建一个进度条对象
        progress = tqdm(total=len(swc_files), desc='Processing SWC Files', unit='file')

        # 将任务提交到线程池
        futures = [executor.submit(process_file, swc_file, gt_swc_dir, swc_dir, result_swc_dir)
                   for swc_file in swc_files]

        # 当每个任务完成时更新进度条
        for future in concurrent.futures.as_completed(futures):
            progress.update(1)  # 更新进度条

        # 关闭进度条
        progress.close()





