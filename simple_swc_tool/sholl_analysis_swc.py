import os
import numpy as np
import subprocess
from pylib import file_io
import platform
import tifffile
import pandas as pd
from scipy.ndimage import label
import matplotlib.pyplot as plt
import seaborn as sns
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

def find_resolution(filename, df):
    filename = int(filename.split('.')[0].split('_')[0])
    for i in range(len(df)):
        if int(df.iloc[i, 0]) == filename:
            return df.iloc[i, 43]

def swc2img(swc_path, img_shape, out_path, v3d_path=r"/home/kfchen/Vaa3D-x.1.1.4_Ubuntu/Vaa3D-x"):
    # "Usage v3d -x swc_to_maskimage_sphere_unit -f swc_to_maskimage -i <input.swc> [-p <sz0> <sz1> <sz2>] [-o <output_image.raw>]\n"
    # "Usage v3d -x swc_to_maskimage_sphere_unit -f swc_filter -i <input.tif> <input.swc> [-o <output_image.raw>]\n"

    cmd_str = f'xvfb-run -a -s "-screen 0 640x480x16" {v3d_path} -x swc_to_maskimage_sphere_unit -f swc_to_maskimage -i {swc_path} ' \
              f'-p {img_shape[2]} {img_shape[1]} {img_shape[0]} -o {out_path}'
    cmd_str = cmd_str.replace('(', '\(').replace(')', '\)')
    # print(cmd_str)
    subprocess.run(cmd_str, stdout=subprocess.DEVNULL, shell=True)

def sholl_analysis_img(img, start_point):
    # 验证起始点是否在图像内部
    if not (0 <= start_point[0] < img.shape[0] and
            0 <= start_point[1] < img.shape[1] and
            0 <= start_point[2] < img.shape[2]):
        raise ValueError("起始点不在图像范围内。")

    # 提取所有前景点的坐标
    foreground_coords = np.argwhere(img == 1)
    distances = np.sqrt(np.sum((foreground_coords - np.array(start_point)) ** 2, axis=1))
    max_radius = int(np.ceil(distances.max()))

    x = np.arange(img.shape[0])
    y = np.arange(img.shape[1])
    z = np.arange(img.shape[2])
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    voxel_distances = np.sqrt((X - start_point[0]) ** 2 +
                              (Y - start_point[1]) ** 2 +
                              (Z - start_point[2]) ** 2)

    results = {
        "radii": [],
        "num_connected_components": []
    }
    for r in range(1, max_radius + 1):
        # 定义球壳的范围 [r, r+1)
        shell_mask = (voxel_distances >= r) & (voxel_distances < r + 1)

        # 交集掩码
        intersection = shell_mask & img

        if np.any(intersection):
            # 标记26连通域
            labeled, num_features = label(intersection, structure=np.ones((3, 3, 3)))
        else:
            num_features = 0

        results["radii"].append(r)
        results["num_connected_components"].append(num_features)

    return results



def sholl_analysis_swc(swc_file, img_shape, temp_img_file):
    """
    Sholl analysis for SWC file
    :param swc_file: str, input SWC file
    :param output_sholl_result: str, output CSV file
    :param step_size: int, step size for Sholl analysis
    :return: None
    """
    # Read SWC file
    with open(swc_file, "r") as f:
        lines = f.readlines()

    # get_soma
    soma_pos = None
    for line in lines:
        if line.startswith("#"):
            continue
        line = line.strip()
        line = line.split(" ")
        line = [int(line[0]), int(line[1]), float(line[2]), float(line[3]), float(line[4]), float(line[5]), int(line[6])]
        if line[-1] == -1:
            soma_pos = np.array(line[2:5])
            break
    soma_pos = tuple((int(soma_pos[2]), int(soma_pos[1]), int(soma_pos[0])))
    # print(soma_pos)

    # swc to tif
    swc2img(swc_file, img_shape, temp_img_file)

    temp_img = tifffile.imread(temp_img_file)
    # temp_img = np.flip(temp_img, axis=1)
    # print(temp_img.shape)
    # print(temp_img[soma_pos])
    os.remove(temp_img_file)
    return sholl_analysis_img(temp_img>0, soma_pos)

def test_sholl_analysis_swc(swc_file, output_sholl_result, img_dir, neuron_info):
    if(os.path.exists(output_sholl_result)):
        return

    img_file = os.path.join(img_dir, os.path.basename(swc_file).replace(".swc", ".tif"))
    img_shape = tifffile.imread(img_file).shape

    xy_resolution = find_resolution(os.path.basename(swc_file), neuron_info)
    img_shape = [int(img_shape[0]), int(img_shape[1] * xy_resolution / 1000), int(img_shape[2] * xy_resolution / 1000)]

    temp_img_file = output_sholl_result.replace(".npy", ".tif")
    results = sholl_analysis_swc(swc_file, img_shape, temp_img_file)
    np.save(output_sholl_result, results)

def plot_sholl_result(result_file):
    results = np.load(result_file, allow_pickle=True).item()
    plt.figure(figsize=(10, 6))
    sns.barplot(x=results["radii"], y=results["num_connected_components"], color='skyblue')
    plt.title('Sholl Analysis')
    plt.xlabel('Radius')
    plt.ylabel('Number of Connected Components (26-connectivity)')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    plt.close()

def sholl_analysis_dir(swc_dir, output_dir):
    img_dir = "/data/kfchen/trace_ws/to_gu/img"
    neuron_info_file = "/data/kfchen/nnUNet/nnUNet_results/Dataset169_hb_10k/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/ptls10/norm_result/Human_SingleCell_TrackingTable_20240712.csv"
    neuron_info = pd.read_csv(neuron_info_file, encoding='gbk')

    # swc_files = file_io.get_files(swc_dir, ".swc")
    swc_files = [os.path.join(swc_dir, f) for f in os.listdir(swc_dir) if f.endswith(".swc")]
    # swc_files = swc_files[:10]
    # for swc_file in swc_files:
    #     output_sholl_result = os.path.join(output_dir, os.path.basename(swc_file).replace(".swc", ".npy"))
    #     test_sholl_analysis_swc(swc_file, output_sholl_result, img_dir, neuron_info)

    with ThreadPoolExecutor(max_workers=10) as executor:
        list(tqdm(executor.map(test_sholl_analysis_swc,
                               swc_files,
                               [os.path.join(output_dir, os.path.basename(f).replace(".swc", ".npy")) for f in swc_files],
                               [img_dir]*len(swc_files),
                               [neuron_info]*len(swc_files),
                                 ), total=len(swc_files))
             )

    # 可视化
    # for swc_file in swc_files:
    #     output_sholl_result = os.path.join(output_dir, os.path.basename(swc_file).replace(".swc", ".npy"))
    #     plot_sholl_result(output_sholl_result)


if __name__ == "__main__":
    # swc_file = r"/data/kfchen/trace_ws/paper_trace_result/manual/origin_anno_swc_sorted_1um/2364.swc"
    # output_sholl_result = "/data/kfchen/trace_ws/paper_trace_result/manual/2364.npy"
    #
    # test_sholl_analysis_swc(swc_file, output_sholl_result)
    # plot_sholl_result(output_sholl_result)

    swc_dir_list = [
        r"/data/kfchen/trace_ws/paper_trace_result/manual/origin_anno_swc_sorted_1um", # 人工原始标注
        r"/data/kfchen/trace_ws/paper_trace_result/nnunet/newcel_0.1/7_scaled_1um_swc", # proposed pipeline
        r"/data/kfchen/trace_ws/paper_auto_human_neuron_recon/swc_label/1um_swc_lab", # gold standard
    ]

    for swc_dir in swc_dir_list:
        # swc_dir = r"/data/kfchen/trace_ws/paper_trace_result/manual/origin_anno_swc_sorted_1um"
        print(f"processing {swc_dir}")
        output_dir = swc_dir + "_sholl_analysis_result"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        sholl_analysis_dir(swc_dir, output_dir)


