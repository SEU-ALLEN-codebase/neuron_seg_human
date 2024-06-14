import os
from tqdm import tqdm
from simple_swc_tool.swc_io import read_swc, write_swc
import shutil
from nnunetv2.dataset_conversion import generate_mask
import tifffile
import threading
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from PIL import Image
import tifffile as tiff


def rename_swc_folder(swc_folder, new_format="brainid_{num:06}_x_{x:06}_y_{y:06}_z_{z:06}.swc"):
    swc_files = sorted(os.listdir(swc_folder))

    process_bar = tqdm(total=len(swc_files))

    for swc_file in swc_files:
        process_bar.update(1)
        if swc_file.endswith(".swc"):
            brain_name = int(swc_file.split("_")[0])
            # if brain_name not in brain_swc_num:
            #     brain_swc_num[brain_name] = 1
            # else:
            #     brain_swc_num[brain_name] += 1

            point_l = read_swc(os.path.join(swc_folder, swc_file))
            soma_x, soma_y, soma_z = point_l.p[1].x, point_l.p[1].y, point_l.p[1].z
            soma_x, soma_y, soma_z = int(soma_x), int(soma_y), int(soma_z)

            new_name = new_format.format(num=brain_name, x=soma_x, y=soma_y, z=soma_z)
            os.rename(os.path.join(swc_folder, swc_file), os.path.join(swc_folder, new_name))

    process_bar.close()

def find_tif_file(brain_name, soma_x, soma_y, soma_z, crop_tif_folder):
    crop_tif_folder = os.path.join(crop_tif_folder, f"{int(brain_name)}")
    tif_files = os.listdir(crop_tif_folder)
    min_dis = 300
    min_tif_file = None
    for tif_file in tif_files:
        if tif_file.endswith(".tif"):
            tif_x, tif_y, tif_z = tif_file.split("_")[3], tif_file.split("_")[5], tif_file.split("_")[7][:-4]
            tif_x, tif_y, tif_z = int(tif_x), int(tif_y), int(tif_z)
            dis = (tif_x - soma_x) ** 2 + (tif_y - soma_y) ** 2 + (tif_z - soma_z) ** 2
            if dis < min_dis:
                min_dis = dis
                min_tif_file = tif_file

    return min_tif_file

# 偏移
def crop_swc_points(swc_file, result_swc_folder, offset_x, offset_y, offset_z, x_shape=256, y_shape=256, z_shape=256):
    point_l = read_swc(swc_file)
    for point in point_l.p:
        point.x = x_shape - (point.x - offset_x + x_shape / 2)
        point.y = point.y - offset_y + y_shape / 2
        point.z = z_shape - (point.z - offset_z + z_shape / 2)

        if(point.x < 0 or point.x >= x_shape or point.y < 0 or point.y >= y_shape or point.z < 0 or point.z >= z_shape):
            point_l.prune_point(point.n)
    write_swc(os.path.join(result_swc_folder, os.path.basename(swc_file)), point_l)

def generate_mask_from_swc(swc_file, img_file, seg_file):
    target_img_path = img_file
    origin_swc_path = swc_file
    target_lab_path = seg_file

    # flip_path = generate_mask.flip_and_resize_swc(target_img_path, origin_swc_path, scale_factors, pad_width, target_swc_path)
    flip_path = generate_mask.flip_swc(target_img_path, origin_swc_path)
    sort_path = generate_mask.sort_swc(target_img_path, flip_path)

    soma_region_path = generate_mask.simple_soma_region(target_img_path)
    #
    radius_path = generate_mask.calc_radius(target_img_path, sort_path)
    killed_soma_swc_path = generate_mask.kill_point_in_soma(radius_path, soma_region_path)
    #
    ano_path = generate_mask.swc2img(target_img_path, killed_soma_swc_path)
    ano_path3 = generate_mask.or_img(ano_path, soma_region_path)
    #
    dilate_path = generate_mask.dilate_img(ano_path3)
    mask_path = generate_mask.and_img(target_img_path, dilate_path)
    #
    ano_path2 = generate_mask.swc2img(target_img_path, sort_path)
    mask_path2 = generate_mask.or_img(mask_path, ano_path2)
    #
    dust_path = generate_mask.dust_img(mask_path2, 3, target_lab_path)
    dust = tifffile.imread(dust_path)
    tifffile.imwrite(dust_path, dust.astype('uint8'), compression='zlib')

    if(os.path.exists(flip_path)):os.remove(flip_path)
    if (os.path.exists(sort_path)): os.remove(sort_path)
    if(os.path.exists(soma_region_path)): os.remove(soma_region_path)
    if (os.path.exists(radius_path)): os.remove(radius_path)
    if (os.path.exists(killed_soma_swc_path)): os.remove(killed_soma_swc_path)
    if (os.path.exists(ano_path)): os.remove(ano_path)
    if (os.path.exists(ano_path3)): os.remove(ano_path3)
    if (os.path.exists(dilate_path)): os.remove(dilate_path)
    if (os.path.exists(mask_path)): os.remove(mask_path)
    if (os.path.exists(ano_path2)): os.remove(ano_path2)
    if (os.path.exists(mask_path2)): os.remove(mask_path2)


def crop_swc_file(swc_file, new_id, origin_swc_folder, result_swc_folder, crop_tif_folder, result_tif_folder, seg_folder):
    if not swc_file.endswith(".swc"):
        return
    new_swc_name = new_seg_name = f"mb1891_{new_id:04}"
    new_tif_name = f"mb1891_{new_id:04}_0000"

    if(os.path.exists(os.path.join(result_swc_folder, new_swc_name + ".swc")) and \
       os.path.exists(os.path.join(result_tif_folder, new_tif_name + ".tif")) and \
       os.path.exists(os.path.join(seg_folder, new_seg_name + ".tif"))):
        return
    brain_name, soma_x, soma_y, soma_z = swc_file.split("_")[1], swc_file.split("_")[3], swc_file.split("_")[5], \
    swc_file.split("_")[7][:-4]
    brain_name, soma_x, soma_y, soma_z = int(brain_name), int(soma_x), int(soma_y), int(soma_z)
    tif_file = find_tif_file(brain_name, soma_x, soma_y, soma_z, crop_tif_folder)
    if (tif_file is None):
        return

    if(not os.path.exists(os.path.join(result_swc_folder, new_swc_name + ".swc"))):
        # print(tif_file)
        tif_x, tif_y, tif_z = tif_file.split("_")[3], tif_file.split("_")[5], tif_file.split("_")[7][:-4]
        tif_x, tif_y, tif_z = int(tif_x), int(tif_y), int(tif_z)
        crop_swc_points(os.path.join(origin_swc_folder, swc_file), result_swc_folder, tif_x, tif_y, tif_z)

        os.rename(os.path.join(result_swc_folder, os.path.basename(swc_file)),
                  os.path.join(result_swc_folder, new_swc_name + ".swc"))

    if(not os.path.exists(os.path.join(result_tif_folder, new_tif_name + ".tif"))):
        # shutil.copy(os.path.join(crop_tif_folder, str(brain_name), tif_file), os.path.join(result_tif_folder, new_tif_name + ".tif"))
        img = tifffile.imread(os.path.join(crop_tif_folder, str(brain_name), tif_file))
        # to 0-255
        img = (img - img.min()) / (img.max() - img.min()) * 255
        tifffile.imwrite(os.path.join(result_tif_folder, new_tif_name + ".tif"), img.astype('uint8'))

    if(not os.path.exists(os.path.join(seg_folder, new_seg_name + ".tif"))):
        generate_mask_from_swc(os.path.join(result_swc_folder, new_swc_name + ".swc"),
                               os.path.join(result_tif_folder, new_tif_name + ".tif"),
                               os.path.join(seg_folder, new_seg_name + ".tif"))


def crop_swc_folder(origin_swc_folder, result_swc_folder, crop_tif_folder, result_tif_folder, seg_folder):
    if(not os.path.exists(result_swc_folder)):
        os.makedirs(result_swc_folder)
    if(not os.path.exists(result_tif_folder)):
        os.makedirs(result_tif_folder)
    if(not os.path.exists(seg_folder)):
        os.makedirs(seg_folder)
    swc_files = sorted(os.listdir(origin_swc_folder))
    # swc_files = swc_files[:10]
    new_ids = [i for i in range(1, len(swc_files) + 1)]

    # process_bar = tqdm(total=len(swc_files))
    # for swc_file, new_id in zip(swc_files, new_ids):
    #     process_bar.update(1)
    #     crop_swc_file(swc_file, new_id, origin_swc_folder, result_swc_folder, crop_tif_folder, result_tif_folder, seg_folder)
    # process_bar.close()

    # 设置进度条
    process_bar = tqdm(total=len(swc_files))

    # 使用 ThreadPoolExecutor 来并发处理文件
    with ThreadPoolExecutor(max_workers=12) as executor:
        # 创建一个字典来保存每个提交的任务的future
        future_to_file = {
            executor.submit(crop_swc_file, swc_file, new_id, origin_swc_folder, result_swc_folder, crop_tif_folder,
                            result_tif_folder, seg_folder): swc_file for swc_file, new_id in zip(swc_files, new_ids)}

        # 通过as_completed迭代future
        for future in as_completed(future_to_file):
            swc_file = future_to_file[future]
            try:
                data = future.result()
            except Exception as exc:
                print(f'{swc_file} generated an exception: {exc}')
            else:
                # 更新进度条
                process_bar.update(1)

    # 关闭进度条
    process_bar.close()

def compute_mip(image_path):
    """计算给定图像路径的最大强度投影（MIP）"""
    img = tiff.imread(image_path)
    mip = np.max(img, axis=1)  # 假设Z轴为0轴
    return Image.fromarray(mip)

def mip_images(image_folder, seg_folder, output_folder):
    """处理图像和分割结果，生成并保存MIP拼接图"""
    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 获取图像和分割结果文件名列表
    image_files = sorted([f for f in os.listdir(image_folder) if f.endswith('.tif')])
    seg_files = sorted([f for f in os.listdir(seg_folder) if f.endswith('.tif')])

    for img_file, seg_file in zip(image_files, seg_files):
        # 计算图像和分割的MIP
        img_mip = compute_mip(os.path.join(image_folder, img_file))
        seg_mip = compute_mip(os.path.join(seg_folder, seg_file))

        # 拼接图像和分割的MIP
        combined_image = Image.new('RGB', (img_mip.width + seg_mip.width, img_mip.height))
        combined_image.paste(img_mip, (0, 0))
        combined_image.paste(seg_mip, (img_mip.width, 0))

        # 保存结果
        combined_image.save(os.path.join(output_folder, f'combined_{img_file}'))


# 随机分测试集和训练集
def split_train_test(source_folder, train_folder, test_folder, train_ratio=0.8):
    tif_files = sorted(os.listdir(source_folder))
    for tif_file in tif_files:
        if tif_file.endswith(".tif"):
            if np.random.rand() < train_ratio:
                shutil.copy(os.path.join(source_folder, tif_file), os.path.join(train_folder, tif_file))
            else:
                shutil.copy(os.path.join(source_folder, tif_file), os.path.join(test_folder, tif_file))


if __name__ == '__main__':
    manual_swc_folder = r"/PBshare/SEU-ALLEN/Users/KaifengChen/1891/refined_swc"
    # rename_swc_folder(manual_swc_folder)

    crop_tif_folder = r"/PBshare/SEU-ALLEN/Users/xq3"
    result_swc_folder = r"/PBshare/SEU-ALLEN/Users/KaifengChen/1891/crop_swc"
    result_tif_folder = r"/PBshare/SEU-ALLEN/Users/KaifengChen/1891/crop_tif"
    seg_folder = r"/PBshare/SEU-ALLEN/Users/KaifengChen/1891/seg"
    crop_swc_folder(manual_swc_folder, result_swc_folder, crop_tif_folder, result_tif_folder, seg_folder)

    mip_output_folder = r"/PBshare/SEU-ALLEN/Users/KaifengChen/1891/mip"
    # mip_images(result_tif_folder, seg_folder, mip_output_folder)

    train_dir = "/data/kfchen/nnUNet/nnUNet_raw/Dataset401_mb1891/imagesTr"
    test_dir = "/data/kfchen/nnUNet/nnUNet_raw/Dataset401_mb1891/imagesTs"
    source_dir = "/data/kfchen/nnUNet/nnUNet_raw/Dataset401_mb1891/tif"
    split_train_test(source_dir, train_dir, test_dir, train_ratio=0.8)

