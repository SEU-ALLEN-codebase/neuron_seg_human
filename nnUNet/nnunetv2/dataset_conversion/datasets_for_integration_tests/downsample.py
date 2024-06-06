import nibabel as nib
import numpy as np
import os
import scipy.ndimage
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm


def resample_image(image, zoom_factor=0.5, interpolation='cubic'):
    """对图像进行下采样。

    参数:
        image (Nifti1Image): 要下采样的nii图像。
        zoom_factor (float): 下采样因子，小于1表示缩小。
        interpolation (str): 插值方法，'cubic'为双三次插值。
    返回:
        Nifti1Image: 下采样后的图像。
    """
    # 提取图像数据
    data = image.get_fdata()
    # 执行下采样
    if interpolation == 'cubic':
        resampled_data = scipy.ndimage.zoom(data, zoom_factor, order=3)  # 双三次插值
    else:
        resampled_data = scipy.ndimage.zoom(data, zoom_factor, order=0)  # 默认为最邻近插值
        resampled_data = (resampled_data - resampled_data.min()) / (resampled_data.max() - resampled_data.min())
    # 创建新的NIfTI图像
    resampled_image = nib.Nifti1Image(resampled_data, image.affine)
    return resampled_image


def process_image_file(file_path, output_folder, interpolation):
    """处理单个图像文件的下采样并保存。

    参数:
        file_path (str): 输入图像的文件路径。
        output_folder (str): 输出图像的文件夹路径。
        interpolation (str): 插值方法。
    """
    image = nib.load(file_path)
    resampled_image = resample_image(image, zoom_factor=0.5, interpolation=interpolation)
    output_path = os.path.join(output_folder, os.path.basename(file_path))
    nib.save(resampled_image, output_path)


def resample_folder(input_folder, output_folder, interpolation):
    """使用多线程处理文件夹中所有nii.gz图像的下采样。

    参数:
        input_folder (str): 输入图像的文件夹路径。
        output_folder (str): 输出图像的文件夹路径。
        interpolation (str): 插值方法。
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    files = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith('.nii.gz')]
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        list(tqdm(executor.map(lambda file: process_image_file(file, output_folder, interpolation), files),
                  total=len(files)))


# 设置文件夹路径
img_folder = 'img'
seg_folder = 'seg'
output_img_folder = 'resampled_img'
output_seg_folder = 'resampled_seg'

# 处理图像文件夹
resample_folder(img_folder, output_img_folder, interpolation='cubic')
# 处理分割文件夹
resample_folder(seg_folder, output_seg_folder, interpolation='cubic')
