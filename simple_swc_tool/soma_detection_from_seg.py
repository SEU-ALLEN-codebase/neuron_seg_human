import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage import measure, morphology
import tifffile
from skimage.transform import resize
import os
import pandas as pd
from scipy.optimize import curve_fit

def find_resolution(df, filename):
    # print(filename)
    filename = filename.split('.')[0]
    for i in range(len(df)):
        if df.iloc[i, 2] == filename:
            return df.iloc[i, 3]
    return (1,1,1)


def exponential_decay(x, a, b, c):
    return a * np.exp(-b * (x-2)) + c

def test_soma_detectison(process_file_pair):
    img_file, seg_file, neuron_info_df, mip_file = process_file_pair

    if(os.path.exists(mip_file.replace('.tif', '_MIP.png'))):
        return
    seg = tifffile.imread(seg_file).astype(np.uint8)
    origin_seg = seg.copy()

    resolution = find_resolution(neuron_info_df, os.path.basename(seg_file))
    resolution = resolution.split(', ')[1]
    resolution = (1, float(resolution), float(resolution))
    # print(resolution)
    # resolution = (1, xy_resolution/1000, xy_resolution/1000)
    origin_img_size = seg.shape
    seg = resize(seg, (seg.shape[0]*resolution[0], seg.shape[1]*resolution[1], seg.shape[2]*resolution[2]), order=0)
    seg = (seg - seg.min()) / (seg.max() - seg.min())
    seg = np.where(seg > 0, 1, 0).astype(np.uint8)
    # 填补空洞
    seg = ndimage.binary_fill_holes(seg).astype(int)

    # 定义核半径列表
    # kernel_radii = [2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10]
    min_radii, max_radii= 2, 15
    kernel_radii = np.linspace(min_radii, max_radii, 20)


    # 存储高频能量占比和高频幅值平均值
    high_freq_ratios = []
    high_freq_averages = []

    # 存储每个阶段的MIP图像
    mip_images = []

    for radius in kernel_radii:
        # print(f"Processing with kernel radius: {radius}")

        # 创建球形结构元素
        struct = morphology.ball(radius)

        # 形态学开运算
        opened_img = morphology.opening(seg, struct)
        opened_img = opened_img * seg
        # eroded_img = morphology.binary_erosion(img, struct)
        # opened_img = eroded_img

        if(np.sum(opened_img) == 0):
            # print(f"Kernel radius {radius} is the best.")
            break
        # save
        # tifffile.imsave(f"C:\\Users\\12626\\Desktop\\fig4\\new\\opened_img_{radius}.tif", opened_img.astype("uint8"))

        # 提取表面网格
        verts, faces, normals, values = measure.marching_cubes(opened_img, level=0)

        # 计算傅里叶描述子
        # 将三维表面网格展开为一维信号
        signal = verts.flatten()

        # 进行傅里叶变换
        F = np.fft.fft(signal)
        N = len(F)

        # 计算幅值谱
        F_magnitude = np.abs(F)

        # 总能量
        E_total = np.sum(F_magnitude ** 2)

        # 设定频率阈值（例如，前10%的频率作为低频）
        k_threshold = int(0.1 * N)

        # 高频能量
        E_high = np.sum(F_magnitude[k_threshold:] ** 2)

        # 高频能量占比
        R = E_high / E_total
        high_freq_ratios.append(R)

        # 平均高频幅值
        A_high = np.mean(F_magnitude[k_threshold:])
        high_freq_averages.append(A_high)

        # 生成沿z轴的MIP
        mip = np.max(opened_img, axis=2)
        mip_images.append(mip)

        # if(high_freq_averages[0] * 0.1 > A_high):
        #     print(f"Kernel radius {radius} is the best.")
        #     break

    fig, ax1 = plt.subplots()
    kernel_radii = kernel_radii[:len(high_freq_ratios)]
    high_freq_averages = np.array(high_freq_averages)
    high_freq_averages = (high_freq_averages - high_freq_averages.min()) / (high_freq_averages.max() - high_freq_averages.min())

    popt, pcov = curve_fit(exponential_decay, kernel_radii, high_freq_averages, p0=(10, 1, 0), maxfev=2000)

    # 生成拟合曲线数据
    x_fit = np.linspace(kernel_radii.min(), kernel_radii.max(), 100)
    y_fit = exponential_decay(x_fit, *popt)

    fitted_gradients = np.gradient(y_fit, x_fit)
    fitted_gradients = (fitted_gradients - fitted_gradients.min()) / (fitted_gradients.max() - fitted_gradients.min())

    # poly_degree = 2
    # coeffs = np.polyfit(kernel_radii, high_freq_averages, poly_degree)
    # poly_fit = np.poly1d(coeffs)
    #
    # # 生成平滑的核半径范围用于绘制拟合曲线
    # kernel_radii_smooth = np.linspace(min(kernel_radii), max(kernel_radii), 100)
    # high_freq_averages_fit = poly_fit(kernel_radii_smooth)


    # # 移动平均
    # window_size = int(len(kernel_radii) / 3)
    # high_freq_averages_smooth = np.convolve(high_freq_averages, np.ones(window_size) / window_size, mode='same')
    # average_gradients = np.gradient(high_freq_averages_smooth, kernel_radii)
    # second_derivatives = np.gradient(average_gradients, kernel_radii)

    # # 找到梯度极大值点
    # max_gradient_index = np.argmin(average_gradients)
    # best_radius = kernel_radii[max_gradient_index]
    # for i in range(max_gradient_index, len(average_gradients)-1):
    #     if average_gradients[i] > average_gradients[i + 1]:
    #         best_radius = kernel_radii[i]
    #         break

    #
    # color = 'tab:red'
    # ax1.set_xlabel('Kernel Radius')
    # ax1.set_ylabel('High-Frequency Energy Ratio', color=color)
    # ax1.plot(kernel_radii, high_freq_ratios, marker='o', color=color)
    # ax1.tick_params(axis='y', labelcolor=color)
    # ax1.set_title('High-Frequency Metrics vs. Kernel Radius')

    ax2 = ax1.twinx()  # 共享x轴

    color = 'tab:blue'
    ax2.set_ylabel('Average High-Frequency Amplitude', color=color)
    ax2.plot(kernel_radii, high_freq_averages, marker='s', color=color)
    # ax2.plot(kernel_radii_smooth, high_freq_averages_fit, color='tab:cyan', linestyle='--',
    #          label=f'Poly Degree {poly_degree} Fit')
    # ax2.plot(kernel_radii, high_freq_averages_smooth, color='tab:cyan', linestyle='--',
    #          label=f'Moving Average')
    ax2.plot(x_fit, y_fit, color='red', label='Exponential Decay Fit')
    ax2.tick_params(axis='y', labelcolor=color)

    ax3 = ax1.twinx()
    ax3.spines['right'].set_position(('outward', 60))
    color = 'tab:green'
    ax3.set_ylabel('Gradient of Avg High-Freq Amplitude', color=color)
    ax3.plot(x_fit, fitted_gradients, color=color, label='Gradient of Avg High-Freq Amplitude')
    # plot line y
    ax3.axhline(y=0.9, color='gray', linestyle='--')
    # gradient == 0.15
    best_radius = x_fit[np.argmin(np.abs(fitted_gradients - 0.9))]
    ax3.axvline(x=best_radius, color='gray', linestyle='--')

    # ax3 = ax1.twinx()
    # # 调整右侧y轴的位置
    # ax3.spines['right'].set_position(('outward', 60))
    # color = 'tab:green'
    # ax3.set_ylabel('Gradient of Avg High-Freq Amplitude', color=color)
    # # ax3.plot(kernel_radii, second_derivatives, marker='^', color=color, label='Gradient of Avg High-Freq Amplitude')
    # ax3.tick_params(axis='y', labelcolor=color)

    # 添加图例
    # fig.legend(loc='upper right', bbox_to_anchor=(1, 1), bbox_transform=ax1.transAxes)

    fig.tight_layout()
    plt.savefig(mip_file.replace('.tif', '_A_high.png'))
    plt.close()

    # fig, axes = plt.subplots(1, len(kernel_radii), figsize=(15, 5))
    #
    # for i, (radius, mip) in enumerate(zip(kernel_radii, mip_images)):
    #     axes[i].imshow(mip, cmap='gray')
    #     axes[i].set_title(f'Radius {radius}')
    #     axes[i].axis('off')
    #
    # plt.suptitle('MIP Images with Different Kernel Radii')
    # plt.show()

    # high_freq_averages_fit的最低点
    result_open_img = morphology.opening(seg, morphology.ball(best_radius))
    result_open_img = result_open_img * seg
    result_img = resize(result_open_img, origin_img_size, order=0)
    result_img = np.where(result_img > 0, 1, 0).astype(np.uint8)
    # plt
    img = tifffile.imread(img_file)
    mip_list = [
        img.max(axis=0),
        img.max(axis=1),
        img.max(axis=2),
        origin_seg.max(axis=0),
        origin_seg.max(axis=1),
        origin_seg.max(axis=2),
        result_img.max(axis=0),
        result_img.max(axis=1),
        result_img.max(axis=2)
    ]



    #
    fig, axes = plt.subplots(3, 3, figsize=(18, 12))
    for i, ax in enumerate(axes.flat):
        ax.imshow(mip_list[i], cmap='gray')
        if(i < 3):
            ax.imshow(mip_list[i+6], cmap='jet', alpha=0.5)  # 调整alpha以控制透明度
        if(i >= 3 and i < 6):
            ax.imshow(mip_list[i+3], cmap='jet', alpha=0.5)  # 调整alpha以控制透明度

        ax.set_title(f'MIP {i+1}')
        ax.axis('off')
    plt.tight_layout()
    # plt.show()
    plt.savefig(mip_file.replace('.tif', '_MIP.png'))
    plt.close()



if __name__ == "__main__":
    dataset_name = "Dataset179_deflu_no_aug"
    name_mapping_file = r"/data/kfchen/nnUNet/nnUNet_raw/" + dataset_name + "/name_mapping.csv"
    df = pd.read_csv(name_mapping_file)

    img_dir = r"/data/kfchen/nnUNet/nnUNet_raw/" + dataset_name + "/imagesTr"
    mip_dir = r"/data/kfchen/nnUNet/nnUNet_results/" + dataset_name + "/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/source500/soma_detection_result"
    if(not os.path.exists(mip_dir)):
        os.makedirs(mip_dir)

    seg_dir = r"/data/kfchen/nnUNet/nnUNet_results/" + dataset_name + "/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/source500/validation"
    seg_files = [f for f in os.listdir(seg_dir) if f.endswith('.tif')]
    # seg_files = seg_files[:10]

    process_file_pairs = []
    for seg_file in seg_files:
        process_file_pairs.append((os.path.join(img_dir, seg_file.replace(".tif", "_0000.tif")), os.path.join(seg_dir, seg_file), df, os.path.join(mip_dir, seg_file)))

    # for process_file_pair in process_file_pairs:
    #     test_soma_detectison(process_file_pair)
    from multiprocessing import Pool
    with Pool(10) as p:
        p.map(test_soma_detectison, process_file_pairs)
