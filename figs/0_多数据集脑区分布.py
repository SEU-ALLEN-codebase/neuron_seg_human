import tifffile
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
from scipy.ndimage import gaussian_filter
import cv2

dataset_info = {
    'Proposed':{
        'brain_region': ["superior frontal gyrus", "middle frontal gyrus", "inferior frontal gyrus",
                        "parietal lobe", "inferior parietal lobe",
                        "superior temporal gyrus", "middle temporal gyrus",
                        'occipital lobe',
                        'temporal lobe',
                        'frontal lobe',
        ],
        'num': 8676,
    },
    'Jacobs':{
        'brain_region': [
            'parietal lobe', 'frontal lobe', 'superior temporal gyrus', 'inferior frontal gyrus'
        ],
        'num': 2621,
    },
    # 'Ellis':{
    #     'brain_region': [
    #         # 涉及到神经元样本培养
    #     ],
    #     'num': 1042,
    # },
    'Vdheuvel':{
        'brain_region': [
            # SFG, MFG, OL, PL, IPL, FL, TL,
            'superior frontal gyrus', 'middle frontal gyrus', 'occipital lobe', 'parietal lobe', 'inferior parietal lobe',
            'frontal lobe', 'temporal lobe',
        ],
        'num': 376,
    },
    'Allen':{
        'brain_region': [
            # IFG MTG tl
            'inferior frontal gyrus', 'middle temporal gyrus', 'temporal lobe',
        ],
        'num': 303,
    },
    "Mechawar":{
        'brain_region': [
            'frontal lobe',
        ],
        'num': 240,
    },
    "Helmstaedter":{
        'brain_region': [
            'frontal lobe', 'temporal lobe',
        ],
        'num': 215,
    },
    # "Ataman": {
    #     'brain_region': [
    #     ],
    #     'num': 208,
    # },
    # others: 921

}
# interested_categories = [89, 52, 73, 60, 61, 79, 54, 85, 54, 73]
# categories_names = ["superior frontal gyrus", "middle frontal gyrus", "inferior frontal gyrus",
#                     "parietal lobe", "inferior parietal lobe",
#                     "superior temporal gyrus", "middle temporal gyrus",
#                     'occipital lobe',
#                     'temporal lobe',
#                     'frontal lobe',
#                     ]

Set3_colors = list(plt.cm.get_cmap('Set3').colors[:10])
Set3_colors[-2] = plt.cm.get_cmap('Set1').colors[-1]
atlas_info = {
    "superior frontal gyrus": {
        "atlas_id": [89],
        "color": [0, 0, 255]
    },
    "middle frontal gyrus": {
        "atlas_id": [52],
        "color": [0, 255, 0]
    },
    "inferior frontal gyrus": {
        "atlas_id": [73],
        "color": [255, 0, 0]
    },
    "parietal lobe": {
        "atlas_id": [60],
        "color": [255, 255, 0]
    },
    "inferior parietal lobe": {
        "atlas_id": [61],
        "color": [255, 0, 255]
    },
    "superior temporal gyrus": {
        "atlas_id": [79],
        "color": [0, 255, 255]
    },
    "middle temporal gyrus": {
        "atlas_id": [54],
        "color": [255, 255, 0]
    },
    'occipital lobe': {
        "atlas_id": [85],
        "color": [0, 255, 255]
    },
    'temporal lobe': {
        "atlas_id": [54, 79],
        "color": [255, 0, 255]
    },
    'frontal lobe': {
        "atlas_id": [89, 52, 73],
        "color": [255, 0, 0]
    },
}
# for i in range(10):
#     atlas_info[categories_names[i]]['color'] = colors[i]
for i, key in enumerate(atlas_info.keys()):
    atlas_info[key]['color'] = Set3_colors[i]



atlas_file = "/data/kfchen/trace_ws/atlas_test/mni_annotation.tif"
atlas = tifffile.imread(atlas_file)
atlas = atlas.astype(np.uint8)

def get_projection(img, view_direction):
    z_shape, y_shape, x_shape = img.shape
    if view_direction == 'xz':
        projection = np.ones((z_shape, x_shape), dtype=np.uint8) * 255
        for x, z in np.ndindex((x_shape, z_shape)):
            y_idx = np.argmax(img[z, :, x] != 0)
            if y_idx != 0:
                projection[z, x] = img[z, y_idx, x]
    elif(view_direction == 'xy'):
        projection = np.ones((z_shape, x_shape), dtype=np.uint8) * 255
        for x, y in np.ndindex((x_shape, y_shape)):
            z_idx = np.argmax(img[:, y, x] != 0)
            if z_idx != 0:
                projection[y, x] = img[z_idx, y, x]
    elif(view_direction == 'yz'):
        projection = np.ones((z_shape, x_shape), dtype=np.uint8) * 255
        for y, z in np.ndindex((y_shape, z_shape)):
            x_idx = np.argmax(img[z, y, :] != 0)
            if x_idx != 0:
                projection[y, z] = img[z, y, x_idx]
    return projection.astype(np.uint8)

def upsample_and_smooth(projection, factor=2):
    projection = zoom(projection, factor, order=0)
    projection = gaussian_filter(projection, sigma=1)
    return projection

def get_background():
    projection = get_projection(atlas, 'xy')
    projection = upsample_and_smooth(projection)
    projection = projection.astype(np.float32)
    projection = (projection - projection.min()) / (projection.max() - projection.min()) * 0.5 + 0.5
    projection = np.stack([projection, projection, projection], axis=-1)
    return projection

def get_brain_region_color_mask(current_brain_region, alpha = 0.7):
    img = atlas
    img = zoom(img, (2, 2, 2), order=0)
    colored_mip = np.zeros((img.shape[1], img.shape[2], 3), dtype=np.float32)
    # colored_mip.astype(np.float32)
    for row in range(colored_mip.shape[0]):
        for col in range(colored_mip.shape[1]):
            if colored_mip[row, col, :].sum() == 0:
                colored_mip[row, col, :] = 1

    for category in current_brain_region:
        # mask = (img == atlas_info[category]['atlas_id'][0])
        mask = np.zeros_like(img)
        for atlas_id in atlas_info[category]['atlas_id']:
            curr_mask = (img == atlas_id)
            mask = mask + curr_mask

        mip_image = np.max(mask, axis=0).astype(np.uint8)
        mip_image = cv2.medianBlur(mip_image, 15)
        if(len(atlas_info[category]['atlas_id']) > 1):
            current_alpha = alpha * 3
            mip_image = mip_image * 255
            edges = cv2.Canny(mip_image, 10, 200)  # 阈值可根据实际需求调整
            kernel = np.ones((2, 2), np.uint8)
            edges = cv2.dilate(edges, kernel, iterations=5)
            edges = edges / 255
            mip_image = edges
        else:
            current_alpha = alpha

        # color = atlas_info[categories_names[category]]['color']
        mip_image = mip_image.astype(np.float32)

        color = atlas_info[category]['color']
        for j in range(3):
            if (len(atlas_info[category]['atlas_id']) == 1):
                colored_mip[:, :, j] = colored_mip[:, :, j] * (1 - current_alpha * mip_image) + current_alpha * color[j] * mip_image
            else:
                # print(np.max(color[j]), np.min(color[j]))
                # print(np.max(colored_mip[:, :, j]), np.min(colored_mip[:, :, j]))
                # print(np.max(mip_image), np.min(mip_image))
                # print(np.max((color[j] * mip_image)), np.min((color[j] * mip_image)))
                colored_mip[:, :, j] = np.where(mip_image, color[j] * mip_image, colored_mip[:, :, j])
    # colored_mip = np.clip(colored_mip, 0, 1)

    return colored_mip

if __name__ == "__main__":
    # 设置清晰度
    plt.rcParams['savefig.dpi'] = 300
    dataset_keys = list(dataset_info.keys())
    if(False):
        col = 2
        row = int(np.ceil(len(dataset_keys) / col))
    else:
        row = 1
        col = int(np.ceil(len(dataset_keys) / row))
    ax_fig_size = (3, 3)

    fig, axes = plt.subplots(row, col, figsize=(col*ax_fig_size[0], row*ax_fig_size[1]))
    axes = axes.flatten()
    for i, key in enumerate(dataset_keys):
        ax = axes[i]
        # ax.imshow(get_background(), cmap='gray', aspect='equal', vmin=0, vmax=128)
        background = get_background()
        color_mask = get_brain_region_color_mask(dataset_info[key]['brain_region'])
        # 叠加
        back_ground_alpha = 0.2
        final_image = background * back_ground_alpha + color_mask * (1 - back_ground_alpha)
        ax.imshow(final_image)
        ax.set_title(key + " (n=" + str(dataset_info[key]['num']) + ")", fontsize=15)
        ax.axis('off')

    # 关闭多余的子图
    for i in range(len(dataset_keys), len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()
    # plt.show()
    plt.savefig("/data/kfchen/trace_ws/atlas_test/brain_region_distribution.png")
    plt.close()

