from v3dpy.terafly import TeraflyInterface
import numpy as np
import tifffile
import os
import pandas as pd
from collections import deque
from skimage.transform import resize
import joblib
from tqdm import tqdm

mouse_neuron_info_file = "/data/kfchen/trace_ws/img_noise_test/seu1876/41467_2024_54745_MOESM3_ESM.xlsx"
mouse_neuron_info_df = pd.read_excel(mouse_neuron_info_file)


def crop_tera_and_swc(neuron_dir, lim, target_img_dir, target_swc_dir):
    def find_xy_resolution(neuron_dir):
        brain_id = os.path.basename(neuron_dir).split('_')[0]
        if(brain_id == 'pre'):
            brain_id = os.path.basename(neuron_dir).split('_')[1]
        brain_id = int(brain_id)
        xy_resolution = mouse_neuron_info_df.loc[
            mouse_neuron_info_df['Image ID'] == int(brain_id), 'Resolution_XY (𝜇𝑚/voxel)'].values[0]
        return xy_resolution

    def find_tera_dir(neuron_dir, rank=0):
        tera_dir_root = os.path.join(neuron_dir, 'L0Data')
        tera_dirs = os.listdir(tera_dir_root)
        total_x_shapes = [f.split('x')[0][4:] for f in tera_dirs]
        total_y_shapes = [f.split('x')[1] for f in tera_dirs]
        total_z_shapes = [f.split('x')[2][:-1] for f in tera_dirs]
        total_sizes = [(int(x)*int(y)*int(z)) for x, y, z in zip(total_x_shapes, total_y_shapes, total_z_shapes)]
        tera_df = pd.DataFrame()
        tera_df['dir'], tera_df['size'] = tera_dirs, total_sizes
        tera_df = tera_df.sort_values(by='size', ascending=False)
        tera_dir = os.path.join(tera_dir_root, tera_df.iloc[rank]['dir'])
        return tera_dir

    def find_swc_file(neuron_dir):
        swc_dir = os.path.join(neuron_dir, 'L2Data')
        swc_files = [f for f in os.listdir(swc_dir) if f.endswith('.swc') and "refined" in f]
        swc_file = os.path.join(swc_dir, swc_files[0])
        # print(swc_file)
        return swc_file

    def crop_img(swc, lim):
        soma = swc[swc.type == 1 & (swc.parent == -1)].iloc[0]
        soma_x, soma_y, soma_z = soma.x, soma.y, soma.z
        # print("???", soma_x, soma_y, soma_z)

        t = TeraflyInterface(find_tera_dir(neuron_dir))
        x_lim, y_lim, z_lim = lim
        start = np.array([soma_x - x_lim / 2, soma_y - y_lim / 2, soma_z - z_lim / 2])
        end = np.array([soma_x + x_lim / 2, soma_y + y_lim / 2, soma_z + z_lim / 2])
        # 4D image, indexed by c, z, y, x
        img = t.get_sub_volume(start[0], end[0], start[1], end[1], start[2], end[2])
        return img

    def crop_swc(swc, lim):
        def bfs(soma_id, df):
            """广度优先搜索，查找从给定soma开始的所有可达节点"""
            visited = set()  # 用来存储已经访问过的节点
            queue = deque([soma_id])  # 队列初始化为只包含soma的ID

            while queue:
                current_node = queue.popleft()  # 弹出队列中的第一个元素
                if current_node not in visited:
                    visited.add(current_node)
                    # 获取当前节点的所有子节点
                    children = df[df.parent == current_node]
                    for child_id in children['id']:
                        if child_id not in visited:
                            queue.append(child_id)

            return visited

        x_lim, y_lim, z_lim = lim
        df = swc
        soma = df[df.type == 1 & (df.parent == -1)]
        assert len(soma) == 1
        soma = soma.iloc[0]
        soma_x, soma_y, soma_z = soma[['x', 'y', 'z']]

        x_min, x_max = soma_x - x_lim / 2, soma_x + x_lim / 2
        y_min, y_max = soma_y - y_lim / 2, soma_y + y_lim / 2
        z_min, z_max = soma_z - z_lim / 2, soma_z + z_lim / 2
        df_crop = df[(df.x >= x_min) & (df.x <= x_max) &
                     (df.y >= y_min) & (df.y <= y_max) &
                     (df.z >= z_min) & (df.z <= z_max)]

        soma_id = soma['id']
        reachable_nodes = bfs(soma_id, df_crop)
        df_crop = df_crop[df_crop['id'].isin(reachable_nodes)]

        # relocate swc position
        df_crop.x = df_crop.x - x_min
        df_crop.y = df_crop.y - y_min
        df_crop.z = df_crop.z - z_min

        return df_crop

    target_img_file = os.path.join(target_img_dir, os.path.basename(neuron_dir).split('.')[0] + '.tif')
    target_swc_file = os.path.join(target_swc_dir, os.path.basename(neuron_dir).split('.')[0] + '.swc')
    if(os.path.exists(target_img_file) and os.path.exists(target_swc_file)):
        return

    try:
        swc_file = find_swc_file(neuron_dir)
        # print(os.path.exists(swc_file))
        swc = pd.read_csv(swc_file, sep=' ', header=None, comment='#',
                         names=['id', 'type', 'x', 'y', 'z', 'radius', 'parent'],
                         dtype={'id': int, 'type': int, 'x': float, 'y': float, 'z': float, 'radius': float, 'parent': int})

        xy_resolution = find_xy_resolution(neuron_dir)
        # print(xy_resolution)

        # crop & save img
        img = crop_img(swc, lim).astype(np.float32)[0]
        img = (img - img.min()) / (img.max() - img.min())
        img = resize(img, (img.shape[0], int(img.shape[1]*xy_resolution), int(img.shape[2]*xy_resolution)), order=2)
        img = (img - img.min()) / (img.max() - img.min()) * 255
        img = np.flip(img, axis=1)
        img = img.astype('uint8')
        # print(img.shape)
        # tifffile.imwrite('/data/kfchen/trace_ws/img_noise_test/tera_crop.tif', img)
        tifffile.imwrite(target_img_file, img)

        # crop swc
        swc = crop_swc(swc, lim)
        # resize swc
        swc.x = swc.x * xy_resolution
        swc.y = swc.y * xy_resolution
        # swc.to_csv('/data/kfchen/trace_ws/img_noise_test/tera_crop.swc', sep=' ', header=False, index=False)
        # swc.to_csv(os.path.join(target_swc_dir, os.path.basename(neuron_dir).split('.')[0] + '.swc'), sep=' ', header=False, index=False)
        swc.to_csv(target_swc_file, sep=' ', header=False, index=False)
    except Exception as e:
        print(e, neuron_dir)


if __name__ == '__main__':
    target_img_dir = "/data/kfchen/trace_ws/img_noise_test/seu1876/cropped_img_1um_bigger"
    target_swc_dir = "/data/kfchen/trace_ws/img_noise_test/seu1876/cropped_swc_1um_bigger"
    os.makedirs(target_img_dir, exist_ok=True)
    os.makedirs(target_swc_dir, exist_ok=True)

    # neuron_dir = "/PBshare/SEU-ALLEN/Projects/fullNeurons/L0/R1740/17109_1701_x8048_y22277.swc"
    source_root = ['/PBshare/SEU-ALLEN/Projects/fullNeurons/L0/R1740', "/PBshare/SEU-ALLEN/Projects/fullNeurons/L0/R151"]
    neuron_dirs = []
    for root in source_root:
        neuron_dirs.extend([os.path.join(root, f) for f in os.listdir(root) if f.endswith('.swc')])

    # for neuron_dir in neuron_dirs:
    #     crop_tera_and_swc(neuron_dir, (512, 512, 128), target_img_dir, target_swc_dir)

    joblib.Parallel(n_jobs=20)(
        joblib.delayed(crop_tera_and_swc)(neuron_dir, (1024, 1024, 128), target_img_dir, target_swc_dir) for neuron_dir in tqdm(neuron_dirs))