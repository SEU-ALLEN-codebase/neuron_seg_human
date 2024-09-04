import tifffile
from pathlib import Path
import neurom as nm
from neurom import morphmath as mm
from nnUNet.scripts.mip import get_mip_swc
import numpy as np
import os

PACKAGE_DIR = Path(__file__).resolve().parent.parent


def sec_len(sec):
    """Return the length of a section."""
    return mm.section_length(sec.points)

def save_as_swc(neuron, file_path, sections_to_remove=[]):
    """Save the neuron data to a SWC file."""
    node_id = 1
    with open(file_path, 'w') as f:
        f.write("# SWC file saved from modified NeuroM data\n")
        soma = neuron.soma
        f.write(f"{node_id} 1 {soma.center[0]:.2f} {soma.center[1]:.2f} {soma.center[2]:.2f} {soma.radius:.2f} -1\n")

        dfs_stack = []

        for sec in nm.iter_sections(neuron):
            # print(sec.parent)
            if(sec.parent is None):
                dfs_stack.append((sec, node_id))

        while len(dfs_stack) > 0:
            top_sec, parent_id = dfs_stack.pop()
            if top_sec in sections_to_remove:
                continue
            if (top_sec.parent is None):
                node_id = node_id + 1
                f.write(f"{node_id} 2 {top_sec.points[0][0]:.2f} {top_sec.points[0][1]:.2f} {top_sec.points[0][2]:.2f} {top_sec.points[0][3]:.2f} {parent_id}\n")
                for point in top_sec.points[1:]:
                    node_id = node_id + 1
                    f.write(f"{node_id} 2 {point[0]:.2f} {point[1]:.2f} {point[2]:.2f} {point[3]:.2f} {node_id - 1}\n")
            else:
                node_id = node_id + 1
                f.write(
                    f"{node_id} 2 {top_sec.points[1][0]:.2f} {top_sec.points[1][1]:.2f} {top_sec.points[1][2]:.2f} {top_sec.points[1][3]:.2f} {parent_id}\n")
                for point in top_sec.points[2:]:
                    node_id = node_id + 1
                    f.write(f"{node_id} 2 {point[0]:.2f} {point[1]:.2f} {point[2]:.2f} {point[3]:.2f} {node_id - 1}\n")

            for child in top_sec.children:
                dfs_stack.append((child, node_id))


def prune_neuron(swc_file, swc_save_file, mip_swc_file, img_file):
    m = nm.load_morphology(swc_file)

    total_neurite_length = sum(sec_len(s) for s in nm.iter_sections(m))
    # print("Total neurite length (sections):", total_neurite_length)

    del_thres = 0.01
    del_l_thres = 10
    sections_to_remove = []

    for i, s in enumerate(nm.iter_sections(m)):
        s_lengths = mm.section_length(s.points)
        if len(s.children) > 0:  # 非叶子section
            continue
        # print(f"section {i} length: {s_lengths}")
        if s_lengths < del_l_thres or len(s.points) < 3: # 删除长度小于阈值的叶子section
            # print(f"delete section {i} length: {s_lengths}")
            # print(f"delete section {i} points: {len(s.points)}")
            # 记录需要删除的叶子section
            sections_to_remove.append(s)

    # # 删除记录的叶子sections
    # for sec in sections_to_remove:
    #     parent = sec.parent
    #     if parent:
    #         parent.children.remove(sec)

    # 保存修改后的结构

    save_as_swc(m, swc_save_file, sections_to_remove)

    # mip

    img = tifffile.imread(img_file)
    swc_mip1 = get_mip_swc(swc_file, img)
    swc_mip2 = get_mip_swc(swc_save_file, img)
    concat_img = np.concatenate((swc_mip1, swc_mip2), axis=1)
    tifffile.imsave(mip_swc_file, concat_img)

def compare_swc(swc_file1, swc_file2):
    m1 = nm.load_morphology(swc_file1)
    total_neurite_length1 = sum(sec_len(s) for s in nm.iter_sections(m1))
    m2 = nm.load_morphology(swc_file2)
    total_neurite_length2 = sum(sec_len(s) for s in nm.iter_sections(m2))
    # print("length1: ", total_neurite_length1, "length2: ", total_neurite_length2)
    print(f"{total_neurite_length2/total_neurite_length1:.2f}")


if __name__ == '__main__':
    # swc_dir = r"/data/kfchen/nnUNet/nnUNet_results/Dataset169_hb_10k/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/ptls10/validation_traced/v3dswc"
    # swc_files = os.listdir(swc_dir)
    # for swc_file in swc_files:
    #     swc_file = os.path.join(swc_dir, swc_file)
    #     result_lines = []
    #     with open(swc_file, 'r') as f:
    #         lines = f.readlines()
    #         for line in lines:
    #             if line.startswith("#"):
    #                 continue
    #             line = line.split()
    #             if(line[6] == '-1'):
    #                 line[1] = '1'
    #             else:
    #                 line[1] = '2'
    #             # print(line)
    #             result_lines.append(line)
    #     # save
    #     with open(swc_file, 'w') as f:
    #         for line in result_lines:
    #             f.write(' '.join(line) + '\n')
    # exit()

    # swc_dir = r"/data/kfchen/trace_ws/neurom_ws/new_sort/2_sort"
    # pruned_swc_dir = r"/data/kfchen/trace_ws/neurom_ws/new_sort/pruned_swc"

    swc_dir = r"/data/kfchen/nnUNet/nnUNet_results/Dataset169_hb_10k/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/ptls10/norm_result/unified_GS"
    pruned_swc_dir = r"/data/kfchen/nnUNet/nnUNet_results/Dataset169_hb_10k/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/ptls10/norm_result/pruned_unified_GS"

    mip_swc_dir = r"/data/kfchen/trace_ws/neurom_ws/mip"
    img_dir = r"/data/kfchen/trace_ws/to_gu/img"
    if(os.path.exists(pruned_swc_dir) == False):
        os.mkdir(pruned_swc_dir)
    if (os.path.exists(mip_swc_dir) == False):
        os.mkdir(mip_swc_dir)

    swc_files = os.listdir(swc_dir)
    for swc_file in swc_files:
        swc_file = os.path.join(swc_dir, swc_file)
        swc_save_file = os.path.join(pruned_swc_dir, os.path.basename(swc_file))
        mip_swc_file = os.path.join(mip_swc_dir, os.path.basename(swc_file).replace('.swc', '_mip.png'))
        img_file = os.path.join(img_dir, os.path.basename(swc_file).replace('.swc', '.tif'))

        prune_neuron(swc_file, swc_save_file, mip_swc_file, img_file)

        # compare_swc(swc_file, swc_save_file)



