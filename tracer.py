"""End-to-end neuron reconstruction from segmentation volumes to SWC.

This script implements a complete *image → segmentation → morphology (SWC)* pipeline
that is designed to be used after nnUNetv2 inference.

High-level workflow
-------------------
Given a directory of 3D neuron segmentation volumes (one file per neuron), the
pipeline performs the following steps for each volume:

1. Prepare segmentation volumes under ``0_seg/``.
2. Skeletonize the binary segmentation (``1_skel_seg/``).
3. Detect the soma region and generate a soma marker
   (``2_soma_region/`` + ``4_soma_marker/``).
4. Merge skeleton and soma into a single mask (``3_skel_with_soma/``).
5. Call Vaa3D APP2 to trace a tree structure (``5_swc/``).
6. Rescale coordinates to 1 µm resolution (``7_scaled_1um_swc/``).

The final SWC files in ``7_scaled_1um_swc/`` are then copied into the user-specified
output directory.

Command-line usage
------------------

    python tracer.py \\
        --seg_dir /path/to/seg_dir \\
        --out_swc_dir /path/to/output_swc \\
        --xy_resolution 1000 \\
        --z_resolution 1000

where:

* ``seg_dir``: directory containing 3D segmentation volumes (``.tif`` or ``.nii.gz``),
  typically the post-processed nnUNetv2 predictions.
* ``out_swc_dir``: directory that will receive the final 1 µm resolution SWC files.
* ``xy_resolution`` / ``z_resolution``: image resolution in nm/pixel. These are used
  to rescale coordinates from voxel units to µm.
"""

import argparse
import glob
import os
import shutil
import subprocess
import sys
from functools import partial
from multiprocessing import Pool
from typing import Optional

import networkx as nx
import numpy as np
import pandas as pd
import SimpleITK as sitk
import tifffile
from skimage import morphology, transform, measure
from scipy import ndimage
from scipy.ndimage import binary_dilation
from skimage.measure import regionprops
from skimage.morphology import skeletonize_3d
from skimage.transform import resize

import cupy as cp
import cupyx


#: Path to the Vaa3D executable used for APP2 tracing and radius estimation.
v3d_path = r"/home/kfchen/Vaa3D-x.1.1.4_Ubuntu/Vaa3D-x"

#: Maximum number of worker processes used when converting segmentation files.
MAX_PROCESSES = 16


class swcPoint:
    """Basic SWC node representation."""

    def __init__(
        self,
        sample_number,
        structure_identifier,
        x_position,
        y_position,
        z_position,
        radius,
        parent_sample,
    ):
        self.n = sample_number
        self.si = 0  # structure_identifier
        self.x = x_position
        self.y = y_position
        self.z = z_position
        self.r = radius
        self.p = parent_sample
        self.s = []  # sons
        self.fn = -1  # fiber number
        self.conn = []  # connect points in other fiber
        self.mp = []  # match point in other swc
        self.neighbor = []  # neighbors closer than a distance
        self.ishead = False
        self.istail = False
        self.swcNeig = []
        self.swcMatchP = []
        self.i = 0
        self.visited = 0
        self.pruned = False
        self.depth = 0


class swcP_list:
    """Container for a list of swcPoint nodes."""

    def __init__(self):
        self.p = []
        self.count = 0

    def prune_point(self, p_id):
        prune_queue = [p_id]
        while prune_queue:
            p = prune_queue.pop(0)
            self.p[p].pruned = True
            for s in self.p[p].s:
                if not self.p[s].pruned:
                    prune_queue.append(s)


class SmartSomaRegionFinder_v2:
    """GPU-accelerated soma region detector based on frequency analysis."""

    def find_resolution(self, df, filename):
        for i in range(len(df)):
            if df.iloc[i, 1] == filename or df.iloc[i, 1] == filename.replace(".tif", ""):
                return df.iloc[i, 3]
        print(filename)
        sys.exit(1)

    def exponential_decay(self, x, a, b, c):
        return a * np.exp(-b * (x - 2)) + c

    def calc_high_freq(self, seg, kernel_radii):
        high_freq_ratios = []
        high_freq_averages = []

        for radius in kernel_radii:
            struct = morphology.ball(radius)
            opened_img = morphology.opening(seg, struct)
            opened_img = opened_img * seg

            if np.sum(opened_img) == 0:
                break

            verts, faces, normals, values = measure.marching_cubes(opened_img, level=0)
            signal = verts.flatten()
            F = np.fft.fft(signal)
            N = len(F)
            F_magnitude = np.abs(F)
            E_total = np.sum(F_magnitude**2)
            k_threshold = int(0.1 * N)
            E_high = np.sum(F_magnitude[k_threshold:] ** 2)
            R = E_high / E_total
            high_freq_ratios.append(R)
            A_high = np.mean(F_magnitude[k_threshold:])
            high_freq_averages.append(A_high)

        return high_freq_ratios, high_freq_averages

    def calc_high_freq_gpu(self, seg, kernel_radii):
        high_freq_ratios = []
        high_freq_averages = []

        for radius in kernel_radii:
            struct = morphology.ball(radius)

            seg_gpu = cp.asarray(seg)
            struct_gpu = cp.asarray(struct)

            opened_img_gpu = cupyx.scipy.ndimage.binary_opening(seg_gpu, struct_gpu)
            opened_img_gpu = opened_img_gpu * seg_gpu

            opened_img = cp.asnumpy(opened_img_gpu)

            if np.sum(opened_img) == 0:
                break

            verts, faces, normals, values = measure.marching_cubes(opened_img, level=0)
            signal = verts.flatten()

            signal_gpu = cp.asarray(signal)
            F_gpu = cp.fft.fft(signal_gpu)
            F_magnitude_gpu = cp.abs(F_gpu)

            F_magnitude = cp.asnumpy(F_magnitude_gpu)
            N = len(F_magnitude)

            E_total = np.sum(F_magnitude**2)
            k_threshold = int(0.1 * N)
            E_high = np.sum(F_magnitude[k_threshold:] ** 2)
            R = E_high / E_total
            high_freq_ratios.append(R)
            A_high = np.mean(F_magnitude[k_threshold:])
            high_freq_averages.append(A_high)

        return high_freq_ratios, high_freq_averages

    def largest_connected_component(self, binary_image):
        labeled_image, num_features = ndimage.label(binary_image)
        component_sizes = np.bincount(labeled_image.ravel())
        component_sizes[0] = 0
        largest_component_label = np.argmax(component_sizes)
        largest_component = (labeled_image == largest_component_label).astype(np.uint8)
        return largest_component

    def test_soma_detectison(self, seg, xy_resolution):
        resolution = (1, float(xy_resolution), float(xy_resolution))
        origin_img_size = seg.shape
        seg = resize(
            seg,
            (
                seg.shape[0] * resolution[0],
                seg.shape[1] * resolution[1],
                seg.shape[2] * resolution[2],
            ),
            order=0,
        )
        seg = (seg - seg.min()) / (seg.max() - seg.min())
        seg = np.where(seg > 0, 1, 0).astype(np.uint8)
        seg = ndimage.binary_fill_holes(seg).astype(int)

        min_radii, max_radii = 2, 15
        kernel_radii = np.linspace(min_radii, max_radii, 20)

        high_freq_ratios, high_freq_averages = self.calc_high_freq_gpu(seg, kernel_radii)

        kernel_radii = kernel_radii[: len(high_freq_ratios)]
        if len(kernel_radii) == 2:
            kernel_radii = np.append(
                kernel_radii, (kernel_radii[1] + kernel_radii[0]) / 2
            )
            high_freq_ratios, high_freq_averages = self.calc_high_freq_gpu(seg, kernel_radii)
        elif len(kernel_radii) == 1:
            kernel_radii = np.append(kernel_radii, kernel_radii[0] + 0.1)
            kernel_radii = np.append(kernel_radii, kernel_radii[0] + 0.2)
            high_freq_ratios, high_freq_averages = self.calc_high_freq_gpu(seg, kernel_radii)
        elif len(kernel_radii) == 0:
            return None

        high_freq_averages = np.array(high_freq_averages)
        high_freq_averages = (high_freq_averages - high_freq_averages.min()) / (
            high_freq_averages.max() - high_freq_averages.min()
        )
        try:
            from scipy.optimize import curve_fit

            popt, pcov = curve_fit(
                self.exponential_decay,
                kernel_radii,
                high_freq_averages,
                p0=[1, 1, 0],
                maxfev=10000,
            )
        except Exception:
            return None

        x_fit = np.linspace(kernel_radii.min(), kernel_radii.max(), 100)
        y_fit = self.exponential_decay(x_fit, *popt)

        fitted_gradients = np.gradient(y_fit, x_fit)
        fitted_gradients = (fitted_gradients - fitted_gradients.min()) / (
            fitted_gradients.max() - fitted_gradients.min()
        ) - 1

        best_radius = x_fit[np.argmin(np.abs(fitted_gradients - 0.9))]

        result_open_img = morphology.opening(seg, morphology.ball(best_radius))
        result_open_img = result_open_img * seg
        result_img = resize(result_open_img, origin_img_size, order=0)
        result_img = self.largest_connected_component(result_img)
        result_img = np.where(result_img > 0, 1, 0).astype(np.uint8)
        return result_img


def read_swc(swc_name):
    """Read an SWC file into a swcP_list structure."""
    point_l = swcP_list()
    with open(swc_name, "r") as f:
        lines = f.readlines()

    swcPoint_number = -1
    point_list = []
    list_map = np.zeros(500000)

    for line in lines:
        if line[0] == "#":
            continue
        temp_line = line.split()
        point_list.append(temp_line)
        swcPoint_number = swcPoint_number + 1
        list_map[int(temp_line[0])] = swcPoint_number

    swcPoint_number = 0
    for point in point_list:
        swcPoint_number = swcPoint_number + 1
        point[0] = swcPoint_number
        point[1] = int(point[1])
        point[2] = float(point[2])
        point[3] = float(point[3])
        point[4] = float(point[4])
        point[5] = float(point[5])
        point[6] = int(point[6])
        if point[6] != -1:
            point[6] = int(list_map[int(point[6])]) + 1

    point_l.p.append(swcPoint(0, 0, 0, 0, 0, 0, 0))

    for point in point_list:
        temp_swcPoint = swcPoint(
            point[0],
            point[1],
            point[2],
            point[3],
            point[4],
            point[5],
            point[6],
        )
        point_l.p.append(temp_swcPoint)
    for point in point_list:
        temp_swcPoint = swcPoint(
            point[0],
            point[1],
            point[2],
            point[3],
            point[4],
            point[5],
            point[6],
        )
        if temp_swcPoint.p != -1:
            parent = point_l.p[int(temp_swcPoint.p)]
            parent.s.append(temp_swcPoint.n)
        if point[0] == 1:
            point_l.p[int(point[0])].depth = 0
        else:
            point_l.p[int(point[0])].depth = parent.depth + 1

    return point_l


def write_swc(
    filepath,
    point_l,
    fiber_l=None,
    reversal: bool = False,
    limit=None,
    overlay: bool = False,
    number_offset: int = 0,
):
    """Write a swcP_list structure back to SWC file."""
    if limit is None:
        limit = [1000, 1000, 1000]

    lines = []
    for temp_p in point_l.p:
        if temp_p.n == 0:
            continue
        if fiber_l and fiber_l.f[temp_p.fn - 1].pruned:
            continue
        if temp_p.pruned:
            continue

        if reversal:
            line = "%d %d %f %f %f %f %d\n" % (
                temp_p.n + number_offset,
                temp_p.si,
                temp_p.x,
                limit[1] - temp_p.y,
                temp_p.z,
                temp_p.r,
                temp_p.p + number_offset,
            )
        else:
            line = "%d %d %f %f %f %f %d\n" % (
                temp_p.n + number_offset,
                temp_p.si,
                temp_p.x,
                temp_p.y,
                temp_p.z,
                temp_p.r,
                temp_p.p + number_offset,
            )
        lines.append(line)

    if overlay and os.path.exists(filepath):
        os.remove(filepath)
    with open(filepath, mode="a") as file_handle:
        file_handle.writelines(lines)


class FileProcessingStep:
    """A single logical step in the file-based processing pipeline.

    Each step:

    * Consumes one or more input files (e.g. segmentation, soma mask).
    * Produces one or more output files (e.g. skeleton, SWC).
    * Delegates actual work to a Python function (``process_function``).

    The step assumes a simple convention: for a given neuron, all intermediate
    files share the same *base* filename but live in different sub-directories
    (``0_seg/``, ``1_skel_seg/``, ``5_swc/`` etc.).
    """

    def __init__(
        self,
        step_name,
        process_function,
        file_name,
        input_dirs,
        output_dirs=None,
        supplementary_files=None,
    ):
        self.step_name = step_name
        self.process_function = process_function
        self.file_name = file_name
        self.input_dirs = input_dirs
        self.output_dirs = output_dirs
        self.supplementary_files = supplementary_files if supplementary_files else []

    def count_function_parameters(self, function):
        import inspect

        signature = inspect.signature(function)
        return len(signature.parameters)

    def execute(self):
        # Ensure all output directories exist.
        for output_dir in self.output_dirs:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)

        # Map base filename into each input_dir / output_dir, adjusting extension
        # for swc/marker folders.
        input_files = [os.path.join(input_dir, self.file_name) for input_dir in self.input_dirs]
        for i in range(len(input_files)):
            if self.input_dirs[i].endswith("swc"):
                input_files[i] = input_files[i].replace(".tif", ".swc")
            elif self.input_dirs[i].endswith("marker"):
                input_files[i] = input_files[i].replace(".tif", ".marker")

        output_files = [os.path.join(output_dir, self.file_name) for output_dir in self.output_dirs]
        for i in range(len(output_files)):
            if self.output_dirs[i].endswith("swc"):
                output_files[i] = output_files[i].replace(".tif", ".swc")
            elif self.output_dirs[i].endswith("marker"):
                output_files[i] = output_files[i].replace(".tif", ".marker")

        # Dispatch to processing function with 2 or 3 parameters.
        if self.count_function_parameters(self.process_function) == 2:
            self.process_function(input_files, output_files)
        elif self.count_function_parameters(self.process_function) == 3:
            self.process_function(input_files, output_files, self.supplementary_files)
        else:
            raise ValueError("Process function must accept 2 or 3 parameters.")


class FileProcessingPipeline:
    """Container for an ordered list of :class:`FileProcessingStep`.

    The pipeline executes all steps in the order they were added. Each step
    typically reads from the output directories of previous steps and writes
    new intermediate results. The ``root_dir`` usually corresponds to a
    directory containing sub-folders such as ``0_seg/``, ``1_skel_seg/``, etc.
    """

    def __init__(self, root_dir, file_name, meta_info=None):
        if not root_dir:
            raise ValueError("Root directory must be specified.")
        self.root_dir = root_dir
        self.steps = []
        self.file_name = file_name
        self.meta_info = meta_info

    def add_step(self, step_name, process_function, input_steps, output_steps, meta_info=None):
        input_dirs = [os.path.join(self.root_dir, input_step) for input_step in input_steps]
        output_dirs = [os.path.join(self.root_dir, output_step) for output_step in output_steps]
        step = FileProcessingStep(
            step_name,
            process_function,
            self.file_name,
            input_dirs,
            output_dirs,
            meta_info,
        )
        self.steps.append(step)

    def run(self):
        if not self.steps:
            print("No steps to run in the pipeline.")
            return

        for step in self.steps:
            step.execute()


def copy_file_from_seg_result(origin_seg_file, seg_dir, name_mapping_file):
    """Copy or convert a segmentation file into a tif and rename via mapping CSV."""

    def get_full_name(file_name, df):
        full_name = df[df["nnunet_name"] == file_name]["full_name"]
        result = str(full_name.values[0])
        if not result.endswith(".tif"):
            result += ".tif"
        return result

    if (not origin_seg_file.endswith(".tif")) and (not origin_seg_file.endswith(".nii.gz")):
        return

    os.makedirs(seg_dir, exist_ok=True)

    file_name = os.path.basename(origin_seg_file)
    if file_name.endswith(".nii.gz"):
        seg = sitk.ReadImage(origin_seg_file)
        seg = sitk.GetArrayFromImage(seg)
        seg = (np.where(seg > 0, 1, 0) * 255).astype("uint8")
        file_name = file_name.replace("_pred0.nii.gz", ".tif")
        tifffile.imwrite(os.path.join(seg_dir, file_name), seg, compression="zlib")
    else:
        seg = tifffile.imread(origin_seg_file)
        seg = (np.where(seg > 0, 1, 0) * 255).astype("uint8")
        tifffile.imwrite(os.path.join(seg_dir, file_name), seg, compression="zlib")

    df = pd.read_csv(name_mapping_file)
    full_name = get_full_name(file_name.split(".")[0], df)
    if full_name is None:
        print(f"Error: {file_name} not found in the name mapping file.")
        return

    new_file_path = os.path.join(seg_dir, full_name)
    os.rename(os.path.join(seg_dir, file_name), new_file_path)


def prepare_seg_files(origin_seg_dir, seg_dir, name_mapping_file):
    """Batch-convert / copy nnUNet segmentation results to a flat tif folder."""
    origin_seg_files = [
        os.path.join(origin_seg_dir, f) for f in os.listdir(origin_seg_dir)
    ]

    pool = Pool(MAX_PROCESSES)
    func = partial(copy_file_from_seg_result, seg_dir=seg_dir, name_mapping_file=name_mapping_file)
    pool.map(func, origin_seg_files)
    pool.close()
    pool.join()


class AutoTracePipeline(FileProcessingPipeline):
    """High-level pipeline for automatic neuron tracing from segmentation.

    This class wires together several :class:`FileProcessingStep` instances to
    implement the complete *segmentation → SWC* workflow for a **single**
    neuron volume ``file_name`` located under ``0_seg/``.

    The main steps are:

    * ``1_skel_seg``: 3D skeletonization of the segmentation.
    * ``2_soma_region`` + ``4_soma_marker``: soma localization and mask/marker
      generation (optionally guided by soma position meta-info).
    * ``3_skel_with_soma``: combining soma and skeleton into a single mask.
    * ``5_swc``: Vaa3D APP2-based tracing to generate a raw SWC tree.
    * ``7_scaled_1um_swc``: coordinate rescaling to 1 µm.
    """

    def __init__(self, root_dir, file_name, meta_info=None):
        if meta_info is None:
            print("Warning: meta_info is empty. Default values will be used.")
            meta_info = {
                "xy_resolution": 500,
                "z_resolution": 1000,
                "soma_x": 0,
                "soma_y": 0,
                "soma_z": 0,
            }
            meta_info = pd.DataFrame([meta_info])

        super().__init__(root_dir, file_name, meta_info)

        self.add_step("1_skel_seg", self.skel_seg, ["0_seg"], ["1_skel_seg"])
        self.add_step(
            "2_soma_region",
            self.get_soma_region,
            ["0_seg"],
            ["2_soma_region", "4_soma_marker"],
        )
        self.add_step(
            "3_skel_with_soma",
            self.skel_with_soma,
            ["1_skel_seg", "2_soma_region"],
            ["3_skel_with_soma"],
        )
        self.add_step(
            "5_swc",
            self.trace_app2_with_soma,
            ["3_skel_with_soma", "4_soma_marker"],
            ["5_swc"],
        )
        self.add_step(
            "7_rescale_xy_resolution",
            self.rescale_xy_resolution,
            ["5_swc"],
            ["7_scaled_1um_swc"],
        )

    def find_resolution(self):
        return self.meta_info["xy_resolution"]

    def skel_seg(self, input_files, output_files):
        """Skeletonize a binary segmentation volume."""
        if not input_files:
            return
        seg_file = input_files[0]
        skel_file = output_files[0]

        if (not os.path.exists(seg_file)) or os.path.exists(skel_file):
            return

        data = tifffile.imread(seg_file).astype("uint8")
        skel = skeletonize_3d(data).astype("uint8")
        tifffile.imwrite(skel_file, skel * 255, compression="zlib")

    def get_soma_region(self, input_files, output_files):
        """Estimate soma region and generate soma marker."""

        def compute_centroid(mask):
            labeled_mask = morphology.label(mask)
            props = regionprops(labeled_mask)
            if len(props) > 0:
                return props[0].centroid
            return None

        def find_soma_region_in_roi(image, center, block_size=50):
            xy_resolution = float(self.meta_info["xy_resolution"].values[0])
            z_resolution = float(self.meta_info["z_resolution"].values[0])

            if center == (0, 0, 0):
                center = (
                    int(image.shape[0] / 2),
                    int(image.shape[1] / 2),
                    int(image.shape[2] / 2),
                )
                block_size = block_size * 2

            background = np.zeros_like(image)
            z_start = max(int(center[0] - block_size / (z_resolution / 1000)), 0)
            z_end = min(int(center[0] + block_size / (z_resolution / 1000)), image.shape[0])
            y_start = max(int(center[1] - block_size / (xy_resolution / 1000)), 0)
            y_end = min(int(center[1] + block_size / (xy_resolution / 1000)), image.shape[1])
            x_start = max(int(center[2] - block_size / (xy_resolution / 1000)), 0)
            x_end = min(int(center[2] + block_size / (xy_resolution / 1000)), image.shape[2])
            image = image[z_start:z_end, y_start:y_end, x_start:x_end]

            somaregionfinder = SmartSomaRegionFinder_v2()
            soma_region = somaregionfinder.test_soma_detectison(
                image, float(xy_resolution / 1000)
            )

            full_ellipsoid_image = np.zeros_like(background)
            if soma_region is None:
                return None
            full_ellipsoid_image[z_start:z_end, y_start:y_end, x_start:x_end] = soma_region
            return full_ellipsoid_image

        if not input_files:
            return
        seg_file = input_files[0]
        soma_region_file = output_files[0]
        soma_marker_file = output_files[1]

        if (not os.path.exists(seg_file)) or os.path.exists(soma_region_file):
            return

        soma_x = int(float(self.meta_info["soma_x"].values[0]))
        soma_y = int(float(self.meta_info["soma_y"].values[0]))
        soma_z = int(float(self.meta_info["soma_z"].values[0]))

        seg = tifffile.imread(seg_file)
        seg = np.flip(seg, axis=1)

        soma_region = find_soma_region_in_roi(seg, (soma_z, soma_y, soma_x))
        if soma_region is None:
            return

        centroid = compute_centroid(soma_region)
        calc_soma_x, calc_soma_y, calc_soma_z, calc_soma_r = (
            centroid[2],
            centroid[1],
            centroid[0],
            1,
        )
        calc_soma_y = soma_region.shape[1] - calc_soma_y
        marker_str = (
            f"{calc_soma_x}, {calc_soma_y}, {calc_soma_z}, {calc_soma_r}, 1, , , 255,0,0"
        )

        with open(soma_marker_file, "w") as f:
            f.write(marker_str)

        soma_region = np.where(soma_region > 0, 1, 0).astype("uint8")
        soma_region = np.flip(soma_region, axis=1)
        tifffile.imwrite(soma_region_file, soma_region * 255, compression="zlib")

    def skel_with_soma(self, input_files, output_files):
        """Merge skeleton and soma region."""
        if not input_files:
            return
        skel_file = input_files[0]
        soma_region_file = input_files[1]
        skel_with_soma_file = output_files[0]

        if (
            (not os.path.exists(skel_file))
            or (not os.path.exists(soma_region_file))
            or os.path.exists(skel_with_soma_file)
        ):
            return

        skel = tifffile.imread(skel_file).astype("uint8")
        soma = tifffile.imread(soma_region_file).astype("uint8")
        skelwithsoma = np.logical_or(skel, soma)
        skelwithsoma = (skelwithsoma * 255).astype("uint8")
        tifffile.imwrite(skel_with_soma_file, skelwithsoma, compression="zlib")

    def trace_app2_with_soma(self, input_files, output_files):
        """Trace morphology using Vaa3D APP2."""

        def process_path(path):
            return path.replace("\\", "/")

        if not input_files:
            return
        img_file = input_files[0]
        somamarker_file = "NULL"
        swc_file = output_files[0]
        if (not os.path.exists(img_file)) or os.path.exists(swc_file):
            return
        ini_swc_path = img_file.replace(".tif", ".tif_ini.swc")

        resample = 1
        gsdt = 1
        b_RadiusFrom2D = 1
        b_256cube = 0

        try:
            if sys.platform == "linux":
                cmd = (
                    f'xvfb-run -a -s "-screen 0 640x480x16" "{v3d_path}" '
                    f'-x vn2 -f app2 -i "{img_file}" -o "{swc_file}" '
                    f'-p "{somamarker_file}" 0 10 {b_256cube} {b_RadiusFrom2D} '
                    f"{gsdt} 1 {resample} 0 0"
                )
                cmd = process_path(cmd)
                subprocess.run(cmd, stdout=subprocess.DEVNULL, shell=True)
            else:
                pass
        except Exception:
            pass

        if os.path.exists(ini_swc_path):
            os.remove(ini_swc_path)

    def prune_fiber_in_soma(self, point_l, soma_region):
        """Prune branches inside soma region."""
        edge_p_list = []
        x_limit, y_limit, z_limit = (
            soma_region.shape[2],
            soma_region.shape[1],
            soma_region.shape[0],
        )

        for p in point_l.p:
            if p.n == 0 or p.n == 1:
                continue
            x, y, z = p.x, p.y, p.z
            y = soma_region.shape[1] - y

            x = min(int(x), x_limit - 1)
            y = min(int(y), y_limit - 1)
            z = min(int(z), z_limit - 1)

            if soma_region[int(z), int(y), int(x)]:
                edge_p_list.append(p)

        for p in edge_p_list:
            if len(p.s) == 0:
                temp_p = point_l.p[p.n]
                while True:
                    if temp_p.n == 1:
                        break
                    if temp_p.pruned is True:
                        break
                    if not len(temp_p.s) == 1:
                        break
                    point_l.p[temp_p.n].pruned = True
                    temp_p = point_l.p[temp_p.p]
        for p in point_l.p:
            for s in list(p.s):
                if point_l.p[s].pruned is True:
                    p.s.remove(s)

        return point_l

    def connect_to_soma_file(self, input_files, output_files):
        """Reconnect tree to soma region and write new SWC."""
        if not input_files:
            return
        swc_file = input_files[0]
        soma_refion_file = input_files[1]
        conn_swc_file = output_files[0]

        if (
            (not os.path.exists(swc_file))
            or (not os.path.exists(soma_refion_file))
            or os.path.exists(conn_swc_file)
        ):
            return

        soma_region = tifffile.imread(soma_refion_file).astype("uint8")
        soma_region = binary_dilation(soma_region, iterations=4).astype("uint8")
        x_limit, y_limit, z_limit = (
            soma_region.shape[2],
            soma_region.shape[1],
            soma_region.shape[0],
        )

        point_l = read_swc(swc_file)

        labeled_img, num_objects = ndimage.label(soma_region)
        if len(point_l.p) <= 1:
            return
        for obj_id in range(1, num_objects + 1):
            obj_img = np.where(labeled_img == obj_id, 1, 0)
            x, y, z = point_l.p[1].x, point_l.p[1].y, point_l.p[1].z
            if obj_img[int(z), int(y), int(x)]:
                soma_region = obj_img
                del obj_img
                break
            del obj_img

        labeled_img, num_objects = ndimage.label(soma_region)
        if num_objects > 1:
            write_swc(conn_swc_file, point_l)
            del soma_region, point_l
            return

        point_l = self.prune_fiber_in_soma(point_l, soma_region)

        edge_p_list = []
        for p in point_l.p:
            if p.n == 0 or p.n == 1:
                continue
            x, y, z = p.x, p.y, p.z
            y = soma_region.shape[1] - y

            x = min(int(x), x_limit - 1)
            y = min(int(y), y_limit - 1)
            z = min(int(z), z_limit - 1)

            if soma_region[int(z), int(y), int(x)]:
                edge_p_list.append(p)

        for p in edge_p_list:
            temp_p = point_l.p[p.p]
            while True:
                if temp_p.n == 1:
                    break
                if temp_p.pruned is True:
                    break
                point_l.p[temp_p.n].pruned = True
                temp_p = point_l.p[temp_p.p]

        for p in edge_p_list:
            if point_l.p[p.n].pruned is False:
                if not len(point_l.p[p.n].s):
                    point_l.p[p.n].pruned = True
                else:
                    point_l.p[p.n].p = 1
                    point_l.p[1].s.append(p.n)
            else:
                for s in point_l.p[p.n].s:
                    point_l.p[s].p = 1
                    point_l.p[1].s.append(s)

        if os.path.exists(conn_swc_file):
            os.remove(conn_swc_file)
        write_swc(conn_swc_file, point_l)
        del soma_region, point_l

    def rescale_xy_resolution(self, input_files, output_files):
        """Rescale SWC coordinates to 1 µm resolution."""
        if not input_files:
            return

        swc_file = input_files[0]
        unified_swc_file = output_files[0]

        if (not os.path.exists(swc_file)) or os.path.exists(unified_swc_file):
            return

        xy_resolution = float(self.meta_info["xy_resolution"].values[0])
        z_resolution = float(self.meta_info["z_resolution"].values[0])
        swc_point_list = pd.read_csv(
            swc_file,
            delim_whitespace=True,
            comment="#",
            header=None,
            names=["id", "type", "x", "y", "z", "radius", "parent"],
        )
        swc_point_list["x"] = swc_point_list["x"] * xy_resolution / 1000
        swc_point_list["y"] = swc_point_list["y"] * xy_resolution / 1000
        swc_point_list["z"] = swc_point_list["z"] * z_resolution / 1000

        swc_point_list.to_csv(unified_swc_file, sep=" ", header=False, index=False)

    def get_estimated_radius(self, input_files, output_files):
        """Estimate radius from image and smooth along tree (optional step)."""
        import tempfile

        def v3d_get_radius(img_path, swc_path, out_path):
            with tempfile.TemporaryDirectory() as temp_dir:
                img_filename = os.path.basename(img_path).split("_")[0] + ".tif"
                swc_filename = os.path.basename(swc_path).split("_")[0] + ".swc"
                output_filename = os.path.basename(out_path).split("_")[0] + ".swc"

                img_cache_path = os.path.join(temp_dir, img_filename)
                swc_cache_path = os.path.join(temp_dir, swc_filename)
                out_cache_path = os.path.join(temp_dir, output_filename)

                shutil.copy(img_path, img_cache_path)
                shutil.copy(swc_path, swc_cache_path)

                radius2d = 1
                cmd_str = (
                    f'xvfb-run -a -s "-screen 0 640x480x16" {v3d_path} '
                    f"-x neuron_radius -f neuron_radius -i {img_cache_path} "
                    f"{swc_cache_path} -o {out_cache_path} -p 10 {radius2d}"
                )
                cmd_str = cmd_str.replace("(", r"\(").replace(")", r"\)")
                subprocess.run(cmd_str, stdout=subprocess.DEVNULL, shell=True)

                shutil.copy(out_cache_path, out_path)

        def load_swc_to_undirected_graph(swc_file_path):
            df = pd.read_csv(
                swc_file_path,
                delim_whitespace=True,
                comment="#",
                header=None,
                names=["id", "type", "x", "y", "z", "radius", "parent"],
            )
            G = nx.Graph()

            for _, row in df.iterrows():
                G.add_node(
                    row["id"],
                    pos=(row["x"], row["y"], row["z"]),
                    radius=row["radius"],
                    type=row["type"],
                    parent=row["parent"],
                )
                if row["parent"] != -1:
                    G.add_edge(row["parent"], row["id"])

            return G

        def find_nearest_node(G, target_pos):
            nearest_node = None
            min_distance = float("inf")

            for node in G.nodes(data=True):
                pos = node[1]["pos"]
                distance = np.linalg.norm(np.array(pos) - np.array(target_pos))
                if distance < min_distance:
                    nearest_node = node[0]
                    min_distance = distance

            return nearest_node

        def export_to_swc_dfs(G, root_pos, output_filename):
            if os.path.exists(output_filename):
                os.remove(output_filename)

            start_node = find_nearest_node(G, root_pos)

            potential_root = max(G.nodes, key=lambda x: G.degree(x))
            potential_root_degree = G.degree(potential_root)
            potential_root_list = [
                node for node in G.nodes if G.degree(node) == potential_root_degree
            ]
            for node in potential_root_list:
                if G.degree(node) > 4 and len(potential_root_list) == 1:
                    start_node = node
                elif nx.shortest_path_length(G, start_node, node) < 3:
                    start_node = node
                elif np.linalg.norm(
                    np.array(G.nodes[node]["pos"]) - np.array(root_pos)
                ) < 10:
                    start_node = node

            with open(output_filename, "w") as f:
                f.write("# SWC file generated from DFS traversal\n")
                f.write("# Columns: id type x y z radius parent\n")

                new_id = 1
                visited = set()
                stack = [(start_node, -1)]

                while stack:
                    node, parent_id = stack.pop()
                    if node not in visited:
                        visited.add(node)
                        node_data = G.nodes[node]
                        pos = node_data["pos"]
                        radius = node_data["radius"]
                        node_type = 1 if parent_id == -1 else 3

                        f.write(
                            f"{new_id} {node_type} {pos[0]} {pos[1]} {pos[2]} "
                            f"{radius} {parent_id}\n"
                        )

                        current_parent_id = new_id
                        new_id += 1

                        for neighbor in G.neighbors(node):
                            if neighbor not in visited:
                                stack.append((neighbor, current_parent_id))

        def calc_node_dist(G, node1, node2):
            pos1 = np.array(G.nodes[node1]["pos"])
            pos2 = np.array(G.nodes[node2]["pos"])
            return np.linalg.norm(pos1 - pos2)

        def gaussian_smoothing_radius_tree(G, sigma=1.0):
            smoothed_values = {}
            soma_r = G.nodes[1]["radius"]
            for node in G.nodes:
                neighbors = list(G.neighbors(node))
                weights = []
                values = []
                for neighbor in neighbors:
                    distance = calc_node_dist(G, node, neighbor)
                    weight = np.exp(-(distance**2) / (2 * sigma**2))
                    weights.append(weight)
                    values.append(G.nodes[neighbor]["radius"])
                self_weight = np.exp(0)
                total_weight = self_weight + sum(weights)
                weighted_sum = (
                    G.nodes[node]["radius"] * self_weight
                    + sum(w * v for w, v in zip(weights, values))
                )
                smoothed_values[node] = weighted_sum / total_weight
            nx.set_node_attributes(G, smoothed_values, "radius")
            G.nodes[1]["radius"] = soma_r
            return G

        def smoothing_swc_file(swc_file_path, output_filename):
            G = load_swc_to_undirected_graph(swc_file_path)
            G = gaussian_smoothing_radius_tree(G)
            root_pos = G.nodes[1]["pos"]
            export_to_swc_dfs(G, root_pos, output_filename)

        if not input_files:
            return

        swc_file = input_files[0]
        seg_file = input_files[1]
        radius_swc_file = output_files[0]

        if (
            (not os.path.exists(swc_file))
            or (not os.path.exists(seg_file))
            or os.path.exists(radius_swc_file)
        ):
            return

        seg = tifffile.imread(seg_file).astype("uint8")
        seg = np.flip(seg, axis=1)
        origin_shape = seg.shape
        xy_resolution = self.meta_info["xy_resolution"].values[0]
        img_shape = [
            int(origin_shape[0]),
            int(origin_shape[1] * xy_resolution / 1000),
            int(origin_shape[2] * xy_resolution / 1000),
        ]
        seg = transform.resize(seg, img_shape, order=0, anti_aliasing=False)
        seg = np.where(seg > 0, 255, 0).astype("uint8")
        temp_img_file = radius_swc_file.replace(".swc", "_temp.tif")
        tifffile.imwrite(temp_img_file, seg)

        v3d_get_radius(temp_img_file, swc_file, radius_swc_file)
        try:
            smoothing_swc_file(radius_swc_file, radius_swc_file)
        except Exception:
            print(f"swc radius estimate error: {radius_swc_file}")
            if os.path.exists(radius_swc_file):
                os.remove(radius_swc_file)
        os.remove(temp_img_file)


def run_tracing(
    seg_dir: str,
    out_swc_dir: str,
    xy_resolution: float = 1000.0,
    z_resolution: float = 1000.0,
) -> None:
    """Run the tracing pipeline on all segmentation volumes in ``seg_dir``.

    This is the convenient, batch-oriented entrypoint used both by the CLI and
    by Python callers. It will:

    1. Normalize the input directory into a ``0_seg/`` folder under a common
       working directory.
    2. Build a shared ``meta_info`` DataFrame (resolution + optional soma seed).
    3. For each segmentation volume in ``0_seg/``, instantiate and run an
       :class:`AutoTracePipeline`.
    4. Copy all final SWC files from ``7_scaled_1um_swc/`` into ``out_swc_dir``.
    """
    seg_dir = os.path.abspath(seg_dir)
    out_swc_dir = os.path.abspath(out_swc_dir)
    os.makedirs(out_swc_dir, exist_ok=True)

    parent = os.path.dirname(seg_dir)
    if os.path.basename(seg_dir) == "0_seg":
        work_dir = parent
        seg_work_dir = seg_dir
    else:
        work_dir = parent
        seg_work_dir = os.path.join(work_dir, "0_seg")
        os.makedirs(seg_work_dir, exist_ok=True)
        for f in glob.glob(os.path.join(seg_dir, "*")):
            if os.path.isfile(f):
                dst = os.path.join(seg_work_dir, os.path.basename(f))
                if not os.path.exists(dst):
                    shutil.copy(f, dst)

    meta_info = pd.DataFrame(
        [
            {
                "xy_resolution": xy_resolution,
                "z_resolution": z_resolution,
                "soma_x": 0,
                "soma_y": 0,
                "soma_z": 0,
            }
        ]
    )

    seg_files = sorted(
        [
            f
            for f in os.listdir(seg_work_dir)
            if f.endswith(".tif") or f.endswith(".nii.gz")
        ]
    )

    if not seg_files:
        raise RuntimeError(f"No .tif or .nii.gz files found in {seg_work_dir}")

    for file_name in seg_files:
        pipeline = AutoTracePipeline(work_dir, file_name, meta_info=meta_info)
        pipeline.run()

    final_swc_dir = os.path.join(work_dir, "7_scaled_1um_swc")
    if os.path.isdir(final_swc_dir):
        for swc in glob.glob(os.path.join(final_swc_dir, "*.swc")):
            shutil.copy(swc, os.path.join(out_swc_dir, os.path.basename(swc)))


def main(argv: Optional[list] = None) -> None:
    """Command-line interface wrapper for :func:`run_tracing`.

    Parameters
    ----------
    argv:
        Optional list of arguments. If ``None``, defaults to ``sys.argv[1:]``.
        This is primarily useful for programmatic/testing use.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Trace neuron morphologies from 3D segmentation volumes to SWC "
            "using the built-in AutoTracePipeline."
        )
    )
    parser.add_argument(
        "--seg_dir",
        required=True,
        help="Directory containing segmentation volumes (.tif or .nii.gz).",
    )
    parser.add_argument(
        "--out_swc_dir",
        required=True,
        help="Directory to write final 1um SWC files.",
    )
    parser.add_argument(
        "--xy_resolution",
        type=float,
        default=1000.0,
        help="In-plane resolution in nm (default: 1000).",
    )
    parser.add_argument(
        "--z_resolution",
        type=float,
        default=1000.0,
        help="Z resolution in nm (default: 1000).",
    )

    args = parser.parse_args(argv)
    run_tracing(
        seg_dir=args.seg_dir,
        out_swc_dir=args.out_swc_dir,
        xy_resolution=args.xy_resolution,
        z_resolution=args.z_resolution,
    )


if __name__ == "__main__":
    main()

