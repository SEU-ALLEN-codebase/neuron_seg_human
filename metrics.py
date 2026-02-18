"""Morphological metrics utilities for neuron SWC files.

This module wraps Vaa3D's ``global_neuron_feature`` plugin to compute
L-measure-like global morphology features from SWC files, and provides:

1. Batch feature extraction for a folder of SWC files.
2. Simple comparison/visualization between two feature CSVs.
3. A small CLI entrypoint for convenient use.

Typical usages
--------------

1. Extract L-measure-like features from a folder of SWC files into a CSV:

    python metrics.py extract \\
        --swc_dir /path/to/swc_dir \\
        --out_csv /path/to/features.csv

2. Compare the feature distributions of two CSV files and plot them:

    python metrics.py compare \\
        --gf1 /path/to/features_group1.csv \\
        --gf2 /path/to/features_group2.csv \\
        --out_png /path/to/compare.png
"""

import argparse
import atexit
import glob
import os
import shutil
import subprocess
import tempfile
from multiprocessing import Manager, Pool
from subprocess import TimeoutExpired
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch


# Feature name list (fixed order), used as DataFrame/CSV column names
__FEAT_NAMES22__ = [
    "Nodes",
    "SomaSurface",
    "Stems",
    "Bifurcations",
    "Branches",
    "Tips",
    "OverallWidth",
    "OverallHeight",
    "OverallDepth",
    "AverageDiameter",
    "Length",
    "Surface",
    "Volume",
    "MaxEuclideanDistance",
    "MaxPathDistance",
    "MaxBranchOrder",
    "AverageContraction",
    "AverageFragmentation",
    "AverageParent-daughterRatio",
    "AverageBifurcationAngleLocal",
    "AverageBifurcationAngleRemote",
    "HausdorffDimension",
]

# Mapping Vaa3D raw output names to cleaner column names
FEAT_NAME_DICT = {
    "N_node": "Nodes",
    "Soma_surface": "SomaSurface",
    "N_stem": "Stems",
    "Number of Bifurcatons": "Bifurcations",
    "Number of Branches": "Branches",
    "Number of Tips": "Tips",
    "Overall Width": "OverallWidth",
    "Overall Height": "OverallHeight",
    "Overall Depth": "OverallDepth",
    "Average Diameter": "AverageDiameter",
    "Total Length": "Length",
    "Total Surface": "Surface",
    "Total Volume": "Volume",
    "Max Euclidean Distance": "MaxEuclideanDistance",
    "Max Path Distance": "MaxPathDistance",
    "Max Branch Order": "MaxBranchOrder",
    "Average Contraction": "AverageContraction",
    "Average Fragmentation": "AverageFragmentation",
    "Average Parent-daughter Ratio": "AverageParent-daughterRatio",
    "Average Bifurcation Angle Local": "AverageBifurcationAngleLocal",
    "Average Bifurcation Angle Remote": "AverageBifurcationAngleRemote",
    "Hausdorff Dimension": "HausdorffDimension",
}


# Global temp directory for SWC copies used by Vaa3D (removed on exit)
TEMP_DIR = tempfile.mkdtemp(prefix="vaa3d_tmp_")
atexit.register(lambda: shutil.rmtree(TEMP_DIR, ignore_errors=True))


def _create_temp_copy(src_swc: str) -> str:
    """Create a temporary copy of an SWC file with a safe path (no spaces)."""
    temp_name = f"tmp_{os.urandom(4).hex()}.swc"
    temp_path = os.path.join(TEMP_DIR, temp_name)
    shutil.copy2(src_swc, temp_path)
    return temp_path


def calc_global_features(
    swc_file: str,
    vaa3d: str = "/opt/Vaa3D_x.1.1.4_ubuntu/Vaa3D-x",
    timeout: int = 60,
):
    """Call Vaa3D to compute global morphology features for a single SWC.

    Parameters
    ----------
    swc_file:
        Path to the input SWC file.
    vaa3d:
        Path to the Vaa3D executable that has the ``global_neuron_feature`` plugin.
    timeout:
        Maximum number of seconds allowed for the Vaa3D process.

    Returns
    -------
    list
        A list of feature values ordered according to ``__FEAT_NAMES22__``.
    """
    if " " in swc_file:
        temp_path = _create_temp_copy(swc_file)
        cmd_str = (
            f'xvfb-run -a -s "-screen 0 640x480x16" {vaa3d} '
            f'-x global_neuron_feature -f compute_feature -i "{temp_path}"'
        )
    else:
        cmd_str = (
            f'xvfb-run -a -s "-screen 0 640x480x16" {vaa3d} '
            f'-x global_neuron_feature -f compute_feature -i "{swc_file}"'
        )

    try:
        p = subprocess.run(
            cmd_str,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            text=True,
        )
    except TimeoutExpired:
        print(f"Timeout ({timeout}s) for file: {swc_file}")
        raise RuntimeError(f"Vaa3D timed out on {swc_file}")

    # Parse plugin stdout into a dict
    output = p.stdout.splitlines()
    info_dict = {}
    for s in output:
        if ":" not in s:
            continue
        try:
            it1, it2 = s.split(":", 1)
            it1 = it1.strip()
            it2 = it2.strip()
            if it1 and it2:
                info_dict[it1] = (
                    float(it2) if it2.replace(".", "", 1).isdigit() else it2
                )
        except ValueError:
            print(f"Ignoring malformed line in {swc_file}: {s}")
            continue

    required_keys = [
        "N_node",
        "Soma_surface",
        "N_stem",
        "Number of Bifurcatons",
        "Number of Branches",
        "Number of Tips",
        "Overall Width",
        "Overall Height",
        "Overall Depth",
        "Average Diameter",
        "Total Length",
        "Total Surface",
        "Total Volume",
        "Max Euclidean Distance",
        "Max Path Distance",
        "Max Branch Order",
        "Average Contraction",
        "Average Fragmentation",
        "Average Parent-daughter Ratio",
        "Average Bifurcation Angle Local",
        "Average Bifurcation Angle Remote",
        "Hausdorff Dimension",
    ]
    for key in required_keys:
        if key not in info_dict:
            raise ValueError(
                f"Missing required key '{key}' in Vaa3D output for {swc_file}"
            )

    features = [
        int(info_dict["N_node"]),
        info_dict["Soma_surface"],
        int(info_dict["N_stem"]),
        int(info_dict["Number of Bifurcatons"]),
        int(info_dict["Number of Branches"]),
        int(info_dict["Number of Tips"]),
        info_dict["Overall Width"],
        info_dict["Overall Height"],
        info_dict["Overall Depth"],
        info_dict["Average Diameter"],
        info_dict["Total Length"],
        info_dict["Total Surface"],
        info_dict["Total Volume"],
        info_dict["Max Euclidean Distance"],
        info_dict["Max Path Distance"],
        int(info_dict["Max Branch Order"]),
        info_dict["Average Contraction"],
        info_dict["Average Fragmentation"],
        info_dict["Average Parent-daughter Ratio"],
        info_dict["Average Bifurcation Angle Local"],
        info_dict["Average Bifurcation Angle Remote"],
        info_dict["Hausdorff Dimension"],
    ]
    return features


def _wrapper(swcfile, prefix, out_dict, robust=True, timeout=60):
    """Multiprocessing helper: compute features for one SWC and store in a shared dict."""
    try:
        features = calc_global_features(swcfile, timeout=timeout)
        out_dict[prefix] = features
    except Exception as e:
        if robust:
            print(f"Error processing {swcfile}: {str(e)}")
        else:
            raise


def calc_global_features_from_folder(
    swc_dir: str,
    outfile: Optional[str] = None,
    robust: bool = True,
    nprocessors: int = 8,
    timeout: int = 60,
):
    """Batch-compute global neuron features for all SWC files in a folder.

    Parameters
    ----------
    swc_dir:
        Directory containing ``*.swc`` files.
    outfile:
        Optional path to write a CSV file of features. If ``None``, only the
        DataFrame is returned.
    robust:
        If ``True``, errors on individual SWCs are logged but do not abort the
        whole batch. If ``False``, the first error will raise.
    nprocessors:
        Number of worker processes to use for parallel extraction.
    timeout:
        Per-file timeout in seconds for the Vaa3D call.

    Returns
    -------
    pandas.DataFrame
        DataFrame with one row per SWC and columns ``['', *__FEAT_NAMES22__]``.
    """

    def is_valid_swc(filepath: str) -> bool:
        with open(filepath, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    return True
        return False

    with Manager() as manager:
        out_dict = manager.dict()
        arg_list = []

        for swcfile in glob.glob(os.path.join(swc_dir, "*.swc")):
            prefix = os.path.splitext(os.path.basename(swcfile))[0]
            if not is_valid_swc(swcfile):
                print(prefix)
                continue
            arg_list.append((swcfile, prefix, out_dict, robust, timeout))

        print("Starts to calculate...")
        with Pool(processes=nprocessors) as pool:
            pool.starmap(_wrapper, arg_list)

        print("Aggregating all features")
        features_all = [[k, *v] for k, v in out_dict.items()]

        df = pd.DataFrame(features_all, columns=["", *__FEAT_NAMES22__])
        if outfile is not None:
            df.to_csv(outfile, float_format="%g", index=False)
        return df


def compare_features(
    gf_file1: str,
    gf_file2: str,
    outfile: str,
    features=None,
    label1: Optional[str] = None,
    label2: Optional[str] = None,
    details: bool = True,
):
    """Compare two global feature CSVs and save feature distribution plots as PNG."""
    df1 = pd.read_csv(gf_file1, index_col=0)
    df2 = pd.read_csv(gf_file2, index_col=0)
    if label1 is None:
        label1 = os.path.basename(gf_file1)[:5]
    if label2 is None:
        label2 = os.path.basename(gf_file2)[:5]

    if features is None:
        features = df1.columns

    sns.set_theme(style="ticks", font_scale=1.3)
    nrows = int(np.ceil(len(features) / 4))
    plt.figure(figsize=(12, 12 / 4 * nrows))

    for i, feature in enumerate(features, 1):
        plt.subplot(nrows, 4, i)

        values1 = df1[feature].values
        values2 = df2[feature].values

        if details:
            mean1 = np.mean(values1)
            mean2 = np.mean(values2)
            std1 = np.std(values1)
            std2 = np.std(values2)
            print(f"{feature} - {label1}: mean={mean1:.2f}, std={std1:.2f}")
            print(f"{feature} - {label2}: mean={mean2:.2f}, std={std2:.2f}")
            print("-----")

        combined = np.concatenate([values1, values2])
        bins = np.linspace(min(combined), max(combined), 40)

        plt.hist(
            values1, bins=bins, alpha=0.5, label=label1, color="blue", density=True
        )
        plt.hist(
            values2, bins=bins, alpha=0.5, label=label2, color="orange", density=True
        )
        plt.title(feature)

    # Legend in an extra subplot
    plt.subplot(nrows, 4, len(features) + 1)
    plt.axis("off")
    legend_handles = [
        Patch(facecolor="blue", edgecolor="blue", alpha=0.5, label=label1),
        Patch(facecolor="orange", edgecolor="orange", alpha=0.5, label=label2),
    ]
    plt.legend(
        handles=legend_handles,
        loc="center",
        fontsize=12,
        title="Legend",
        title_fontsize="13",
    )

    plt.tight_layout()
    plt.suptitle(
        f"Feature Comparison between {label1} and {label2}", fontsize=16, y=1.02
    )
    plt.savefig(outfile, dpi=300)
    plt.close()


def _build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute and compare L-measure-like morphology features from SWC files using Vaa3D."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Sub-command: extract — batch extract features from a folder of SWCs into a CSV
    p_extract = subparsers.add_parser(
        "extract", help="Extract global features from a folder of SWC files."
    )
    p_extract.add_argument(
        "--swc_dir", required=True, help="Directory containing input SWC files."
    )
    p_extract.add_argument(
        "--out_csv",
        required=True,
        help="Output CSV path to save extracted features.",
    )
    p_extract.add_argument(
        "--vaa3d",
        default="/opt/Vaa3D_x.1.1.4_ubuntu/Vaa3D-x",
        help="Path to Vaa3D executable (default: %(default)s).",
    )
    p_extract.add_argument(
        "--nproc",
        type=int,
        default=8,
        help="Number of processes for parallel extraction (default: 8).",
    )
    p_extract.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="Per-file timeout in seconds for Vaa3D (default: 60).",
    )

    # Sub-command: compare — compare two feature CSVs and output histogram PNG
    p_compare = subparsers.add_parser(
        "compare", help="Compare two feature CSVs and plot distributions."
    )
    p_compare.add_argument("--gf1", required=True, help="First feature CSV file.")
    p_compare.add_argument("--gf2", required=True, help="Second feature CSV file.")
    p_compare.add_argument(
        "--out_png", required=True, help="Output PNG file for comparison plots."
    )
    p_compare.add_argument(
        "--label1",
        default=None,
        help="Optional label for the first group (default: inferred from filename).",
    )
    p_compare.add_argument(
        "--label2",
        default=None,
        help="Optional label for the second group (default: inferred from filename).",
    )

    return parser


def main(argv: Optional[list] = None) -> None:
    parser = _build_cli_parser()
    args = parser.parse_args(argv)

    if args.command == "extract":
        # Slight wrapper around calc_global_features_from_folder to allow passing
        # a custom Vaa3D path via environment variable override.
        if args.vaa3d:
            # Monkey-patch default argument only for this call
            def _calc(swc_file, timeout):
                return calc_global_features(swc_file, vaa3d=args.vaa3d, timeout=timeout)

            global _wrapper

            def _wrapper(swcfile, prefix, out_dict, robust=True, timeout=60):
                try:
                    features = _calc(swcfile, timeout=timeout)
                    out_dict[prefix] = features
                except Exception as e:
                    if robust:
                        print(f"Error processing {swcfile}: {str(e)}")
                    else:
                        raise

        calc_global_features_from_folder(
            swc_dir=args.swc_dir,
            outfile=args.out_csv,
            robust=True,
            nprocessors=args.nproc,
            timeout=args.timeout,
        )
    elif args.command == "compare":
        compare_features(
            gf_file1=args.gf1,
            gf_file2=args.gf2,
            outfile=args.out_png,
            label1=args.label1,
            label2=args.label2,
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

