# ACT-H8K Cortical Neuron Morphologies: Production & Evaluation Tools

This repository provides the methods and tools for **producing and evaluating** the [ACT-H8K human cortical neuron morphologies dataset](https://zenodo.org/records/15189542).

---

## Environment

- **OS**: Ubuntu 20.04  
- **Python**: 3.10  
- **GPU**: ≥12 GB VRAM (recommended for 3D full‑resolution nnUNetv2 training)  
- **VAA3D**: Vaa3D-x.1.1.4_Ubuntu (optional, for visualization)

---

## Quick Start

Follow these steps to set up the environment (≈30 min) and train your models.

### 1. Install Dependencies

#### Install `skelrec`
Follow the official installation instructions for `skelrec` from its GitHub repository: `https://github.com/MIC-DKFZ/Skeleton-Recall`
    

#### Replace nnUNetv2 Files
Locate your nnUNetv2 installation path (e.g., via `python -c "import nnunetv2, inspect; print(inspect.getfile(nnunetv2))"`). Copy the following files from this repository into the corresponding locations in that installation:

- `nnunetv2/training/loss/compound_losses.py`
- `nnunetv2/training/loss/dice.py`
- `nnunetv2/training/nnUNetTrainer/variants/loss/nnUNetTrainerConnectivityEnhancement.py`

### 2. Train a Model

Configure your dataset according to the official nnUNet documentation (folder structure + `dataset.json`). Then run:

```bash
nnUNetv2_train DATASET_NAME_OR_ID 3d_fullres FOLD -tr nnUNetTrainerConnectivityEnhancement
```

After training completes, you will obtain the segmentation results.

---

## Tracing

The `tracer.py` script converts segmentation results into neuron morphologies (SWC format). See the script for detailed usage.

---

## Evaluation

`metrics.py` provides tools to evaluate the quality of reconstructed morphologies against ground truth.

---

## Demo: From Image to SWC

Assume you have completed the installation, integrated the trainer, and organized your 3D images/labels in nnUNetv2 format (see the `example/` directory for sample files). The following steps show the full pipeline.

1. **Plan and Preprocess**
   ```bash
   nnUNetv2_plan_and_preprocess -d DATASET_NAME_OR_ID -c 3d_fullres --verify_dataset_integrity
   ```

2. **(Optional) Train** (≈8 hours on an RTX 3090)
   ```bash
   nnUNetv2_train DATASET_NAME_OR_ID 3d_fullres 0 -tr nnUNetTrainerConnectivityEnhancement
   ```

3. **Inference: Image → Segmentation** (≈30 seconds on an RTX 3090)
   ```bash
   nnUNetv2_predict -d DATASET_NAME_OR_ID -c 3d_fullres -f 0 \
     -i /path/to/imagesTs \
     -o /path/to/segmentation_output \
     -tr nnUNetTrainerConnectivityEnhancement
   ```

4. **Tracing: Segmentation → SWC** (≈30 seconds)
   ```bash
   python tracer.py \
     --seg_dir /path/to/segmentation_output \
     --out_swc_dir /path/to/output_swc
   ```

Adjust `DATASET_NAME_OR_ID`, fold index (0–4), and all paths to match your setup. The example input `example_image.tif` should produce `example_segment.tif` and `example_morphology.swc`.
