-----

# ACT-H8K Cortical Neuron Morphologies: Production & Evaluation Tools

This project provides the necessary methods and tools for **producing and evaluating** the [ACT-H8K human cortical neuron morphologies](https://www.google.com/search?q=https://zenodo.org/records/your-zenodo-link) dataset.

-----

## 🚀 Getting Started

Follow these steps to set up the environment and train your models.

### Prerequisites

1.  **Install `skelrec`**:
    Follow the official installation instructions for `skelrec` from its GitHub repository:
    `https://github.com/MIC-DKFZ/Skeleton-Recall`

2.  **Replace `nnunetv2` Files**:
    Navigate to your `nnunetv2` installation directory and replace the following files with the ones provided in this project:

      * `nnunetv2/training/loss/compound_losses.py`
      * `nnunetv2/training/loss/dice.py`
      * `nnunetv2/training/nnUNetTrainer/variants/loss/nnUNetTrainerConnectivityEnhancement.py`

### Training

1.  **Configure Your Dataset**:
    Set up your dataset following the official `nnunet` documentation.

2.  **Start Training**:
    Use the following command to begin training with the connectivity enhancement features:

    ```bash
    nnUNetv2_train DATASET_NAME_OR_ID 3d_fullres FOLD -tr nnUNetTrainerConnectivityEnhancement
    ```

    Once the training is complete, you will obtain the segmentation results.

-----

## 🔎 Tracing

For details on the tracing process, refer to the `tracer.py` file. This script handles the post-segmentation tracing to generate neuron morphologies.

## 📊 Evaluation

The evaluation metrics and tools are located in the `metrics.py` file. Use this script to assess the quality of the reconstructed neuron morphologies against ground truth data.

-----
