# Instructions for NEU-RSDDS-AUG Training and Testing

This document provides instructions on how to set up the environment and run the training and testing scripts for the RGB-D salient object detection task on the NEU-RSDDS-AUG dataset.

## 1. Environment Setup

It is recommended to use `conda` to manage the environment. You can create and activate a new conda environment with all the necessary dependencies using the provided `environment.yaml` file.

```bash
# Create the conda environment from the YAML file
conda env create -f environment.yaml

# Activate the newly created environment
conda activate pytorch183
```

### Required Libraries

If you prefer to set up the environment manually, here are the key libraries you need to install. You can use `pip` or `conda` to install them.

- `python==3.7`
- `pytorch==1.12.1`
- `torchvision==0.13.1`
- `monai==1.1.0`
- `numpy`
- `Pillow`
- `scikit-image`
- `tqdm`
- `matplotlib`

You can install them using pip:
```bash
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
pip install monai numpy Pillow scikit-image tqdm matplotlib
```
*Note: The exact PyTorch installation command may vary depending on your system and CUDA version. Please refer to the official PyTorch website for the correct command.*

## 2. Download Pre-trained Weights

**This is a mandatory step.** The training script requires a pre-trained model file that must be downloaded manually.

1.  **Create the required directory:**
    ```bash
    mkdir -p work_dir_cod/SAM/
    ```

2.  **Download the model weights:**
    Use the following command to download the `sam_vit_b_01ec64.pth` file into the created directory.

    **URL:** `https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth`

    ```bash
    wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth -O work_dir_cod/SAM/sam_vit_b_01ec64.pth
    ```
    The final path must be exactly `work_dir_cod/SAM/sam_vit_b_01ec64.pth` for the scripts to find it.

## 3. Dataset Preparation

Make sure the NEU-RSDDS-AUG dataset is structured as follows relative to the project root directory:

```
../datasets/
└── NEU-RSDDS-AUG/
    ├── Image_train/      # .bmp files
    ├── Depth_train/      # .tiff files
    ├── GT_train/         # .png files
    ├── Image_test/       # .bmp files
    └── Depth_test/       # .tiff files
```

The scripts are configured to look for the dataset at `../datasets/NEU-RSDDS-AUG/`.

## 4. Running the Scripts

Ensure you have activated the correct conda environment (`conda activate pytorch183`) before running the scripts.

The scripts are configured to save all outputs (logs, checkpoints, predictions) into a `hy-tmp` directory created within the project folder.

### Step 1: Training

To start the training process, run the `Mytrain.py` script. The script will:
- Load the training data from the `NEU-RSDDS-AUG` dataset.
- Train the model for 100 epochs.
- Save the training log to `./hy-tmp/output/result.log`.
- Save the final model checkpoint to `./hy-tmp/output/checkpoint.pth`.
- Save a plot of the training loss to `./hy-tmp/output/train_loss.png`.

```bash
python Mytrain.py
```

### Step 2: Testing

After the training is complete and the `checkpoint.pth` file has been generated, you can run the testing script. The script will:
- Load the test data from the `NEU-RSDDS-AUG` dataset.
- Load the trained model from `./hy-tmp/output/checkpoint.pth`.
- Generate prediction masks for each test image.
- Save the predicted masks as PNG files in the `./hy-tmp/output/predictions/` directory, resized to their original dimensions.
- Append the testing log to `./hy-tmp/output/result.log`.

```bash
python Mytest.py
```