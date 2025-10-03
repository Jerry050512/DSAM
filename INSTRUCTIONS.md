# Environment Setup and Usage Guide

This guide provides instructions on how to set up the environment, download necessary files, and run the training and testing scripts for this project.

## 1. Environment Setup

It is recommended to use a virtual environment to manage dependencies. You can create one using `conda` or `venv`.

### Using conda:
```bash
conda create -n myenv python=3.10
conda activate myenv
```

### Installing Dependencies:
Install the required Python packages using pip:
```bash
pip install torch torchvision monai timm opencv-python tifffile tqdm matplotlib
```
**Note:** Ensure you have a version of `torch` that is compatible with your CUDA installation if you are using a GPU.

## 2. Download Pre-trained Weights

This model uses pre-trained weights for both the SAM model and the PVT backbone.

1.  **SAM ViT-B Weights:**
    -   Download the weights from: [https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth)
    -   Create a directory `work_dir_cod/SAM/` in the root of the project.
    -   Place the downloaded file (`sam_vit_b_01ec64.pth`) into this directory.

2.  **PVT V2 B2 Weights:**
    -   The original repository mentions a `pvt_v2_b2.pth` file. If you have access to it, place it in the same `work_dir_cod/SAM/` directory. If not, the model will initialize this part with random weights.

## 3. Dataset Preparation

The scripts are configured to use the **NEU-RSSDDS-AUG** dataset with a specific directory structure.

-   Place the dataset in a directory named `datasets` at the same level as the project's root directory.
-   The final structure should look like this:
    ```
    ../
    ├── your_project_directory/
    │   ├── Mytrain.py
    │   ├── Mytest.py
    │   └── ...
    └── datasets/
        └── NEU-RSDDS-AUG/
            ├── Image_train/ (contains .bmp files)
            ├── Depth_train/ (contains .tiff files)
            ├── GT_train/    (contains .png files)
            ├── Image_test/  (contains .bmp files)
            └── Depth_test/  (contains .tiff files)
    ```

## 4. Training the Model

To train the model, run the `Mytrain.py` script from the project's root directory:

```bash
python Mytrain.py
```

-   **Device:** The script is set to use `cuda:0`. You can change the device by editing the `device` variable in `Mytrain.py`.
-   **Logging:** All training logs will be saved to `/hy-tmp/output/result.log`.
-   **Checkpoints:** The model checkpoint will be saved to `/hy-tmp/output/checkpoint.pth` upon completion and at specified intervals, overwriting the previous version.

## 5. Testing the Model

After training, you can run inference on the test set using the `Mytest.py` script:

```bash
python Mytest.py
```

-   **Device:** The script is set to use `cuda:0`. You can change the device by editing the `device` variable in `Mytest.py`.
-   **Model Loading:** The script will load the trained model from `/hy-tmp/output/checkpoint.pth`.
-   **Predictions:** Predicted segmentation masks will be saved as PNG files in the `/hy-tmp/output/predictions/` directory. The filenames will correspond to the original test image names.
-   **Logging:** All testing logs will be appended to `/hy-tmp/output/result.log`.

**Important:** The output paths (`/hy-tmp/...`) are set as per the requirements. If you do not have write access to this directory, you can change the `output_dir` variable in both `Mytrain.py` and `Mytest.py` to a local path (e.g., `./hy-tmp/output`).