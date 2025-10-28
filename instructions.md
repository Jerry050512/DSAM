# Instructions for Training and Testing

This document provides instructions on how to set up the environment, prepare the dataset, and run the training and testing scripts for the RGB-D salient object detection model.

## 1. Environment Setup

It is recommended to use a virtual environment (e.g., conda or venv) to manage dependencies.

### Required Libraries

Install the required Python libraries using pip:

```bash
pip install torch torchvision torchaudio
pip install opencv-python
pip install numpy
pip install matplotlib
pip install tqdm
pip install monai
pip install segment-anything
```

### Pre-trained SAM Weights

The model architecture is based on the Segment Anything Model (SAM). The pre-trained weights for the `vit_b` model are required for optimal performance.

- **Download the weights:** [SAM ViT-B Checkpoint](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth)
- **Required Path:** The downloaded weights file must be placed at the following location in the project directory:
  ```
  work_dir_cod/SAM/sam_vit_b_01ec64.pth
  ```
  You may need to create the `work_dir_cod/SAM` directories if they do not exist.

## 2. Dataset Preparation

The model is designed to be trained and tested on the **NEU-RSSDDS-AUG** dataset. The dataset must be organized in the following directory structure:

```
../datasets/
└── NEU-RSDDS-AUG/
    ├── Image_train/      # (.bmp format)
    ├── Depth_train/      # (.tiff format)
    ├── GT_train/         # (.png format)
    ├── Image_test/       # (.bmp format)
    └── Depth_test/       # (.tiff format)
```

## 3. Training

To start the training process, run the `Mytrain.py` script from the root of the project directory.

```bash
python Mytrain.py
```

- **Logging:** All training progress, including epoch losses, will be logged to `/hy-tmp/output/result.log`.
- **Checkpoints:** The model checkpoint will be saved to `/hy-tmp/output/checkpoint.pth`. This file is overwritten at each save interval specified in the script.
- **Loss Curve:** A plot of the training loss will be saved to `/hy-tmp/output/train_loss.png` upon completion.

## 4. Testing

After training, you can run the testing script to generate prediction masks on the test set.

```bash
python Mytest.py
```

- **Model Loading:** The script will automatically load the trained model from `/hy-tmp/output/checkpoint.pth`.
- **Predictions:** The output segmentation masks will be saved as PNG files in the `/hy-tmp/output/predictions/` directory. Each prediction will have the same filename as its corresponding input image and will be resized to the original image dimensions.
- **Logging:** Testing progress will be appended to the same log file: `/hy-tmp/output/result.log`.