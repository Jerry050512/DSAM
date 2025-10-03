import numpy as np
import cv2
import tifffile
import os

# Create dummy data for training
train_image_dir = './datasets/NEU-RSDDS-AUG/Image_train'
train_depth_dir = './datasets/NEU-RSDDS-AUG/Depth_train'
train_gt_dir = './datasets/NEU-RSDDS-AUG/GT_train'

for i in range(2):
    filename_base = f'train_{i}'
    # image
    img = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
    cv2.imwrite(os.path.join(train_image_dir, f'{filename_base}.bmp'), img)
    # depth
    depth = np.random.rand(256, 256).astype(np.float32)
    tifffile.imwrite(os.path.join(train_depth_dir, f'{filename_base}.tiff'), depth)
    # ground truth
    gt = np.zeros((256, 256), dtype=np.uint8)
    gt[50:200, 50:200] = 255
    cv2.imwrite(os.path.join(train_gt_dir, f'{filename_base}.png'), gt)


# Create dummy data for testing
test_image_dir = './datasets/NEU-RSDDS-AUG/Image_test'
test_depth_dir = './datasets/NEU-RSDDS-AUG/Depth_test'

for i in range(2):
    filename_base = f'test_{i}'
    # image
    img = np.random.randint(0, 256, (300, 400, 3), dtype=np.uint8)
    cv2.imwrite(os.path.join(test_image_dir, f'{filename_base}.bmp'), img)
    # depth
    depth = np.random.rand(300, 400).astype(np.float32)
    tifffile.imwrite(os.path.join(test_depth_dir, f'{filename_base}.tiff'), depth)

print("Dummy data created successfully.")