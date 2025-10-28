import os
import cv2
import tifffile
import torch
from torch.utils.data import Dataset
import numpy as np

class NEUDataset(Dataset):
    def __init__(self, root_dir, mode='train', transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.mode = mode

        if self.mode == 'train':
            self.image_dir = os.path.join(root_dir, 'Image_train')
            self.depth_dir = os.path.join(root_dir, 'Depth_train')
            self.gt_dir = os.path.join(root_dir, 'GT_train')
        elif self.mode == 'test':
            self.image_dir = os.path.join(root_dir, 'Image_test')
            self.depth_dir = os.path.join(root_dir, 'Depth_test')
            self.gt_dir = None # No ground truth for testing
        else:
            raise ValueError(f"Invalid mode '{mode}'. Choose 'train' or 'test'.")

        self.image_files = sorted([f for f in os.listdir(self.image_dir) if f.endswith('.bmp')])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # File names
        image_name = self.image_files[idx]
        depth_name = image_name.replace('.bmp', '.tiff')
        gt_name = image_name.replace('.bmp', '.png')

        # Paths
        image_path = os.path.join(self.image_dir, image_name)
        depth_path = os.path.join(self.depth_dir, depth_name)

        # Loading data
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        depth = tifffile.imread(depth_path)

        sample = {'image': image, 'depth': depth, 'name': image_name.split('.')[0]}

        if self.gt_dir:
            gt_name = image_name.replace('.bmp', '.png')
            gt_path = os.path.join(self.gt_dir, gt_name)
            gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
            sample['gt'] = gt
        else:
            # For test mode, we might not have GTs.
            sample['gt'] = np.zeros(image.shape[:2], dtype=np.uint8)

        if self.transform:
            sample = self.transform(sample)

        return sample