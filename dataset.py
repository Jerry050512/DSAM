import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class CustomDataset(Dataset):
    def __init__(self, root_dir, train=True):
        self.root_dir = root_dir
        self.train = train
        if self.train:
            self.image_dir = os.path.join(root_dir, 'Image_train')
            self.depth_dir = os.path.join(root_dir, 'Depth_train')
            self.gt_dir = os.path.join(root_dir, 'GT_train')
        else:
            self.image_dir = os.path.join(root_dir, 'Image_test')
            self.depth_dir = os.path.join(root_dir, 'Depth_test')

        self.image_files = sorted([f for f in os.listdir(self.image_dir) if f.endswith('.bmp')])

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((1024, 1024), antialias=True),
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)

        depth_name = img_name.replace('.bmp', '.tiff')
        depth_path = os.path.join(self.depth_dir, depth_name)

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_size = image.shape[:2]

        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        # Convert depth to float, normalize, and stack to 3 channels
        if depth is not None:
            depth = (depth / 65535.0).astype(np.float32)
            depth = np.stack([depth] * 3, axis=-1)
        else:
            depth = np.zeros((original_size[0], original_size[1], 3), dtype=np.float32)

        if self.transform:
            image = self.transform(image)
            depth = self.transform(depth)

        if self.train:
            gt_name = img_name.replace('.bmp', '.png')
            gt_path = os.path.join(self.gt_dir, gt_name)
            gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
            if gt is not None:
                # Convert to binary mask of 0s and 1s
                gt = (gt > 0).astype(np.uint8)
                gt = self.transform(gt)
            else:
                gt = torch.zeros((1, 1024, 1024), dtype=torch.float32)
            gt = (gt > 0).float()
            return image, depth, gt
        else:
            return image, depth, img_name, original_size