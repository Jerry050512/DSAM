import numpy as np
import matplotlib.pyplot as plt
import os
import logging
import sys
import cv2
from PIL import Image

join = os.path.join
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
import monai
import torch.nn as nn
from segment_anything import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide
from segment_anything.modeling.CWDLoss import CriterionCWD
from torch.nn import functional as F
from torchvision.models.mobilenetv2 import InvertedResidual
from dataset import NEUDataset

# set seeds
torch.manual_seed(2024)
np.random.seed(2024)

def setup_logging(log_path):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[
                            logging.FileHandler(log_path, mode='a'),
                            logging.StreamHandler(sys.stdout)
                        ])

def _upsample_like_1024(src):
    src = F.interpolate(src, size=(1024, 1024), mode='bilinear', align_corners=False)
    return src

# prepare environment
output_dir = '/hy-tmp/output'
log_file = os.path.join(output_dir, 'result.log')
setup_logging(log_file)
logging.info("Starting new training session.")


# paths and parameters
device = 'cuda:0'
dataset_path = '../datasets/NEU-RSDDS-AUG'
model_type = 'vit_b'
# The user should download the pre-trained weights to this path
pretrained_checkpoint = 'work_dir_cod/SAM/sam_vit_b_01ec64.pth'
model_save_path = os.path.join(output_dir, 'checkpoint.pth')
os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

# Load SAM model
try:
    sam_model = sam_model_registry[model_type](checkpoint=pretrained_checkpoint).to(device)
    logging.info(f"Loaded pre-trained SAM model from {pretrained_checkpoint}")
except FileNotFoundError:
    logging.warning(f"Pre-trained checkpoint not found at {pretrained_checkpoint}.")
    logging.warning("Please download it from https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth and place it in the 'work_dir_cod/SAM/' directory.")
    logging.warning("Initializing model with random weights.")
    sam_model = sam_model_registry[model_type](checkpoint=None).to(device)


sam_model.train()

# Setup optimizer
try:
    trainable_params = list(sam_model.mask_decoder.parameters()) + \
                       list(sam_model.pvt.parameters()) + \
                       list(sam_model.BC.parameters()) + \
                       list(sam_model.DWT.parameters()) + \
                       list(sam_model.ME.parameters())
    optimizer = torch.optim.Adam(trainable_params, lr=1e-5, weight_decay=0)
    logging.info("Optimizer created with parameters from mask_decoder, pvt, BC, DWT, and ME.")
except AttributeError as e:
    logging.warning(f"Could not add all module parameters to optimizer: {e}. Falling back to optimizing mask_decoder only.")
    optimizer = torch.optim.Adam(sam_model.mask_decoder.parameters(), lr=1e-5, weight_decay=0)

seg_loss = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')
CWD_loss = CriterionCWD(norm_type='channel', divergence='kl', temperature=4.0)

num_epochs = 10
losses = []
batch_size = 2

# Create dataset and dataloader
try:
    train_dataset = NEUDataset(root_dir=dataset_path, mode='train')
except FileNotFoundError:
    logging.error(f"Dataset not found at {dataset_path}. Please check the path.")
    sys.exit(1)


def collate_fn(batch):
    images, depths, gts, boxes, names, orig_sizes = [], [], [], [], [], []
    sam_trans = ResizeLongestSide(sam_model.image_encoder.img_size)

    for item in batch:
        # Original size
        img = item['image']
        H, W, _ = img.shape
        orig_sizes.append((H, W))

        # Image
        resize_img = sam_trans.apply_image(img)
        resize_img_tensor = torch.as_tensor(resize_img.transpose(2, 0, 1)).float()
        input_image = sam_model.preprocess(resize_img_tensor.to(device))
        images.append(input_image)

        # Depth
        depth = item['depth']
        # 确保深度图像为支持的数据类型
        if depth.dtype not in [np.uint8, np.float32]:
            if depth.dtype == np.uint16:
                depth = depth.astype(np.float32) / 65535.0
            else:
                depth = depth.astype(np.float32)
        if depth.ndim == 2: # Convert grayscale to 3-channel
            depth = np.stack((depth,)*3, axis=-1)
        resize_depth = sam_trans.apply_image(depth)
        resize_depth_tensor = torch.as_tensor(resize_depth.transpose(2, 0, 1)).float()
        input_depth = sam_model.preprocess(resize_depth_tensor.to(device))
        depths.append(input_depth)

        # Ground Truth - MODIFIED
        gt = item['gt']
        # Ensure gt is a NumPy array before applying transform, if it's not already
        if not isinstance(gt, np.ndarray):
            gt = np.array(gt)
        if gt.ndim == 2:
            gt = cv2.resize(gt, (sam_model.image_encoder.img_size, sam_model.image_encoder.img_size), interpolation=cv2.INTER_NEAREST)
        else:
            gt = cv2.resize(gt, (sam_model.image_encoder.img_size, sam_model.image_encoder.img_size), interpolation=cv2.INTER_NEAREST)
            gt = gt[:, :, 0]  # Convert to single channel if needed
        if gt.dtype == np.uint8:
            gt = gt.astype(np.float32) / 255.0
        gts.append(torch.tensor(gt).unsqueeze(0))

        # Bounding box from GT
        y_indices, x_indices = np.where(gt > 0)
        if len(y_indices) > 0:
            x_min, x_max = np.min(x_indices), np.max(x_indices)
            y_min, y_max = np.min(y_indices), np.max(y_indices)
            # add perturbation
            x_min = max(0, x_min - np.random.randint(0, 20))
            x_max = min(W, x_max + np.random.randint(0, 20))
            y_min = max(0, y_min - np.random.randint(0, 20))
            y_max = min(H, y_max + np.random.randint(0, 20))
            bbox = np.array([x_min, y_min, x_max, y_max])
        else: # If mask is empty, use a dummy box of the whole image
            bbox = np.array([0, 0, W, H])

        box = sam_trans.apply_boxes(bbox, (H, W))
        boxes.append(torch.as_tensor(box, dtype=torch.float))

        names.append(item['name'])

    return {
        'image': torch.stack(images),
        'depth': torch.stack(depths),
        'gt': torch.stack(gts),
        'box': torch.stack(boxes),
        'name': names,
        'original_size': orig_sizes
    }

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=0)


for epoch in range(num_epochs):
    epoch_loss = 0

    pbar = tqdm(train_dataloader, desc=f"Epoch {epoch}/{num_epochs-1}")

    for step, batch in enumerate(pbar):
        # Move data to device
        images = batch['image'].to(device)
        depths = batch['depth'].to(device)
        gt2D = batch['gt'].to(device)
        box_torch = batch['box'].to(device)

        with torch.no_grad():
            if len(box_torch.shape) == 2:
                box_torch = box_torch[:, None, :]

            image_embedding = sam_model.image_encoder(images)
            depth_embedding = sam_model.image_encoder(depths)

            sparse_embeddings, dense_embeddings_box = sam_model.prompt_encoder(
                points=None,
                boxes=box_torch,
                masks=None,
            )

        # These modules are part of the training process
        input_image_1024 = F.interpolate(images, size=(1024, 1024), mode='bilinear', align_corners=False)
        pvt_embedding = sam_model.pvt(input_image_1024)[3]

        bc_embedding, pvt_64 = sam_model.BC(pvt_embedding)
        distill_loss = CWD_loss(bc_embedding, depth_embedding)

        hybrid_embedding = torch.cat([pvt_64, bc_embedding], dim=1)
        high_frequency = sam_model.DWT(hybrid_embedding)

        dense_embeddings, sparse_embeddings = sam_model.ME(dense_embeddings_box,
                                                           high_frequency, sparse_embeddings)

        # Predicted masks
        mask_predictions, _ = sam_model.mask_decoder(
            image_embeddings=image_embedding,
            image_pe=sam_model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )

        final_mask = sam_model.loop_finer(mask_predictions, depth_embedding, depth_embedding)
        mask_predictions = 0.1 * final_mask + 0.9 * mask_predictions
        mask_predictions = torch.sigmoid(mask_predictions)

        gt2D = 1 - gt2D

        gt2D = F.interpolate(gt2D.float(), size=mask_predictions.shape[-2:], mode='nearest')

        loss = 0.9 * seg_loss(mask_predictions, gt2D.float()) + 0.1 * distill_loss

        pbar.set_postfix({'loss': loss.item()})

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    if len(train_dataloader) > 0:
      epoch_loss /= len(train_dataloader)

    losses.append(epoch_loss)
    logging.info(f'EPOCH: {epoch}, Loss: {epoch_loss}')

    # Save the model checkpoint, overwriting the previous one
    if epoch >= 80 and epoch % 10 == 0:
        torch.save(sam_model.state_dict(), model_save_path)
        logging.info(f"Model checkpoint saved to {model_save_path} at epoch {epoch}")

# Save final model
torch.save(sam_model.state_dict(), model_save_path)
logging.info(f"Final model saved to {model_save_path}")

# plot loss
plt.figure()
plt.plot(losses)
plt.title('Dice + Cross Entropy Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.savefig(os.path.join(output_dir, 'train_loss.png'))
plt.close()

logging.info("Training finished.")