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
from torch.utils.data import DataLoader
import torch.nn.functional as F
from segment_anything import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide
from dataset import NEUDataset

# Set seeds
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

# Prepare environment
output_dir = '/hy-tmp/output'
log_file = os.path.join(output_dir, 'result.log')
setup_logging(log_file)
logging.info("Starting new testing session.")

# Paths and parameters
device = 'cuda:0'
dataset_path = '../datasets/NEU-RSDDS-AUG'
model_type = 'vit_b'
checkpoint_path = os.path.join(output_dir, 'checkpoint.pth')
predictions_path = os.path.join(output_dir, 'predictions')
os.makedirs(predictions_path, exist_ok=True)

# Load SAM model
logging.info("Loading model...")
sam_model = sam_model_registry[model_type](checkpoint=None).to(device)

try:
    sam_model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    logging.info(f"Loaded trained model from {checkpoint_path}")
except FileNotFoundError:
    logging.error(f"Checkpoint not found at {checkpoint_path}. A trained model is required for testing.")
    sys.exit(1)
except Exception as e:
    logging.error(f"Failed to load model: {e}")
    sys.exit(1)

sam_model.eval()

# Create dataset and dataloader
logging.info("Loading test dataset...")
try:
    test_dataset = NEUDataset(root_dir=dataset_path, mode='test')
except Exception as e:
    logging.error(f"Failed to load dataset at {dataset_path}. Error: {e}")
    sys.exit(1)

def test_collate_fn(batch):
    images, depths, names, orig_sizes, input_sizes = [], [], [], [], []
    sam_trans = ResizeLongestSide(sam_model.image_encoder.img_size)

    for item in batch:
        img = item['image']
        H, W, _ = img.shape
        orig_sizes.append((H, W))
        names.append(item['name'])

        # Image
        resize_img = sam_trans.apply_image(img)
        input_sizes.append(resize_img.shape[:2])
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

    return {
        'image': torch.stack(images),
        'depth': torch.stack(depths),
        'name': names,
        'original_size': orig_sizes,
        'input_size': input_sizes
    }

test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=test_collate_fn, num_workers=0)

logging.info("Starting inference...")
with torch.no_grad():
    for batch in tqdm(test_dataloader, desc="Testing"):
        images = batch['image'].to(device)
        depths = batch['depth'].to(device)
        names = batch['name']
        orig_sizes = batch['original_size']
        input_sizes = batch['input_size']

        # Generate a bounding box prompt for the whole image
        H, W = orig_sizes[0]
        box_np = np.array([[0, 0, W, H]])

        sam_trans = ResizeLongestSide(sam_model.image_encoder.img_size)
        box = sam_trans.apply_boxes(box_np, (H, W))
        box_torch = torch.as_tensor(box, dtype=torch.float, device=device)
        if len(box_torch.shape) == 2:
            box_torch = box_torch[:, None, :]

        # Model Forward Pass
        image_embedding = sam_model.image_encoder(images)
        depth_embedding = sam_model.image_encoder(depths)

        sparse_embeddings, dense_embeddings_box = sam_model.prompt_encoder(
            points=None, boxes=box_torch, masks=None)

        input_image_1024 = F.interpolate(images, size=(1024, 1024), mode='bilinear', align_corners=False)
        pvt_embedding = sam_model.pvt(input_image_1024)[3]

        bc_embedding, pvt_64 = sam_model.BC(pvt_embedding)
        hybrid_embedding = torch.cat([pvt_64, bc_embedding], dim=1)
        high_frequency = sam_model.DWT(hybrid_embedding)
        dense_embeddings, sparse_embeddings = sam_model.ME(dense_embeddings_box, high_frequency, sparse_embeddings)

        mask_predictions, _ = sam_model.mask_decoder(
            image_embeddings=image_embedding,
            image_pe=sam_model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False)

        final_mask = sam_model.loop_finer(mask_predictions, depth_embedding, depth_embedding)
        mask_predictions = 0.1 * final_mask + 0.9 * mask_predictions

        # Post-process and save masks
        mask_predictions = 1 - mask_predictions
        upscaled_masks = sam_model.postprocess_masks(mask_predictions, input_sizes[0], orig_sizes[0]).squeeze()
        binary_mask = (torch.sigmoid(upscaled_masks) > 0.5).cpu().numpy().astype(np.uint8)
        final_mask_image = binary_mask * 255

        # Save the mask
        filename = f"{names[0]}.png"
        save_path = os.path.join(predictions_path, filename)
        Image.fromarray(final_mask_image).save(save_path)
        logging.info(f"Processed {names[0]}, saved prediction to {save_path}")

logging.info("Testing finished.")