import numpy as np
import matplotlib.pyplot as plt
import os
import logging
from tqdm import tqdm
import monai
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from segment_anything import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide
from segment_anything.modeling.CWDLoss import CriterionCWD

from PIL import Image
import skimage.io as io

# set seeds
torch.manual_seed(2024)
np.random.seed(2024)

# --- Configuration ---
# Paths
DATA_ROOT = './datasets/NEU-RSDDS-AUG/'
TRAIN_IMG_DIR = os.path.join(DATA_ROOT, 'Image_train')
TRAIN_DEPTH_DIR = os.path.join(DATA_ROOT, 'Depth_train')
TRAIN_GT_DIR = os.path.join(DATA_ROOT, 'GT_train')
OUTPUT_DIR = './hy-tmp/output/'
LOG_FILE = os.path.join(OUTPUT_DIR, 'result.log')
CHECKPOINT_PATH = os.path.join(OUTPUT_DIR, 'checkpoint.pth')
LOSS_PLOT_PATH = os.path.join(OUTPUT_DIR, 'train_loss.png')

# SAM Model
MODEL_TYPE = 'vit_b'
SAM_CHECKPOINT = 'work_dir_cod/SAM/sam_vit_b_01ec64.pth'

# Training Hyperparameters
DEVICE = 'cuda:0'
NUM_EPOCHS = 100
BATCH_SIZE = 8 # Restored to original hyperparameter
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 0

# --- Logger Setup ---
os.makedirs(OUTPUT_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --- Dataset Definition ---
class NEUDataset(Dataset):
    def __init__(self, image_dir, depth_dir, gt_dir, transform=None):
        self.image_dir = image_dir
        self.depth_dir = depth_dir
        self.gt_dir = gt_dir
        self.transform = transform
        self.image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.bmp')])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        img_name = self.image_files[index]
        base_name = os.path.splitext(img_name)[0]

        # File paths
        img_path = os.path.join(self.image_dir, img_name)
        depth_path = os.path.join(self.depth_dir, base_name + '.tiff')
        gt_path = os.path.join(self.gt_dir, base_name + '.png')

        # Load data
        image = np.array(Image.open(img_path).convert('RGB'))
        depth_16bit = io.imread(depth_path)
        gt2D = np.array(Image.open(gt_path).convert('L'))

        # Normalize depth data from 16-bit to 8-bit for model input
        if depth_16bit.dtype == np.uint16:
            depth_normalized = (depth_16bit / 65535.0 * 255.0).astype(np.uint8)
        else:
            # If not 16-bit, assume it's already in a compatible format (e.g., uint8)
            depth_normalized = depth_16bit.astype(np.uint8)

        if gt2D.max() > 0:
            gt2D = (gt2D / gt2D.max() * 255).astype(np.uint8)

        gt2D = (gt2D > 0).astype(np.uint8)

        # Generate bounding box from GT mask
        y_indices, x_indices = np.where(gt2D > 0)
        if len(y_indices) > 0:
            x_min, x_max = np.min(x_indices), np.max(x_indices)
            y_min, y_max = np.min(y_indices), np.max(y_indices)
            # Add perturbation
            H, W = gt2D.shape
            x_min = max(0, x_min - np.random.randint(0, 20))
            x_max = min(W, x_max + np.random.randint(0, 20))
            y_min = max(0, y_min - np.random.randint(0, 20))
            y_max = min(H, y_max + np.random.randint(0, 20))
            bbox = np.array([x_min, y_min, x_max, y_max])
        else:
            # Fallback for empty masks: use the whole image
            H, W = gt2D.shape
            bbox = np.array([0, 0, W, H])

        return image, depth_normalized, gt2D[None, :, :], torch.tensor(bbox).float()

# --- Main Training Script ---
def main():
    logger.info("--- Starting Training ---")

    # Prepare SAM model
    logger.info(f"Loading SAM model: {MODEL_TYPE}")
    sam_model = sam_model_registry[MODEL_TYPE](checkpoint=SAM_CHECKPOINT).to(DEVICE)
    sam_model.train()
    sam_trans = ResizeLongestSide(sam_model.image_encoder.img_size)

    # Optimizer and Loss
    optimizer = torch.optim.Adam(sam_model.mask_decoder.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    seg_loss = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')
    cwd_loss = CriterionCWD(norm_type='channel', divergence='kl', temperature=4.0)

    # Data Loading
    logger.info("Setting up dataset and dataloader...")
    train_dataset = NEUDataset(TRAIN_IMG_DIR, TRAIN_DEPTH_DIR, TRAIN_GT_DIR)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    losses = []
    logger.info(f"Starting training for {NUM_EPOCHS} epochs...")

    for epoch in range(NUM_EPOCHS):
        epoch_loss = 0
        pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")

        for step, (images, depths, gt2Ds, boxes) in enumerate(pbar):

            with torch.no_grad():
                # Preprocess images and depths for the encoder
                input_images = []
                input_depths = []
                for i in range(images.shape[0]):
                    # Preprocess RGB image
                    img_np = images[i].numpy().astype(np.uint8)
                    resized_img = sam_trans.apply_image(img_np)
                    resized_img_tensor = torch.as_tensor(resized_img.transpose(2, 0, 1)).to(DEVICE)
                    input_images.append(sam_model.preprocess(resized_img_tensor))

                    # Preprocess depth map (convert to 3-channel)
                    depth_np = depths[i].numpy()
                    if depth_np.ndim == 2:
                        depth_np_3c = np.repeat(depth_np[:, :, np.newaxis], 3, axis=2)
                    else:
                        depth_np_3c = depth_np
                    resized_depth = sam_trans.apply_image(depth_np_3c)
                    resized_depth_tensor = torch.as_tensor(resized_depth.transpose(2, 0, 1)).to(DEVICE)
                    input_depths.append(sam_model.preprocess(resized_depth_tensor))

                input_image_torch = torch.stack(input_images)
                input_depth_torch = torch.stack(input_depths)

                # Generate embeddings
                image_embedding = sam_model.image_encoder(input_image_torch)
                depth_embedding = sam_model.image_encoder(input_depth_torch)

                # Process bounding boxes
                box_torch = sam_trans.apply_boxes_torch(boxes, (gt2Ds.shape[-2], gt2Ds.shape[-1])).to(DEVICE)
                if len(box_torch.shape) == 2:
                    box_torch = box_torch[:, None, :]

                sparse_embeddings, dense_embeddings_box = sam_model.prompt_encoder(
                    points=None, boxes=box_torch, masks=None
                )

                # This part is for pvt_embedding, which seems to require a different input size
                # Using the resized image before SAM's specific preprocessing
                pvt_input_images = F.interpolate(input_image_torch, size=(1024, 1024), mode='bilinear', align_corners=False)
                pvt_embedding = sam_model.pvt(pvt_input_images)[3]


            # Forward pass through the decoder
            bc_embedding, pvt_64 = sam_model.BC(pvt_embedding)
            distill_loss = cwd_loss(bc_embedding, depth_embedding)
            hybrid_embedding = torch.cat([pvt_64, bc_embedding], dim=1)
            high_frequency = sam_model.DWT(hybrid_embedding)
            dense_embeddings, sparse_embeddings = sam_model.ME(dense_embeddings_box, high_frequency, sparse_embeddings)

            mask_predictions, _ = sam_model.mask_decoder(
                image_embeddings=image_embedding,
                image_pe=sam_model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
            )

            final_mask = sam_model.loop_finer(mask_predictions, depth_embedding, depth_embedding)
            mask_predictions = 0.1 * final_mask + 0.9 * mask_predictions

            # Calculate loss
            loss = 0.9 * seg_loss(mask_predictions, gt2Ds.to(DEVICE)) + 0.1 * distill_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix({"loss": loss.item()})

        epoch_loss /= (step + 1)
        losses.append(epoch_loss)
        logger.info(f'EPOCH: {epoch+1}, Loss: {epoch_loss}')

        # Save checkpoint, overwriting previous one
        if (epoch + 1) >= 80 and (epoch + 1) % 10 == 0:
            logger.info(f"Saving checkpoint at epoch {epoch+1} to {CHECKPOINT_PATH}")
            torch.save(sam_model.state_dict(), CHECKPOINT_PATH)

    # Plot and save loss curve
    plt.figure()
    plt.plot(losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig(LOSS_PLOT_PATH)
    plt.close()
    logger.info(f"Loss curve saved to {LOSS_PLOT_PATH}")

if __name__ == '__main__':
    main()
