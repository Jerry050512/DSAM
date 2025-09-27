import numpy as np
import os
import logging
from tqdm import tqdm
import torch
import torch.nn.functional as F
from segment_anything import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide
from PIL import Image
import skimage.io as io

# --- Configuration ---
# Paths
DATA_ROOT = './datasets/NEU-RSDDS-AUG/'
TEST_IMG_DIR = os.path.join(DATA_ROOT, 'Image_test')
TEST_DEPTH_DIR = os.path.join(DATA_ROOT, 'Depth_test')
OUTPUT_DIR = './hy-tmp/output/'
PREDICTION_DIR = os.path.join(OUTPUT_DIR, 'predictions/')
LOG_FILE = os.path.join(OUTPUT_DIR, 'result.log')
CHECKPOINT_PATH = os.path.join(OUTPUT_DIR, 'checkpoint.pth')

# SAM Model
MODEL_TYPE = 'vit_b'

# Testing Hyperparameters
DEVICE = 'cuda:0'

# --- Logger Setup ---
# Ensure the logger appends to the existing log file from training
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def predict_mask(sam_model, image_np, depth_np):
    """
    Run model inference on a single image and depth map.
    """
    sam_trans = ResizeLongestSide(sam_model.image_encoder.img_size)
    original_height, original_width = image_np.shape[:2]

    # Preprocess RGB image
    resized_img = sam_trans.apply_image(image_np)
    resized_img_tensor = torch.as_tensor(resized_img.transpose(2, 0, 1)).to(DEVICE)
    input_image = sam_model.preprocess(resized_img_tensor[None, ...])

    # Preprocess depth map (convert to 3-channel if needed)
    if depth_np.ndim == 2:
        depth_np_3c = np.repeat(depth_np[:, :, np.newaxis], 3, axis=2)
    else:
        depth_np_3c = depth_np
    resized_depth = sam_trans.apply_image(depth_np_3c)
    resized_depth_tensor = torch.as_tensor(resized_depth.transpose(2, 0, 1)).to(DEVICE)
    input_depth = sam_model.preprocess(resized_depth_tensor[None, ...])

    with torch.no_grad():
        # --- Start of Forward Pass Logic (mirrors Mytrain.py) ---

        # Generate embeddings
        image_embedding = sam_model.image_encoder(input_image)
        depth_embedding = sam_model.image_encoder(input_depth)

        # Create a bounding box prompt for the whole image
        box_np = np.array([[0, 0, original_width, original_height]])
        box_torch = torch.as_tensor(box_np, dtype=torch.float, device=DEVICE)
        box_torch = sam_trans.apply_boxes_torch(box_torch, (original_height, original_width))
        if len(box_torch.shape) == 2:
            box_torch = box_torch[:, None, :]

        sparse_embeddings, dense_embeddings_box = sam_model.prompt_encoder(
            points=None, boxes=box_torch, masks=None
        )

        pvt_input_images = F.interpolate(input_image, size=(1024, 1024), mode='bilinear', align_corners=False)
        pvt_embedding = sam_model.pvt(pvt_input_images)[3]

        # Forward pass through the decoder
        bc_embedding, pvt_64 = sam_model.BC(pvt_embedding)
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

        # --- End of Forward Pass Logic ---

        # Upscale mask to the original image size
        upscaled_mask = sam_model.postprocess_masks(
            mask_predictions,
            input_size=input_image.shape[-2:],
            original_size=(original_height, original_width)
        ).to(DEVICE)

        # Convert to binary mask
        binary_mask = (torch.sigmoid(upscaled_mask) > 0.5).squeeze().cpu().numpy().astype(np.uint8)

    return binary_mask


def main():
    logger.info("--- Starting Testing ---")

    # Load Model
    logger.info(f"Loading model from {CHECKPOINT_PATH}")
    if not os.path.exists(CHECKPOINT_PATH):
        logger.error(f"Checkpoint not found at {CHECKPOINT_PATH}. Please run training first.")
        return

    sam_model = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH).to(DEVICE)
    sam_model.eval()

    # Create prediction directory if it doesn't exist
    os.makedirs(PREDICTION_DIR, exist_ok=True)

    test_files = sorted([f for f in os.listdir(TEST_IMG_DIR) if f.endswith('.bmp')])
    logger.info(f"Found {len(test_files)} images to test.")

    for img_name in tqdm(test_files, desc="Testing"):
        base_name = os.path.splitext(img_name)[0]
        img_path = os.path.join(TEST_IMG_DIR, img_name)
        depth_path = os.path.join(TEST_DEPTH_DIR, base_name + '.tiff')

        logger.info(f"Processing {img_name}...")

        if not os.path.exists(depth_path):
            logger.warning(f"Depth map not found for {img_name}, skipping.")
            continue

        # Load image and depth map
        image_np = np.array(Image.open(img_path).convert('RGB'))
        depth_16bit = io.imread(depth_path)

        # Normalize depth data from 16-bit to 8-bit for model input
        if depth_16bit.dtype == np.uint16:
            depth_normalized = (depth_16bit / 65535.0 * 255.0).astype(np.uint8)
        else:
            # If not 16-bit, assume it's already in a compatible format (e.g., uint8)
            depth_normalized = depth_16bit.astype(np.uint8)

        # Get prediction
        predicted_mask = predict_mask(sam_model, image_np, depth_normalized)

        # Save the mask as a PNG file
        mask_img = Image.fromarray(predicted_mask * 255)
        save_path = os.path.join(PREDICTION_DIR, base_name + '.png')
        mask_img.save(save_path)
        logger.info(f"Saved prediction to {save_path}")

    logger.info("--- Testing Finished ---")

if __name__ == '__main__':
    main()