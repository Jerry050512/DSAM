import numpy as np
import os
import sys
import torch
from segment_anything import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide
from tqdm import tqdm
import cv2
from torch.utils.data import DataLoader
from dataset import CustomDataset
from torch.nn import functional as F

# --- Configuration ---
torch.manual_seed(2024)
np.random.seed(2024)

# --- Path and Model Configuration ---
output_dir = '/hy-tmp/output'
log_file = os.path.join(output_dir, 'result.log')
predictions_dir = os.path.join(output_dir, 'predictions')
model_load_path = os.path.join(output_dir, 'checkpoint.pth')
dataset_root = '../datasets/NEU-RSDDS-AUG'
model_type = 'vit_b'
device = 'cuda' # Use 'cpu' for development, will be changed to 'cuda' for submission

# --- Setup Output and Logging ---
os.makedirs(predictions_dir, exist_ok=True)
sys.stdout = open(log_file, 'a')
sys.stderr = sys.stdout

print("\n--- Initializing Testing ---")
print(f"Device: {device}")
print(f"Dataset Root: {dataset_root}")
print(f"Predictions Directory: {predictions_dir}")

# --- Model ---
sam_model = sam_model_registry[model_type]().to(device)
try:
    sam_model.load_state_dict(torch.load(model_load_path, map_location=device))
    print("Model loaded successfully from checkpoint.")
except Exception as e:
    print(f"Could not load model from checkpoint: {e}. Using randomly initialized model.")
sam_model.eval()

# --- Data Loading ---
try:
    test_dataset = CustomDataset(root_dir=dataset_root, train=False)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    print("Test dataset loaded successfully.")
except Exception as e:
    print(f"Error loading test dataset: {e}")
    sys.exit(1)

# --- Testing Loop ---
sam_trans = ResizeLongestSide(sam_model.image_encoder.img_size)

print("--- Starting Testing Loop ---")
with torch.no_grad():
    for image, depth, img_name, original_size in tqdm(test_dataloader, desc="Testing"):
        image = image.to(device)
        depth = depth.to(device)
        img_name = img_name[0]
        original_h, original_w = original_size[0].item(), original_size[1].item()

        # --- Forward pass using the full architecture ---
        image_embedding = sam_model.image_encoder(image)
        depth_embedding = sam_model.image_encoder(depth)

        # Use a full-image bounding box as the prompt
        box = torch.tensor([[0, 0, 1024, 1024]], device=device)
        box_torch = box
        # box_torch = sam_trans.apply_boxes_torch(box, (original_h, original_w))

        sparse_embeddings, dense_embeddings_box = sam_model.prompt_encoder(
            points=None, boxes=box_torch, masks=None
        )

        pvt_embedding = sam_model.pvt(image)[3]
        bc_embedding, pvt_64 = sam_model.BC(pvt_embedding)

        hybrid_embedding = torch.cat([pvt_64, bc_embedding], dim=1)
        high_frequency = sam_model.DWT(hybrid_embedding)

        dense_embeddings, sparse_embeddings = sam_model.ME(
            dense_embeddings_box, high_frequency, sparse_embeddings
        )

        mask_predictions, _ = sam_model.mask_decoder(
            image_embeddings=image_embedding,
            image_pe=sam_model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )

        final_mask = sam_model.loop_finer(mask_predictions, depth_embedding, depth_embedding)
        seg_prob = 0.1 * final_mask + 0.9 * mask_predictions

        # Upsample, convert to binary mask, and resize to original dimensions
        upsampled_mask = F.interpolate(
            seg_prob,
            size=(original_h, original_w),
            mode="bilinear",
            align_corners=False,
        )

        binary_mask = (torch.sigmoid(upsampled_mask) > 0.5).cpu().numpy().squeeze().astype(np.uint8)

        # Save the final prediction
        save_path = os.path.join(predictions_dir, os.path.splitext(img_name)[0] + '.png')
        cv2.imwrite(save_path, binary_mask * 255)

print("--- Testing Complete ---")
print(f"Predictions saved to: {predictions_dir}")

# Close the log file
sys.stdout.close()
sys.stdout = sys.__stdout__
sys.stderr = sys.__stderr__