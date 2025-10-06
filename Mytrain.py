import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import monai
from segment_anything import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide
from segment_anything.modeling.CWDLoss import CriterionCWD
from torch.nn import functional as F
from dataset import CustomDataset

# --- Configuration ---
torch.manual_seed(2024)
np.random.seed(2024)

# --- Path and Model Configuration ---
output_dir = '/hy-tmp/output'
log_file = os.path.join(output_dir, 'result.log')
model_save_path = os.path.join(output_dir, 'checkpoint.pth')
dataset_root = '../datasets/NEU-RSDDS-AUG'
model_type = 'vit_b'
checkpoint_path = 'work_dir_cod/SAM/sam_vit_b_01ec64.pth'
device = 'cuda' # Use 'cpu' for development, will be changed to 'cuda' for submission

# --- Setup Output and Logging ---
os.makedirs(output_dir, exist_ok=True)
sys.stdout = open(log_file, 'w')
sys.stderr = sys.stdout

print("--- Initializing Training ---")
print(f"Device: {device}")
print(f"Dataset Root: {dataset_root}")
print(f"Output Directory: {output_dir}")

# --- Model and Optimizer ---
# Note: In the final version, checkpoint will be loaded if it exists.
# For now, we initialize without it to allow running without the file.
sam_model = sam_model_registry[model_type](checkpoint=checkpoint_path).to(device)
sam_model.train()

optimizer = torch.optim.Adam(sam_model.mask_decoder.parameters(), lr=1e-5, weight_decay=0)
seg_loss = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')
cwd_loss = CriterionCWD(norm_type='channel', divergence='kl', temperature=4.0)

# --- Data Loading ---
try:
    train_dataset = CustomDataset(root_dir=dataset_root, train=True)
    # Use a smaller batch size for CPU development
    train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    print("Dataset loaded successfully.")
except Exception as e:
    print(f"Error loading dataset: {e}")
    sys.exit(1)

# --- Training Loop ---
num_epochs = 100
losses = []
sam_trans = ResizeLongestSide(sam_model.image_encoder.img_size)

print("--- Starting Training Loop ---")
for epoch in range(num_epochs):
    epoch_loss = 0

    for image, depth, gt2D in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        image = image.to(device)
        depth = depth.to(device)
        gt2D = gt2D.to(device)

        with torch.no_grad():
            # Generate embeddings
            image_embedding = sam_model.image_encoder(image)
            depth_embedding = sam_model.image_encoder(depth)

            # Generate bounding boxes
            gt2D_np = gt2D.cpu().numpy()
            batch_boxes = []
            for mask_np in gt2D_np:
                y_indices, x_indices = np.where(mask_np.squeeze() > 0)
                if len(y_indices) == 0:
                    H, W = mask_np.shape[-2:]
                    batch_boxes.append([0, 0, W, H])
                    continue

                x_min, x_max = np.min(x_indices), np.max(x_indices)
                y_min, y_max = np.min(y_indices), np.max(y_indices)
                H, W = mask_np.shape[-2:]
                x_min = max(0, x_min - np.random.randint(0, 20))
                x_max = min(W, x_max + np.random.randint(0, 20))
                y_min = max(0, y_min - np.random.randint(0, 20))
                y_max = min(H, y_max + np.random.randint(0, 20))
                batch_boxes.append([x_min, y_min, x_max, y_max])

            boxes = torch.tensor(batch_boxes, device=device)
            box_torch = sam_trans.apply_boxes_torch(boxes, gt2D.shape[-2:])

            sparse_embeddings, dense_embeddings_box = sam_model.prompt_encoder(
                points=None, boxes=box_torch, masks=None
            )

            # Generate PVT embedding from the RGB image
            pvt_embedding = sam_model.pvt(image)[3]

        # --- Forward Pass using the full architecture ---
        bc_embedding, pvt_64 = sam_model.BC(pvt_embedding)
        distill_loss = cwd_loss(bc_embedding, depth_embedding)

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

        # Combine original prediction with refined mask
        mask_predictions = 0.1 * final_mask + 0.9 * mask_predictions

        # Upsample prediction to match ground truth
        upsampled_masks = F.interpolate(
            mask_predictions, size=gt2D.shape[-2:], mode="bilinear", align_corners=False
        )

        # Calculate combined loss
        loss = 0.9 * seg_loss(upsampled_masks, gt2D) + 0.1 * distill_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    epoch_loss /= len(train_dataloader)
    losses.append(epoch_loss)
    print(f'EPOCH: {epoch+1}, Loss: {epoch_loss}')

    # Overwrite checkpoint at specified interval
    if epoch >= 80 and epoch % 10 == 0:
        print(f"Saving checkpoint at epoch {epoch+1}")
        torch.save(sam_model.state_dict(), model_save_path)

print("--- Training Complete ---")

# Plot and save loss curve
plt.figure()
plt.plot(losses)
plt.title('Training Loss Curve')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.savefig(os.path.join(output_dir, 'train_loss.png'))
plt.close()
print(f"Loss curve saved to {os.path.join(output_dir, 'train_loss.png')}")

# Close the log file
sys.stdout.close()
sys.stdout = sys.__stdout__
sys.stderr = sys.__stderr__