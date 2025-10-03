import torch
import os
from segment_anything import sam_model_registry

# Parameters
model_type = 'vit_b'
device = 'cpu'
output_dir = './hy-tmp/output'
checkpoint_path = os.path.join(output_dir, 'checkpoint.pth')
os.makedirs(output_dir, exist_ok=True)

# Initialize model
print(f"Initializing model: {model_type}")
sam_model = sam_model_registry[model_type](checkpoint=None).to(device)

# Save model state
print(f"Saving dummy checkpoint to: {checkpoint_path}")
torch.save(sam_model.state_dict(), checkpoint_path)
print("Dummy checkpoint created successfully.")