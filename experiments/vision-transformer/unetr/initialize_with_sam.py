import torch
from torch_em.model.unetr import build_unetr_with_sam_initialization

# FIXME this doesn't work yet
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = build_unetr_with_sam_initialization(
    model_type="vit_h",
    checkpoint_path="/scratch/usr/nimanwai/models/segment-anything/checkpoints/sam_vit_h_4b8939.pth")
model.to(device)

x = torch.randn(1, 3, 1024, 1024).to(device=device)

y = model(x)
print(y.shape)
