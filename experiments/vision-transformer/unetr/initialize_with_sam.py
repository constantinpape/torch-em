import torch
from torch_em.model import UNETR

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = UNETR(encoder="vit_h", out_channels=1,
              encoder_checkpoint_path="/scratch/usr/nimanwai/models/segment-anything/checkpoints/sam_vit_h_4b8939.pth")
model.to(device)

x = torch.randn(1, 3, 1024, 1024).to(device=device)

y = model(x)
print(y.shape)
