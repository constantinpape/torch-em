import torch
from torch_em.model.unetr import build_unetr_with_sam_intialization

# FIXME this doesn't work yet
model = build_unetr_with_sam_intialization()
x = torch.randn(1, 3, 1024, 1024)

y = model(x)
print(y.shape)
