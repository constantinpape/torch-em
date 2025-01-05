"""Neural network architectures for classifcation, segmentation or image-to-image translation tasks.
"""

from .unet import AnisotropicUNet, UNet2d, UNet3d
from .probabilistic_unet import ProbabilisticUNet
from .unetr import UNETR
from .vit import get_vision_transformer
from .vim import get_vimunet_model
