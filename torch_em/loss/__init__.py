from .affinity_side_loss import AffinitySideLoss
from .combined_loss import CombinedLoss
from .contrastive import ContrastiveLoss
from .dice import DiceLoss, dice_score
from .spoco_loss import SPOCOLoss
from .wrapper import ApplyAndRemoveMask, ApplyMask, LossWrapper, MaskIgnoreLabel

EMBEDDING_LOSSES = (
    ContrastiveLoss, SPOCOLoss
)
