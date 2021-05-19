from .contrastive import ContrastiveLoss
from .dice import DiceLoss, dice_score
from .wrapper import ApplyAndRemoveMask, LossWrapper

EMBEDDING_LOSSES = (
    ContrastiveLoss,
)
