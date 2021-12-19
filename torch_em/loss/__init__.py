from .affinity_side_loss import AffinitySideLoss
from .combined_loss import CombinedLoss
from .contrastive import ContrastiveLoss
from .dice import DiceLoss, dice_score
from .wrapper import ApplyAndRemoveMask, LossWrapper

EMBEDDING_LOSSES = (
    ContrastiveLoss,
)
