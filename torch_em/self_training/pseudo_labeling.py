from typing import Optional

import torch


class DefaultPseudoLabeler:
    """Compute pseudo labels based on model predictions, typically from a teacher model.

    Args:
        activation: Activation function applied to the teacher prediction.
        confidence_threshold: Threshold for computing a mask for filtering the pseudo labels.
            If None is given no mask will be computed.
        threshold_from_both_sides: Whether to include both values bigger than the threshold
            and smaller than 1 - the thrhesold, or only values bigger than the threshold, in the mask.
            The former should be used for binary labels, the latter for for multiclass labels.
    """
    def __init__(
        self,
        activation: Optional[torch.nn.Module] = None,
        confidence_threshold: Optional[float] = None,
        threshold_from_both_sides: bool = True,
    ):
        self.activation = activation
        self.confidence_threshold = confidence_threshold
        self.threshold_from_both_sides = threshold_from_both_sides
        # TODO serialize the class names and kwargs for activation instead
        self.init_kwargs = {
            "activation": None, "confidence_threshold": confidence_threshold,
            "threshold_from_both_sides": threshold_from_both_sides
        }

    def _compute_label_mask_both_sides(self, pseudo_labels):
        upper_threshold = self.confidence_threshold
        lower_threshold = 1.0 - self.confidence_threshold
        mask = ((pseudo_labels >= upper_threshold) + (pseudo_labels <= lower_threshold)).to(dtype=torch.float32)
        return mask

    def _compute_label_mask_one_side(self, pseudo_labels):
        mask = (pseudo_labels >= self.confidence_threshold)
        return mask

    def __call__(self, teacher: torch.nn.Module, input_: torch.Tensor) -> torch.Tensor:
        """Compute pseudo-labels.

        Args:
            teacher: The teacher model.
            input_: The input for this batch.

        Returns:
            The pseudo-labels.
        """
        pseudo_labels = teacher(input_)
        if self.activation is not None:
            pseudo_labels = self.activation(pseudo_labels)
        if self.confidence_threshold is None:
            label_mask = None
        else:
            label_mask = self._compute_label_mask_both_sides(pseudo_labels) if self.threshold_from_both_sides\
                else self._compute_label_mask_one_side(pseudo_labels)
        return pseudo_labels, label_mask


class ProbabilisticPseudoLabeler:
    """Compute pseudo labels from a Probabilistic UNet.

    Args:
        activation: Activation function applied to the teacher prediction.
        confidence_threshold: Threshold for computing a mask for filterign the pseudo labels.
            If none is given no mask will be computed.
        threshold_from_both_sides: Whether to include both values bigger than the threshold
            and smaller than 1 - the thrhesold, or only values bigger than the threshold, in the mask.
            The former should be used for binary labels, the latter for for multiclass labels.
        prior_samples: The number of times to sample from the model distribution per input.
        consensus_masking: Whether to activate consensus masking in the label filter.
            If False, the weighted consensus response (weighted per-pixel response) is returned.
            If True, the masked consensus response (complete aggrement of pixels) is returned.
    """
    def __init__(
        self,
        activation: Optional[torch.nn.Module] = None,
        confidence_threshold: Optional[float] = None,
        threshold_from_both_sides: bool = True,
        prior_samples: int = 16,
        consensus_masking: bool = False,
    ):
        self.activation = activation
        self.confidence_threshold = confidence_threshold
        self.threshold_from_both_sides = threshold_from_both_sides
        self.prior_samples = prior_samples
        self.consensus_masking = consensus_masking
        # TODO serialize the class names and kwargs for activation instead
        self.init_kwargs = {
            "activation": None, "confidence_threshold": confidence_threshold,
            "threshold_from_both_sides": threshold_from_both_sides
        }

    def _compute_label_mask_both_sides(self, pseudo_labels):
        upper_threshold = self.confidence_threshold
        lower_threshold = 1.0 - self.confidence_threshold
        mask = [
            torch.where((sample >= upper_threshold) + (sample <= lower_threshold), torch.tensor(1.), torch.tensor(0.))
            for sample in pseudo_labels
        ]
        return mask

    def _compute_label_mask_one_side(self, pseudo_labels):
        mask = [
            torch.where((sample >= self.confidence_threshold), torch.tensor(1.), torch.tensor(0.))
            for sample in pseudo_labels
        ]
        return mask

    def __call__(self, teacher: torch.nn.Module, input_: torch.Tensor) -> torch.Tensor:
        """Compute pseudo-labels.

        Args:
            teacher: The teacher model. Must be a `torch_em.model.probabilistic_unet.ProbabilisticUNet`.
            input_: The input for this batch.

        Returns:
            The pseudo-labels.
        """
        teacher.forward(input_)
        if self.activation is not None:
            pseudo_labels = [self.activation(teacher.sample()) for _ in range(self.prior_samples)]
        else:
            pseudo_labels = [teacher.sample() for _ in range(self.prior_samples)]
        pseudo_labels = torch.stack(pseudo_labels, dim=0).sum(dim=0)/self.prior_samples

        if self.confidence_threshold is None:
            label_mask = None
        else:
            label_mask = self._compute_label_mask_both_sides(pseudo_labels) if self.threshold_from_both_sides \
                else self._compute_label_mask_one_side(pseudo_labels)
            label_mask = torch.stack(label_mask, dim=0).sum(dim=0)/self.prior_samples
            if self.consensus_masking:
                label_mask = torch.where(label_mask == 1, 1, 0)

        return pseudo_labels, label_mask
