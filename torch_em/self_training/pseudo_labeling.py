import torch


class DefaultPseudoLabeler:
    """Compute pseudo labels.

    Parameters:
        activation [nn.Module, callable] - activation function applied to the teacher prediction.
        confidence_threshold [float] - threshold for computing a mask for filterign the pseudo labels.
            If none is given no mask will be computed (default: None)
        threshold_from_both_sides [bool] - whether to include both values bigger than the threshold
            and smaller than 1 - it, or only values bigger than it in the mask.
            The former should be used for binary labels, the latter for for multiclass labels (default: False)
    """
    def __init__(self, activation=None, confidence_threshold=None, threshold_from_both_sides=True):
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

    def __call__(self, teacher, input_):
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
    """Compute pseudo labels from the Probabilistic UNet.

    Parameters:
        activation [nn.Module, callable] - activation function applied to the teacher prediction.
        confidence_threshold [float] - threshold for computing a mask for filterign the pseudo labels.
            If none is given no mask will be computed (default: None)
        threshold_from_both_sides [bool] - whether to include both values bigger than the threshold
            and smaller than 1 - it, or only values bigger than it in the mask.
            The former should be used for binary labels, the latter for for multiclass labels (default: False)
        prior_samples [int] - the number of times we want to sample from the
            prior distribution per inputs (default: 16)
        consensus_masking [bool] - whether to activate consensus masking in the label filter (default: False)
            If false, the weighted consensus response (weighted per-pixel response) is returned
            If true, the masked consensus response (complete aggrement of pixels) is returned
    """
    def __init__(self, activation=None, confidence_threshold=None, threshold_from_both_sides=True,
                 prior_samples=16, consensus_masking=False):
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
        mask = [torch.where((sample >= upper_threshold) + (sample <= lower_threshold),
                            torch.tensor(1.),
                            torch.tensor(0.)) for sample in pseudo_labels]
        return mask

    def _compute_label_mask_one_side(self, pseudo_labels):
        mask = [torch.where((sample >= self.confidence_threshold),
                            torch.tensor(1.),
                            torch.tensor(0.)) for sample in pseudo_labels]
        return mask

    def __call__(self, teacher, input_):
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
