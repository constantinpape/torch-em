

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
