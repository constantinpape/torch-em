from typing import Optional

import torch
import torch_em
import torch.nn as nn
from torch_em.loss import DiceLoss


class DefaultSelfTrainingLoss(nn.Module):
    """Loss function for self training.

    This loss takes as input a model and its input, as well as (pseudo) labels and potentially
    a mask for the labels. It then runs prediction with the model and compares the outputs
    to the (pseudo) labels using an internal loss function. Typically, the labels are derived
    from the predictions of a teacher model, and the model passed is the student model.

    Args:
        loss: The internal loss function to use for comparing predictions of the teacher and student model.
        activation: The activation function to be applied to the prediction before passing it to the loss.
    """
    def __init__(self, loss: nn.Module = torch_em.loss.DiceLoss(), activation: Optional[nn.Module] = None):
        super().__init__()
        self.activation = activation
        self.loss = loss
        # TODO serialize the class names and kwargs instead
        self.init_kwargs = {}

    def __call__(
        self, model: nn.Module, input_: torch.Tensor, labels: torch.Tensor, label_filter: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute the loss for self-training.

        Args:
            model: The model.
            input_: The model inputs for this batch.
            labels: The (pseudo) labels for this batch.
            label_filter: A mask to exclude from the loss computation.

        Returns:
            The loss value.
        """
        prediction = model(input_)
        if self.activation is not None:
            prediction = self.activation(prediction)
        if label_filter is None:
            loss = self.loss(prediction, labels)
        else:
            loss = self.loss(prediction * label_filter, labels * label_filter)
        return loss


class DefaultSelfTrainingLossAndMetric(nn.Module):
    """Loss and metric function for self training.

    Similar to `DefaultSelfTrainingLoss`, but computes loss and metric value in one call
    to avoid running prediction with the model twice.

    Args:
        loss: The internal loss function to use for comparing predictions of the teacher and student model.
        metric: The internal metric function to use for comparing predictions of the teacher and student model.
        activation: The activation function to be applied to the prediction before passing it to the loss.
    """
    def __init__(
        self,
        loss: nn.Module = torch_em.loss.DiceLoss(),
        metric: nn.Module = torch_em.loss.DiceLoss(),
        activation: Optional[nn.Module] = None
    ):
        super().__init__()
        self.activation = activation
        self.loss = loss
        self.metric = metric
        # TODO serialize the class names and dicts instead
        self.init_kwargs = {}

    def __call__(self, model, input_, labels, label_filter=None):
        prediction = model(input_)
        if self.activation is not None:
            prediction = self.activation(prediction)
        if label_filter is None:
            loss = self.loss(prediction, labels)
        else:
            loss = self.loss(prediction * label_filter, labels * label_filter)
        metric = self.metric(prediction, labels)
        return loss, metric


def _l2_regularisation(m):
    l2_reg = None
    for W in m.parameters():
        if l2_reg is None:
            l2_reg = W.norm(2)
        else:
            l2_reg = l2_reg + W.norm(2)
    return l2_reg


class ProbabilisticUNetLoss(nn.Module):
    """Training loss for ProbabilisticUNet.

    Computes the ELBO loss: reconstruction term plus beta-weighted KL divergence,
    with L2 regularisation on the posterior, prior, and fcomb weights.
    Labels are sliced to model.output_channels to support multi-rater inputs.

    Args:
        loss: Reserved. Must be None; the ELBO objective is always used.
    """
    def __init__(self, loss: Optional[nn.Module] = None) -> None:
        super().__init__()
        self.loss = loss

    def __call__(
        self, model: nn.Module, input_: torch.Tensor, labels: torch.Tensor, label_filter: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        model(input_, labels)
        labels = labels[:, :model.output_channels, ...]

        if self.loss is None:
            elbo = model.elbo(labels, label_filter)
            reg_loss = (
                _l2_regularisation(model.posterior)
                + _l2_regularisation(model.prior)
                + _l2_regularisation(model.fcomb.layers)
            )
            loss = -elbo + 1e-5 * reg_loss
        else:
            raise NotImplementedError("Custom loss is not supported; pass loss=None to use the ELBO.")

        return loss


class ProbabilisticUNetLossAndMetric(nn.Module):
    """Training loss and validation metric for ProbabilisticUNet.

    Computes the ELBO loss and a sample-averaged Dice metric in a single forward pass.
    Draws prior_samples segmentation hypotheses, averages them, and evaluates against labels.
    Labels are sliced to model.output_channels to support multi-rater inputs.

    Args:
        loss: Reserved. Must be None; the ELBO objective is always used.
        metric: Metric function applied to averaged prior samples vs. labels.
        activation: Activation applied to prior samples before metric computation.
        prior_samples: Number of prior samples to average for the metric.
    """
    def __init__(
        self,
        loss: Optional[nn.Module] = None,
        metric: nn.Module = DiceLoss(),
        activation: Optional[nn.Module] = torch.nn.Sigmoid(),
        prior_samples: int = 16,
    ) -> None:
        super().__init__()
        self.activation = activation
        self.metric = metric
        self.loss = loss
        self.prior_samples = prior_samples

    def __call__(
        self, model: nn.Module, input_: torch.Tensor, labels: torch.Tensor, label_filter: Optional[torch.Tensor] = None
    ):
        model(input_, labels)
        labels = labels[:, :model.output_channels, ...]

        if self.loss is None:
            elbo = model.elbo(labels, label_filter)
            reg_loss = (
                _l2_regularisation(model.posterior)
                + _l2_regularisation(model.prior)
                + _l2_regularisation(model.fcomb.layers)
            )
            loss = -elbo + 1e-5 * reg_loss
        else:
            raise NotImplementedError("Custom loss is not supported; pass loss=None to use the ELBO.")

        samples_per_distribution = []
        for _ in range(self.prior_samples):
            samples = model.sample()
            if self.activation is not None:
                samples = self.activation(samples)
            samples_per_distribution.append(samples)

        avg_samples = torch.stack(samples_per_distribution, dim=0).mean(dim=0)
        metric = self.metric(avg_samples, labels)

        return loss, metric


class SelfTrainingLossWithInvertibleAugmentations(nn.Module):
    """Loss function for self-training with invertible augmentations.

    Variant of `DefaultSelfTrainingLoss` for use with `FixMatchTrainerWithInvertibleAugmentations`
    and `MeanTeacherTrainerWithInvertibleAugmentations`. Unlike `DefaultSelfTrainingLoss`, this loss
    receives pre-computed predictions directly rather than a model and input, because the trainer
    already applies the model and invertible augmentations before calling the loss.

    Args:
        loss: The internal loss function used to compare student predictions to pseudo-labels.
        activation: Optional activation applied to the prediction before the loss.
    """
    def __init__(self, loss: nn.Module = torch_em.loss.DiceLoss(), activation: Optional[nn.Module] = None):
        super().__init__()
        self.activation = activation
        self.loss = loss
        # TODO serialize the class names and kwargs instead
        self.init_kwargs = {}

    def __call__(
        self,
        prediction: torch.Tensor,
        labels: torch.Tensor,
        label_filter: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute the self-training loss.

        Args:
            prediction: Student model predictions, already mapped to the reference frame
                via the inverse augmentation transform.
            labels: The (pseudo) labels, mapped to the same reference frame.
            label_filter: Optional mask or weight tensor. Where provided, both prediction
                and labels are multiplied by this tensor before the loss is computed.

        Returns:
            The loss value.
        """

        if self.activation is not None:
            prediction = self.activation(prediction)
        if label_filter is None:
            loss = self.loss(prediction, labels)
        else:
            loss = self.loss(prediction * label_filter, labels * label_filter)
        return loss


class SelfTrainingLossAndMetricWithInvertibleAugmentations(nn.Module):
    """Loss and metric function for self-training with invertible augmentations.

    Variant of `DefaultSelfTrainingLossAndMetric` for use with
    `FixMatchTrainerWithInvertibleAugmentations` and `MeanTeacherTrainerWithInvertibleAugmentations`.
    Computes both loss and metric in a single call from pre-computed predictions, avoiding a
    second forward pass. Used during validation where the trainer already holds the predictions.

    Args:
        loss: The internal loss function used to compare student predictions to pseudo-labels.
        metric: The internal metric function used to evaluate student predictions against pseudo-labels.
        activation: Optional activation applied to the prediction before the loss and metric.
    """
    def __init__(
        self,
        loss: nn.Module = torch_em.loss.DiceLoss(),
        metric: nn.Module = torch_em.loss.DiceLoss(),
        activation: Optional[nn.Module] = None
    ):
        super().__init__()
        self.activation = activation
        self.loss = loss
        self.metric = metric
        # TODO serialize the class names and dicts instead
        self.init_kwargs = {}

    def __call__(
        self,
        prediction: torch.Tensor,
        labels: torch.Tensor,
        label_filter: Optional[torch.Tensor] = None,
    ):
        """Compute the self-training loss.

        Args:
            prediction: Student model predictions, already mapped to the reference frame
                via the inverse augmentation transform.
            labels: The (pseudo) labels, mapped to the same reference frame.
            label_filter: Optional mask or weight tensor. Where provided, both prediction
                and labels are multiplied by this tensor before the loss is computed.

        Returns:
            The loss and metric value.
        """
        if self.activation is not None:
            prediction = self.activation(prediction)
        if label_filter is None:
            loss = self.loss(prediction, labels)
        else:
            loss = self.loss(prediction * label_filter, labels * label_filter)
        metric = self.metric(prediction, labels)
        return loss, metric


class UniMatchv2Loss(nn.Module):
    """Loss function for `UniMatchv2Trainer`.

    Extends `SelfTrainingLossWithInvertibleAugmentations` to support the two-student-view scheme
    of UniMatch v2. When `pred_dim=2`, `prediction` is expected to be a stacked tensor of two
    student predictions `[pred_s1_inv, pred_s2_inv]`, and the loss is averaged over both views.
    When `pred_dim=1`, it falls back to the standard single-prediction behaviour.

    Args:
        loss: The internal loss function used to compare student predictions to pseudo-labels.
        activation: Optional activation applied to the predictions before the loss.
    """
    def __init__(self, loss: nn.Module = DiceLoss(), activation: Optional[nn.Module] = None):
        super().__init__()
        self.activation = activation
        self.loss = loss
        self.init_kwargs = {}

    def __call__(
        self,
        prediction: torch.Tensor,
        labels: torch.Tensor,
        label_filter: Optional[torch.Tensor] = None,
        pred_dim: int = 1,
    ) -> torch.Tensor:
        """Compute the UniMatch v2 self-training loss.

        Args:
            prediction: Student predictions mapped to the reference frame. When `pred_dim=2`,
                a stacked tensor of shape `(2, B, C, ...)` holding the two strong-view predictions.
                When `pred_dim=1`, a standard `(B, C, ...)` prediction tensor.
            labels: The (pseudo) labels, mapped to the same reference frame.
            label_filter: Optional mask or weight tensor applied to both prediction and labels
                before the loss is computed.
            pred_dim: Number of student views. Use `2` for the standard UniMatch v2 dual-view
                training and `1` for single-view inference or validation.

        Returns:
            The loss value.
        """

        assert pred_dim in (1, 2), "pred_dim must be either 1 or 2"

        if self.activation is not None:
            prediction = self.activation(prediction)

        if pred_dim == 2:
            if label_filter is None:
                loss = (self.loss(prediction[0], labels) + self.loss(prediction[1], labels)) / 2
            else:
                loss = (self.loss(
                    prediction[0] * label_filter, labels * label_filter
                ) + self.loss(prediction[1] * label_filter, labels * label_filter)) / 2
            return loss

        else:
            if label_filter is None:
                loss = self.loss(prediction, labels)
            else:
                loss = self.loss(prediction * label_filter, labels * label_filter)
            return loss


class UniMatchv2LossAndMetric(nn.Module):
    """Loss and metric function for `UniMatchv2Trainer`.

    Extends `SelfTrainingLossAndMetricWithInvertibleAugmentations` to support the two-student-view scheme
    of UniMatch v2. `pred_dim` depends on how many views the student model processes
    at the same time. Supports the same dual-view `pred_dim=2` convention: when two student
    predictions are stacked, loss and metric are each averaged over both views.
    When `pred_dim=1`, it falls back to the standard single-prediction behaviour.

    Args:
        loss: The internal loss function used to compare student predictions to pseudo-labels.
        metric: The internal metric function used to evaluate student predictions against pseudo-labels.
        activation: Optional activation applied to the predictions before the loss and metric.
    """
    def __init__(
        self,
        loss: nn.Module = DiceLoss(),
        metric: nn.Module = DiceLoss(),
        activation: Optional[nn.Module] = None
    ):
        super().__init__()
        self.activation = activation
        self.loss = loss
        self.metric = metric
        self.init_kwargs = {}

    def __call__(
        self,
        prediction: torch.Tensor,
        labels: torch.Tensor,
        label_filter: Optional[torch.Tensor] = None,
        pred_dim: int = 1,
    ):
        """Compute the UniMatch v2 self-training loss.

        Args:
            prediction: Student predictions mapped to the reference frame. When `pred_dim=2`,
                a stacked tensor of shape `(2, B, C, ...)` holding the two strong-view predictions.
                When `pred_dim=1`, a standard `(B, C, ...)` prediction tensor.
            labels: The (pseudo) labels, mapped to the same reference frame.
            label_filter: Optional mask or weight tensor applied to both prediction and labels
                before the loss is computed.
            pred_dim: Number of student views. Use `2` for the standard UniMatch v2 dual-view
                training and `1` for single-view inference or validation.

        Returns:
            The loss and metric value.
        """

        if self.activation is not None:
            prediction = self.activation(prediction)

        assert pred_dim in (1, 2), "pred_dim must be either 1 or 2"

        if pred_dim == 2:
            assert len(prediction) == 2, "only implemented for list of len 2"
            if label_filter is None:
                loss = (self.loss(prediction[0], labels) + self.loss(prediction[1], labels)) / 2
            else:
                loss = (self.loss(
                    prediction[0] * label_filter, labels * label_filter
                ) + self.loss(prediction[1] * label_filter, labels * label_filter)) / 2
            metric = (self.metric(prediction[0], labels) + self.metric(prediction[1], labels)) / 2
            return loss, metric

        else:
            if label_filter is None:
                loss = self.loss(prediction, labels)
            else:
                loss = self.loss(prediction * label_filter, labels * label_filter)
            metric = self.metric(prediction, labels)
            return loss, metric
