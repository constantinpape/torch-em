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


# TODO: The probabilistic U-Net related code should be refactored to `torch_em.loss`
# and should be documented properly.


def l2_regularisation(m):
    """@private
    """
    l2_reg = None
    for W in m.parameters():
        if l2_reg is None:
            l2_reg = W.norm(2)
        else:
            l2_reg = l2_reg + W.norm(2)
    return l2_reg


class ProbabilisticUNetLoss(nn.Module):
    """@private
    """
    # """Loss function for Probabilistic UNet

    # Args:
    #     # TODO : Implement a generic utility function for all Probabilistic UNet schemes (ELBO, GECO, etc.)
    #     loss [nn.Module] - the loss function to be used. (default: None)
    # """
    def __init__(self, loss=None):
        super().__init__()
        self.loss = loss

    def __call__(self, model, input_, labels, label_filter=None):
        model.forward(input_, labels)

        if self.loss is None:
            elbo = model.elbo(labels, label_filter)
            reg_loss = l2_regularisation(model.posterior) + l2_regularisation(model.prior) + \
                l2_regularisation(model.fcomb.layers)
            loss = -elbo + 1e-5 * reg_loss

        return loss


class ProbabilisticUNetLossAndMetric(nn.Module):
    """@private
    """
    # """Loss and metric function for Probabilistic UNet.

    # Args:
    #     # TODO : Implement a generic utility function for all Probabilistic UNet schemes (ELBO, GECO, etc.)
    #     loss [nn.Module] - the loss function to be used. (default: None)

    #     metric [nn.Module] - the metric function to be used. (default: torch_em.loss.DiceLoss)
    #     activation [nn.Module, callable] - the activation function to be applied to the prediction
    #         before evaluating the average predictions. (default: None)
    # """
    def __init__(self, loss=None, metric=DiceLoss(), activation=torch.nn.Sigmoid(), prior_samples=16):
        super().__init__()
        self.activation = activation
        self.metric = metric
        self.loss = loss
        self.prior_samples = prior_samples

    def __call__(self, model, input_, labels, label_filter=None):
        model.forward(input_, labels)

        if self.loss is None:
            elbo = model.elbo(labels, label_filter)
            reg_loss = l2_regularisation(model.posterior) + l2_regularisation(model.prior) + \
                l2_regularisation(model.fcomb.layers)
            loss = -elbo + 1e-5 * reg_loss

        samples_per_distribution = []
        for _ in range(self.prior_samples):
            samples = model.sample(testing=False)
            if self.activation is not None:
                samples = self.activation(samples)
            samples_per_distribution.append(samples)

        avg_samples = torch.stack(samples_per_distribution, dim=0).sum(dim=0) / len(samples_per_distribution)
        metric = self.metric(avg_samples, labels)

        return loss, metric
