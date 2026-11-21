import torch

from .default_trainer import DefaultTrainer


class HybridUNETRDeepSupervisionTrainer(DefaultTrainer):
    """Train HybridUNETR with the three auxiliary heads of the `cagd_3d` decoder.

    Import this class from `torch_em.trainer.hybrid_unetr_trainer` and pass it as
    `trainer_class` to `torch_em.default_segmentation_trainer`. The same loss and
    target are used for each prediction. Set `deep_supervision=True` in the model's
    `decoder_kwargs`. Validation uses the main prediction.
    """

    aux_weights = (0.4, 0.3, 0.2)  # MedCAGD weights, deepest head first.

    def _forward_and_loss(self, x, y):
        if not self.model.training:
            return super()._forward_and_loss(x, y)

        outputs = self.model(x)
        if isinstance(outputs, torch.Tensor):
            raise ValueError("HybridUNETRDeepSupervisionTrainer requires deep_supervision=True.")
        prediction, auxiliary_outputs = outputs
        if len(auxiliary_outputs) != len(self.aux_weights):
            raise ValueError("Expected one loss weight per auxiliary prediction.")

        if self._iteration % self.log_image_interval == 0 and prediction.requires_grad:
            prediction.retain_grad()

        loss = self.loss(prediction, y)
        for weight, auxiliary in zip(self.aux_weights, auxiliary_outputs):
            loss = loss + weight * self.loss(auxiliary, y)
        return prediction, loss
