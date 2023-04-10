import warnings

import numpy as np
import torch
import torch_em


class ClassificationTrainer(torch_em.trainer.DefaultTrainer):
    def _validate_impl(self, forward_context):
        self.model.eval()

        loss_val = 0.0

        # we use the syntax from sklearn.metrics to compute metrics
        # over all the preditions
        y_true, y_pred = [], []

        with torch.no_grad():
            for x, y in self.val_loader:
                x, y = x.to(self.device), y.to(self.device)
                with forward_context():
                    pred, loss = self._forward_and_loss(x, y)
                loss_val += loss.item()
                y_true.append(y.detach().cpu().numpy())
                y_pred.append(pred.max(1)[1].detach().cpu().numpy())

        if torch.isnan(pred).any():
            warnings.warn("Predictions are NaN")
        loss_val /= len(self.val_loader)

        y_true, y_pred = np.concatenate(y_true), np.concatenate(y_pred)
        metric_val = self.metric(y_true, y_pred)

        if self.logger is not None:
            self.logger.log_validation(self._iteration, metric_val, loss_val, x, y, pred, y_true, y_pred)
        return metric_val
