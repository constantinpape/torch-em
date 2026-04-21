import time

import torch
from torch_em.self_training.mean_teacher import MeanTeacherTrainerWithInvertibleAugmentations

import os
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


class UniMatchv2Trainer(MeanTeacherTrainerWithInvertibleAugmentations):
    """
    Trainer for semi-supervised learning and domain adaptation following the UniMatch v2 framework.
    """

    def __init__(
        self, complementary_dropout, **kwargs
    ):
        super().__init__(**kwargs)
        self.complementary_dropout = complementary_dropout

        self.teacher.eval()

    def unetr_decoder_prediction(self, model, features, input_shape, original_shape):

        z9 = model.deconv1(features)
        z6 = model.deconv2(z9)
        z3 = model.deconv3(z6)
        z0 = model.deconv4(z3)

        updated_from_encoder = [z9, z6, z3]

        x = model.base(features)
        x = model.decoder(x, encoder_inputs=updated_from_encoder)
        x = model.deconv_out(x)

        x = torch.cat([x, z0], dim=1)
        x = model.decoder_head(x)

        x = model.out_conv(x)
        if model.final_activation is not None:
            x = model.final_activation(x)

        x = model.postprocess_masks(x, input_shape, original_shape)
        return x

    def predict_with_comp_drop(self, model, input_):
        batch_size = input_.shape[0]
        original_shape = input_.shape[2:]

        x, input_shape = model.preprocess(input_)

        if len(original_shape) == 2:
            encoder_output = model.encoder(x)
            if isinstance(encoder_output[-1], list):
                features, _ = encoder_output
            else:
                features = encoder_output
        if len(original_shape) == 3:
            depth = input_.shape[-3]
            features = torch.stack([model.encoder(x[:, :, i])[0] for i in range(depth)], dim=2)

        features_dim = features.shape[1]

        binom = torch.distributions.binomial.Binomial(probs=0.5)

        dropout_mask1 = binom.sample((int(batch_size/2), features_dim)).to(input_.device) * 2.0
        if len(original_shape) == 2:
            dropout_mask1 = dropout_mask1.unsqueeze(-1).unsqueeze(-1)
        if len(original_shape) == 3:
            dropout_mask1 = dropout_mask1.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

        dropout_mask2 = 2.0 - dropout_mask1
        dropout_mask = torch.cat([dropout_mask1, dropout_mask2])

        # NOTE: in the UniMatch v2 code some samples of the batch stay unchanged!
        # Keep some samples unchanged (code block not tested)
        # dropout_prob = 0.5
        # num_kept = int(batch_size * (1 - dropout_prob))
        # kept_indexes = torch.randperm(batch_size, device=input_.device)[:num_kept]

        # dropout_mask1[kept_indexes, :] = 1.0
        # dropout_mask2[kept_indexes, :] = 1.0

        dropped_features = features * dropout_mask

        pred = self.unetr_decoder_prediction(model, dropped_features, input_shape, original_shape)

        return pred

    def _train_epoch_unsupervised(
        self, progress, forward_context, backprop
    ):
        self.model.train()

        n_iter = 0
        t_per_iter = time.time()

        for x_u in self.unsupervised_train_loader:

            x_u = x_u.to(self.device, non_blocking=True)

            x_u_w = self.augmenter.weak.transform(x_u)
            x_u_s1, x_u_s2 = self.augmenter.strong1.transform(x_u), self.augmenter.strong2.transform(x_u)

            # Compute the pseudo labels (unsupervised teacher prediction)
            with forward_context(), torch.no_grad():
                pseudo_labels, label_filter = self.pseudo_labeler(self.teacher, x_u_w)
                pseudo_labels_inv = self.augmenter.weak.reverse_transform(pseudo_labels)
                label_filter_inv = self.augmenter.weak.reverse_transform(label_filter)

            # Perform unsupervised training
            with forward_context():
                if self.complementary_dropout:
                    pred_s1, pred_s2 = self.predict_with_comp_drop(self.model, torch.cat((x_u_s1, x_u_s2))).chunk(2)
                else:
                    pred_s1, pred_s2 = self.model(torch.cat((x_u_s1, x_u_s2))).chunk(2)
                pred_s1_inv = self.augmenter.strong1.reverse_transform(pred_s1)
                pred_s2_inv = self.augmenter.strong2.reverse_transform(pred_s2)
                unsupervised_loss = self.unsupervised_loss(
                    torch.stack((pred_s1_inv, pred_s2_inv)),
                    pseudo_labels_inv,
                    label_filter_inv,
                    pred_dim=2,
                )

            backprop(unsupervised_loss)

            if self.logger is not None:
                self.logger.log_train_unsupervised(
                    self._iteration,
                    unsupervised_loss,
                    x_u,
                    pred_s1_inv,
                    pred_s2_inv,
                    pseudo_labels_inv,
                    label_filter_inv,
                )
                self.logger.log_train_augmentations(
                    self._iteration,
                    x_u_w,
                    x_u_s1,
                    x_u_s2,
                    pseudo_labels,
                    pred_s1,
                    pred_s2,
                )

                lr = [pm["lr"] for pm in self.optimizer.param_groups][0]
                self.logger.log_lr(self._iteration, lr)
                if self.pseudo_labeler.confidence_threshold is not None:
                    self.logger.log_ct(self._iteration, self.pseudo_labeler.confidence_threshold)

            with torch.no_grad():
                self._momentum_update()  # EMA update of the teacher

            self._iteration += 1
            n_iter += 1
            if self._iteration >= self.max_iteration:
                break
            progress.update(1)

            self.augmenter.reset_all()

        t_per_iter = (time.time() - t_per_iter) / n_iter
        return t_per_iter

    def _train_epoch_semisupervised(
        self, progress, forward_context, backprop
    ):
        train_loader = zip(self.supervised_train_loader, self.unsupervised_train_loader)
        self.model.train()

        n_iter = 0
        t_per_iter = time.time()

        for i, ((x_s, y_s), x_u) in enumerate(train_loader):

            x_s, y_s = x_s.to(self.device, non_blocking=True), y_s.to(self.device, non_blocking=True)
            x_u = x_u.to(self.device, non_blocking=True)

            x_u_w = self.augmenter.weak.transform(x_u)
            x_u_s1, x_u_s2 = self.augmenter.strong1.transform(x_u), self.augmenter.strong2.transform(x_u)

            self.optimizer.zero_grad()
            # supervised loss (supervised student prediction)
            pred_s = self.model(x_s)
            supervised_loss = self.supervised_loss(pred_s, y_s)

            backprop(supervised_loss)

            # Compute the pseudo labels (unsupervised teacher prediction)
            with forward_context(), torch.no_grad():
                pseudo_labels, label_filter = self.pseudo_labeler(self.teacher, x_u_w)
                pseudo_labels_inv = self.augmenter.weak.reverse_transform(pseudo_labels)
                label_filter_inv = self.augmenter.weak.reverse_transform(label_filter)

            # Perform unsupervised training
            with forward_context():
                # if self.complementary_dropout:
                #     pred_s1, pred_s2 = self.predict_with_comp_drop(self.model, torch.cat((x_u_s1, x_u_s2))).chunk(2)
                # else:
                #     pred_s1, pred_s2 = self.model(torch.cat((x_u_s1, x_u_s2))).chunk(2)
                pred_s1 = self.model(x_u_s1)
                pred_s2 = pred_s1
                pred_s1_inv = self.augmenter.strong1.reverse_transform(pred_s1)
                pred_s2_inv = self.augmenter.strong2.reverse_transform(pred_s2)
                unsupervised_loss = self.unsupervised_loss(
                    torch.stack((pred_s1_inv, pred_s2_inv)),
                    pseudo_labels_inv,
                    label_filter_inv,
                    pred_dim=2,
                )

            backprop(unsupervised_loss)

            if self.logger is not None:
                self.logger.log_train_supervised(
                    self._iteration, supervised_loss, x_s, y_s, pred_s
                )
                self.logger.log_train_unsupervised(
                    self._iteration,
                    unsupervised_loss,
                    x_u,
                    pred_s1_inv,
                    pred_s2_inv,
                    pseudo_labels_inv,
                    label_filter_inv,
                )
                self.logger.log_train_augmentations(
                    self._iteration,
                    x_u_w,
                    x_u_s1,
                    x_u_s2,
                    pseudo_labels,
                    pred_s1,
                    pred_s2,
                )

                lr = [pm["lr"] for pm in self.optimizer.param_groups][0]
                self.logger.log_lr(self._iteration, lr)
                if self.pseudo_labeler.confidence_threshold is not None:
                    self.logger.log_ct(self._iteration, self.pseudo_labeler.confidence_threshold)

            with torch.no_grad():
                self._momentum_update()  # EMA update of the teacher

            self._iteration += 1
            n_iter += 1
            if self._iteration >= self.max_iteration:
                break
            progress.update(1)

            self.augmenter.reset_all()

        t_per_iter = (time.time() - t_per_iter) / n_iter
        return t_per_iter

    def _validate_supervised(self, forward_context):
        metric_val = 0.0
        loss_val = 0.0

        for x, y in self.supervised_val_loader:
            x, y = (
                x.to(self.device, non_blocking=True),
                y.to(self.device, non_blocking=True)
            )

            with forward_context():
                pred = self.model(x)
                loss, metric = self.supervised_loss_and_metric(pred, y)
                loss_val += loss.item()
            metric_val += metric.item()

        metric_val /= len(self.supervised_val_loader)
        loss_val /= len(self.supervised_val_loader)

        if self.logger is not None:
            self.logger.log_validation_supervised(
                self._iteration, metric_val, loss_val, x, y, pred
            )

        return metric_val

    def _validate_unsupervised(self, forward_context):
        metric_val = 0.0
        loss_val = 0.0

        for x in self.unsupervised_val_loader:
            x = x.to(self.device, non_blocking=True)

            # apply augmentations
            x_w = self.augmenter.weak.transform(x)
            x_s1, x_s2 = self.augmenter.strong1.transform(x), self.augmenter.strong2.transform(x)

            # Compute the pseudo labels (unsupervised teacher prediction)
            with forward_context():
                pseudo_labels, label_filter = self.pseudo_labeler(self.teacher, x_w)
                pseudo_labels_inv = self.augmenter.weak.reverse_transform(pseudo_labels)
                label_filter_inv = self.augmenter.weak.reverse_transform(label_filter)

                if self.complementary_dropout:
                    pred_s1, pred_s2 = self.predict_with_comp_drop(self.model, torch.cat((x_s1, x_s2))).chunk(2)
                else:
                    pred_s1, pred_s2 = self.model(torch.cat((x_s1, x_s2))).chunk(2)
                pred_s1_inv = self.augmenter.strong1.reverse_transform(pred_s1)
                pred_s2_inv = self.augmenter.strong2.reverse_transform(pred_s2)

                loss, metric = self.unsupervised_loss_and_metric(
                    torch.stack((pred_s1_inv, pred_s2_inv)),
                    pseudo_labels_inv,
                    label_filter_inv,
                    pred_dim=2,
                )
            loss_val += loss.item()
            metric_val += metric.item()

            self.augmenter.reset_all()

        metric_val /= len(self.unsupervised_val_loader)
        loss_val /= len(self.unsupervised_val_loader)

        if self.logger is not None:
            self.logger.log_validation_unsupervised(
                self._iteration,
                metric_val,
                loss_val,
                x,
                pred_s1_inv,
                pred_s2_inv,
                pseudo_labels_inv,
                label_filter_inv,
            )

            self.logger.log_validation_augmentations(
                self._iteration,
                x_w,
                x_s1,
                x_s2,
                pseudo_labels,
                pred_s1,
                pred_s2,
            )

        self.pseudo_labeler.step(metric_val, self._epoch)

        return metric_val

    def _validate_impl(self, forward_context):
        self.model.eval()

        with torch.no_grad():

            if self.supervised_val_loader is None:
                supervised_metric = None
            else:
                supervised_metric = self._validate_supervised(forward_context)

            if self.unsupervised_val_loader is None:
                unsupervised_metric = None
            else:
                unsupervised_metric = self._validate_unsupervised(forward_context)

        if unsupervised_metric is None:
            metric = supervised_metric
        elif supervised_metric is None:
            metric = unsupervised_metric
        else:
            metric = (supervised_metric + unsupervised_metric) / 2

        return metric
