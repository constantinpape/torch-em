"""@private
"""

# This code is based on:
#   1. The original TensorFlow implementation: https://github.com/SimonKohl/probabilistic_unet
#   2. PyTorch adaptation from https://github.com/stefanknegt/Probabilistic-Unet-Pytorch
#   3. PhiSeg's benchmarking script from https://github.com/annawundram/glaucoma-diagnosis-pipeline.

from typing import Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Independent, kl

from torch_em.model import UNet2d
from torch_em.model.unet import get_norm_layer
from torch_em.loss.dice import DiceLossWithLogits


def init_weights(m: nn.Module) -> None:
    """@private
    """
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        nn.init.trunc_normal_(m.bias, std=0.001)


def init_weights_orthogonal_normal(m: nn.Module) -> None:
    """@private
    """
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        nn.init.orthogonal_(m.weight)
        nn.init.trunc_normal_(m.bias, std=0.001)


class Encoder(nn.Module):
    """
    A convolutional neural network, consisting of len(num_filters) times a block of no_convs_per_block
    convolutional layers, after each block a pooling operation is performed.
    And after each convolutional layer a non-linear (ReLU) activation function is applied.
    """
    def __init__(
        self,
        input_channels: int,
        num_filters: List[int],
        no_convs_per_block: int,
        posterior: bool = False,
        num_classes: Optional[int] = None,
        norm: Optional[str] = None,
    ):
        super().__init__()

        self.input_channels = input_channels
        self.num_filters = num_filters

        if posterior:
            # To accommodate for the mask concatenated at the channel axis, increase input_channels.
            assert num_classes is not None
            self.input_channels += num_classes

        layers = []
        output_dim = None

        for i in range(len(self.num_filters)):
            # First block: input_channels -> num_filters[i]; subsequent: prev_output -> num_filters[i].
            input_dim = self.input_channels if i == 0 else output_dim
            output_dim = num_filters[i]

            if i != 0:
                layers.append(nn.AvgPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=True))

            layers.append(nn.Conv2d(input_dim, output_dim, kernel_size=3))
            if norm is not None:
                layers.append(get_norm_layer(norm, 2, output_dim))
            layers.append(nn.ReLU(inplace=True))

            for _ in range(no_convs_per_block - 1):
                layers.append(nn.Conv2d(output_dim, output_dim, kernel_size=3))
                if norm is not None:
                    layers.append(get_norm_layer(norm, 2, output_dim))
                layers.append(nn.ReLU(inplace=True))

        self.layers = nn.Sequential(*layers)
        self.layers.apply(init_weights)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.layers(input)


class AxisAlignedConvGaussian(nn.Module):
    """A convolutional network that parametrizes a Gaussian distribution with axis aligned covariance matrix.
    """
    def __init__(
        self,
        input_channels: int,
        num_filters: List[int],
        no_convs_per_block: int,
        latent_dim: int,
        posterior: bool = False,
        num_classes: Optional[int] = None,
        use_onehot: bool = False,
        norm: Optional[str] = None,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.use_onehot = use_onehot

        self.encoder = Encoder(
            input_channels,
            num_filters,
            no_convs_per_block,
            posterior=posterior,
            num_classes=num_classes,
            norm=norm,
        )

        self.conv_layer = nn.Conv2d(num_filters[-1], 2 * self.latent_dim, (1, 1), stride=1)
        # Orthogonal init + truncated-normal bias per the original paper's training details.
        nn.init.orthogonal_(self.conv_layer.weight, gain=1)
        nn.init.trunc_normal_(self.conv_layer.bias, std=0.001)

    def forward(self, input: torch.Tensor, segm: Optional[torch.Tensor] = None) -> Independent:
        # Posterior: encode segm and concatenate with image along channel axis.
        # One-hot encodes class-index labels to remove spurious ordinal relationships.
        # Centering (- 0.5) keeps inputs zero-mean in both the one-hot and raw binary cases.
        if segm is not None:
            if self.use_onehot:
                segm = F.one_hot(segm.squeeze(1).long(), self.num_classes).permute(0, 3, 1, 2).float() - 0.5
            else:
                segm = segm.float() - 0.5
            input = torch.cat((input, segm), dim=1)

        encoding = self.encoder(input)

        # Global average pool to (B, C, 1, 1), then squeeze spatial dims to (B, C).
        encoding = encoding.mean(dim=(2, 3), keepdim=True)
        mu_log_sigma = self.conv_layer(encoding).squeeze(3).squeeze(2)

        mu = mu_log_sigma[:, :self.latent_dim]
        log_sigma = mu_log_sigma[:, self.latent_dim:]

        # This is a multivariate normal with diagonal covariance matrix sigma
        # https://github.com/pytorch/pytorch/pull/11178
        dist = Independent(Normal(loc=mu, scale=torch.exp(log_sigma)), 1)
        return dist


class Fcomb(nn.Module):
    """A sequence of 1x1 convolutions that combines the UNet feature map with a sample from the latent space
    by broadcasting z to spatial size and concatenating along the channel axis.
    """
    def __init__(
        self,
        num_filters: List[int],
        latent_dim: int,
        num_output_channels: int,
        no_convs_fcomb: int,
    ):
        super().__init__()

        layers = []
        layers.append(nn.Conv2d(num_filters[0] + latent_dim, num_filters[0], kernel_size=1))
        layers.append(nn.ReLU(inplace=True))

        for _ in range(no_convs_fcomb - 2):
            layers.append(nn.Conv2d(num_filters[0], num_filters[0], kernel_size=1))
            layers.append(nn.ReLU(inplace=True))

        self.layers = nn.Sequential(*layers)
        self.last_layer = nn.Conv2d(num_filters[0], num_output_channels, kernel_size=1)

        self.layers.apply(init_weights_orthogonal_normal)
        self.last_layer.apply(init_weights_orthogonal_normal)

    def forward(self, feature_map: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Z is (batch_size x latent_dim) and feature_map is (batch_size x no_channels x H x W).
        Broadcast Z to batch_size x latent_dim x H x W and concatenate with the feature map.
        """
        H, W = feature_map.shape[2], feature_map.shape[3]
        z = z.view(z.shape[0], z.shape[1], 1, 1).expand(-1, -1, H, W)
        feature_map = torch.cat((feature_map, z), dim=1)
        return self.last_layer(self.layers(feature_map))


class ProbabilisticUNet(nn.Module):
    """This network implementation for the Probabilistic UNet of Kohl et al. (https://arxiv.org/abs/1806.05034).
    This generative segmentation heuristic uses UNet combined with a conditional variational
    autoencoder enabling to efficiently produce an unlimited number of plausible hypotheses.

    For multi-class segmentation (output_channels > 1), the posterior encoder automatically receives
    a one-hot encoded label with output_channels classes. For binary segmentation (output_channels == 1),
    the posterior receives num_raters raw binary channels centered at -0.5.

    Args:
        input_channels: Number of channels in the image (1 for grayscale and 3 for RGB). The default is set to 1.
        output_channels: Number of channels to predict. The default is set to 1.
        num_raters: Number of annotators providing binary labels (only used when output_channels == 1).
            The default is set to 1.
        num_filters: Number of filters per encoder level. The default is set to [32, 64, 128, 192].
        latent_dim: Dimension of the latent space. The default is set to 6.
        no_convs_per_block: Number of convolutions per block in the prior/posterior encoder. The default is set to 3.
        no_convs_fcomb: Number of convolutions in the feature combination module. The default is set to 4.
        norm: Normalization type for the prior and posterior encoder. The default is set to 'InstanceNorm'.
        beta: Weighting factor for the KL divergence term in the ELBO (loss = reconstruction + beta * KL).
            beta=1.0 is the standard VAE objective with no extra regularization. The original paper
            used beta=10.0 for the LIDC-IDRI multi-rater chest X-ray task after task-specific tuning,
            which over-regularizes the latent space for most other tasks. Set beta > 1 to encourage a
            more structured latent space at the cost of reconstruction quality, or beta < 1 to
            prioritize reconstruction. The default is set to 1.0.
        consensus_masking: Whether to apply consensus masking in the reconstruction loss. The default is set to False.
        rl_swap: Whether to use dice loss instead of BCE/CE for reconstruction. The default is set to False.
        device: Device to place the model on. The default is set to None.
    """

    def __init__(
        self,
        input_channels: int = 1,
        output_channels: int = 1,
        num_raters: int = 1,
        num_filters: List[int] = [32, 64, 128, 192],
        latent_dim: int = 6,
        no_convs_per_block: int = 3,
        no_convs_fcomb: int = 4,
        norm: Optional[str] = "InstanceNorm",
        beta: float = 1.0,
        consensus_masking: bool = False,
        rl_swap: bool = False,
        device: Optional[torch.device] = None,
    ):
        super().__init__()

        self.input_channels = input_channels
        self.output_channels = output_channels
        self.num_raters = num_raters
        self.num_filters = num_filters
        self.latent_dim = latent_dim
        self.no_convs_per_block = no_convs_per_block
        self.no_convs_fcomb = no_convs_fcomb
        self.beta = beta
        self.consensus_masking = consensus_masking
        self.rl_swap = rl_swap
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device is None else device

        self.unet = UNet2d(
            in_channels=self.input_channels,
            out_channels=None,
            depth=len(self.num_filters),
            initial_features=num_filters[0],
            norm=norm,
        ).to(self.device)

        self.prior = AxisAlignedConvGaussian(
            self.input_channels,
            self.num_filters,
            self.no_convs_per_block,
            self.latent_dim,
            norm=norm,
        ).to(self.device)

        # Multi-class: one-hot encode the class-index label with output_channels classes.
        # Binary: concatenate num_raters raw binary channels centered at -0.5.
        use_onehot = output_channels > 1
        posterior_seg_channels = output_channels if use_onehot else num_raters
        self.posterior = AxisAlignedConvGaussian(
            self.input_channels,
            self.num_filters,
            self.no_convs_per_block,
            self.latent_dim,
            posterior=True,
            num_classes=posterior_seg_channels,
            use_onehot=use_onehot,
            norm=norm,
        ).to(self.device)

        self.fcomb = Fcomb(
            self.num_filters,
            self.latent_dim,
            self.output_channels,
            self.no_convs_fcomb,
        ).to(self.device)

    def _check_shape(self, patch: torch.Tensor) -> None:
        spatial_shape = tuple(patch.shape)[2:]
        depth = len(self.num_filters)
        factor = [2**depth] * len(spatial_shape)
        if any(sh % fac != 0 for sh, fac in zip(spatial_shape, factor)):
            msg = f"Invalid shape for Probabilistic U-Net: {spatial_shape} is not divisible by {factor}"
            raise ValueError(msg)

    def forward(self, patch: torch.Tensor, segm: Optional[torch.Tensor] = None) -> None:
        """Construct prior latent space for patch and run patch through UNet,
        in case training is True also construct posterior latent space
        """
        self._check_shape(patch)

        if segm is not None:
            self.posterior_latent_space = self.posterior.forward(patch, segm)

        self.prior_latent_space = self.prior.forward(patch)
        self.unet_features = self.unet.forward(patch)

    def sample(self) -> torch.Tensor:
        """Sample a segmentation from the prior and decode it through the UNet feature map.

        Draws a latent vector z from the prior p(z|x) and passes it through fcomb together
        with the UNet features to produce one segmentation hypothesis. Calling this N times
        yields N diverse hypotheses for the same input image.
        """
        z_prior = self.prior_latent_space.sample()
        return self.fcomb.forward(self.unet_features, z_prior)

    def reconstruct(
        self,
        use_posterior_mean: bool = False,
        calculate_posterior: bool = False,
        z_posterior: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Reconstruct a segmentation from a posterior sample (decoding a posterior sample) and UNet feature map
        use_posterior_mean: use posterior_mean instead of sampling z_q
        calculate_posterior: use a provided sample or sample from posterior latent space
        """
        if use_posterior_mean:
            z_posterior = self.posterior_latent_space.mean
        else:
            if calculate_posterior:
                z_posterior = self.posterior_latent_space.rsample()
        return self.fcomb.forward(self.unet_features, z_posterior)

    def kl_divergence(
        self,
        analytic: bool = True,
        calculate_posterior: bool = False,
        z_posterior: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Calculate the KL divergence between the posterior and prior KL(Q||P)
        analytic: calculate KL analytically or via sampling from the posterior
        calculate_posterior: if we use samapling to approximate KL we can sample here or supply a sample
        """
        if analytic:
            # Neeed to add this to torch source code, see: https://github.com/pytorch/pytorch/issues/13545
            kl_div = kl.kl_divergence(self.posterior_latent_space, self.prior_latent_space)
        else:
            if calculate_posterior:
                z_posterior = self.posterior_latent_space.rsample()
            log_posterior_prob = self.posterior_latent_space.log_prob(z_posterior)
            log_prior_prob = self.prior_latent_space.log_prob(z_posterior)
            kl_div = log_posterior_prob - log_prior_prob
        return kl_div

    def elbo(
        self,
        segm: torch.Tensor,
        consm: Optional[torch.Tensor] = None,
        analytic_kl: bool = True,
        reconstruct_posterior_mean: bool = False,
    ) -> torch.Tensor:
        """Calculate the evidence lower bound of the log-likelihood of P(Y|X)
        consm: consensus response
        """

        # Reconstruction criterion:
        #   - DiceLossWithLogits when rl_swap=True (returns a scalar directly)
        #   - BCE for single output channel (Bernoulli log-likelihood)
        #   - CE for multi-class (Categorical log-likelihood)
        if self.rl_swap:
            criterion = DiceLossWithLogits()
            use_bce, use_dice = False, True
        else:
            if self.output_channels == 1:
                criterion = nn.BCEWithLogitsLoss(reduction="none")
                use_bce, use_dice = True, False
            else:
                criterion = nn.CrossEntropyLoss(reduction="none")
                use_bce, use_dice = False, False

        z_posterior = self.posterior_latent_space.rsample()

        kl_term = torch.mean(
            self.kl_divergence(analytic=analytic_kl, calculate_posterior=False, z_posterior=z_posterior)
        )

        reconstruction = self.reconstruct(
            use_posterior_mean=reconstruct_posterior_mean,
            calculate_posterior=False,
            z_posterior=z_posterior
        )

        # Squeeze trailing channel dim: CE needs (B, H, W) long, BCE needs (B, 1, H, W) float.
        segm_t = segm.float() if use_bce else segm.squeeze(1).long()
        if self.consensus_masking is True and consm is not None:
            consm_t = consm if use_bce else consm.squeeze(1)
            target = (segm_t * consm_t) if use_bce else (segm_t * consm_t.long())
            reconstruction_loss = criterion(reconstruction, target)
        else:
            reconstruction_loss = criterion(reconstruction, segm_t)

        if use_dice:
            # DiceLossWithLogits already returns a scalar
            return -(reconstruction_loss + self.beta * kl_term)

        # Sum over spatial dims, mean over batch - keeps loss scale independent of batch size.
        reconstruction_loss = reconstruction_loss.sum(dim=tuple(range(1, reconstruction_loss.dim()))).mean()
        return -(reconstruction_loss + self.beta * kl_term)
